import Foundation
import MLX

// Exact port of mlx_vlm/models/kernels.py `bicubic_interpolate` (the _HAS_METAL
// path). PyTorch-style non-antialiased bicubic: a = -0.75, half-pixel coordinate
// mapping, align_corners = false, with border weight-sum renormalization. This
// reproduces the reference pos-emb interpolation bit-for-bit (mlx-swift's
// Upsample(.cubic) differs at the grid borders).

enum MetalBicubic {

    private static let header = """
        float cubic_kernel_a(float x, float a) {
            float absx = fabs(x);
            float absx2 = absx * absx;
            float absx3 = absx2 * absx;
            if (absx <= 1.0f) {
                return (a + 2.0f) * absx3 - (a + 3.0f) * absx2 + 1.0f;
            } else if (absx < 2.0f) {
                return a * absx3 - 5.0f * a * absx2 + 8.0f * a * absx - 4.0f * a;
            }
            return 0.0f;
        }

        float cubic_kernel(float x) {
            return cubic_kernel_a(x, -0.75f);
        }

        float cubic_kernel_antialias(float x, float scale) {
            return cubic_kernel_a(x / scale, -0.5f);
        }
    """

    private static let source = """
        uint x_out = thread_position_in_grid.x;
        uint y_out = thread_position_in_grid.y;
        uint bc_idx = thread_position_in_grid.z;

        int batch_size = dims[0];
        int channels = dims[1];
        int in_h = dims[2];
        int in_w = dims[3];
        int out_h = dims[4];
        int out_w = dims[5];

        float scale_h = params[0];
        float scale_w = params[1];
        bool align_corners = params[2] > 0.5f;
        bool use_antialias = params[3] > 0.5f;
        float filter_scale_h = params[4];
        float filter_scale_w = params[5];
        float support = params[6];

        if (x_out >= (uint)out_w || y_out >= (uint)out_h || bc_idx >= (uint)(batch_size * channels))
            return;

        int c = bc_idx % channels;
        int b = bc_idx / channels;

        float x_in, y_in;
        if (align_corners && out_w > 1 && out_h > 1) {
            x_in = float(x_out) * (in_w - 1) / (out_w - 1);
            y_in = float(y_out) * (in_h - 1) / (out_h - 1);
        } else {
            x_in = ((float(x_out) + 0.5f) / float(out_w)) * float(in_w) - 0.5f;
            y_in = ((float(y_out) + 0.5f) / float(out_h)) * float(in_h) - 0.5f;
        }

        float support_h = use_antialias ? support * filter_scale_h : support;
        float support_w = use_antialias ? support * filter_scale_w : support;

        int y_start = int(floor(y_in - support_h)) + 1;
        int y_end = int(floor(y_in + support_h)) + 1;
        int x_start = int(floor(x_in - support_w)) + 1;
        int x_end = int(floor(x_in + support_w)) + 1;

        y_start = max(0, y_start);
        y_end = min(in_h, y_end);
        x_start = max(0, x_start);
        x_end = min(in_w, x_end);

        float result = 0.0f;
        float weight_sum = 0.0f;

        for (int y_pos = y_start; y_pos < y_end; y_pos++) {
            float dy = float(y_pos) - y_in;
            float wy = use_antialias ?
                cubic_kernel_antialias(dy, filter_scale_h) :
                cubic_kernel(dy);

            for (int x_pos = x_start; x_pos < x_end; x_pos++) {
                float dx = float(x_pos) - x_in;
                float wx = use_antialias ?
                    cubic_kernel_antialias(dx, filter_scale_w) :
                    cubic_kernel(dx);

                float weight = wy * wx;
                int input_offset = ((b * channels + c) * in_h + y_pos) * in_w + x_pos;
                result += input[input_offset] * weight;
                weight_sum += weight;
            }
        }

        if (weight_sum > 1e-8f) {
            result /= weight_sum;
        }

        int output_offset = ((b * channels + c) * out_h + y_out) * out_w + x_out;
        output[output_offset] = result;
    """

    private static let kernel = MLXFast.metalKernel(
        name: "bicubic_interpolation_antialias",
        inputNames: ["input", "dims", "params"],
        outputNames: ["output"],
        source: source,
        header: header)

    /// Bicubic interpolate `x` (NCHW: [B, C, H, W]) to (outH, outW).
    /// Computes in float32 and casts back to the input dtype, matching the Python
    /// reference (`result.astype(input_dtype)`).
    static func interpolate(_ x: MLXArray, outH: Int, outW: Int) -> MLXArray {
        let inputDType = x.dtype
        let b = x.dim(0)
        let c = x.dim(1)
        let inH = x.dim(2)
        let inW = x.dim(3)
        let scaleH = Float(outH) / Float(inH)
        let scaleW = Float(outW) / Float(inW)

        let xFlat = x.asType(.float32).reshaped([-1])
        let dims = MLXArray([Int32(b), Int32(c), Int32(inH), Int32(inW), Int32(outH), Int32(outW)])
        let params = MLXArray([scaleH, scaleW, 0.0, 0.0, 1.0, 1.0, 2.0] as [Float])

        let (tgW, tgH) = threadgroup(outW: outW, outH: outH)
        let out = kernel(
            [xFlat, dims, params],
            grid: (outW, outH, b * c),
            threadGroup: (tgW, tgH, 1),
            outputShapes: [[b * c * outH * outW]],
            outputDTypes: [.float32])
        return out[0].reshaped([b, c, outH, outW]).asType(inputDType)
    }

    /// Mirror of kernels.py get_optimal_threadgroup (result is threadgroup-invariant).
    private static func threadgroup(outW: Int, outH: Int) -> (Int, Int) {
        func largestPow2(_ n: Int) -> Int {
            var w = 1 << (Int.bitWidth - 1 - n.leadingZeroBitCount)
            if w > n { w /= 2 }
            return max(1, w)
        }
        var width = largestPow2(min(1024, outW))
        var height = largestPow2(min(1024, outH))
        while width * height > 1024 {
            if width >= height { width /= 2 } else { height /= 2 }
        }
        return (max(8, width), max(8, height))
    }
}
