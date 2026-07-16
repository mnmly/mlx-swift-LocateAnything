# ``MLXLocateAnything``

Fast visual grounding with NVIDIA's LocateAnything model, ported to Swift/MLX for on-device inference on Apple Silicon.

## Overview

Give the model an image and a natural-language instruction; it returns bounding
boxes (or points) for the described objects — multi-object detection, referring
grounding, GUI-element grounding, text localization, and pointing. The full graph
(MoonViT vision tower + Qwen2 decoder + connector) plus **Parallel Box Decoding**
(atomic single-forward box prediction) are ported and verified against the Python
`mlx-vlm` reference.

``LocateAnythingSession`` is the one entry point both the CLI and the SwiftUI demo
consume. It takes a materialized `CGImage` (never a file path), so a sandboxed GUI
behaves identically to an unsandboxed CLI.

```swift
import MLXLocateAnything

// Downloads the 8-bit weights to ~/.cache/huggingface on first use.
let session = try await LocateAnythingSession.fromPretrained()

let image = try LocateAnythingImageProcessor.loadCGImage(path: "street.jpg")
let result = try session.predict(
    image: image,
    question: "Locate all the instances that matches the following description: car.")

for box in result.boxes {
    print(box.label ?? "?", box.boxPixels)   // pixel [x1, y1, x2, y2]
}
```

Coordinate tokens `<0>`…`<1000>` encode normalized `0…1000` coordinates; a
``DetectedBox`` exposes both normalized (`box`) and pixel (`boxPixels`) rectangles.

## Topics

### Running inference

- ``LocateAnythingSession``
- ``LocateAnythingResult``
- ``DetectedBox``
- ``PBDMode``

### Loading the model

- ``LocateAnythingSession/fromPretrained(repoId:)``
- ``LocateAnythingSession/download(repoId:progress:)``
- ``LocateAnythingSession/load(directory:)``
- ``LocateAnythingSession/cachedDirectory(repoId:)``

### Preprocessing & prompts

- ``LocateAnythingImageProcessor``
- ``ProcessedImage``
- ``LocateAnythingPrompt``

### Model architecture

- ``LocateAnythingModel``
- ``VisionModel``
- ``LanguageModel``
- ``PBDDecoder``

### Configuration

- ``LocateAnythingConfiguration``
- ``TextConfiguration``
- ``VisionConfiguration``
- ``QuantizationConfig``

### Weights & parity

- ``LocateAnythingWeights``
- ``LocateAnythingParity``
- ``LocateAnythingError``
```
