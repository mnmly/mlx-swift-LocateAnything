// swift-tools-version: 6.1
// Swift 6 language mode: complete strict-concurrency checking for every target.
import PackageDescription

let package = Package(
    name: "mlx-swift-LocateAnything",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
    ],
    products: [
        .library(name: "MLXLocateAnything", targets: ["MLXLocateAnything"]),
        .executable(name: "locateanything-cli", targets: ["locateanything-cli"]),
    ],
    dependencies: [
        // Reuse the shared VLM/LLM infrastructure (KVCache with trim, generation,
        // tokenizer loading, quantized weight loading, MediaProcessing).
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", .upToNextMinor(from: "3.32.3")),
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.32.3")),
        .package(url: "https://github.com/huggingface/swift-huggingface", from: "0.9.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.3.0"),
    ],
    targets: [
        .target(
            name: "MLXLocateAnything",
            dependencies: [
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "Sources/MLXLocateAnything"
        ),
        .executableTarget(
            name: "locateanything-cli",
            dependencies: [
                "MLXLocateAnything",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Sources/locateanything-cli"
        ),
        .testTarget(
            name: "MLXLocateAnythingTests",
            dependencies: ["MLXLocateAnything"],
            path: "Tests/MLXLocateAnythingTests"
        ),
    ]
)

// Pull in swift-docc-plugin only when generating documentation.
if Context.environment["SPI_GENERATE_DOCS"] == "1" || Context.environment["BUILD_DOC"] == "1" {
    package.dependencies.append(
        .package(url: "https://github.com/swiftlang/swift-docc-plugin", from: "1.4.3")
    )
}
