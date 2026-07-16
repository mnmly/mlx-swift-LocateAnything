# mlx-swift-LocateAnything

Swift/MLX port of NVIDIA's LocateAnything visual-grounding VLM (MoonViT vision +
Qwen2 decoder + connector + Parallel Box Decoding). Verified against the Python
`mlx-vlm` reference.

## Layout

- `Sources/MLXLocateAnything/` — the library. `LocateAnythingSession` is the single
  shared driver both frontends consume (swift-cli-gui-shared-driver): it takes a
  materialized `CGImage`, never a file path.
- `Sources/locateanything-cli/` — CLI (`predict` / `bench` / `parity` subcommands).
- `Examples/LocateAnythingDemo/` — SwiftUI macOS app (Xcode project), consumes the
  same `LocateAnythingSession`.

## Build & test

- Library / CLI: `swift build` (Release: `swift build -c release`).
- Runtime (MLX Metal): run the binary or `xcodebuild` — `swift test` can't load
  MLX's `default.metallib`, so MLX-dependent tests run via
  `xcodebuild -scheme mlx-swift-LocateAnything-Package -destination 'platform=macOS' test`.
  Pure tests (`ConfigTests`, `PromptTests`) run under `swift test`.
- GUI: `xcodebuild -project Examples/LocateAnythingDemo/LocateAnythingDemo.xcodeproj -scheme LocateAnythingDemo -destination 'platform=macOS' build`.

## Documentation

`MLXLocateAnything` ships DocC-generated reference docs (see
`Sources/MLXLocateAnything/Documentation.docc/` and `Scripts/build_docs.sh`).
**`///` doc comments on public/`open` symbols are published** to the static site
and (with `EMIT_LLMS_TXT=1`) into `docs/llms.txt`.

When you add or modify a `public` or `open` declaration:

- Write a `///` doc comment. One-sentence summary, then a paragraph if the *why*
  is non-obvious. Skip restating what the signature already says.
- Document each parameter with `- Parameter name:` (use the **internal** name when
  there's an external label — DocC warns otherwise).
- Cross-reference related symbols with double-backtick links, e.g.
  `` ``LocateAnythingSession/predict(image:question:maxNewTokens:mode:visualPromptBox:visualPromptBoxFormat:)`` ``.
  DocC link syntax is signature-sensitive: `foo(_:)` and `foo(_:_:)` differ.
- When you add a new top-level public symbol, file it under the appropriate
  `## Topics` group in `Sources/MLXLocateAnything/Documentation.docc/MLXLocateAnything.md`
  (topics are organized by *user task*, not alphabetically).

Verify before declaring documentation work done:

```bash
BUILD_DOC=1 Scripts/build_docs.sh    # exit 0, no unresolved-link warnings on prose
```

## Convention

Weights: `mlx-community/LocateAnything-3B-8bit` (8-bit affine, group 64; only the
language model is quantized). The legacy tokenizer (no `tokenizer.json`) is
synthesized natively in Swift on load (`TokenizerSetup.swift`). Detect prompt
format: `"Locate all the instances that matches the following description: {classes joined by </c>}."`
