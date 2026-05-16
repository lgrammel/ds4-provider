# @lgrammel/ds4-provider

DS4 provider for the Vercel AI SDK.

The npm package includes the pinned DS4 native sources and compiles the Node.js
native addon during installation. GGUF model files are not bundled; download the
model separately and pass its path when creating or loading the model.

## Requirements

- Node.js 18 or newer
- A working native build toolchain for `node-gyp`
- A DS4-compatible GGUF model file

On macOS, installation builds the Metal backend. Other platforms build the CPU
backend from the bundled DS4 sources.
