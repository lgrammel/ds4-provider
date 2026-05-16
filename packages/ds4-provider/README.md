# @lgrammel/ds4-provider

AI SDK provider for running DeepSeek V4 Flash locally through
[DwarfStar 4](https://github.com/antirez/ds4). The package bundles the DS4
native sources and builds the Node.js addon on install; you only need to provide
a compatible GGUF model file.

## Getting Started

Install the package and the AI SDK:

```sh
pnpm add @lgrammel/ds4-provider ai
```

Install requirements:

- Node.js 18 or newer.
- A working native build toolchain for `node-gyp`.
- A DS4-compatible DeepSeek V4 Flash GGUF.

Download a model using DS4's downloader:

```sh
git clone https://github.com/antirez/ds4.git
cd ds4
./download_model.sh q2-imatrix
```

The DS4 project recommends `q2-imatrix` for 96/128 GB RAM machines and
`q4-imatrix` for machines with at least 256 GB RAM. The script downloads from
`https://huggingface.co/antirez/deepseek-v4-gguf` and creates
`./ds4flash.gguf`.

Point the provider at the downloaded file:

```sh
export DS4_MODEL_PATH=/path/to/ds4/ds4flash.gguf
```

## Agent TUI Example

Install the example dependencies:

```sh
pnpm add @lgrammel/ds4-provider @lgrammel/agent-tui ai zod dotenv
pnpm add -D tsx typescript @types/node
```

Create `agent.ts`:

```ts
import "dotenv/config";

import { runAgentTUI } from "@lgrammel/agent-tui";
import { ds4 } from "@lgrammel/ds4-provider";
import { ToolLoopAgent, tool } from "ai";
import { z } from "zod";

const model = ds4({
  modelId: "deepseek-v4-flash",
  modelPath: process.env.DS4_MODEL_PATH ?? "./ds4flash.gguf",
  contextSize: Number(process.env.DS4_CONTEXT_SIZE ?? 32768),
});

await runAgentTUI({
  name: "DS4 Agent",
  agent: new ToolLoopAgent({
    model,
    instructions: "You are a concise local coding agent.",
    tools: {
      localTime: tool({
        description: "Get the current local time.",
        inputSchema: z.object({}),
        execute: () => ({ time: new Date().toLocaleString() }),
      }),
    },
  }),
  reasoning: "auto-collapsed",
  contextSize: Number(process.env.DS4_CONTEXT_SIZE ?? 32768),
});
```

Run it:

```sh
DS4_MODEL_PATH=/path/to/ds4/ds4flash.gguf pnpm tsx agent.ts
```

## Credit

This package is a Node.js / AI SDK integration for
[DwarfStar 4](https://github.com/antirez/ds4), the local inference engine by
Salvatore Sanfilippo (`antirez`) for DeepSeek V4 Flash. DS4 provides the native
runtime, model format expectations, Metal/CUDA work, and GGUF download flow; this
package only wraps that engine for use from JavaScript.
