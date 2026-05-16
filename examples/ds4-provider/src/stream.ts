import { streamText } from "ai";

import { model } from "./model.js";

const result = streamText({
  model,
  prompt: "Write three short tips for using local coding models.",
});

for await (const delta of result.textStream) {
  process.stdout.write(delta);
}

process.stdout.write("\n");
