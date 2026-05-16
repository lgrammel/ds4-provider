import { runAgentTUI } from "@lgrammel/agent-tui";
import { ToolLoopAgent } from "ai";

import { model } from "./model.js";

await runAgentTUI({
  name: "DS4 Tool Loop Agent",
  agent: new ToolLoopAgent({
    model,
    instructions:
      "You are a concise local DS4 agent. Answer in markdown and keep " +
      "responses practical for someone experimenting with local inference.",
  }),
  reasoning: "auto-collapsed",
  contextSize: Number(process.env.DS4_CONTEXT_SIZE ?? 32768),
});
