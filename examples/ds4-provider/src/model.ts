import "dotenv/config";

import { ds4 } from "@lgrammel/ds4-provider";

const modelPath = process.env.DS4_MODEL_PATH ?? "./ds4flash.gguf";

export const model = ds4({
  modelId: process.env.DS4_MODEL ?? "deepseek-v4-flash",
  modelPath,
  mtpPath: process.env.DS4_MTP_PATH,
  contextSize: process.env.DS4_CONTEXT_SIZE ? Number(process.env.DS4_CONTEXT_SIZE) : 32768,
  debug: process.env.DS4_DEBUG === "1",
});
