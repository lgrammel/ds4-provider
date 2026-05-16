import { generateText } from "ai";

import { model } from "./model.js";

const result = await generateText({
  model,
  prompt: "Write a haiku about local inference.",
});

console.log(result.text);
