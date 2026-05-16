import { strict as assert } from "node:assert";

import { parseGeneratedContent } from "../src/ds4-language-model.js";

const dsmlToolCall =
  "\n\n<｜DSML｜tool_calls>\n" +
  '<｜DSML｜invoke name="bash">\n' +
  '<｜DSML｜parameter name="command" string="true">cd /tmp && git diff 2>/dev/null</｜DSML｜parameter>\n' +
  '<｜DSML｜parameter name="timeout" string="false">10</｜DSML｜parameter>\n' +
  "</｜DSML｜invoke>\n" +
  "</｜DSML｜tool_calls>";

{
  const parsed = parseGeneratedContent(`need a tool${dsmlToolCall}`, false);
  assert.equal(parsed.finishReason?.unified, "tool-calls");
  assert.equal(parsed.toolCalls.length, 1);
  assert.equal(parsed.toolCalls[0]?.toolName, "bash");
  assert.deepEqual(JSON.parse(parsed.toolCalls[0]?.input ?? "{}"), {
    command: "cd /tmp && git diff 2>/dev/null",
    timeout: 10,
  });
  assert.equal(parsed.content.find((part) => part.type === "text")?.text, "need a tool");
}

{
  const parsed = parseGeneratedContent(`${dsmlToolCall}\nfinal text`, true);
  assert.equal(parsed.toolCalls.length, 0);
  assert.equal(parsed.content[0]?.type, "reasoning");
}

{
  const parsed = parseGeneratedContent(`thinking${dsmlToolCall.replace('name="bash"', "")}`, false);
  assert.equal(parsed.toolCalls.length, 0);
  assert.equal(parsed.content[0]?.type, "text");
  assert.match(parsed.content[0]?.type === "text" ? parsed.content[0].text : "", /tool_calls/);
}
