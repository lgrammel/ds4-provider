import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../src/native-binding.js", () => ({
  cancelGeneration: vi.fn(),
  generate: vi.fn(),
  generateStream: vi.fn(),
  loadModel: vi.fn(),
  unloadModel: vi.fn(),
}));

import {
  DS4LanguageModel,
  convertMessages,
  parseGeneratedContent,
} from "../src/ds4-language-model.js";
import { ds4 } from "../src/ds4-provider.js";
import {
  generate,
  loadModel,
  unloadModel,
  type GenerateOptions,
  type GenerateResult,
} from "../src/native-binding.js";

const dsmlToolCall =
  "\n\n<｜DSML｜tool_calls>\n" +
  '<｜DSML｜invoke name="bash">\n' +
  '<｜DSML｜parameter name="command" string="true">cd /tmp && git diff 2>/dev/null</｜DSML｜parameter>\n' +
  '<｜DSML｜parameter name="timeout" string="false">10</｜DSML｜parameter>\n' +
  "</｜DSML｜invoke>\n" +
  "</｜DSML｜tool_calls>";

describe("parseGeneratedContent", () => {
  it("parses DSML tool calls and preserves raw string parameter text", () => {
    const parsed = parseGeneratedContent(`need a tool${dsmlToolCall}`, false);

    expect(parsed.finishReason?.unified).toBe("tool-calls");
    expect(parsed.toolCalls).toHaveLength(1);
    expect(parsed.toolCalls[0]?.toolName).toBe("bash");
    expect(JSON.parse(parsed.toolCalls[0]?.input ?? "{}")).toEqual({
      command: "cd /tmp && git diff 2>/dev/null",
      timeout: 10,
    });
    expect(parsed.content.find((part) => part.type === "text")).toMatchObject({
      text: "need a tool",
    });
  });

  it("treats unclosed initial reasoning as reasoning instead of parsing tools", () => {
    const parsed = parseGeneratedContent(`${dsmlToolCall}\nfinal text`, true);

    expect(parsed.toolCalls).toHaveLength(0);
    expect(parsed.content[0]?.type).toBe("reasoning");
  });

  it("parses tool calls after closed initial reasoning", () => {
    const parsed = parseGeneratedContent(`need a tool</think>${dsmlToolCall}`, true);

    expect(parsed.finishReason?.unified).toBe("tool-calls");
    expect(parsed.content[0]).toMatchObject({ type: "reasoning", text: "need a tool" });
    expect(parsed.toolCalls).toHaveLength(1);
  });

  it("parses DSML blocks with DSLS invoke tags and empty object parameters", () => {
    const parsed = parseGeneratedContent(
      "I need current info.</think>\n\n" +
        "<｜DSML｜tool_calls>\n" +
        '<｜DSLS｜invoke name="currentDate">\n' +
        '<｜DSLS｜parameter name="" string="false">{}</｜DSLS｜parameter>\n' +
        "</｜DSLS｜invoke>\n" +
        '<｜DSLS｜invoke name="webSearch">\n' +
        '<｜DSLS｜parameter name="query" string="true">Berlin news yesterday</｜DSLS｜parameter>\n' +
        "</｜DSLS｜invoke>\n" +
        "</｜DSML｜tool_calls>",
      true,
    );

    expect(parsed.finishReason?.unified).toBe("tool-calls");
    expect(parsed.toolCalls.map((toolCall) => toolCall.toolName)).toEqual([
      "currentDate",
      "webSearch",
    ]);
    expect(JSON.parse(parsed.toolCalls[0]?.input ?? "")).toEqual({});
    expect(JSON.parse(parsed.toolCalls[1]?.input ?? "")).toEqual({
      query: "Berlin news yesterday",
    });
  });

  it("keeps invalid DSML output as text", () => {
    const parsed = parseGeneratedContent(
      `thinking${dsmlToolCall.replace('name="bash"', "")}`,
      false,
    );

    expect(parsed.toolCalls).toHaveLength(0);
    expect(parsed.content[0]?.type).toBe("text");
    expect(parsed.content[0]?.type === "text" ? parsed.content[0].text : "").toMatch(/tool_calls/);
  });

  it("parses plain XML-style tool calls", () => {
    const parsed = parseGeneratedContent(
      'done\n<tool_calls><invoke name="search"><parameter name="query">a &amp; b</parameter></invoke></tool_calls>',
    );

    expect(parsed.toolCalls[0]?.toolName).toBe("search");
    expect(JSON.parse(parsed.toolCalls[0]?.input ?? "{}")).toEqual({ query: "a & b" });
    expect(parsed.content.find((part) => part.type === "text")).toMatchObject({ text: "done" });
  });
});

describe("convertMessages", () => {
  it("prepends DSML tool instructions for function tools and formats unsupported user parts", () => {
    const converted = convertMessages(
      [
        {
          role: "user",
          content: [
            { type: "text", text: "describe this " },
            {
              type: "file",
              data: new Uint8Array([1, 2, 3]),
              filename: "image.png",
              mediaType: "image/png",
            },
          ],
        },
      ],
      {
        toolChoice: { type: "tool", toolName: "search" },
        tools: [
          {
            type: "function",
            name: "search",
            description: "Search docs",
            inputSchema: {
              type: "object",
              properties: { query: { type: "string" } },
              required: ["query"],
            },
          },
        ],
      },
    );

    expect(converted[0]?.role).toBe("system");
    expect(converted[0]?.content).toContain('You must call the "search" tool.');
    expect(converted[1]).toEqual({
      role: "user",
      content: "describe this [Unsupported file part omitted]",
    });
  });

  it("formats assistant tool calls as DSML and escapes closing parameter tags", () => {
    const converted = convertMessages([
      {
        role: "assistant",
        content: [
          { type: "text", text: "calling" },
          {
            type: "tool-call",
            toolCallId: "call-1",
            toolName: "shell",
            input: {
              command: "echo '</｜DSML｜parameter>'",
              retries: 2,
            },
          },
        ],
      },
    ]);

    expect(converted).toHaveLength(1);
    expect(converted[0]?.content).toContain('<｜DSML｜invoke name="shell">');
    expect(converted[0]?.content).toContain("echo '&lt;/｜DSML｜parameter>'</｜DSML｜parameter>");
    expect(converted[0]?.content).toContain(
      '<｜DSML｜parameter name="retries" string="false">2</｜DSML｜parameter>',
    );
  });
});

describe("DS4 provider", () => {
  beforeEach(() => {
    vi.mocked(loadModel).mockReset();
    vi.mocked(generate).mockReset();
    vi.mocked(unloadModel).mockReset();
  });

  it("creates language models with the default model id", () => {
    const model = ds4({ modelPath: "/models/ds4.gguf" });

    expect(model).toBeInstanceOf(DS4LanguageModel);
    expect(model.modelId).toBe("deepseek-v4-flash");
  });

  it("builds native generate options from AI SDK call options", async () => {
    vi.mocked(loadModel).mockResolvedValue(42);
    vi.mocked(generate).mockResolvedValue({
      text: "answer",
      promptTokens: 3,
      completionTokens: 4,
      finishReason: "length",
    } satisfies GenerateResult);

    const model = ds4({
      modelPath: "/models/ds4.gguf",
      backend: "cpu",
      contextSize: 4096,
      topK: 40,
      minP: 0.05,
      seed: 123,
    });
    const result = await model.doGenerate({
      prompt: [{ role: "user", content: [{ type: "text", text: "hello" }] }],
      maxOutputTokens: 10,
      temperature: 0.2,
      topP: 0.8,
      stopSequences: ["END"],
      reasoning: "none",
    });

    expect(loadModel).toHaveBeenCalledWith({
      modelPath: "/models/ds4.gguf",
      mtpPath: undefined,
      contextSize: 4096,
      threads: undefined,
      backend: "cpu",
      mtpDraftTokens: undefined,
      mtpMargin: undefined,
      warmWeights: undefined,
      quality: undefined,
      debug: undefined,
    });
    expect(generate).toHaveBeenCalledWith(42, {
      messages: [{ role: "user", content: "hello" }],
      maxTokens: 10,
      temperature: 0.2,
      topP: 0.8,
      topK: 40,
      minP: 0.05,
      seed: 123,
      stopSequences: ["END"],
    } satisfies GenerateOptions);
    expect(result.finishReason).toEqual({ unified: "length", raw: "length" });
    expect(result.usage.outputTokens.total).toBe(4);
  });

  it("unloads an initialized model on dispose", async () => {
    vi.mocked(loadModel).mockResolvedValue(7);
    vi.mocked(generate).mockResolvedValue({
      text: "ok",
      promptTokens: 1,
      completionTokens: 1,
      finishReason: "stop",
    });

    const model = ds4({ modelPath: "/models/ds4.gguf" });
    await model.doGenerate({ prompt: [{ role: "user", content: [{ type: "text", text: "hi" }] }] });
    await model.dispose();

    expect(unloadModel).toHaveBeenCalledWith(7);
  });
});
