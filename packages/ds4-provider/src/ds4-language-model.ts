import type {
  LanguageModelV4,
  LanguageModelV4CallOptions,
  LanguageModelV4Content,
  LanguageModelV4FinishReason,
  LanguageModelV4GenerateResult,
  LanguageModelV4Message,
  LanguageModelV4StreamPart,
  LanguageModelV4StreamResult,
  LanguageModelV4ToolCall,
  LanguageModelV4Usage,
  SharedV4Warning,
} from "@ai-sdk/provider";

import {
  cancelGeneration,
  type DS4ReasoningMode,
  generate,
  generateStream,
  loadModel,
  unloadModel,
  type ChatMessage,
  type GenerateOptions,
  type GenerateResult,
  type LoadModelOptions,
} from "./native-binding.js";
import { DS4StreamProjector } from "./ds4-stream-projector.js";

export interface DS4ProviderSettings {
  /**
   * Path to a DS4-compatible GGUF model.
   */
  modelPath: string;
  /**
   * Optional MTP GGUF file path.
   */
  mtpPath?: string;
  /**
   * AI SDK model id.
   *
   * @default "deepseek-v4-flash"
   */
  modelId?: string;
  contextSize?: number;
  threads?: number;
  backend?: "metal" | "cuda" | "cpu";
  mtpDraftTokens?: number;
  mtpMargin?: number;
  warmWeights?: boolean;
  quality?: boolean;
  /**
   * Show native DS4 startup logs on stderr.
   *
   * @default false
   */
  debug?: boolean;
  topK?: number;
  minP?: number;
  seed?: number;
}

export interface DS4LanguageModelConfig extends DS4ProviderSettings {
  modelId: string;
}

interface ToolParseResult {
  content: LanguageModelV4Content[];
  finishReason?: LanguageModelV4FinishReason;
  toolCalls: LanguageModelV4ToolCall[];
  toolDsml?: string;
}

type AssistantToolCallPromptPart = Extract<
  Extract<LanguageModelV4Message, { role: "assistant" }>["content"][number],
  { type: "tool-call" }
>;

type ToolResultPromptPart = Extract<
  Extract<LanguageModelV4Message, { role: "tool" }>["content"][number],
  { type: "tool-result" }
>;

export class DS4LanguageModel implements LanguageModelV4 {
  readonly specificationVersion = "v4";
  readonly provider = "ds4";
  readonly modelId: string;
  readonly supportedUrls = {};

  private readonly topK?: number;
  private readonly minP?: number;
  private readonly seed?: number;
  private readonly toolReplay = new Map<string, string>();
  private readonly toolReplayIds = new Map<string, string[]>();
  private modelHandle: number | null = null;
  private initPromise: Promise<void> | null = null;

  constructor(private readonly config: DS4LanguageModelConfig) {
    this.modelId = config.modelId;
    this.topK = config.topK;
    this.minP = config.minP;
    this.seed = config.seed;
  }

  async doGenerate(options: LanguageModelV4CallOptions): Promise<LanguageModelV4GenerateResult> {
    throwIfAborted(options.abortSignal);

    const handle = await this.ensureModelLoaded();
    const generateOptions = this.buildGenerateOptions(options);
    const result = await this.runWithAbortSignal(handle, options.abortSignal, () =>
      generate(handle, generateOptions),
    );

    const parsed = parseGeneratedContent(result.text, generateOptions.thinkMode !== undefined);
    this.rememberToolCalls(parsed);

    return {
      content: parsed.content,
      finishReason: parsed.finishReason ?? convertFinishReason(result.finishReason),
      usage: convertUsage(result),
      warnings: getWarnings(options),
      request: {
        body: generateOptions,
      },
      response: {
        modelId: this.modelId,
      },
    };
  }

  async doStream(options: LanguageModelV4CallOptions): Promise<LanguageModelV4StreamResult> {
    throwIfAborted(options.abortSignal);

    const handle = await this.ensureModelLoaded();
    const generateOptions = this.buildGenerateOptions(options);
    const textId = crypto.randomUUID();
    const reasoningId = crypto.randomUUID();
    const warnings = getWarnings(options);
    const stream = this.createStream(
      handle,
      generateOptions,
      textId,
      reasoningId,
      warnings,
      options,
    );

    return {
      stream,
      request: {
        body: generateOptions,
      },
    };
  }

  async dispose(): Promise<void> {
    if (this.modelHandle !== null) {
      unloadModel(this.modelHandle);
      this.modelHandle = null;
    }
  }

  private async ensureModelLoaded(): Promise<number> {
    if (this.modelHandle !== null) {
      return this.modelHandle;
    }

    this.initPromise ??= (async () => {
      const loadOptions: LoadModelOptions = {
        modelPath: this.config.modelPath,
        mtpPath: this.config.mtpPath,
        contextSize: this.config.contextSize,
        threads: this.config.threads,
        backend: this.config.backend,
        mtpDraftTokens: this.config.mtpDraftTokens,
        mtpMargin: this.config.mtpMargin,
        warmWeights: this.config.warmWeights,
        quality: this.config.quality,
        debug: this.config.debug,
      };

      this.modelHandle = await loadModel(loadOptions);
    })();

    await this.initPromise;
    this.initPromise = null;

    if (this.modelHandle === null) {
      throw new Error("Failed to load DS4 model");
    }

    return this.modelHandle;
  }

  private buildGenerateOptions(options: LanguageModelV4CallOptions): GenerateOptions {
    const body: GenerateOptions = {
      messages: this.convertMessages(options.prompt, options),
    };

    if (options.maxOutputTokens !== undefined) {
      body.maxTokens = options.maxOutputTokens;
    }
    if (options.temperature !== undefined) {
      body.temperature = options.temperature;
    }
    if (options.topP !== undefined) {
      body.topP = options.topP;
    }
    if (this.topK !== undefined) {
      body.topK = this.topK;
    }
    if (this.minP !== undefined) {
      body.minP = this.minP;
    }
    if (this.seed !== undefined) {
      body.seed = this.seed;
    }
    if (options.stopSequences?.length) {
      body.stopSequences = options.stopSequences;
    }
    const thinkMode = getThinkMode(options);
    if (thinkMode !== "none") {
      body.thinkMode = thinkMode;
    }

    return body;
  }

  private createStream(
    handle: number,
    generateOptions: GenerateOptions,
    textId: string,
    reasoningId: string,
    warnings: SharedV4Warning[],
    options: LanguageModelV4CallOptions,
  ): ReadableStream<LanguageModelV4StreamPart> {
    return new ReadableStream<LanguageModelV4StreamPart>({
      start: async (controller) => {
        let generatedText = "";
        const hasTools = shouldParseToolOutput(options);
        const projector = new DS4StreamProjector({
          textId,
          reasoningId,
          initialInReasoning:
            generateOptions.thinkMode !== undefined && generateOptions.thinkMode !== "none",
          hasTools,
          stopSequences: generateOptions.stopSequences,
        });

        try {
          controller.enqueue({ type: "stream-start", warnings });

          const result = await this.runWithAbortSignal(handle, options.abortSignal, () =>
            generateStream(handle, generateOptions, (delta) => {
              generatedText += delta;
              if (hasTools && isCompleteDsmlToolCall(generatedText, generateOptions)) {
                cancelGeneration(handle);
              }
              for (const part of projector.update(generatedText)) {
                controller.enqueue(part);
              }
            }),
          );

          generatedText = result.text;
          for (const part of projector.update(generatedText, true)) {
            controller.enqueue(part);
          }
          for (const part of projector.finish()) {
            controller.enqueue(part);
          }

          const parsed = parseGeneratedContent(
            generatedText,
            generateOptions.thinkMode !== undefined,
            projector.getToolCallIds(),
          );
          this.rememberToolCalls(parsed);

          for (const toolCall of parsed.toolCalls) {
            controller.enqueue(toolCall);
          }

          controller.enqueue({
            type: "finish",
            finishReason: parsed.finishReason ?? convertFinishReason(result.finishReason),
            usage: convertUsage(result),
          });
          controller.close();
        } catch (error) {
          controller.error(error);
        }
      },
    });
  }

  private async runWithAbortSignal<T>(
    handle: number,
    signal: AbortSignal | undefined,
    run: () => Promise<T>,
  ): Promise<T> {
    throwIfAborted(signal);

    if (!signal) {
      return run();
    }

    return new Promise<T>((resolve, reject) => {
      let settled = false;
      let abortListener: (() => void) | undefined;

      const settle = (callback: () => void) => {
        if (settled) {
          return;
        }
        settled = true;
        if (abortListener) {
          signal.removeEventListener("abort", abortListener);
        }
        callback();
      };

      abortListener = () => {
        cancelGeneration(handle);
        settle(() => reject(createAbortError()));
      };

      signal.addEventListener("abort", abortListener, { once: true });

      if (signal.aborted) {
        abortListener();
        return;
      }

      run().then(
        (result) => settle(() => resolve(result)),
        (error: unknown) => settle(() => reject(error)),
      );
    });
  }

  private convertMessages(
    messages: LanguageModelV4Message[],
    options: Pick<LanguageModelV4CallOptions, "toolChoice" | "tools"> = {},
  ): ChatMessage[] {
    return convertMessages(messages, options, this.toolReplay, this.toolReplayIds);
  }

  private rememberToolCalls(parsed: ToolParseResult): void {
    if (!parsed.toolDsml) {
      return;
    }

    for (const toolCall of parsed.toolCalls) {
      this.toolReplay.set(toolCall.toolCallId, parsed.toolDsml);
    }
    this.toolReplayIds.set(
      parsed.toolDsml,
      parsed.toolCalls.map((toolCall) => toolCall.toolCallId),
    );
  }
}

export function convertMessages(
  messages: LanguageModelV4Message[],
  options: Pick<LanguageModelV4CallOptions, "toolChoice" | "tools"> = {},
  toolReplay?: ReadonlyMap<string, string>,
  toolReplayIds?: ReadonlyMap<string, string[]>,
): ChatMessage[] {
  const result: ChatMessage[] = [];
  const toolInstructions = formatToolInstructions(options);

  if (toolInstructions) {
    result.push({ role: "system", content: toolInstructions });
  }

  for (const message of messages) {
    switch (message.role) {
      case "system":
        result.push({ role: "system", content: message.content });
        break;
      case "user":
        result.push({
          role: "user",
          content: message.content
            .map((part) => {
              if (part.type === "text") {
                return part.text;
              }
              return `[Unsupported ${part.type} part omitted]`;
            })
            .join(""),
        });
        break;
      case "assistant": {
        const content = formatAssistantContent(message.content, toolReplay, toolReplayIds);
        if (content.length > 0) {
          result.push({ role: "assistant", content });
        }
        break;
      }
      case "tool": {
        const content = message.content.map(formatToolResult).join("\n\n");
        if (content.length > 0) {
          result.push({ role: "user", content });
        }
        break;
      }
    }
  }

  return result;
}

const THINK_OPEN = "<think>";
const THINK_CLOSE = "</think>";
const DSML = "｜DSML｜";
const DSLS = "｜DSLS｜";
const DSML_SHORT = "DSML｜";
const DSML_TOOL_CALLS_OPEN = `<${DSML}tool_calls>`;
const DSML_TOOL_CALLS_CLOSE = `</${DSML}tool_calls>`;
const DSML_INVOKE_OPEN = `<${DSML}invoke`;
const DSML_INVOKE_CLOSE = `</${DSML}invoke>`;
const DSML_PARAMETER_OPEN = `<${DSML}parameter`;
const DSML_PARAMETER_CLOSE = `</${DSML}parameter>`;
const DSLS_TOOL_CALLS_OPEN = `<${DSLS}tool_calls>`;
const DSLS_TOOL_CALLS_CLOSE = `</${DSLS}tool_calls>`;
const DSLS_INVOKE_OPEN = `<${DSLS}invoke`;
const DSLS_INVOKE_CLOSE = `</${DSLS}invoke>`;
const DSLS_PARAMETER_OPEN = `<${DSLS}parameter`;
const DSLS_PARAMETER_CLOSE = `</${DSLS}parameter>`;
const DSML_TOOL_CALLS_OPEN_SHORT = `<${DSML_SHORT}tool_calls>`;
const DSML_TOOL_CALLS_CLOSE_SHORT = `</${DSML_SHORT}tool_calls>`;
const DSML_INVOKE_OPEN_SHORT = `<${DSML_SHORT}invoke`;
const DSML_INVOKE_CLOSE_SHORT = `</${DSML_SHORT}invoke>`;
const DSML_PARAMETER_OPEN_SHORT = `<${DSML_SHORT}parameter`;
const DSML_PARAMETER_CLOSE_SHORT = `</${DSML_SHORT}parameter>`;
const PLAIN_TOOL_CALLS_OPEN = "<tool_calls>";
const PLAIN_TOOL_CALLS_CLOSE = "</tool_calls>";
const PLAIN_INVOKE_OPEN = "<invoke";
const PLAIN_INVOKE_CLOSE = "</invoke>";
const PLAIN_PARAMETER_OPEN = "<parameter";
const PLAIN_PARAMETER_CLOSE = "</parameter>";
const TOOL_RESULT_CLOSE = "</tool_result>";

interface DsmlSyntax {
  toolCallsOpen: string;
  toolCallsClose: string;
  invokeOpen: string;
  invokeClose: string;
  parameterOpen: string;
  parameterClose: string;
}

interface DsmlToolCall {
  toolName: string;
  input: string;
}

interface DsmlParseResult {
  contentText: string;
  reasoningText?: string;
  calls: DsmlToolCall[];
  rawDsml?: string;
  invalidToolCall?: boolean;
}

const DSML_SYNTAXES: DsmlSyntax[] = [
  {
    toolCallsOpen: DSML_TOOL_CALLS_OPEN,
    toolCallsClose: DSML_TOOL_CALLS_CLOSE,
    invokeOpen: DSML_INVOKE_OPEN,
    invokeClose: DSML_INVOKE_CLOSE,
    parameterOpen: DSML_PARAMETER_OPEN,
    parameterClose: DSML_PARAMETER_CLOSE,
  },
  {
    toolCallsOpen: DSML_TOOL_CALLS_OPEN,
    toolCallsClose: DSML_TOOL_CALLS_CLOSE,
    invokeOpen: DSLS_INVOKE_OPEN,
    invokeClose: DSLS_INVOKE_CLOSE,
    parameterOpen: DSLS_PARAMETER_OPEN,
    parameterClose: DSLS_PARAMETER_CLOSE,
  },
  {
    toolCallsOpen: DSLS_TOOL_CALLS_OPEN,
    toolCallsClose: DSLS_TOOL_CALLS_CLOSE,
    invokeOpen: DSLS_INVOKE_OPEN,
    invokeClose: DSLS_INVOKE_CLOSE,
    parameterOpen: DSLS_PARAMETER_OPEN,
    parameterClose: DSLS_PARAMETER_CLOSE,
  },
  {
    toolCallsOpen: DSML_TOOL_CALLS_OPEN_SHORT,
    toolCallsClose: DSML_TOOL_CALLS_CLOSE_SHORT,
    invokeOpen: DSML_INVOKE_OPEN_SHORT,
    invokeClose: DSML_INVOKE_CLOSE_SHORT,
    parameterOpen: DSML_PARAMETER_OPEN_SHORT,
    parameterClose: DSML_PARAMETER_CLOSE_SHORT,
  },
  {
    toolCallsOpen: PLAIN_TOOL_CALLS_OPEN,
    toolCallsClose: PLAIN_TOOL_CALLS_CLOSE,
    invokeOpen: PLAIN_INVOKE_OPEN,
    invokeClose: PLAIN_INVOKE_CLOSE,
    parameterOpen: PLAIN_PARAMETER_OPEN,
    parameterClose: PLAIN_PARAMETER_CLOSE,
  },
];

function pushContent(
  content: LanguageModelV4Content[],
  type: "text" | "reasoning",
  text: string,
): void {
  if (text.length === 0) {
    return;
  }

  content.push(
    type === "text"
      ? {
          type,
          text,
          providerMetadata: undefined,
        }
      : {
          type,
          text,
          providerMetadata: undefined,
        },
  );
}

export function parseGeneratedContent(
  text: string,
  initialInReasoning = false,
  toolCallIds: string[] = [],
): ToolParseResult {
  const parsed = parseGeneratedMessage(text, initialInReasoning);
  const content: LanguageModelV4Content[] = [];
  pushContent(content, "reasoning", parsed.reasoningText ?? "");
  pushContent(content, "text", parsed.contentText);

  const toolCalls = parsed.calls.map(
    (call, index): LanguageModelV4ToolCall => ({
      type: "tool-call",
      toolCallId: toolCallIds[index] ?? crypto.randomUUID(),
      toolName: call.toolName,
      input: call.input,
    }),
  );
  content.push(...toolCalls);

  return {
    content,
    finishReason: toolCalls.length > 0 ? { unified: "tool-calls", raw: "tool-calls" } : undefined,
    toolCalls,
    toolDsml: parsed.rawDsml,
  };
}

function isCompleteDsmlToolCall(text: string, generateOptions: GenerateOptions): boolean {
  const requireThinkingClosed = generateOptions.thinkMode !== undefined;
  const searchStart = requireThinkingClosed
    ? text.lastIndexOf(THINK_CLOSE) + THINK_CLOSE.length
    : 0;

  if (requireThinkingClosed && searchStart < THINK_CLOSE.length) {
    return false;
  }

  const found = findDsmlToolStart(text, searchStart);
  if (!found) {
    return false;
  }

  return found.syntaxes.some(
    (syntax) =>
      text.indexOf(syntax.toolCallsClose, found.start + syntax.toolCallsOpen.length) !== -1,
  );
}

function shouldParseToolOutput(options: Pick<LanguageModelV4CallOptions, "tools">): boolean {
  return options.tools?.some((tool) => tool.type === "function") ?? false;
}

function formatAssistantContent(
  content: Extract<LanguageModelV4Message, { role: "assistant" }>["content"],
  toolReplay?: ReadonlyMap<string, string>,
  toolReplayIds?: ReadonlyMap<string, string[]>,
): string {
  let text = "";
  const toolCalls: AssistantToolCallPromptPart[] = [];

  for (const part of content) {
    if (part.type === "text") {
      text += part.text;
    } else if (part.type === "reasoning") {
      text += `${THINK_OPEN}${part.text}${THINK_CLOSE}`;
    } else if (part.type === "tool-call") {
      toolCalls.push(part);
    }
  }

  if (toolCalls.length === 0) {
    return text;
  }

  const replayed = getReplayedDsml(toolCalls, toolReplay, toolReplayIds);
  return `${text}${replayed ?? formatAssistantToolCalls(toolCalls)}`;
}

function formatToolInstructions(
  options: Pick<LanguageModelV4CallOptions, "toolChoice" | "tools">,
): string | undefined {
  const tools = options.tools?.filter((tool) => tool.type === "function") ?? [];
  if (tools.length === 0) {
    return undefined;
  }

  const toolChoice = options.toolChoice ?? { type: "auto" as const };
  const toolChoiceInstruction =
    toolChoice.type === "none"
      ? "Do not call any tools for this response."
      : toolChoice.type === "required"
        ? "You must call one of the available tools."
        : toolChoice.type === "tool"
          ? `You must call the ${JSON.stringify(toolChoice.toolName)} tool.`
          : "Call a tool only when it is needed to answer the user.";

  return [
    "## Tools",
    "",
    "You have access to a set of tools to help answer the user question. " +
      `You can invoke tools by writing a "${DSML_TOOL_CALLS_OPEN}" block like the following:`,
    "",
    DSML_TOOL_CALLS_OPEN,
    `${DSML_INVOKE_OPEN} name="$TOOL_NAME">`,
    `${DSML_PARAMETER_OPEN} name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE${DSML_PARAMETER_CLOSE}`,
    "...",
    DSML_INVOKE_CLOSE,
    `${DSML_INVOKE_OPEN} name="$TOOL_NAME2">`,
    "...",
    DSML_INVOKE_CLOSE,
    DSML_TOOL_CALLS_CLOSE,
    "",
    'String parameters should be specified as raw text and set `string="true"`.',
    "Preserve characters such as `>`, `&`, and `&&` exactly; never replace normal string characters with XML or HTML entity escapes.",
    `Only if a string value itself contains the exact closing parameter tag \`${DSML_PARAMETER_CLOSE}\`, write that tag as \`&lt;/${DSML}parameter>\` inside the value.`,
    'For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.',
    "",
    `If thinking_mode is enabled (triggered by ${THINK_OPEN}), you MUST output your complete reasoning inside ${THINK_OPEN}...${THINK_CLOSE} BEFORE any tool calls or final response.`,
    "",
    `Otherwise, output directly after ${THINK_CLOSE} with tool calls or final response.`,
    "",
    "### Available Tool Schemas",
    "",
    tools.map(formatToolSchema).join("\n"),
    "",
    "You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls. Use the exact parameter names from the schemas.",
    toolChoiceInstruction,
  ].join("\n");
}

function formatToolSchema(
  tool: Extract<NonNullable<LanguageModelV4CallOptions["tools"]>[number], { type: "function" }>,
): string {
  return JSON.stringify({
    name: tool.name,
    description: tool.description,
    input_schema: tool.inputSchema,
  });
}

function getReplayedDsml(
  toolCalls: AssistantToolCallPromptPart[],
  toolReplay?: ReadonlyMap<string, string>,
  toolReplayIds?: ReadonlyMap<string, string[]>,
): string | undefined {
  if (!toolReplay || toolCalls.length === 0) {
    return undefined;
  }

  const first = toolReplay.get(toolCalls[0].toolCallId);
  if (!first) {
    return undefined;
  }

  const ids = toolReplayIds?.get(first);
  if (ids && ids.length === toolCalls.length) {
    const sameOrder = toolCalls.every((part, index) => part.toolCallId === ids[index]);
    if (!sameOrder) {
      return undefined;
    }
  }

  return toolCalls.every((part) => toolReplay.get(part.toolCallId) === first) ? first : undefined;
}

function formatAssistantToolCalls(parts: AssistantToolCallPromptPart[]): string {
  if (parts.length === 0) {
    return "";
  }

  return `\n\n${DSML_TOOL_CALLS_OPEN}\n${parts.map(formatAssistantToolCall).join("")}${DSML_TOOL_CALLS_CLOSE}`;
}

function formatAssistantToolCall(part: AssistantToolCallPromptPart): string {
  return `${DSML_INVOKE_OPEN} name="${escapeDsmlAttribute(part.toolName)}">\n${formatDsmlArguments(
    stringifyToolInput(part.input),
  )}${DSML_INVOKE_CLOSE}\n`;
}

function formatToolResult(
  part: Extract<LanguageModelV4Message, { role: "tool" }>["content"][number],
): string {
  if (part.type !== "tool-result") {
    return "";
  }

  return `<tool_result>${escapeToolResultText(
    JSON.stringify({
      id: part.toolCallId,
      name: part.toolName,
      result: formatToolResultOutput(part.output),
    }),
  )}${TOOL_RESULT_CLOSE}`;
}

function formatToolResultOutput(part: ToolResultPromptPart["output"]): unknown {
  switch (part.type) {
    case "text":
    case "json":
    case "error-text":
    case "error-json":
      return part.value;
    case "execution-denied":
      return { error: "execution-denied", reason: part.reason };
    case "content":
      return part.value.map((content) => {
        switch (content.type) {
          case "text":
            return { type: "text", text: content.text };
          case "file":
            return {
              type: "file",
              mediaType: content.mediaType,
              filename: content.filename,
            };
          case "custom":
            return { type: "custom" };
        }
      });
  }
}

function formatDsmlArguments(input: string): string {
  const parsed = parseJsonObject(input);
  if (!parsed) {
    return `${DSML_PARAMETER_OPEN} name="arguments" string="true">${escapeDsmlParameterText(
      input,
    )}${DSML_PARAMETER_CLOSE}\n`;
  }

  let result = "";
  for (const [key, value] of Object.entries(parsed)) {
    const isString = typeof value === "string";
    const renderedValue = isString ? value : JSON.stringify(value);
    result += `${DSML_PARAMETER_OPEN} name="${escapeDsmlAttribute(key)}" string="${
      isString ? "true" : "false"
    }">${isString ? escapeDsmlParameterText(renderedValue) : escapeDsmlJsonLiteral(renderedValue)}${
      DSML_PARAMETER_CLOSE
    }\n`;
  }
  return result;
}

function parseGeneratedMessage(text: string, requireThinkingClosed: boolean): DsmlParseResult {
  const source = text ?? "";
  const searchStart = requireThinkingClosed
    ? source.lastIndexOf(THINK_CLOSE) + THINK_CLOSE.length
    : 0;

  if (requireThinkingClosed && searchStart < THINK_CLOSE.length) {
    const content: LanguageModelV4Content[] = [];
    pushReasoningAndText(content, source, true);
    return partsToDsmlParseResult(content);
  }

  const found = findDsmlToolStart(source, searchStart);
  if (!found) {
    const content: LanguageModelV4Content[] = [];
    pushReasoningAndText(content, source, requireThinkingClosed);
    return partsToDsmlParseResult(content);
  }

  const contentTextEnd = trimTrailingWhitespace(source.slice(0, found.start)).length;
  const beforeTool = source.slice(0, contentTextEnd);
  const parsedPrefix: LanguageModelV4Content[] = [];
  pushReasoningAndText(parsedPrefix, beforeTool, requireThinkingClosed);
  const base = partsToDsmlParseResult(parsedPrefix);

  let parsedTool: { calls: DsmlToolCall[]; rawDsml: string } | undefined;
  for (const syntax of found.syntaxes) {
    parsedTool = parseDsmlToolCalls(source, found.start, syntax);
    if (parsedTool) {
      break;
    }
  }
  if (!parsedTool) {
    return {
      contentText: base.contentText + source.slice(found.start),
      reasoningText: base.reasoningText,
      calls: [],
      invalidToolCall: true,
    };
  }

  return {
    contentText: base.contentText,
    reasoningText: base.reasoningText,
    calls: parsedTool.calls,
    rawDsml: parsedTool.rawDsml,
  };
}

function partsToDsmlParseResult(content: LanguageModelV4Content[]): DsmlParseResult {
  let contentText = "";
  let reasoningText = "";
  for (const part of content) {
    if (part.type === "text") {
      contentText += part.text;
    } else if (part.type === "reasoning") {
      reasoningText += part.text;
    }
  }
  return {
    contentText,
    reasoningText: reasoningText.length > 0 ? reasoningText : undefined,
    calls: [],
  };
}

function pushReasoningAndText(
  content: LanguageModelV4Content[],
  text: string,
  initialInReasoning = false,
): void {
  let remaining = text;
  let inReasoning = initialInReasoning || remaining.startsWith(THINK_OPEN);
  if (inReasoning && remaining.startsWith(THINK_OPEN)) {
    remaining = remaining.slice(THINK_OPEN.length);
  }

  while (remaining.length > 0) {
    if (inReasoning) {
      const endIndex = remaining.indexOf(THINK_CLOSE);
      if (endIndex === -1) {
        pushContent(content, "reasoning", remaining);
        return;
      }
      pushContent(content, "reasoning", remaining.slice(0, endIndex));
      remaining = remaining.slice(endIndex + THINK_CLOSE.length);
      inReasoning = false;
      continue;
    }

    const startIndex = remaining.indexOf(THINK_OPEN);
    if (startIndex === -1) {
      pushContent(content, "text", remaining);
      return;
    }
    pushContent(content, "text", remaining.slice(0, startIndex));
    remaining = remaining.slice(startIndex + THINK_OPEN.length);
    inReasoning = true;
  }
}

function findDsmlToolStart(
  text: string,
  fromIndex: number,
): { start: number; syntaxes: DsmlSyntax[] } | undefined {
  let best: { start: number; syntaxes: DsmlSyntax[] } | undefined;
  for (const syntax of DSML_SYNTAXES) {
    const index = text.indexOf(syntax.toolCallsOpen, fromIndex);
    if (index === -1) {
      continue;
    }
    if (!best || index < best.start) {
      best = { start: index, syntaxes: [syntax] };
    } else if (index === best.start) {
      best.syntaxes.push(syntax);
    }
  }
  return best;
}

function parseDsmlToolCalls(
  text: string,
  start: number,
  syntax: DsmlSyntax,
): { calls: DsmlToolCall[]; rawDsml: string } | undefined {
  let index = start;
  if (!text.startsWith(syntax.toolCallsOpen, index)) {
    return undefined;
  }
  index += syntax.toolCallsOpen.length;

  const calls: DsmlToolCall[] = [];
  while (index < text.length) {
    index = skipWhitespace(text, index);
    if (text.startsWith(syntax.toolCallsClose, index)) {
      const end = index + syntax.toolCallsClose.length;
      return {
        calls,
        rawDsml: text.slice(start, end),
      };
    }

    if (!text.startsWith(syntax.invokeOpen, index)) {
      return undefined;
    }

    const invokeTagEnd = text.indexOf(">", index);
    if (invokeTagEnd === -1) {
      return undefined;
    }
    const invokeTag = text.slice(index, invokeTagEnd + 1);
    const toolName = parseXmlAttribute(invokeTag, "name");
    if (!toolName) {
      return undefined;
    }
    index = invokeTagEnd + 1;

    const args: Record<string, unknown> = {};
    while (index < text.length) {
      index = skipWhitespace(text, index);
      if (text.startsWith(syntax.invokeClose, index)) {
        index += syntax.invokeClose.length;
        calls.push({ toolName, input: JSON.stringify(args) });
        break;
      }

      const parsedParam = parseDsmlParameter(text, index, syntax);
      if (!parsedParam) {
        return undefined;
      }
      if (parsedParam.name === "") {
        if (isJsonObject(parsedParam.value)) {
          Object.assign(args, parsedParam.value);
        } else {
          args.arguments = parsedParam.value;
        }
      } else {
        args[parsedParam.name] = parsedParam.value;
      }
      index = parsedParam.end;
    }
  }

  return undefined;
}

function parseDsmlParameter(
  text: string,
  start: number,
  syntax: DsmlSyntax,
): { name: string; value: unknown; end: number } | undefined {
  if (!text.startsWith(syntax.parameterOpen, start)) {
    return undefined;
  }

  const tagEnd = text.indexOf(">", start);
  if (tagEnd === -1) {
    return undefined;
  }

  const tag = text.slice(start, tagEnd + 1);
  const name = parseXmlAttribute(tag, "name");
  if (name === null) {
    return undefined;
  }
  const stringAttribute = parseXmlAttribute(tag, "string");
  const valueStart = tagEnd + 1;
  const valueEnd = text.indexOf(syntax.parameterClose, valueStart);
  if (valueEnd === -1) {
    return undefined;
  }

  const rawValue = text.slice(valueStart, valueEnd);
  const isString = stringAttribute === null || stringAttribute === "true";
  return {
    name,
    value: isString ? unescapeDsmlText(rawValue) : parseJsonValue(rawValue.trim()),
    end: valueEnd + syntax.parameterClose.length,
  };
}

function parseJsonObject(text: string): Record<string, unknown> | undefined {
  const parsed = parseJsonValue(text);
  return isJsonObject(parsed) ? parsed : undefined;
}

function isJsonObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function parseJsonValue(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

function parseXmlAttribute(tag: string, name: string): string | null {
  const match = new RegExp(`${escapeRegExp(name)}="([^"]*)"`).exec(tag);
  return match ? unescapeDsmlText(match[1]) : null;
}

function skipWhitespace(text: string, index: number): number {
  while (index < text.length && /\s/.test(text[index])) {
    index++;
  }
  return index;
}

function trimTrailingWhitespace(text: string): string {
  return text.replace(/\s+$/u, "");
}

function escapeDsmlAttribute(text: string): string {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function escapeDsmlParameterText(text: string): string {
  return text.replaceAll(DSML_PARAMETER_CLOSE, `&lt;/${DSML}parameter>`);
}

function escapeDsmlJsonLiteral(text: string): string {
  return text.replaceAll(DSML_PARAMETER_CLOSE, `\\u003c/${DSML}parameter>`);
}

function escapeToolResultText(text: string): string {
  return text.replaceAll(TOOL_RESULT_CLOSE, "&lt;/tool_result>");
}

function unescapeDsmlText(text: string): string {
  return text
    .replaceAll("&quot;", '"')
    .replaceAll("&apos;", "'")
    .replaceAll("&gt;", ">")
    .replaceAll("&lt;", "<")
    .replaceAll("&amp;", "&");
}

function escapeRegExp(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function stringifyToolInput(input: unknown): string {
  if (input === undefined) {
    return "{}";
  }
  if (typeof input === "string") {
    return input;
  }
  return JSON.stringify(input) ?? "{}";
}

function convertFinishReason(reason: string | null | undefined): LanguageModelV4FinishReason {
  switch (reason) {
    case "stop":
      return { unified: "stop", raw: reason };
    case "length":
      return { unified: "length", raw: reason };
    default:
      return { unified: "other", raw: reason ?? "unknown" };
  }
}

function convertUsage(result: GenerateResult | undefined): LanguageModelV4Usage {
  return {
    inputTokens: {
      total: result?.promptTokens,
      noCache: result?.promptTokens,
      cacheRead: undefined,
      cacheWrite: undefined,
    },
    outputTokens: {
      total: result?.completionTokens,
      text: result?.completionTokens,
      reasoning: undefined,
    },
  };
}

function getWarnings(options: LanguageModelV4CallOptions): SharedV4Warning[] {
  const warnings: SharedV4Warning[] = [];

  if (options.tools?.some((tool) => tool.type === "provider")) {
    warnings.push({
      type: "unsupported",
      feature: "provider tools",
      details: "DS4 only supports client-executed function tools.",
    });
  }

  return warnings;
}

function getThinkMode(options: LanguageModelV4CallOptions): DS4ReasoningMode {
  switch (options.reasoning) {
    case "none":
      return "none";
    case "minimal":
    case "low":
    case "medium":
    case "high":
      return "high";
    case "xhigh":
      return "max";
    case "provider-default":
    case undefined:
      return "high";
  }
}

function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) {
    throw createAbortError();
  }
}

function createAbortError(): Error {
  const error = new Error("The operation was aborted");
  error.name = "AbortError";
  return error;
}
