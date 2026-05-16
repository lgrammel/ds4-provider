import type {
  LanguageModelV4,
  LanguageModelV4CallOptions,
  LanguageModelV4Content,
  LanguageModelV4FinishReason,
  LanguageModelV4GenerateResult,
  LanguageModelV4Message,
  LanguageModelV4StreamPart,
  LanguageModelV4StreamResult,
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

export class DS4LanguageModel implements LanguageModelV4 {
  readonly specificationVersion = "v4";
  readonly provider = "ds4";
  readonly modelId: string;
  readonly supportedUrls = {};

  private readonly topK?: number;
  private readonly minP?: number;
  private readonly seed?: number;
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

    return {
      content: toContent(result.text, generateOptions.thinkMode !== undefined),
      finishReason: convertFinishReason(result.finishReason),
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
      options.abortSignal,
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
      messages: convertMessages(options.prompt),
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
    abortSignal?: AbortSignal,
  ): ReadableStream<LanguageModelV4StreamPart> {
    return new ReadableStream<LanguageModelV4StreamPart>({
      start: async (controller) => {
        let textStarted = false;
        let reasoningStarted = false;
        let inReasoning =
          generateOptions.thinkMode !== undefined && generateOptions.thinkMode !== "none";
        let parserBuffer = "";

        const emitReasoningDelta = (delta: string) => {
          if (delta.length === 0) {
            return;
          }

          if (!reasoningStarted) {
            controller.enqueue({ type: "reasoning-start", id: reasoningId });
            reasoningStarted = true;
          }

          controller.enqueue({ type: "reasoning-delta", id: reasoningId, delta });
        };

        const endReasoning = () => {
          if (!reasoningStarted) {
            return;
          }

          controller.enqueue({ type: "reasoning-end", id: reasoningId });
          reasoningStarted = false;
        };

        const emitTextDelta = (delta: string) => {
          if (delta.length === 0) {
            return;
          }

          if (!textStarted) {
            controller.enqueue({ type: "text-start", id: textId });
            textStarted = true;
          }

          controller.enqueue({ type: "text-delta", id: textId, delta });
        };

        const emitParsedDelta = (delta: string, finish = false) => {
          parserBuffer += delta;

          while (parserBuffer.length > 0) {
            const openingThinkIndex = parserBuffer.indexOf(THINK_OPEN);
            const closingThinkIndex = parserBuffer.indexOf(THINK_CLOSE);

            if (!inReasoning) {
              if (openingThinkIndex === -1) {
                const safeLength = finish
                  ? parserBuffer.length
                  : getSafePrefixLength(parserBuffer, THINK_OPEN);
                if (safeLength === 0) {
                  return;
                }
                emitTextDelta(parserBuffer.slice(0, safeLength));
                parserBuffer = parserBuffer.slice(safeLength);
                continue;
              }

              emitTextDelta(parserBuffer.slice(0, openingThinkIndex));
              parserBuffer = parserBuffer.slice(openingThinkIndex + THINK_OPEN.length);
              inReasoning = true;
              continue;
            }

            if (closingThinkIndex === -1) {
              const safeLength = finish
                ? parserBuffer.length
                : getSafePrefixLength(parserBuffer, THINK_CLOSE);
              if (safeLength === 0) {
                return;
              }
              emitReasoningDelta(parserBuffer.slice(0, safeLength));
              parserBuffer = parserBuffer.slice(safeLength);
              continue;
            }

            emitReasoningDelta(parserBuffer.slice(0, closingThinkIndex));
            parserBuffer = parserBuffer.slice(closingThinkIndex + THINK_CLOSE.length);
            inReasoning = false;
            endReasoning();
          }
        };

        try {
          controller.enqueue({ type: "stream-start", warnings });

          const result = await this.runWithAbortSignal(handle, abortSignal, () =>
            generateStream(handle, generateOptions, (delta) => emitParsedDelta(delta)),
          );
          emitParsedDelta("", true);
          endReasoning();

          if (textStarted) {
            controller.enqueue({ type: "text-end", id: textId });
          }

          controller.enqueue({
            type: "finish",
            finishReason: convertFinishReason(result.finishReason),
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
}

export function convertMessages(messages: LanguageModelV4Message[]): ChatMessage[] {
  const result: ChatMessage[] = [];

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
        const content = message.content
          .map((part) => {
            if (part.type === "text") {
              return part.text;
            }
            if (part.type === "reasoning") {
              return `${THINK_OPEN}${part.text}${THINK_CLOSE}`;
            }
            return "";
          })
          .join("");
        if (content.length > 0) {
          result.push({ role: "assistant", content });
        }
        break;
      }
      case "tool":
        break;
    }
  }

  return result;
}

const THINK_OPEN = "<think>";
const THINK_CLOSE = "</think>";

function toContent(text: string, initialInReasoning = false): LanguageModelV4Content[] {
  const content: LanguageModelV4Content[] = [];
  let remaining = text;
  let inReasoning = initialInReasoning || remaining.startsWith(THINK_OPEN);
  if (inReasoning) {
    if (remaining.startsWith(THINK_OPEN)) {
      remaining = remaining.slice(THINK_OPEN.length);
    }
  }

  while (remaining.length > 0) {
    if (inReasoning) {
      const endIndex = remaining.indexOf(THINK_CLOSE);
      if (endIndex === -1) {
        pushContent(content, "reasoning", remaining);
        break;
      }
      pushContent(content, "reasoning", remaining.slice(0, endIndex));
      remaining = remaining.slice(endIndex + THINK_CLOSE.length);
      inReasoning = false;
      continue;
    }

    const startIndex = remaining.indexOf(THINK_OPEN);
    if (startIndex === -1) {
      pushContent(content, "text", remaining);
      break;
    }
    pushContent(content, "text", remaining.slice(0, startIndex));
    remaining = remaining.slice(startIndex + THINK_OPEN.length);
    inReasoning = true;
  }

  return content;
}

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

function getSafePrefixLength(buffer: string, stopMarker: string): number {
  const maxOverlap = Math.min(buffer.length, stopMarker.length - 1);
  for (let overlap = maxOverlap; overlap > 0; overlap--) {
    if (stopMarker.startsWith(buffer.slice(buffer.length - overlap))) {
      return buffer.length - overlap;
    }
  }
  return buffer.length;
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

  if (options.tools?.length) {
    warnings.push({
      type: "unsupported",
      feature: "tools",
      details: "Tool use is not implemented by this provider yet.",
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
