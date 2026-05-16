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
      content: toTextContent(result.text),
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
    const warnings = getWarnings(options);
    const stream = this.createStream(
      handle,
      generateOptions,
      textId,
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

    return body;
  }

  private createStream(
    handle: number,
    generateOptions: GenerateOptions,
    textId: string,
    warnings: SharedV4Warning[],
    abortSignal?: AbortSignal,
  ): ReadableStream<LanguageModelV4StreamPart> {
    return new ReadableStream<LanguageModelV4StreamPart>({
      start: async (controller) => {
        let textStarted = false;

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

        try {
          controller.enqueue({ type: "stream-start", warnings });

          const result = await this.runWithAbortSignal(handle, abortSignal, () =>
            generateStream(handle, generateOptions, emitTextDelta),
          );

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

function toTextContent(text: string): LanguageModelV4Content[] {
  if (text.length === 0) {
    return [];
  }

  return [
    {
      type: "text",
      text,
      providerMetadata: undefined,
    },
  ];
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
  if (options.reasoning && options.reasoning !== "none") {
    warnings.push({
      type: "unsupported",
      feature: "reasoning",
      details: "Reasoning is not implemented by this provider yet.",
    });
  }

  return warnings;
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
