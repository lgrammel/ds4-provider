import { createRequire } from "node:module";
import { stat } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);

configureMetalSourcePaths();

const binding = require(
  join(__dirname, "..", "build", "Release", "ds4_binding.node"),
) as NativeBinding;

export interface LoadModelOptions {
  modelPath: string;
  mtpPath?: string;
  contextSize?: number;
  threads?: number;
  backend?: "metal" | "cuda" | "cpu";
  mtpDraftTokens?: number;
  mtpMargin?: number;
  warmWeights?: boolean;
  quality?: boolean;
  debug?: boolean;
}

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface GenerateOptions {
  messages: ChatMessage[];
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
  minP?: number;
  seed?: number;
  stopSequences?: string[];
}

export interface GenerateResult {
  text: string;
  promptTokens: number;
  completionTokens: number;
  finishReason: "stop" | "length" | "error";
  errorMessage?: string;
}

interface NativeBinding {
  loadModel(
    options: LoadModelOptions,
    callback: (error: string | null, handle: number | null) => void,
  ): void;
  unloadModel(handle: number): boolean;
  generate(
    handle: number,
    options: GenerateOptions,
    callback: (error: string | null, result: GenerateResult | null) => void,
  ): void;
  generateStream(
    handle: number,
    options: GenerateOptions,
    tokenCallback: (token: string) => void,
    doneCallback: (error: string | null, result: GenerateResult | null) => void,
  ): void;
  cancelGeneration(handle: number): boolean;
  isModelLoaded(handle: number): boolean;
}

export async function loadModel(options: LoadModelOptions): Promise<number> {
  const nativeOptions = omitUndefinedLoadModelOptions(options);

  await Promise.all([
    validateModelPath(nativeOptions.modelPath, "modelPath"),
    nativeOptions.mtpPath ? validateModelPath(nativeOptions.mtpPath, "mtpPath") : undefined,
  ]);

  return new Promise((resolve, reject) => {
    binding.loadModel(nativeOptions, (error, handle) => {
      if (error) {
        reject(new Error(error));
      } else if (handle !== null) {
        resolve(handle);
      } else {
        reject(new Error("Failed to load DS4 model: unknown error"));
      }
    });
  });
}

export function unloadModel(handle: number): boolean {
  return binding.unloadModel(handle);
}

export function generate(handle: number, options: GenerateOptions): Promise<GenerateResult> {
  return new Promise((resolve, reject) => {
    binding.generate(handle, options, (error, result) => {
      if (error) {
        reject(new Error(error));
      } else if (result) {
        if (result.finishReason === "error") {
          reject(new Error(result.errorMessage ?? "DS4 generation failed"));
        } else {
          resolve(result);
        }
      } else {
        reject(new Error("DS4 generation failed: unknown error"));
      }
    });
  });
}

export function generateStream(
  handle: number,
  options: GenerateOptions,
  onToken: (token: string) => void,
): Promise<GenerateResult> {
  return new Promise((resolve, reject) => {
    binding.generateStream(handle, options, onToken, (error, result) => {
      if (error) {
        reject(new Error(error));
      } else if (result) {
        if (result.finishReason === "error") {
          reject(new Error(result.errorMessage ?? "DS4 stream failed"));
        } else {
          resolve(result);
        }
      } else {
        reject(new Error("DS4 stream failed: unknown error"));
      }
    });
  });
}

export function cancelGeneration(handle: number): boolean {
  return binding.cancelGeneration(handle);
}

export function isModelLoaded(handle: number): boolean {
  return binding.isModelLoaded(handle);
}

function configureMetalSourcePaths(): void {
  const metalDir = join(__dirname, "..", "ds4", "metal");
  const sources = {
    DS4_METAL_FLASH_ATTN_SOURCE: "flash_attn.metal",
    DS4_METAL_DENSE_SOURCE: "dense.metal",
    DS4_METAL_MOE_SOURCE: "moe.metal",
    DS4_METAL_DSV4_HC_SOURCE: "dsv4_hc.metal",
    DS4_METAL_UNARY_SOURCE: "unary.metal",
    DS4_METAL_DSV4_KV_SOURCE: "dsv4_kv.metal",
    DS4_METAL_DSV4_ROPE_SOURCE: "dsv4_rope.metal",
    DS4_METAL_DSV4_MISC_SOURCE: "dsv4_misc.metal",
    DS4_METAL_ARGSORT_SOURCE: "argsort.metal",
    DS4_METAL_CPY_SOURCE: "cpy.metal",
    DS4_METAL_CONCAT_SOURCE: "concat.metal",
    DS4_METAL_GET_ROWS_SOURCE: "get_rows.metal",
    DS4_METAL_SUM_ROWS_SOURCE: "sum_rows.metal",
    DS4_METAL_SOFTMAX_SOURCE: "softmax.metal",
    DS4_METAL_REPEAT_SOURCE: "repeat.metal",
    DS4_METAL_GLU_SOURCE: "glu.metal",
    DS4_METAL_NORM_SOURCE: "norm.metal",
    DS4_METAL_BIN_SOURCE: "bin.metal",
    DS4_METAL_SET_ROWS_SOURCE: "set_rows.metal",
  };

  for (const [envName, fileName] of Object.entries(sources)) {
    process.env[envName] ??= join(metalDir, fileName);
  }
}

function omitUndefinedLoadModelOptions(options: LoadModelOptions): LoadModelOptions {
  const nativeOptions: LoadModelOptions = {
    modelPath: options.modelPath,
  };

  if (options.mtpPath !== undefined) {
    nativeOptions.mtpPath = options.mtpPath;
  }
  if (options.contextSize !== undefined) {
    nativeOptions.contextSize = options.contextSize;
  }
  if (options.threads !== undefined) {
    nativeOptions.threads = options.threads;
  }
  if (options.backend !== undefined) {
    nativeOptions.backend = options.backend;
  }
  if (options.mtpDraftTokens !== undefined) {
    nativeOptions.mtpDraftTokens = options.mtpDraftTokens;
  }
  if (options.mtpMargin !== undefined) {
    nativeOptions.mtpMargin = options.mtpMargin;
  }
  if (options.warmWeights !== undefined) {
    nativeOptions.warmWeights = options.warmWeights;
  }
  if (options.quality !== undefined) {
    nativeOptions.quality = options.quality;
  }
  if (options.debug !== undefined) {
    nativeOptions.debug = options.debug;
  }

  return nativeOptions;
}

async function validateModelPath(path: string, optionName: "modelPath" | "mtpPath"): Promise<void> {
  if (path.startsWith("~/")) {
    throw new Error(
      `Failed to load DS4 model: ${optionName} uses '~', which is not expanded automatically. ` +
        `Pass an absolute path instead. Received: ${path}`,
    );
  }

  let file;
  try {
    file = await stat(path);
  } catch (error) {
    if (error instanceof Error && "code" in error && error.code === "ENOENT") {
      throw new Error(`Failed to load DS4 model: ${optionName} file does not exist at ${path}`);
    }

    throw error;
  }

  if (!file.isFile()) {
    throw new Error(
      `Failed to load DS4 model: expected ${optionName} to be a file but found a directory at ${path}`,
    );
  }
}
