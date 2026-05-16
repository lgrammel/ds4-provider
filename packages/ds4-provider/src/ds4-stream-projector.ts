import type { LanguageModelV4StreamPart } from "@ai-sdk/provider";

export interface DS4StreamProjectorOptions {
  textId: string;
  reasoningId: string;
  initialInReasoning: boolean;
  hasTools: boolean;
  stopSequences?: string[];
}

type ProjectorPart = Exclude<
  LanguageModelV4StreamPart,
  { type: "stream-start" } | { type: "finish" } | { type: "response-metadata" }
>;

type StreamMode = "thinking" | "text" | "tool" | "suppress";
type ToolMode = "between-invokes" | "between-params" | "param-value" | "done" | "error";

interface DsmlSyntax {
  toolCallsOpen: string;
  toolCallsClose: string;
  invokeOpen: string;
  invokeClose: string;
  parameterOpen: string;
  parameterClose: string;
}

interface ToolStream {
  active: boolean;
  mode: ToolMode;
  syntax: DsmlSyntax;
  parsePos: number;
  index: number;
  ids: string[];
  currentId?: string;
  currentName?: string;
  argsOpen: boolean;
  firstProperty: boolean;
  paramName?: string;
  paramIsString: boolean;
  paramRawObject: boolean;
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

export class DS4StreamProjector {
  private mode: StreamMode;
  private emitPos = 0;
  private checkedThinkPrefix = false;
  private textStarted = false;
  private reasoningStarted = false;
  private tool?: ToolStream;
  private readonly stopSequences: string[];

  constructor(private readonly options: DS4StreamProjectorOptions) {
    this.mode = options.initialInReasoning ? "thinking" : "text";
    this.stopSequences = options.stopSequences?.filter((stop) => stop.length > 0) ?? [];
  }

  update(raw: string, final = false): ProjectorPart[] {
    const safeRaw = raw.slice(0, stopListStreamSafeLength(this.stopSequences, raw, final));
    const parts: ProjectorPart[] = [];

    if (this.mode === "thinking") {
      this.updateThinking(parts, safeRaw, final);
    }

    if (this.mode === "text") {
      this.updateText(parts, safeRaw, final);
    }

    if (this.mode === "tool") {
      this.updateTool(parts, safeRaw);
    }

    return parts;
  }

  finish(): ProjectorPart[] {
    const parts: ProjectorPart[] = [];
    if (this.reasoningStarted) {
      parts.push({ type: "reasoning-end", id: this.options.reasoningId });
      this.reasoningStarted = false;
    }
    if (this.textStarted) {
      parts.push({ type: "text-end", id: this.options.textId });
      this.textStarted = false;
    }
    return parts;
  }

  getToolCallIds(): string[] {
    return this.tool?.ids ?? [];
  }

  private updateThinking(parts: ProjectorPart[], raw: string, final: boolean): void {
    if (!this.checkedThinkPrefix) {
      if (raw.length < THINK_OPEN.length && THINK_OPEN.startsWith(raw) && !final) {
        return;
      }
      if (raw.startsWith(THINK_OPEN)) {
        this.emitPos = THINK_OPEN.length;
      }
      this.checkedThinkPrefix = true;
    }

    const close = raw.indexOf(THINK_CLOSE, this.emitPos);
    let limit: number;
    if (close !== -1) {
      limit = close;
    } else if (final) {
      limit = raw.length;
    } else {
      const hold = THINK_CLOSE.length - 1;
      limit = raw.length > hold ? raw.length - hold : this.emitPos;
      limit = utf8StreamSafeLength(raw, this.emitPos, limit, false);
    }

    if (limit > this.emitPos) {
      this.emitReasoning(parts, raw.slice(this.emitPos, limit));
      this.emitPos = limit;
    }

    if (close !== -1) {
      this.emitPos = close + THINK_CLOSE.length;
      this.endReasoning(parts);
      this.mode = "text";
    } else if (final) {
      this.endReasoning(parts);
      this.mode = "suppress";
    }
  }

  private updateText(parts: ProjectorPart[], raw: string, final: boolean): void {
    const tool = this.options.hasTools ? findDsmlToolStart(raw, this.emitPos) : undefined;
    const limit = textStreamSafeLimit(raw, this.emitPos, raw.length, this.options.hasTools, final);

    if (limit > this.emitPos) {
      this.emitText(parts, raw.slice(this.emitPos, limit));
      this.emitPos = limit;
    }

    if (tool) {
      this.endText(parts);
      this.emitPos = tool.start;
      this.tool = {
        active: true,
        mode: "between-invokes",
        syntax: tool.syntax,
        parsePos: tool.start + tool.syntax.toolCallsOpen.length,
        index: 0,
        ids: [],
        argsOpen: false,
        firstProperty: true,
        paramIsString: true,
        paramRawObject: false,
      };
      this.mode = "tool";
      this.updateTool(parts, raw);
    } else if (final) {
      this.endText(parts);
      this.mode = "suppress";
    }
  }

  private updateTool(parts: ProjectorPart[], raw: string): void {
    const tool = this.tool;
    if (!tool?.active) {
      return;
    }

    while (tool.active && tool.mode !== "done" && tool.mode !== "error") {
      if (tool.mode === "between-invokes") {
        tool.parsePos = skipWhitespace(raw, tool.parsePos);
        if (tool.parsePos >= raw.length) {
          return;
        }
        if (rawFullLit(raw, tool.parsePos, tool.syntax.toolCallsClose)) {
          tool.parsePos += tool.syntax.toolCallsClose.length;
          tool.mode = "done";
          tool.active = false;
          this.mode = "suppress";
          return;
        }
        if (rawPartialAny(raw, tool.parsePos, tool.syntax.toolCallsClose, tool.syntax.invokeOpen)) {
          return;
        }
        if (rawFullLit(raw, tool.parsePos, tool.syntax.invokeOpen)) {
          if (!this.startInvoke(parts, raw, tool)) {
            return;
          }
          continue;
        }
        tool.mode = "error";
        tool.active = false;
        this.mode = "suppress";
        return;
      }

      if (tool.mode === "between-params") {
        tool.parsePos = skipWhitespace(raw, tool.parsePos);
        if (tool.parsePos >= raw.length) {
          return;
        }
        if (rawFullLit(raw, tool.parsePos, tool.syntax.invokeClose)) {
          this.finishInvoke(parts, tool);
          tool.parsePos += tool.syntax.invokeClose.length;
          tool.index++;
          tool.mode = "between-invokes";
          continue;
        }
        if (rawPartialAny(raw, tool.parsePos, tool.syntax.invokeClose, tool.syntax.parameterOpen)) {
          return;
        }
        if (rawFullLit(raw, tool.parsePos, tool.syntax.parameterOpen)) {
          if (!this.startParam(parts, raw, tool)) {
            return;
          }
          continue;
        }
        tool.mode = "error";
        tool.active = false;
        this.mode = "suppress";
        return;
      }

      if (tool.mode === "param-value") {
        if (!this.updateParamValue(parts, raw, tool)) {
          return;
        }
        continue;
      }
    }
  }

  private startInvoke(parts: ProjectorPart[], raw: string, tool: ToolStream): boolean {
    const tagEnd = raw.indexOf(">", tool.parsePos);
    if (tagEnd === -1) {
      return false;
    }

    const tag = raw.slice(tool.parsePos, tagEnd + 1);
    const name = parseXmlAttribute(tag, "name");
    if (!name) {
      tool.mode = "error";
      tool.active = false;
      this.mode = "suppress";
      return false;
    }

    const id = tool.ids[tool.index] ?? crypto.randomUUID();
    tool.ids[tool.index] = id;
    tool.currentId = id;
    tool.currentName = name;
    tool.argsOpen = true;
    tool.firstProperty = true;
    tool.parsePos = tagEnd + 1;
    tool.mode = "between-params";

    parts.push({ type: "tool-input-start", id, toolName: name });
    parts.push({ type: "tool-input-delta", id, delta: "{" });
    return true;
  }

  private finishInvoke(parts: ProjectorPart[], tool: ToolStream): void {
    if (!tool.currentId) {
      return;
    }
    if (tool.argsOpen) {
      parts.push({ type: "tool-input-delta", id: tool.currentId, delta: "}" });
    }
    parts.push({ type: "tool-input-end", id: tool.currentId });
    tool.currentId = undefined;
    tool.currentName = undefined;
    tool.argsOpen = false;
    tool.firstProperty = true;
  }

  private startParam(parts: ProjectorPart[], raw: string, tool: ToolStream): boolean {
    const tagEnd = raw.indexOf(">", tool.parsePos);
    if (tagEnd === -1) {
      return false;
    }

    const tag = raw.slice(tool.parsePos, tagEnd + 1);
    const name = parseXmlAttribute(tag, "name");
    if (name === null || !tool.currentId) {
      tool.mode = "error";
      tool.active = false;
      this.mode = "suppress";
      return false;
    }

    const stringAttribute = parseXmlAttribute(tag, "string");
    tool.paramName = name;
    tool.paramIsString = stringAttribute === null || stringAttribute === "true";
    tool.paramRawObject = name === "" && !tool.paramIsString;
    tool.parsePos = tagEnd + 1;
    tool.mode = "param-value";

    if (!tool.paramRawObject) {
      this.emitPropertyPrefix(parts, tool, name);
      if (tool.paramIsString) {
        parts.push({ type: "tool-input-delta", id: tool.currentId, delta: '"' });
      }
    }
    return true;
  }

  private updateParamValue(parts: ProjectorPart[], raw: string, tool: ToolStream): boolean {
    const end = findLitBounded(
      raw,
      tool.parsePos,
      raw.length - tool.parsePos,
      tool.syntax.parameterClose,
    );
    const limit =
      end === -1
        ? toolParamValueStreamSafeLength(
            raw,
            tool.parsePos,
            raw.length,
            tool.syntax.parameterClose,
            tool.paramIsString,
          )
        : end;

    if (!tool.paramRawObject && limit > tool.parsePos && tool.currentId) {
      const delta = raw.slice(tool.parsePos, limit);
      parts.push({
        type: "tool-input-delta",
        id: tool.currentId,
        delta: tool.paramIsString ? jsonStringFragment(unescapeDsmlText(delta)) : delta,
      });
      tool.parsePos = limit;
    }

    if (end === -1) {
      return false;
    }

    if (tool.paramRawObject && tool.currentId) {
      const rawValue = raw.slice(tool.parsePos, end);
      const objectDelta = jsonObjectParameterDelta(rawValue, tool);
      if (objectDelta.length > 0) {
        parts.push({ type: "tool-input-delta", id: tool.currentId, delta: objectDelta });
      }
    }

    if (!tool.paramRawObject && tool.paramIsString && tool.currentId) {
      parts.push({ type: "tool-input-delta", id: tool.currentId, delta: '"' });
    }

    tool.parsePos = end + tool.syntax.parameterClose.length;
    tool.paramName = undefined;
    tool.paramRawObject = false;
    tool.paramIsString = true;
    tool.mode = "between-params";
    return true;
  }

  private emitPropertyPrefix(parts: ProjectorPart[], tool: ToolStream, name: string): void {
    if (!tool.currentId) {
      return;
    }
    const prefix = `${tool.firstProperty ? "" : ","}${JSON.stringify(name)}:`;
    parts.push({ type: "tool-input-delta", id: tool.currentId, delta: prefix });
    tool.firstProperty = false;
  }

  private emitReasoning(parts: ProjectorPart[], delta: string): void {
    if (delta.length === 0) {
      return;
    }
    if (!this.reasoningStarted) {
      parts.push({ type: "reasoning-start", id: this.options.reasoningId });
      this.reasoningStarted = true;
    }
    parts.push({ type: "reasoning-delta", id: this.options.reasoningId, delta });
  }

  private endReasoning(parts: ProjectorPart[]): void {
    if (!this.reasoningStarted) {
      return;
    }
    parts.push({ type: "reasoning-end", id: this.options.reasoningId });
    this.reasoningStarted = false;
  }

  private emitText(parts: ProjectorPart[], delta: string): void {
    if (delta.length === 0) {
      return;
    }
    if (!this.textStarted) {
      parts.push({ type: "text-start", id: this.options.textId });
      this.textStarted = true;
    }
    parts.push({ type: "text-delta", id: this.options.textId, delta });
  }

  private endText(parts: ProjectorPart[]): void {
    if (!this.textStarted) {
      return;
    }
    parts.push({ type: "text-end", id: this.options.textId });
    this.textStarted = false;
  }
}

function stopListStreamSafeLength(stops: string[], raw: string, final: boolean): number {
  if (raw.length === 0) {
    return 0;
  }

  const stop = findFirstStop(stops, raw);
  if (stop) {
    return utf8StreamSafeLength(raw, 0, stop.position, true);
  }

  if (final || stops.length === 0) {
    return utf8StreamSafeLength(raw, 0, raw.length, final);
  }

  const maxStopLength = Math.max(...stops.map((value) => value.length));
  const hold = Math.max(0, maxStopLength - 1);
  const limit = raw.length > hold ? raw.length - hold : 0;
  return utf8StreamSafeLength(raw, 0, limit, false);
}

function findFirstStop(
  stops: string[],
  raw: string,
): { position: number; length: number } | undefined {
  let best: { position: number; length: number } | undefined;
  for (const stop of stops) {
    const index = raw.indexOf(stop);
    if (index === -1) {
      continue;
    }
    if (!best || index < best.position) {
      best = { position: index, length: stop.length };
    }
  }
  return best;
}

function textStreamSafeLimit(
  raw: string,
  start: number,
  rawLength: number,
  hasTools: boolean,
  final: boolean,
): number {
  if (rawLength <= start) {
    return rawLength;
  }

  let limit = rawLength;
  if (hasTools) {
    const tool = findDsmlToolStart(raw, start);
    if (tool) {
      limit = trimToolSeparatorWhitespace(raw, start, tool.start);
      return utf8StreamSafeLength(raw, start, limit, true);
    }

    if (!final) {
      while (limit > start && /\s/u.test(raw[limit - 1] ?? "")) {
        limit--;
      }

      const maxMarker = 80;
      const scan = rawLength - start > maxMarker ? rawLength - maxMarker : start;
      for (let index = rawLength; index > scan; index--) {
        if (raw[index - 1] === "<") {
          const marker = index - 1;
          if (marker < limit) {
            limit = marker;
          }
          break;
        }
      }
      limit = trimToolSeparatorWhitespace(raw, start, limit);
    }
  }

  return utf8StreamSafeLength(raw, start, limit, final);
}

function toolParamValueStreamSafeLength(
  raw: string,
  start: number,
  rawLength: number,
  paramEnd: string,
  isString: boolean,
): number {
  let limit = rawLength;
  const hold = paramEnd.length - 1;
  limit = rawLength > hold ? rawLength - hold : start;
  if (isString) {
    limit = dsmlEntityStreamSafeLength(raw, start, limit);
  }
  return utf8StreamSafeLength(raw, start, limit, false);
}

function dsmlEntityStreamSafeLength(raw: string, start: number, limit: number): number {
  const entities = ["&amp;", "&lt;", "&gt;", "&quot;", "&apos;"];
  const scan = Math.max(start, limit - 6);
  for (let index = limit; index > scan; index--) {
    if (raw[index - 1] !== "&") {
      continue;
    }
    const suffix = raw.slice(index - 1, limit);
    if (entities.some((entity) => entity.startsWith(suffix) && suffix.length < entity.length)) {
      return index - 1;
    }
  }
  return limit;
}

function utf8StreamSafeLength(raw: string, start: number, limit: number, final: boolean): number {
  if (final || limit <= start) {
    return limit;
  }

  const last = raw.charCodeAt(limit - 1);
  if (last >= 0xd800 && last <= 0xdbff) {
    return limit - 1;
  }
  return limit;
}

function findDsmlToolStart(
  raw: string,
  fromIndex: number,
): { start: number; syntax: DsmlSyntax } | undefined {
  let best: { start: number; syntax: DsmlSyntax } | undefined;
  for (const syntax of DSML_SYNTAXES) {
    const index = raw.indexOf(syntax.toolCallsOpen, fromIndex);
    if (index === -1) {
      continue;
    }
    if (!best || index < best.start) {
      best = { start: index, syntax };
    }
  }
  return best;
}

function trimToolSeparatorWhitespace(raw: string, start: number, limit: number): number {
  let result = limit;
  while (result > start && /\s/u.test(raw[result - 1] ?? "")) {
    result--;
  }
  return result;
}

function rawFullLit(raw: string, position: number, literal: string): boolean {
  return raw.startsWith(literal, position);
}

function rawPartialLit(raw: string, position: number, literal: string): boolean {
  if (position > raw.length || raw.length - position >= literal.length) {
    return false;
  }
  return literal.startsWith(raw.slice(position));
}

function rawPartialAny(raw: string, position: number, first: string, second: string): boolean {
  return rawPartialLit(raw, position, first) || rawPartialLit(raw, position, second);
}

function findLitBounded(raw: string, start: number, length: number, literal: string): number {
  const end = start + length;
  const index = raw.indexOf(literal, start);
  return index === -1 || index + literal.length > end ? -1 : index;
}

function skipWhitespace(raw: string, index: number): number {
  let result = index;
  while (result < raw.length && /\s/u.test(raw[result] ?? "")) {
    result++;
  }
  return result;
}

function parseXmlAttribute(tag: string, name: string): string | null {
  const match = new RegExp(`${escapeRegExp(name)}="([^"]*)"`).exec(tag);
  return match ? unescapeDsmlText(match[1] ?? "") : null;
}

function jsonStringFragment(text: string): string {
  return JSON.stringify(text).slice(1, -1);
}

function jsonObjectParameterDelta(rawValue: string, tool: ToolStream): string {
  const text = rawValue.trim();
  const prefix = tool.firstProperty ? "" : ",";

  try {
    const value = JSON.parse(text) as unknown;
    if (isJsonObject(value)) {
      const inner = JSON.stringify(value).slice(1, -1);
      if (inner.length === 0) {
        return "";
      }
      tool.firstProperty = false;
      return `${prefix}${inner}`;
    }
  } catch {
    // Fall back below.
  }

  tool.firstProperty = false;
  return `${prefix}"arguments":${text.length > 0 ? text : "null"}`;
}

function isJsonObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
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
