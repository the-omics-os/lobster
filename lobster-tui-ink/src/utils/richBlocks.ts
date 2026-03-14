export interface CodeBlockData {
  title?: string;
  language?: string;
  code: string;
}

export interface RichTableColumn {
  name: string;
  width?: number;
  maxWidth?: number;
  justify?: "left" | "right" | "center";
  noWrap?: boolean;
  overflow?: "crop" | "ellipsis";
}

export interface RichTableData {
  title?: string;
  headers: string[];
  rows: string[][];
  columns?: RichTableColumn[];
}

export interface AlertBlockData {
  level: "success" | "warning" | "error" | "info";
  title?: string;
  message: string;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

function asString(value: unknown): string | undefined {
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function asStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.map((entry) => String(entry ?? ""));
}

function normalizeColumns(value: unknown): RichTableColumn[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .filter(isRecord)
    .map((column) => {
      const justify =
        column.justify === "left" ||
        column.justify === "right" ||
        column.justify === "center"
          ? column.justify
          : undefined;
      const overflow =
        column.overflow === "crop" || column.overflow === "ellipsis"
          ? column.overflow
          : undefined;

      return {
        name: String(column.name ?? ""),
        width:
          typeof column.width === "number" && Number.isFinite(column.width)
            ? column.width
            : undefined,
        maxWidth:
          typeof column.maxWidth === "number" && Number.isFinite(column.maxWidth)
            ? column.maxWidth
            : typeof column.max_width === "number" && Number.isFinite(column.max_width)
              ? column.max_width
              : undefined,
        justify,
        noWrap:
          typeof column.noWrap === "boolean"
            ? column.noWrap
            : typeof column.no_wrap === "boolean"
              ? column.no_wrap
              : undefined,
        overflow,
      } satisfies RichTableColumn;
    })
    .filter((column) => column.name.length > 0);
}

export function coerceCodeBlockData(value: unknown): CodeBlockData | null {
  if (!isRecord(value)) return null;

  const code = asString(value.code) ?? asString(value.content) ?? asString(value.text);
  if (!code) return null;

  return {
    title: asString(value.title),
    language: asString(value.language),
    code,
  };
}

export function coerceRichTableData(value: unknown): RichTableData | null {
  if (!isRecord(value)) return null;

  const columns = normalizeColumns(value.columns);
  const headers = asStringArray(value.headers);
  const rows = Array.isArray(value.rows)
    ? value.rows.map((row) => asStringArray(row))
    : [];

  const normalizedHeaders =
    headers.length > 0
      ? headers
      : columns.length > 0
        ? columns.map((column) => column.name)
        : [];

  if (normalizedHeaders.length === 0 || rows.length === 0) {
    return null;
  }

  return {
    title: asString(value.title),
    headers: normalizedHeaders,
    rows,
    columns: columns.length > 0 ? columns : undefined,
  };
}

export function coerceAlertBlockData(value: unknown): AlertBlockData | null {
  if (!isRecord(value)) return null;

  const message = asString(value.message) ?? asString(value.text);
  if (!message) return null;

  const level =
    value.level === "success" ||
    value.level === "warning" ||
    value.level === "error" ||
    value.level === "info"
      ? value.level
      : "info";

  return {
    level,
    title: asString(value.title),
    message,
  };
}
