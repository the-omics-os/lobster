/**
 * Message history hydration from the durable message envelope (protocol §3).
 *
 * Fetches GET /sessions/{id}/messages, converts to ThreadMessageLike[],
 * and deduplicates by message_id.
 */

import type { ThreadMessageLike } from "@assistant-ui/react-ink";
import type { ReadonlyJSONObject } from "assistant-stream/utils";
import type { AppConfig } from "../config.js";
import { freshAuthHeaders } from "../config.js";
import type {
  AlertBlockData,
  CodeBlockData,
  RichTableColumn,
  RichTableData,
} from "./richBlocks.js";

/** Raw message from the REST API (durable envelope). */
interface DurableMessage {
  message_id: string;
  session_id: string;
  message_index: number;
  role: "user" | "assistant" | "tool" | "system";
  content_blocks?: ContentBlock[];
  content?: string;
  tool_calls?: ToolCall[];
  scientific_review?: unknown;
  follow_ups?: string[];
  created_at: string;
}

interface ContentBlock {
  type: string;
  text?: string;
  title?: string;
  language?: string;
  code?: string;
  content?: string;
  headers?: string[];
  rows?: string[][];
  columns?: Array<Record<string, unknown>>;
  message?: string;
  level?: string;
  tool_call_id?: string;
  tool_name?: string;
  args?: Record<string, unknown>;
  result?: unknown;
}

interface ToolCall {
  tool_call_id: string;
  tool_name: string;
  args?: Record<string, unknown>;
  result?: unknown;
  is_error?: boolean;
}

/**
 * Fetch and convert message history for a session.
 * Returns ThreadMessageLike[] suitable for `initialMessages`.
 */
export async function hydrateMessages(
  config: AppConfig,
  sessionId: string
): Promise<ThreadMessageLike[]> {
  const url = `${config.apiUrl}/sessions/${sessionId}/messages`;
  const auth = await freshAuthHeaders(config);
  const headers: Record<string, string> = {
    Accept: "application/json",
    ...auth,
  };

  try {
    const resp = await fetch(url, { headers });
    if (!resp.ok) {
      // Session might not exist yet — that's fine
      return [];
    }
    const data = (await resp.json()) as { messages?: DurableMessage[] };
    const messages = data.messages ?? (Array.isArray(data) ? (data as DurableMessage[]) : []);

    // Sort by message_index and dedup by message_id
    const seen = new Set<string>();
    const deduped: DurableMessage[] = [];
    for (const msg of messages.sort((a, b) => a.message_index - b.message_index)) {
      if (!seen.has(msg.message_id)) {
        seen.add(msg.message_id);
        deduped.push(msg);
      }
    }

    return deduped
      .filter((m) => m.role === "user" || m.role === "assistant")
      .map(convertToThreadMessageLike);
  } catch {
    // Network error — no history available
    return [];
  }
}

/** Dedup message_ids seen so far (used during live streaming to skip replays). */
export function createMessageDeduplicator(
  hydrated: ThreadMessageLike[]
): Set<string> {
  const seen = new Set<string>();
  for (const msg of hydrated) {
    if (msg.id) seen.add(msg.id);
  }
  return seen;
}

function convertToThreadMessageLike(msg: DurableMessage): ThreadMessageLike {
  const role = msg.role === "tool" ? "assistant" : msg.role;

  // Prefer content_blocks (structured) over content (plain text)
  if (msg.content_blocks && msg.content_blocks.length > 0) {
    const parts = msg.content_blocks
      .map(blockToPart)
      .filter((p): p is NonNullable<typeof p> => p !== null);

    // Append tool_calls if present
    if (msg.tool_calls) {
      for (const tc of msg.tool_calls) {
        parts.push({
          type: "tool-call" as const,
          toolCallId: tc.tool_call_id,
          toolName: tc.tool_name,
          args: tc.args as ReadonlyJSONObject | undefined,
          result: tc.result,
          isError: tc.is_error ?? false,
        });
      }
    }

    const content: ThreadMessageLike["content"] =
      parts.length > 0 ? (parts as ThreadMessageLike["content"]) : (msg.content ?? "");

    return {
      role: role as "user" | "assistant",
      id: msg.message_id ?? ((msg as unknown as Record<string, unknown>).id as string),
      content,
      createdAt: msg.created_at ? new Date(msg.created_at) : new Date(),
      attachments: [],
    };
  }

  // Fallback: plain text content
  return {
    role: role as "user" | "assistant",
    id: msg.message_id ?? ((msg as unknown as Record<string, unknown>).id as string),
    content: msg.content ?? "",
    createdAt: msg.created_at ? new Date(msg.created_at) : new Date(),
    attachments: [],
  };
}

type MessagePart =
  | { readonly type: "text"; readonly text: string }
  | { readonly type: "data-code"; readonly data: CodeBlockData }
  | { readonly type: "data-table"; readonly data: RichTableData }
  | { readonly type: "data-alert"; readonly data: AlertBlockData }
  | {
      readonly type: "tool-call";
      readonly toolCallId: string;
      readonly toolName: string;
      readonly args?: ReadonlyJSONObject;
      readonly result?: unknown;
      readonly isError?: boolean;
    };

function blockToPart(block: ContentBlock): MessagePart | null {
  switch (block.type) {
    case "text":
      return { type: "text", text: block.text ?? "" };
    case "code": {
      const code = block.code ?? block.content ?? block.text ?? "";
      if (!code) {
        return null;
      }
      return {
        type: "data-code",
        data: {
          title: block.title,
          language: block.language,
          code,
        },
      };
    }
    case "table": {
      const headers = Array.isArray(block.headers) ? block.headers.map(String) : [];
      const rows = Array.isArray(block.rows)
        ? block.rows.map((row) => (Array.isArray(row) ? row.map(String) : []))
        : [];
      const columns = Array.isArray(block.columns)
        ? block.columns
            .filter((column): column is Record<string, unknown> => !!column && typeof column === "object")
            .map<RichTableColumn>((column) => ({
              name: String(column.name ?? ""),
              width:
                typeof column.width === "number" && Number.isFinite(column.width)
                  ? column.width
                  : undefined,
              maxWidth:
                typeof column.max_width === "number" && Number.isFinite(column.max_width)
                  ? column.max_width
                  : typeof column.maxWidth === "number" && Number.isFinite(column.maxWidth)
                    ? column.maxWidth
                    : undefined,
              justify:
                column.justify === "left" ||
                column.justify === "right" ||
                column.justify === "center"
                  ? column.justify
                  : undefined,
              noWrap:
                typeof column.no_wrap === "boolean"
                  ? column.no_wrap
                  : typeof column.noWrap === "boolean"
                    ? column.noWrap
                    : undefined,
              overflow:
                column.overflow === "crop" || column.overflow === "ellipsis"
                  ? column.overflow
                  : undefined,
            }))
            .filter((column) => column.name.length > 0)
        : [];

      const normalizedHeaders =
        headers.length > 0 ? headers : columns.map((column) => column.name);
      if (normalizedHeaders.length === 0 || rows.length === 0) {
        return null;
      }

      return {
        type: "data-table",
        data: {
          title: block.title,
          headers: normalizedHeaders,
          rows,
          columns: columns.length > 0 ? columns : undefined,
        },
      };
    }
    case "alert": {
      const message = block.message ?? block.text ?? block.content ?? "";
      if (!message) {
        return null;
      }
      const level =
        block.level === "success" ||
        block.level === "warning" ||
        block.level === "error" ||
        block.level === "info"
          ? block.level
          : "info";
      return {
        type: "data-alert",
        data: {
          level,
          title: block.title,
          message,
        },
      };
    }
    case "tool_call":
    case "tool-call":
      return {
        type: "tool-call",
        toolCallId: block.tool_call_id ?? "",
        toolName: block.tool_name ?? "unknown",
        args: block.args as ReadonlyJSONObject | undefined,
        result: block.result,
      };
    default:
      return null;
  }
}
