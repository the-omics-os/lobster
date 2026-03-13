/**
 * Message history hydration from the durable message envelope (protocol §3).
 *
 * Fetches GET /sessions/{id}/messages, converts to ThreadMessageLike[],
 * and deduplicates by message_id.
 */

import type { ThreadMessageLike } from "@assistant-ui/react-ink";
import type { ReadonlyJSONObject } from "assistant-stream/utils";
import type { AppConfig } from "../config.js";

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
  const headers: Record<string, string> = {
    Accept: "application/json",
  };
  if (config.token) {
    headers["Authorization"] = `Bearer ${config.token}`;
  }

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
      id: msg.message_id,
      content,
      createdAt: new Date(msg.created_at),
    };
  }

  // Fallback: plain text content
  return {
    role: role as "user" | "assistant",
    id: msg.message_id,
    content: msg.content ?? "",
    createdAt: new Date(msg.created_at),
  };
}

type MessagePart =
  | { readonly type: "text"; readonly text: string }
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
    case "tool_call":
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
