/** Session CRUD — create, resume, list via REST API. */

import type { AppConfig } from "../config.js";
import { freshAuthHeaders } from "../config.js";

export interface Session {
  session_id: string;
  name?: string;
  status?: string;
  created_at: string;
  last_activity?: string;
  message_count?: number;
  client_source?: string;
}

const UUID_RE =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export function isUuidSessionId(value: string): boolean {
  return UUID_RE.test(value);
}

/** Create a new session on the backend. Returns session_id. */
export async function createSession(config: AppConfig): Promise<string> {
  const url = `${config.apiUrl}/sessions`;
  const auth = await freshAuthHeaders(config);
  const resp = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...auth,
    },
    body: JSON.stringify({ name: "CLI Session", client_source: "cli" }),
  });

  if (!resp.ok) {
    throw new Error(`Failed to create session: ${resp.status} ${resp.statusText}`);
  }

  const data = (await resp.json()) as {
    session_id?: string;
    id?: string;
    session?: { session_id?: string };
  };
  // Backend wraps response as SessionResponse: { session: { session_id } }
  const id = data.session?.session_id ?? data.session_id ?? data.id;
  if (!id) {
    throw new Error("Backend returned no session_id");
  }
  return id;
}

/** List sessions for the current user. */
export async function listSessions(config: AppConfig): Promise<Session[]> {
  const url = `${config.apiUrl}/sessions`;
  const auth = await freshAuthHeaders(config);
  const resp = await fetch(url, {
    headers: {
      Accept: "application/json",
      ...auth,
    },
  });

  if (!resp.ok) {
    return [];
  }

  const data = (await resp.json()) as { sessions?: Session[] } | Session[];
  return Array.isArray(data) ? data : (data.sessions ?? []);
}

/** Resolve session: use provided ID, or create a new one. */
export async function resolveSessionId(config: AppConfig): Promise<string> {
  if (config.sessionId) {
    if (config.isCloud && !isUuidSessionId(config.sessionId)) {
      throw new Error(
        `Invalid session ID format: "${config.sessionId}". Expected a UUID.`,
      );
    }
    return config.sessionId;
  }

  try {
    return await createSession(config);
  } catch (error) {
    if (config.isCloud) {
      // In cloud mode, propagate the error — local IDs aren't valid UUIDs
      throw error;
    }
    // Local mode: backend might not be up yet, use a temporary local ID
    return `local_${crypto.randomUUID().slice(0, 12)}`;
  }
}
