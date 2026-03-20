/** Session CRUD — create, resume, list via REST API. */

import type { AppConfig } from "../config.js";
import { freshAuthHeaders } from "../config.js";

export interface Session {
  session_id: string;
  title?: string;
  created_at: string;
  updated_at?: string;
  message_count?: number;
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
