/** Session CRUD — create, resume, list via REST API. */

import type { AppConfig } from "../config.js";
import { authHeaders } from "../config.js";

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
  const resp = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...authHeaders(config),
    },
    body: JSON.stringify({}),
  });

  if (!resp.ok) {
    throw new Error(`Failed to create session: ${resp.status} ${resp.statusText}`);
  }

  const data = (await resp.json()) as { session_id?: string; id?: string };
  const id = data.session_id ?? data.id;
  if (!id) {
    throw new Error("Backend returned no session_id");
  }
  return id;
}

/** List sessions for the current user. */
export async function listSessions(config: AppConfig): Promise<Session[]> {
  const url = `${config.apiUrl}/sessions`;
  const resp = await fetch(url, {
    headers: {
      Accept: "application/json",
      ...authHeaders(config),
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
  } catch {
    // If backend is not available (local mode starting up), use a local ID
    return `local_${crypto.randomUUID().slice(0, 12)}`;
  }
}
