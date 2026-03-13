/** Thin auth-aware REST client for shared backend APIs. */

import type { AppConfig } from "../config.js";
import { authHeaders } from "../config.js";

/** Fetch JSON from a backend endpoint with auth headers. */
export async function apiFetch<T>(
  config: AppConfig,
  path: string,
): Promise<T> {
  const url = `${config.apiUrl}${path}`;
  const resp = await fetch(url, {
    headers: {
      Accept: "application/json",
      ...authHeaders(config),
    },
  });

  if (!resp.ok) {
    throw new Error(`API ${path}: ${resp.status} ${resp.statusText}`);
  }

  return (await resp.json()) as T;
}
