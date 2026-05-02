/** Thin auth-aware REST client for shared backend APIs. */

import type { AppConfig } from "../config.js";
import { freshAuthHeaders } from "../config.js";

interface ApiRequestOptions {
  method?: "GET" | "POST" | "PATCH" | "DELETE";
  body?: unknown;
  timeoutMs?: number;
}

/** Fetch JSON from a backend endpoint with auth headers. */
export async function apiFetch<T>(
  config: AppConfig,
  path: string,
  options: ApiRequestOptions = {},
): Promise<T> {
  const url = `${config.apiUrl}${path}`;
  const method = options.method ?? "GET";
  const controller = typeof options.timeoutMs === "number" ? new AbortController() : undefined;
  const timeoutId = controller
    ? setTimeout(() => controller.abort(), options.timeoutMs)
    : undefined;

  const auth = await freshAuthHeaders(config);

  let resp: Response;
  try {
    resp = await fetch(url, {
      method,
      headers: {
        Accept: "application/json",
        ...(options.body !== undefined ? { "Content-Type": "application/json" } : {}),
        ...auth,
      },
      ...(options.body !== undefined ? { body: JSON.stringify(options.body) } : {}),
      ...(controller ? { signal: controller.signal } : {}),
    });
  } catch (error) {
    if (
      options.timeoutMs !== undefined &&
      error instanceof Error &&
      error.name === "AbortError"
    ) {
      throw new Error(`API ${path}: timed out after ${options.timeoutMs}ms`);
    }
    throw error;
  } finally {
    if (timeoutId !== undefined) {
      clearTimeout(timeoutId);
    }
  }

  if (resp.status === 401) {
    throw new Error(
      `Authentication failed (401). Run 'lobster cloud login' to re-authenticate.`
    );
  }

  if (!resp.ok) {
    throw new Error(`API ${path}: ${resp.status} ${resp.statusText}`);
  }

  if (resp.status === 204) {
    return {} as T;
  }

  return (await resp.json()) as T;
}
