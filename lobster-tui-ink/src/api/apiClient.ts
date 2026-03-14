/** Thin auth-aware REST client for shared backend APIs. */

import type { AppConfig } from "../config.js";
import { authHeaders } from "../config.js";

interface ApiRequestOptions {
  method?: "GET" | "POST";
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

  let resp: Response;
  try {
    resp = await fetch(url, {
      method,
      headers: {
        Accept: "application/json",
        ...(options.body !== undefined ? { "Content-Type": "application/json" } : {}),
        ...authHeaders(config),
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

  if (!resp.ok) {
    throw new Error(`API ${path}: ${resp.status} ${resp.statusText}`);
  }

  return (await resp.json()) as T;
}
