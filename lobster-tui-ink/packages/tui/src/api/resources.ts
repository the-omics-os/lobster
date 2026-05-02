/** Resources catalog API — fetches @mention-able resources (protocol §4.4). */

import type { AppConfig } from "../config.js";
import { apiFetch } from "./apiClient.js";

export interface Resource {
  kind: string;
  id: string;
  label: string;
  description?: string;
  source?: string;
}

interface ResourcesResponse {
  resources: Resource[];
}

let cachedResources: Resource[] | null = null;

/** Fetch resources catalog. Cached for session lifetime. */
export async function fetchResources(config: AppConfig): Promise<Resource[]> {
  if (cachedResources) return cachedResources;

  try {
    const data = await apiFetch<ResourcesResponse>(config, "/resources");
    cachedResources = data.resources ?? [];
    return cachedResources;
  } catch {
    return [];
  }
}

/** Filter resources by prefix (case-insensitive). */
export function filterResources(
  resources: Resource[],
  prefix: string,
): Resource[] {
  const lower = prefix.toLowerCase();
  return resources.filter(
    (r) =>
      r.id.toLowerCase().includes(lower) ||
      r.label.toLowerCase().includes(lower),
  );
}
