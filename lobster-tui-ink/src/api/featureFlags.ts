/** Feature flags — fetched once on startup, cached for session lifetime. */

import type { AppConfig } from "../config.js";
import { apiFetch } from "./apiClient.js";

export interface FeatureFlags {
  projects_datasets: boolean;
  session_state_read: boolean;
  session_state_write: boolean;
  message_ddb_primary: boolean;
  curated_agent_store: boolean;
  playground: boolean;
}

const DEFAULT_FLAGS: FeatureFlags = {
  projects_datasets: false,
  session_state_read: false,
  session_state_write: false,
  message_ddb_primary: false,
  curated_agent_store: false,
  playground: false,
};

let cachedFlags: FeatureFlags | undefined;

/** Fetch feature flags from backend. Returns defaults on error. */
export async function fetchFeatureFlags(
  config: AppConfig,
): Promise<FeatureFlags> {
  if (cachedFlags) return cachedFlags;

  try {
    const data = await apiFetch<Record<string, boolean>>(
      config,
      "/config/flags",
    );
    cachedFlags = { ...DEFAULT_FLAGS, ...data };
  } catch {
    cachedFlags = DEFAULT_FLAGS;
  }

  return cachedFlags;
}

/** Get cached flags (undefined if not yet fetched). */
export function getCachedFlags(): FeatureFlags | undefined {
  return cachedFlags;
}

/** Check if a specific flag is enabled. */
export function isFeatureEnabled(
  flag: keyof FeatureFlags,
): boolean {
  return cachedFlags?.[flag] ?? false;
}
