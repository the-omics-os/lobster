/**
 * Bootstrap endpoint — single fetch for flags, resources, and templates.
 * Falls back to 3 separate fetches if /bootstrap is not available (old backend).
 */

import type { AppConfig } from "../config.js";
import { apiFetch } from "./apiClient.js";
import { fetchFeatureFlags, type FeatureFlags } from "./featureFlags.js";
import { fetchTemplates, type PromptTemplate } from "./templates.js";
import { fetchResources, type Resource } from "./resources.js";

export interface BootstrapData {
  flags: FeatureFlags;
  resources: Resource[];
  templates: PromptTemplate[];
}

/** Fetch bootstrap data. Tries /bootstrap first, falls back to 3 separate fetches. */
export async function fetchBootstrap(config: AppConfig): Promise<BootstrapData> {
  try {
    return await apiFetch<BootstrapData>(config, "/bootstrap");
  } catch {
    // Fallback: old backend doesn't have /bootstrap yet
    const [flags, resources, templates] = await Promise.all([
      fetchFeatureFlags(config),
      fetchResources(config),
      fetchTemplates(config),
    ]);
    return { flags, resources, templates };
  }
}
