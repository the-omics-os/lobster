/** Prompt templates — fetched on new session, displayed as suggestions. */

import type { AppConfig } from "../config.js";
import { apiFetch } from "./apiClient.js";

export interface PromptTemplate {
  id: string;
  title: string;
  template: string;
  category: string;
  placeholders: string[];
}

interface TemplatesResponse {
  templates: PromptTemplate[];
}

/** Fetch prompt templates from backend. Returns empty array on error. */
export async function fetchTemplates(
  config: AppConfig,
): Promise<PromptTemplate[]> {
  try {
    const data = await apiFetch<TemplatesResponse>(
      config,
      "/config/templates",
    );
    return data.templates ?? [];
  } catch {
    return [];
  }
}
