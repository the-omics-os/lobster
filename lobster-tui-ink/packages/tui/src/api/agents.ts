/** Agent store REST API (protocol §4.6). */

import type { AppConfig } from "../config.js";
import { apiFetch } from "./apiClient.js";

export interface AgentSummary {
  name: string;
  version: string;
  description?: string;
  category?: string;
}

export interface AgentDetail extends AgentSummary {
  author?: string;
  capabilities?: string[];
  tools?: string[];
  created_at?: string;
}

interface AgentsListResponse {
  agents: AgentSummary[];
}

/** List all curated agents. */
export async function listAgents(config: AppConfig): Promise<AgentSummary[]> {
  try {
    const data = await apiFetch<AgentsListResponse>(config, "/agents/");
    return data.agents ?? [];
  } catch {
    return [];
  }
}

/** Get detailed info for a specific agent. */
export async function getAgentInfo(
  config: AppConfig,
  name: string,
  version?: string,
): Promise<AgentDetail | null> {
  try {
    const v = version ?? "latest";
    return await apiFetch<AgentDetail>(config, `/agents/${name}/${v}`);
  } catch {
    return null;
  }
}
