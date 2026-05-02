/**
 * Credential file contract shared between Python lobster-ai and TypeScript @omicsos/lobster.
 * File location: ~/.config/omics-os/credentials.json
 * Permissions: 0600 (user read/write only)
 */

export interface StoredCredentials {
  auth_mode: "oauth" | "api_key";
  access_token?: string;
  refresh_token?: string;
  id_token?: string;
  client_id?: string;
  token_expiry?: string;
  api_key?: string;
  endpoint?: string;
  user_id?: string;
  email?: string;
  tier?: "free" | "cloud" | "enterprise";
}

export function isValidCredentials(data: unknown): data is StoredCredentials {
  if (!data || typeof data !== "object") return false;
  const d = data as Record<string, unknown>;
  if (d.auth_mode !== "oauth" && d.auth_mode !== "api_key") return false;
  if (d.auth_mode === "api_key" && typeof d.api_key !== "string") return false;
  if (d.auth_mode === "oauth" && typeof d.access_token !== "string") return false;
  return true;
}
