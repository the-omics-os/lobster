/** Central configuration for API connectivity + auth. */

import { existsSync, readFileSync, writeFileSync, mkdirSync } from "fs";
import { join, dirname } from "path";
import { homedir } from "os";

const CLOUD_API_BASE = "https://app.omics-os.com/api/v1";
const LOCAL_API_BASE = "http://localhost:8000";
const DEFAULT_CLIENT_ID = "7lgldp8e72p2lmpmi3gjbnn9uk";
const CREDENTIALS_PATH = join(
  homedir(),
  ".config",
  "omics-os",
  "credentials.json"
);

export type AuthType = "bearer" | "api-key" | "none";
export type TokenSource = "env" | "cli" | "stored" | "none";

export interface AppConfig {
  apiUrl: string;
  sessionId?: string;
  projectId?: string;
  token?: string;
  authType: AuthType;
  tokenSource: TokenSource;
  isCloud: boolean;
  isResume: boolean;
}

interface StoredCredentials {
  auth_mode?: string;
  token?: string;
  access_token?: string;
  api_key?: string;
  refresh_token?: string;
  client_id?: string;
  token_expiry?: string;
  endpoint?: string;
  id_token?: string;
  [key: string]: unknown;
}

/** Read the full stored credentials object. */
function readFullCredentials(): StoredCredentials | undefined {
  try {
    if (!existsSync(CREDENTIALS_PATH)) return undefined;
    return JSON.parse(readFileSync(CREDENTIALS_PATH, "utf-8")) as StoredCredentials;
  } catch {
    return undefined;
  }
}

/** Write credentials back to disk. */
function writeCredentials(creds: StoredCredentials): void {
  try {
    mkdirSync(dirname(CREDENTIALS_PATH), { recursive: true });
    writeFileSync(CREDENTIALS_PATH, JSON.stringify(creds, null, 2) + "\n", "utf-8");
  } catch {
    // Non-fatal — token refresh persistence failed
  }
}

/** Check if the stored OAuth token is expired (with 60s buffer). */
function isTokenExpired(creds: StoredCredentials): boolean {
  if (creds.auth_mode !== "oauth") return false;
  const expiry = creds.token_expiry;
  if (!expiry) return true;
  try {
    return new Date(expiry).getTime() < Date.now() + 60_000;
  } catch {
    return true;
  }
}

/** Attempt to refresh the OAuth token via the gateway endpoint. Returns new access_token or undefined. */
async function refreshOAuthToken(creds: StoredCredentials): Promise<string | undefined> {
  const refreshToken = creds.refresh_token;
  if (!refreshToken) return undefined;

  const endpoint = (creds.endpoint ?? "https://app.omics-os.com").replace(/\/+$/, "");
  const clientId = creds.client_id ?? DEFAULT_CLIENT_ID;
  const url = `${endpoint}/api/v1/gateway/token/refresh`;

  try {
    const resp = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refresh_token: refreshToken, client_id: clientId }),
      signal: AbortSignal.timeout(15_000),
    });
    if (!resp.ok) return undefined;

    const data = (await resp.json()) as { access_token?: string; id_token?: string; expires_in?: number };
    if (!data.access_token) return undefined;

    // Persist refreshed tokens
    creds.access_token = data.access_token;
    if (data.id_token) creds.id_token = data.id_token;
    creds.token_expiry = new Date(Date.now() + (data.expires_in ?? 3600) * 1000).toISOString();
    writeCredentials(creds);

    return data.access_token;
  } catch {
    return undefined;
  }
}

/** Read stored credentials, auto-refreshing expired OAuth tokens. */
async function readStoredCredentialsAsync(): Promise<{ token: string; authType: AuthType } | undefined> {
  const creds = readFullCredentials();
  if (!creds) return undefined;

  // API key (omk_* prefix)
  if (creds.api_key && creds.api_key.startsWith("omk_")) {
    return { token: creds.api_key, authType: "api-key" };
  }

  // OAuth mode — check expiry and refresh if needed
  if (creds.auth_mode === "oauth") {
    let token = creds.access_token;
    if (isTokenExpired(creds)) {
      const refreshed = await refreshOAuthToken(creds);
      if (refreshed) {
        token = refreshed;
      }
      // If refresh failed, return stale token — server will reject with 401
    }
    if (token) return { token, authType: "bearer" };
    return undefined;
  }

  // Legacy: plain token field
  const jwt = creds.token ?? creds.access_token;
  if (jwt) return { token: jwt, authType: "bearer" };

  return undefined;
}

/** Read stored credentials (sync, no refresh — for initial config resolution). */
function readStoredCredentials(): { token: string; authType: AuthType } | undefined {
  const creds = readFullCredentials();
  if (!creds) return undefined;

  if (creds.api_key && creds.api_key.startsWith("omk_")) {
    return { token: creds.api_key, authType: "api-key" };
  }

  const jwt = creds.token ?? creds.access_token;
  if (jwt) return { token: jwt, authType: "bearer" };

  return undefined;
}

/** Detect auth type from a raw token string. */
function detectAuthType(token: string): AuthType {
  if (token.startsWith("omk_")) return "api-key";
  return "bearer";
}

export function resolveConfig(args: {
  apiUrl?: string;
  sessionId?: string;
  projectId?: string;
  token?: string;
  cloud?: boolean;
  isResume?: boolean;
}): AppConfig {
  const isCloud = args.cloud ?? false;
  const apiUrl = args.apiUrl ?? (isCloud ? CLOUD_API_BASE : LOCAL_API_BASE);
  const isResume = args.isResume ?? false;

  // Token priority: env var > CLI flag > stored credentials > none
  const envToken = process.env.LOBSTER_TOKEN;
  if (envToken) {
    return {
      apiUrl,
      sessionId: args.sessionId,
      projectId: args.projectId,
      token: envToken,
      authType: detectAuthType(envToken),
      tokenSource: "env",
      isCloud,
      isResume,
    };
  }

  if (args.token) {
    return {
      apiUrl,
      sessionId: args.sessionId,
      projectId: args.projectId,
      token: args.token,
      authType: detectAuthType(args.token),
      tokenSource: "cli",
      isCloud,
      isResume,
    };
  }

  if (isCloud) {
    const stored = readStoredCredentials();
    if (stored) {
      return {
        apiUrl,
        sessionId: args.sessionId,
        projectId: args.projectId,
        token: stored.token,
        authType: stored.authType,
        tokenSource: "stored",
        isCloud,
        isResume,
      };
    }
  }

  return {
    apiUrl,
    sessionId: args.sessionId,
    projectId: args.projectId,
    authType: "none",
    tokenSource: "none",
    isCloud,
    isResume,
  };
}

/** Build auth headers for API requests. */
export function authHeaders(config: AppConfig): Record<string, string> {
  if (!config.token) return {};

  switch (config.authType) {
    case "api-key":
      return { "X-API-Key": config.token };
    case "bearer":
      return { Authorization: `Bearer ${config.token}` };
    default:
      return {};
  }
}

/** Build fresh auth headers, refreshing expired OAuth tokens if needed. */
export async function freshAuthHeaders(config: AppConfig): Promise<Record<string, string>> {
  // Explicit tokens (env or CLI) are never replaced by stored credentials
  if (config.tokenSource === "env" || config.tokenSource === "cli") {
    return authHeaders(config);
  }

  // Non-cloud mode — use whatever token exists as-is
  if (!config.isCloud) {
    return authHeaders(config);
  }

  // Cloud mode with stored credentials — attempt refresh if expired
  const stored = await readStoredCredentialsAsync();
  if (stored) {
    const refreshedConfig = { ...config, token: stored.token, authType: stored.authType };
    return authHeaders(refreshedConfig);
  }

  return authHeaders(config);
}

/** Build the stream endpoint URL for a session. */
export function streamUrl(config: AppConfig, sessionId: string): string {
  return `${config.apiUrl}/sessions/${sessionId}/chat/stream`;
}
