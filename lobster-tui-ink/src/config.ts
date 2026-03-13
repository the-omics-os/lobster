/** Central configuration for API connectivity + auth. */

import { existsSync, readFileSync } from "fs";
import { join } from "path";
import { homedir } from "os";

const CLOUD_API_BASE = "https://app.omics-os.com/api/v1";
const LOCAL_API_BASE = "http://localhost:8000";
const CREDENTIALS_PATH = join(
  homedir(),
  ".config",
  "omics-os",
  "credentials.json"
);

export type AuthType = "bearer" | "api-key" | "none";

export interface AppConfig {
  apiUrl: string;
  sessionId?: string;
  token?: string;
  authType: AuthType;
  isCloud: boolean;
}

interface StoredCredentials {
  token?: string;
  access_token?: string;
  api_key?: string;
}

/** Read stored credentials from ~/.config/omics-os/credentials.json. */
function readStoredCredentials(): { token: string; authType: AuthType } | undefined {
  try {
    if (!existsSync(CREDENTIALS_PATH)) return undefined;
    const data = JSON.parse(readFileSync(CREDENTIALS_PATH, "utf-8")) as StoredCredentials;

    // API key (omk_* prefix)
    if (data.api_key && data.api_key.startsWith("omk_")) {
      return { token: data.api_key, authType: "api-key" };
    }

    // JWT / OAuth token
    const jwt = data.token ?? data.access_token;
    if (jwt) {
      return { token: jwt, authType: "bearer" };
    }

    return undefined;
  } catch {
    return undefined;
  }
}

/** Detect auth type from a raw token string. */
function detectAuthType(token: string): AuthType {
  if (token.startsWith("omk_")) return "api-key";
  return "bearer";
}

export function resolveConfig(args: {
  apiUrl?: string;
  sessionId?: string;
  token?: string;
  cloud?: boolean;
}): AppConfig {
  const isCloud = args.cloud ?? false;
  const apiUrl = args.apiUrl ?? (isCloud ? CLOUD_API_BASE : LOCAL_API_BASE);

  // Token priority: CLI flag > stored credentials > none
  if (args.token) {
    return {
      apiUrl,
      sessionId: args.sessionId,
      token: args.token,
      authType: detectAuthType(args.token),
      isCloud,
    };
  }

  if (isCloud) {
    const stored = readStoredCredentials();
    if (stored) {
      return {
        apiUrl,
        sessionId: args.sessionId,
        token: stored.token,
        authType: stored.authType,
        isCloud,
      };
    }
  }

  return {
    apiUrl,
    sessionId: args.sessionId,
    authType: "none",
    isCloud,
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

/** Build the stream endpoint URL for a session. */
export function streamUrl(config: AppConfig, sessionId: string): string {
  return `${config.apiUrl}/sessions/${sessionId}/chat/stream`;
}
