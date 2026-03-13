/** Central configuration for API connectivity. */

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

export interface AppConfig {
  apiUrl: string;
  sessionId?: string;
  token?: string;
  isCloud: boolean;
}

/** Read stored token from ~/.config/omics-os/credentials.json. */
function readStoredToken(): string | undefined {
  try {
    if (!existsSync(CREDENTIALS_PATH)) return undefined;
    const data = JSON.parse(readFileSync(CREDENTIALS_PATH, "utf-8"));
    return (data as { token?: string }).token ?? (data as { access_token?: string }).access_token;
  } catch {
    return undefined;
  }
}

export function resolveConfig(args: {
  apiUrl?: string;
  sessionId?: string;
  token?: string;
  cloud?: boolean;
}): AppConfig {
  const isCloud = args.cloud ?? false;
  const apiUrl = args.apiUrl ?? (isCloud ? CLOUD_API_BASE : LOCAL_API_BASE);

  // For cloud mode: CLI token > stored credentials
  const token = args.token ?? (isCloud ? readStoredToken() : undefined);

  return {
    apiUrl,
    sessionId: args.sessionId,
    token,
    isCloud,
  };
}

/** Build the stream endpoint URL for a session. */
export function streamUrl(config: AppConfig, sessionId: string): string {
  return `${config.apiUrl}/sessions/${sessionId}/chat/stream`;
}
