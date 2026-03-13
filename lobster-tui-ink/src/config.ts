/** Central configuration for API connectivity. */

const CLOUD_API_BASE = "https://app.omics-os.com/api/v1";
const LOCAL_API_BASE = "http://localhost:8000";

export interface AppConfig {
  apiUrl: string;
  sessionId?: string;
  token?: string;
  isCloud: boolean;
}

export function resolveConfig(args: {
  apiUrl?: string;
  sessionId?: string;
  token?: string;
  cloud?: boolean;
}): AppConfig {
  const isCloud = args.cloud ?? false;
  const apiUrl = args.apiUrl ?? (isCloud ? CLOUD_API_BASE : LOCAL_API_BASE);

  return {
    apiUrl,
    sessionId: args.sessionId,
    token: args.token,
    isCloud,
  };
}

/** Build the stream endpoint URL for a session. */
export function streamUrl(config: AppConfig, sessionId: string): string {
  return `${config.apiUrl}/sessions/${sessionId}/chat/stream`;
}
