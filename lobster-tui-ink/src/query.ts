/**
 * Non-interactive query mode.
 * Sends a single message, streams the response as plain text to stdout, then exits.
 */

import type { AppConfig } from "./config.js";
import { authHeaders } from "./config.js";
import { resolveSessionId } from "./api/sessions.js";

export async function runQuery(config: AppConfig, message: string): Promise<void> {
  const sessionId = await resolveSessionId(config);
  const url = `${config.apiUrl}/sessions/${sessionId}/chat/stream`;

  const resp = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
      ...authHeaders(config),
    },
    body: JSON.stringify({
      messages: [{ role: "user", content: message }],
    }),
  });

  if (!resp.ok) {
    process.stderr.write(`Error: ${resp.status} ${resp.statusText}\n`);
    process.exit(1);
  }

  if (!resp.body) {
    process.stderr.write("Error: No response body\n");
    process.exit(1);
  }

  // Parse the ui-message-stream format and extract text chunks
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      // ui-message-stream text prefix: "0:"
      if (line.startsWith("0:")) {
        try {
          const text = JSON.parse(line.slice(2)) as string;
          process.stdout.write(text);
        } catch {
          // Not valid JSON text chunk, skip
        }
      }
    }
  }

  // Process remaining buffer
  if (buffer.startsWith("0:")) {
    try {
      const text = JSON.parse(buffer.slice(2)) as string;
      process.stdout.write(text);
    } catch {
      // Skip
    }
  }

  process.stdout.write("\n");
}
