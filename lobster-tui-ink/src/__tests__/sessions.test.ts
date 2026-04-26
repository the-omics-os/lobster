import { describe, expect, it } from "bun:test";
import { resolveSessionId } from "../api/sessions.js";
import type { AppConfig } from "../config.js";

const baseConfig: AppConfig = {
  apiUrl: "http://localhost:8000",
  authType: "none",
  isCloud: true,
  isResume: true,
};

describe("resolveSessionId", () => {
  it("rejects malformed cloud resume IDs before hitting the backend", async () => {
    await expect(
      resolveSessionId({
        ...baseConfig,
        sessionId: "not-a-uuid",
      }),
    ).rejects.toThrow('Invalid session ID format: "not-a-uuid". Expected a UUID.');
  });

  it("accepts valid cloud UUID resume IDs", async () => {
    const sessionId = "11111111-1111-1111-1111-111111111111";

    await expect(
      resolveSessionId({
        ...baseConfig,
        sessionId,
      }),
    ).resolves.toBe(sessionId);
  });
});
