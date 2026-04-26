import { afterEach, describe, expect, it } from "bun:test";
import { dispatchCommand } from "../commands/dispatcher.js";
import { createInitialState } from "../utils/stateHandlers.js";
import type { AppConfig } from "../config.js";

const baseConfig: AppConfig = {
  apiUrl: "http://localhost:8000",
  authType: "none",
  isCloud: false,
  isResume: false,
};

const originalFetch = globalThis.fetch;

afterEach(() => {
  globalThis.fetch = originalFetch;
});

describe("dispatchCommand", () => {
  it("posts grouped command paths to the backend", async () => {
    const calls: Array<{ url: string; init: RequestInit | undefined }> = [];

    globalThis.fetch = (async (input, init) => {
      calls.push({ url: String(input), init });
      return new Response(JSON.stringify({ blocks: [{ kind: "section", body: "ok" }] }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }) as typeof fetch;

    const result = dispatchCommand("/workspace list", {
      state: createInitialState(),
      config: baseConfig,
      sessionId: "sess_123",
    });

    expect(result).toBeInstanceOf(Promise);
    const resolved = await result;
    expect(resolved?.type).toBe("output");
    expect(calls).toHaveLength(1);
    expect(calls[0]?.url).toBe("http://localhost:8000/sessions/sess_123/commands/workspace/list");
    expect(calls[0]?.init?.method).toBe("POST");
    expect(calls[0]?.init?.body).toBe("{}");
  });

  it("sends nested grouped arguments in the POST body", async () => {
    const calls: Array<{ url: string; init: RequestInit | undefined }> = [];

    globalThis.fetch = (async (input, init) => {
      calls.push({ url: String(input), init });
      return new Response(JSON.stringify({ blocks: [{ kind: "section", body: "ok" }] }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }) as typeof fetch;

    const result = dispatchCommand("/queue clear download", {
      state: createInitialState(),
      config: baseConfig,
      sessionId: "sess_123",
    });

    expect(result).toBeInstanceOf(Promise);
    await result;
    expect(calls).toHaveLength(1);
    expect(calls[0]?.url).toBe("http://localhost:8000/sessions/sess_123/commands/queue/clear");
    expect(calls[0]?.init?.body).toBe(JSON.stringify({ args: "download" }));
  });

  it("preserves free-form arguments for standalone bridged commands", async () => {
    const calls: Array<{ url: string; init: RequestInit | undefined }> = [];

    globalThis.fetch = (async (input, init) => {
      calls.push({ url: String(input), init });
      return new Response(JSON.stringify({ blocks: [{ kind: "section", body: "ok" }] }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }) as typeof fetch;

    const result = dispatchCommand("/read My Data/file.csv", {
      state: createInitialState(),
      config: baseConfig,
      sessionId: "sess_123",
    });

    expect(result).toBeInstanceOf(Promise);
    await result;
    expect(calls).toHaveLength(1);
    expect(calls[0]?.url).toBe("http://localhost:8000/sessions/sess_123/commands/read");
    expect(calls[0]?.init?.body).toBe(JSON.stringify({ args: "My Data/file.csv" }));
  });

  it("lists session names using backend name and last_activity fields", async () => {
    const sessionId = "11111111-1111-1111-1111-111111111111";

    globalThis.fetch = (async () => {
      return new Response(JSON.stringify({
        sessions: [
          {
            session_id: sessionId,
            name: "Cloud Analysis",
            created_at: "2026-03-18T09:00:00Z",
            last_activity: "2026-03-20T10:30:00Z",
          },
        ],
      }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }) as unknown as typeof fetch;

    const result = dispatchCommand("/sessions", {
      state: createInitialState(),
      config: { ...baseConfig, isCloud: true },
      sessionId,
    });

    expect(result).toBeInstanceOf(Promise);
    const resolved = await result;
    expect(resolved?.type).toBe("output");
    expect(resolved?.text).toContain("Cloud Analysis");
    expect(resolved?.text).toContain(sessionId);
    expect(resolved?.text).not.toContain("untitled");
  });

  it("resolves unique cloud session ID prefixes for delete", async () => {
    const targetSessionId = "11111111-1111-1111-1111-111111111111";
    const calls: Array<{ url: string; init: RequestInit | undefined }> = [];

    globalThis.fetch = (async (input, init) => {
      calls.push({ url: String(input), init });

      if (String(input).endsWith("/sessions")) {
        return new Response(JSON.stringify({
          sessions: [
            {
              session_id: targetSessionId,
              name: "Cloud Analysis",
              created_at: "2026-03-18T09:00:00Z",
            },
            {
              session_id: "22222222-2222-2222-2222-222222222222",
              name: "Other Session",
              created_at: "2026-03-17T09:00:00Z",
            },
          ],
        }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }

      return new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }) as typeof fetch;

    const result = dispatchCommand(`/sessions delete ${targetSessionId.slice(0, 8)}`, {
      state: createInitialState(),
      config: { ...baseConfig, isCloud: true },
      sessionId: targetSessionId,
    });

    expect(result).toBeInstanceOf(Promise);
    const resolved = await result;
    expect(resolved?.type).toBe("output");
    expect(calls).toHaveLength(2);
    expect(calls[0]?.url).toBe("http://localhost:8000/sessions");
    expect(calls[1]?.url).toBe(`http://localhost:8000/sessions/${targetSessionId}`);
    expect(calls[1]?.init?.method).toBe("DELETE");
    expect(resolved?.text).toContain(targetSessionId);
  });
});
