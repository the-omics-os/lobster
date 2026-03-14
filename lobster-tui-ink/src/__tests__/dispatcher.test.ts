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
});
