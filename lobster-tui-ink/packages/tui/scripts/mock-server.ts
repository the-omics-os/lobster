#!/usr/bin/env bun
/**
 * Mock DataStream server for testing the Ink TUI.
 * Uses the ui-message-stream protocol (SSE with JSON payloads).
 *
 * Usage: bun run scripts/mock-server.ts
 * Then: bun run src/cli.tsx --api-url=http://localhost:3333
 */

const PORT = 3333;

function sseEvent(data: unknown): string {
  return `data: ${JSON.stringify(data)}\n\n`;
}

async function handleStream(req: Request): Promise<Response> {
  const body = await req.json().catch(() => ({}));
  const messages =
    (body as { messages?: { role: string; content: string }[] }).messages ?? [];
  const userMsg = messages[messages.length - 1]?.content ?? "hello";

  const stream = new ReadableStream({
    async start(controller) {
      const encoder = new TextEncoder();

      // Start message
      controller.enqueue(
        encoder.encode(sseEvent({ type: "start", messageId: "msg_1" }))
      );

      // Stream text word by word
      const response = `Here is a **markdown** response to: "${userMsg}"

- Item one
- Item two

\`\`\`python
print("hello world")
\`\`\`

Done!`;

      const words = response.split(" ");
      for (const word of words) {
        controller.enqueue(
          encoder.encode(sseEvent({ type: "text-delta", textDelta: word + " " }))
        );
        await new Promise((r) => setTimeout(r, 60));
      }

      // Finish
      controller.enqueue(
        encoder.encode(
          sseEvent({
            type: "finish",
            finishReason: "stop",
            usage: { promptTokens: 10, completionTokens: words.length },
          })
        )
      );

      // DONE marker
      controller.enqueue(encoder.encode("data: [DONE]\n\n"));
      controller.close();
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
      "Access-Control-Allow-Origin": "*",
    },
  });
}

Bun.serve({
  port: PORT,
  fetch(req) {
    const url = new URL(req.url);

    if (req.method === "OPTIONS") {
      return new Response(null, {
        headers: {
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Methods": "POST, OPTIONS",
          "Access-Control-Allow-Headers": "Content-Type, Authorization",
        },
      });
    }

    if (url.pathname.endsWith("/chat/stream") && req.method === "POST") {
      return handleStream(req);
    }

    return new Response("Mock DataStream server\n", { status: 200 });
  },
});

console.log(`Mock DataStream server running on http://localhost:${PORT}`);
console.log("Stream endpoint: POST /sessions/{id}/chat/stream");
