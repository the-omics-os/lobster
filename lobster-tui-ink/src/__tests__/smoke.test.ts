import { describe, it, expect } from "bun:test";
import { snapshotMessage } from "../utils/snapshotMessage.js";
import { isLikelySensitive } from "../hooks/useHistory.js";

describe("snapshotMessage", () => {
  it("formats user messages", () => {
    const result = snapshotMessage({
      role: "user",
      content: [{ type: "text", text: "Hello world" }],
      status: { type: "complete" },
    });
    expect(result).toContain("You:");
    expect(result).toContain("Hello world");
  });

  it("formats assistant text messages", () => {
    const result = snapshotMessage({
      role: "assistant",
      content: [{ type: "text", text: "Here is the analysis" }],
      status: { type: "complete" },
    });
    expect(result).toContain("Lobster:");
    expect(result).toContain("Here is the analysis");
  });

  it("formats tool call parts", () => {
    const result = snapshotMessage({
      role: "assistant",
      content: [
        { type: "tool-call", toolName: "search_pubmed", result: "3 results" },
      ],
      status: { type: "complete" },
    });
    expect(result).toContain("search_pubmed");
    expect(result).toContain("done");
  });

  it("formats reasoning parts as collapsed", () => {
    const result = snapshotMessage({
      role: "assistant",
      content: [{ type: "reasoning", text: "Let me think about this..." }],
      status: { type: "complete" },
    });
    expect(result).toContain("Thinking... (collapsed)");
  });

  it("handles multi-part assistant messages", () => {
    const result = snapshotMessage({
      role: "assistant",
      content: [
        { type: "reasoning", text: "thinking..." },
        { type: "tool-call", toolName: "load_data", result: "ok" },
        { type: "text", text: "Analysis complete." },
      ],
      status: { type: "complete" },
    });
    expect(result).toContain("Lobster:");
    expect(result).toContain("Thinking... (collapsed)");
    expect(result).toContain("load_data");
    expect(result).toContain("Analysis complete.");
  });

  it("formats rich data parts", () => {
    const result = snapshotMessage({
      role: "assistant",
      content: [
        {
          type: "data",
          name: "code",
          data: { language: "python", code: 'print("hello")' },
        },
        {
          type: "data",
          name: "alert",
          data: { level: "warning", title: "Careful", message: "Check inputs" },
        },
      ],
      status: { type: "complete" },
    });

    expect(result).toContain("print(\"hello\")");
    expect(result).toContain("Careful");
    expect(result).toContain("Check inputs");
  });
});

describe("isLikelySensitive", () => {
  it("detects Omics-OS API keys", () => {
    expect(isLikelySensitive("omk_test1234567890")).toBe(true);
  });

  it("detects JWT-like tokens", () => {
    expect(isLikelySensitive("eyJhbGciOiJIUzI1NiJ9.payload.sig")).toBe(true);
  });

  it("detects OpenAI-style keys", () => {
    expect(isLikelySensitive("sk-abc123def456")).toBe(true);
  });

  it("detects Bearer tokens", () => {
    expect(isLikelySensitive("Bearer eyJhbG...")).toBe(true);
    expect(isLikelySensitive("bearer sometoken")).toBe(true);
  });

  it("allows normal commands", () => {
    expect(isLikelySensitive("/help")).toBe(false);
    expect(isLikelySensitive("Search PubMed for CRISPR")).toBe(false);
    expect(isLikelySensitive("/data")).toBe(false);
    expect(isLikelySensitive("analyze the RNA-seq dataset")).toBe(false);
  });
});
