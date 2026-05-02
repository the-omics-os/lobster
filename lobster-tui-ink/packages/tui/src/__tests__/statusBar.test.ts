import { describe, expect, it } from "vitest";
import {
  buildFooterParts,
  selectVisibleFooterParts,
} from "../components/StatusBar.js";

describe("StatusBar helpers", () => {
  it("uses completion guidance when the completion menu is visible", () => {
    const parts = buildFooterParts({
      running: true,
      activeAgent: "data_expert_agent",
      tokenUsage: { promptTokens: 1200, completionTokens: 34, duration: 6.2 },
      dataStatus: null,
      sessionId: "89d3d9173b87",
      runtimeInfo: {
        provider: "anthropic",
        model: "claude-sonnet-4-20250514",
      },
      completionVisible: true,
      inputBlocked: false,
      multiline: false,
    });

    expect(parts.map((part) => part.text)).toEqual([
      "data expert agent",
      "anthropic / claude-sonnet-4-20250514",
      "1,234 tok",
      "Tab accept",
      "↑/↓ move",
      "Esc dismiss",
      "6.2s",
      "session 89d3d917",
    ]);
  });

  it("switches to interaction-blocked guidance when HITL input is active", () => {
    const parts = buildFooterParts({
      running: false,
      activeAgent: "supervisor",
      tokenUsage: null,
      dataStatus: null,
      runtimeInfo: undefined,
      completionVisible: false,
      inputBlocked: true,
      multiline: false,
    });

    expect(parts.map((part) => part.text)).toEqual([
      "ready",
      "local",
      "respond above",
      "/help",
    ]);
  });

  it("shows data status when modalities are loading or cold", () => {
    const parts = buildFooterParts({
      running: true,
      activeAgent: "data_expert_agent",
      tokenUsage: null,
      dataStatus: { cold: 1, warm: 2, hot: 3, total: 6 },
      runtimeInfo: undefined,
      completionVisible: false,
      inputBlocked: false,
      multiline: false,
    });

    const texts = parts.map((p) => p.text);
    expect(texts).toContain("2 loading, 1 on disk");
  });

  it("hides data status when all modalities are hot", () => {
    const parts = buildFooterParts({
      running: false,
      activeAgent: null,
      tokenUsage: null,
      dataStatus: { cold: 0, warm: 0, hot: 5, total: 5 },
      runtimeInfo: undefined,
      completionVisible: false,
      inputBlocked: false,
      multiline: false,
    });

    const texts = parts.map((p) => p.text);
    expect(texts).not.toContain(expect.stringContaining("loading"));
    expect(texts).not.toContain(expect.stringContaining("on disk"));
  });

  it("drops low-priority parts on narrow terminals", () => {
    const visible = selectVisibleFooterParts([
      { text: "ready", tone: "success" },
      { text: "local", tone: "muted" },
      { text: "Tab complete", tone: "dim" },
    ], 12);

    expect(visible.map((part) => part.text)).toEqual(["ready"]);
  });
});
