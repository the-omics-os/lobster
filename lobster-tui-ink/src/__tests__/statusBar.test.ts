import { describe, expect, it } from "bun:test";
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

  it("drops low-priority parts on narrow terminals", () => {
    const visible = selectVisibleFooterParts([
      { text: "ready", tone: "success" },
      { text: "local", tone: "muted" },
      { text: "Tab complete", tone: "dim" },
    ], 12);

    expect(visible.map((part) => part.text)).toEqual(["ready"]);
  });
});
