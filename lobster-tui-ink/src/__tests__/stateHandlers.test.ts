import { describe, expect, it } from "bun:test";
import {
  applyStatePatch,
  createInitialState,
  processStatePatch,
} from "../utils/stateHandlers.js";
import {
  applyAppStatePatch,
  clearRunActivity,
  createAppStateStore,
} from "../utils/appStateStore.js";

describe("stateHandlers", () => {
  it("appends activity events instead of replacing them", () => {
    const initial = createInitialState();
    const withStart = applyStatePatch(initial, "activity_events", [
      { type: "tool_start", tool_name: "search_pubmed", tool_call_id: "tc-1" },
    ]);
    const withFinish = applyStatePatch(withStart, "activity_events", [
      { type: "tool_complete", tool_name: "search_pubmed", tool_call_id: "tc-1" },
    ]);

    expect(withFinish.activityEvents).toHaveLength(1);
    expect(withFinish.activityEvents[0]?.type).toBe("tool_complete");
  });

  it("normalizes mapped progress patches", () => {
    const initial = createInitialState();
    const next = applyStatePatch(initial, "progress", {
      download: { label: "Downloading", current: 2, total: 4, done: false },
    });

    expect(next.progress.download).toEqual({
      id: "download",
      label: "Downloading",
      current: 2,
      total: 4,
      done: false,
    });
  });

  it("ignores unknown future schema versions", () => {
    const result = processStatePatch("progress", {
      _v: 999,
      label: "Downloading",
    });

    expect(result).toBeNull();
  });

  it("clears per-run activity without resetting other state", () => {
    const store = createAppStateStore();
    applyAppStatePatch(store, "activity_events", [
      { type: "tool_start", tool_name: "read_file", tool_call_id: "tc-1" },
    ]);
    applyAppStatePatch(store, "progress", {
      read: { label: "Reading", current: 1, total: 4, done: false },
    });
    applyAppStatePatch(store, "session_title", "Debug session");

    clearRunActivity(store);

    expect(store.getState().activityEvents).toEqual([]);
    expect(store.getState().progress).toEqual({});
    expect(store.getState().sessionTitle).toBe("Debug session");
  });
});
