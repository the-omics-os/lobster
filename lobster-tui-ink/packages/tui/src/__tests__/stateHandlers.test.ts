import { describe, expect, it } from "vitest";
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

  it("handles null data for known state keys without crashing", () => {
    const result = processStatePatch("active_agent", null);
    expect(result).toEqual({ key: "active_agent", data: null });

    const initial = createInitialState();
    const next = applyStatePatch(initial, "active_agent", null);
    expect(next.activeAgent).toBeNull();
  });

  it("ignores unknown future schema versions", () => {
    const result = processStatePatch("progress", {
      _v: 999,
      label: "Downloading",
    });

    expect(result).toBeNull();
  });

  it("derives dataStatus summary from modalities patch", () => {
    const initial = createInitialState();
    const next = applyStatePatch(initial, "modalities", [
      { name: "rna_seq", data_status: "hot" },
      { name: "proteomics", data_status: "warm" },
      { name: "genomics", data_status: "cold" },
      { name: "metabolomics" },
    ]);

    expect(next.dataStatus).toEqual({ cold: 1, warm: 1, hot: 2, total: 4 });
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
