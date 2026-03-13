/**
 * Schema-versioned state handler per protocol §1.3.
 *
 * Rules from the cross-surface protocol:
 * 1. Unknown `aui-state:` keys → silently ignore (never crash)
 * 2. Known key with `_v` > supported version → ignore the value
 * 3. New optional fields within known schemas → tolerate (no crash)
 *
 * State patches arrive via the DataStream's `onData` callback with
 * `type: "data-<name>"` for ui-message-stream protocol, or via
 * native `aui-state:` chunks in the data-stream protocol.
 */

/** Supported schema versions for known state keys. */
const SUPPORTED_VERSIONS: Record<string, number> = {
  plots: 1,
  modalities: 1,
  active_agent: 1,
  agent_status: 1,
  activity_events: 1,
  token_usage: 1,
  session_title: 1,
};

/** Known state key names. */
export type StateKey = keyof typeof SUPPORTED_VERSIONS;

/** State patch with optional version field. */
interface StatePatch {
  _v?: number;
  [key: string]: unknown;
}

/** Callback type for state updates. */
export type StateUpdateCallback = (key: StateKey, data: unknown) => void;

/** Accumulated state from patches. */
export interface AppState {
  plots: unknown[];
  modalities: unknown[];
  activeAgent: string | null;
  agentStatus: string | null;
  activityEvents: unknown[];
  tokenUsage: { promptTokens?: number; completionTokens?: number } | null;
  sessionTitle: string | null;
}

export function createInitialState(): AppState {
  return {
    plots: [],
    modalities: [],
    activeAgent: null,
    agentStatus: null,
    activityEvents: [],
    tokenUsage: null,
    sessionTitle: null,
  };
}

/**
 * Process a state patch, applying schema versioning rules.
 * Returns the validated data if accepted, or null if rejected.
 */
export function processStatePatch(
  key: string,
  data: unknown
): { key: StateKey; data: unknown } | null {
  // Rule 1: Unknown keys → silently ignore
  if (!(key in SUPPORTED_VERSIONS)) {
    return null;
  }

  const typedKey = key as StateKey;
  const patch = data as StatePatch;

  // Rule 2: Check _v field — if higher than supported, ignore
  if (patch._v !== undefined && patch._v > SUPPORTED_VERSIONS[typedKey]!) {
    return null;
  }

  // Rule 3: Accept the data (new optional fields are tolerated by nature of unknown)
  return { key: typedKey, data };
}

/**
 * Apply a validated state patch to the app state.
 * Returns a new state object (immutable update).
 */
export function applyStatePatch(
  state: AppState,
  key: StateKey,
  data: unknown
): AppState {
  switch (key) {
    case "plots":
      return { ...state, plots: Array.isArray(data) ? data : [data] };
    case "modalities":
      return { ...state, modalities: Array.isArray(data) ? data : [data] };
    case "active_agent":
      return { ...state, activeAgent: typeof data === "string" ? data : null };
    case "agent_status":
      return { ...state, agentStatus: typeof data === "string" ? data : null };
    case "activity_events":
      return {
        ...state,
        activityEvents: Array.isArray(data) ? data : [data],
      };
    case "token_usage":
      return {
        ...state,
        tokenUsage: data as AppState["tokenUsage"],
      };
    case "session_title":
      return {
        ...state,
        sessionTitle: typeof data === "string" ? data : null,
      };
    default:
      return state;
  }
}

/**
 * Create an onData handler for useDataStreamRuntime.
 * Filters and processes state patches according to protocol §1.3.
 */
export function createDataHandler(
  onStateUpdate: StateUpdateCallback
): (data: { type: string; name: string; data: unknown }) => void {
  return ({ name, data }) => {
    const result = processStatePatch(name, data);
    if (result) {
      onStateUpdate(result.key, result.data);
    }
  };
}
