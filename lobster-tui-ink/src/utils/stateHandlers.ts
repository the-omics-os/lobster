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
  progress: 1,
  alerts: 1,
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

export interface ActivityEvent {
  type: string;
  agent?: string;
  from_agent?: string;
  to_agent?: string;
  tool?: string;
  tool_name?: string;
  id?: string;
  tool_call_id?: string;
  duration_ms?: number;
  error?: boolean | string;
  summary?: string;
  message?: string;
  task_description?: string;
  kind?: string;
  status?: string;
}

export interface ProgressEntry {
  id: string;
  label: string;
  current: number;
  total: number;
  done: boolean;
}

export interface AlertEvent {
  level: "success" | "warning" | "error" | "info";
  title?: string;
  message: string;
}

/** Accumulated state from patches. */
export interface AppState {
  plots: unknown[];
  modalities: unknown[];
  activeAgent: string | null;
  agentStatus: string | null;
  activityEvents: ActivityEvent[];
  progress: Record<string, ProgressEntry>;
  alerts: AlertEvent[];
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
    progress: {},
    alerts: [],
    tokenUsage: null,
    sessionTitle: null,
  };
}

const ACTIVITY_EVENT_LIMIT = 25;
const ALERT_EVENT_LIMIT = 8;

function asArray<T>(value: unknown): T[] {
  if (Array.isArray(value)) {
    return value as T[];
  }
  if (value === null || value === undefined) {
    return [];
  }
  return [value as T];
}

function eventToolKey(event: ActivityEvent) {
  return event.tool_call_id ?? event.id ?? null;
}

function isToolTerminalEvent(event: ActivityEvent) {
  return event.type === "tool_complete" || event.type === "tool_error";
}

function appendActivityEvents(
  current: ActivityEvent[],
  data: unknown,
): ActivityEvent[] {
  const next = [...current];

  for (const rawEvent of asArray<ActivityEvent>(data)) {
    if (!rawEvent || typeof rawEvent !== "object") {
      continue;
    }

    const event = rawEvent as ActivityEvent;
    const toolKey = eventToolKey(event);

    if (toolKey && isToolTerminalEvent(event)) {
      const existingIndex = next.findIndex(
        (candidate) => eventToolKey(candidate) === toolKey,
      );
      if (existingIndex >= 0) {
        next[existingIndex] = {
          ...next[existingIndex],
          ...event,
        };
        continue;
      }
    }

    next.push(event);
  }

  return next.slice(-ACTIVITY_EVENT_LIMIT);
}

function normalizeProgressEntries(data: unknown): ProgressEntry[] {
  const rawEntries: Record<string, unknown>[] = Array.isArray(data)
    ? (data as Record<string, unknown>[])
    : data && typeof data === "object"
      ? Object.entries(data as Record<string, unknown>).map(([id, value]) => ({
          id,
          ...(value && typeof value === "object" ? (value as Record<string, unknown>) : {}),
        }))
      : asArray<Record<string, unknown>>(data);

  return rawEntries
    .filter((entry) => !!entry && typeof entry === "object")
    .map((entry, index) => {
      const label =
        typeof entry.label === "string" && entry.label.trim().length > 0
          ? entry.label
          : `Progress ${index + 1}`;
      const id =
        typeof entry.id === "string" && entry.id.trim().length > 0
          ? entry.id
          : label;

      return {
        id,
        label,
        current:
          typeof entry.current === "number" && Number.isFinite(entry.current)
            ? entry.current
            : 0,
        total:
          typeof entry.total === "number" && Number.isFinite(entry.total)
            ? entry.total
            : -1,
        done: Boolean(entry.done),
      } satisfies ProgressEntry;
    });
}

function mergeProgressState(
  current: Record<string, ProgressEntry>,
  data: unknown,
): Record<string, ProgressEntry> {
  const next = { ...current };

  for (const entry of normalizeProgressEntries(data)) {
    if (entry.done) {
      delete next[entry.id];
      continue;
    }
    next[entry.id] = entry;
  }

  return next;
}

function appendAlerts(current: AlertEvent[], data: unknown): AlertEvent[] {
  const next = [...current];

  for (const rawAlert of asArray<Record<string, unknown>>(data)) {
    if (!rawAlert || typeof rawAlert !== "object") {
      continue;
    }

    const level =
      rawAlert.level === "success" ||
      rawAlert.level === "warning" ||
      rawAlert.level === "error" ||
      rawAlert.level === "info"
        ? rawAlert.level
        : "info";

    const message =
      typeof rawAlert.message === "string" && rawAlert.message.length > 0
        ? rawAlert.message
        : "";

    if (!message) {
      continue;
    }

    next.push({
      level,
      title:
        typeof rawAlert.title === "string" && rawAlert.title.length > 0
          ? rawAlert.title
          : undefined,
      message,
    });
  }

  return next.slice(-ALERT_EVENT_LIMIT);
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
        activityEvents: appendActivityEvents(state.activityEvents, data),
      };
    case "progress":
      return {
        ...state,
        progress: mergeProgressState(state.progress, data),
      };
    case "alerts":
      return {
        ...state,
        alerts: appendAlerts(state.alerts, data),
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
