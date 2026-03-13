/**
 * SSE reconnection with Last-Event-ID support and degraded mode (protocol §5).
 *
 * Tracks the last event ID from the SSE stream. On disconnect, sends
 * Last-Event-ID header for ring buffer replay. Falls back to REST
 * hydration on cache miss (404). Implements tiered degraded mode:
 *   < 5s:       silent reconnect
 *   5-30s:      "Reconnecting..." spinner
 *   30s-2min:   "Connection lost" warning
 *   > 2min:     Print resume instructions and exit
 */

import { useState, useCallback, useRef, useEffect } from "react";
import { existsSync, readFileSync, writeFileSync, mkdirSync } from "fs";
import { join, dirname } from "path";
import { homedir } from "os";

export type DegradedLevel = "none" | "silent" | "reconnecting" | "lost" | "exit";

interface SSEReconnectState {
  lastEventId: string | null;
  degradedLevel: DegradedLevel;
  retryCount: number;
  disconnectedAt: number | null;
}

interface ResumeEntry {
  session_id: string;
  last_event_id: string | null;
  last_message_index: number;
  updated_at: string;
}

const SESSIONS_PATH = join(homedir(), ".config", "omics-os", "sessions.json");
const MAX_RETRIES = 3;

/** Read persisted session resume state. */
function readResumeState(): Record<string, ResumeEntry> {
  try {
    if (!existsSync(SESSIONS_PATH)) return {};
    return JSON.parse(readFileSync(SESSIONS_PATH, "utf-8")) as Record<string, ResumeEntry>;
  } catch {
    return {};
  }
}

/** Persist session resume state. */
function writeResumeState(sessionId: string, entry: ResumeEntry): void {
  try {
    const dir = dirname(SESSIONS_PATH);
    mkdirSync(dir, { recursive: true });
    const state = readResumeState();
    state[sessionId] = entry;
    writeFileSync(SESSIONS_PATH, JSON.stringify(state, null, 2), "utf-8");
  } catch {
    // Best-effort persistence — don't crash on write failure
  }
}

/** Load persisted last_event_id for a session. */
export function loadLastEventId(sessionId: string): string | null {
  const state = readResumeState();
  return state[sessionId]?.last_event_id ?? null;
}

export function useSSEReconnect(sessionId: string | undefined) {
  const [state, setState] = useState<SSEReconnectState>({
    lastEventId: null,
    degradedLevel: "none",
    retryCount: 0,
    disconnectedAt: null,
  });
  const intervalRef = useRef<ReturnType<typeof setInterval> | undefined>(undefined);

  // Track an SSE event ID from the stream
  const trackEventId = useCallback(
    (eventId: string) => {
      setState((prev) => ({ ...prev, lastEventId: eventId }));
      if (sessionId) {
        writeResumeState(sessionId, {
          session_id: sessionId,
          last_event_id: eventId,
          last_message_index: 0,
          updated_at: new Date().toISOString(),
        });
      }
    },
    [sessionId],
  );

  // Called when connection is lost
  const onDisconnect = useCallback(() => {
    setState((prev) => ({
      ...prev,
      disconnectedAt: prev.disconnectedAt ?? Date.now(),
      degradedLevel: "silent",
      retryCount: prev.retryCount + 1,
    }));
  }, []);

  // Called when connection is restored
  const onReconnected = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = undefined;
    }
    setState((prev) => ({
      ...prev,
      degradedLevel: "none",
      retryCount: 0,
      disconnectedAt: null,
    }));
  }, []);

  // Update degraded level based on elapsed time
  useEffect(() => {
    if (state.disconnectedAt === null) return;

    const tick = () => {
      const elapsed = Date.now() - (state.disconnectedAt ?? Date.now());

      if (state.retryCount > MAX_RETRIES && elapsed > 120_000) {
        setState((prev) => ({ ...prev, degradedLevel: "exit" }));
        return;
      }

      if (elapsed < 5_000) {
        setState((prev) => ({ ...prev, degradedLevel: "silent" }));
      } else if (elapsed < 30_000) {
        setState((prev) => ({ ...prev, degradedLevel: "reconnecting" }));
      } else {
        setState((prev) => ({ ...prev, degradedLevel: "lost" }));
      }
    };

    tick();
    intervalRef.current = setInterval(tick, 1000);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = undefined;
      }
    };
  }, [state.disconnectedAt, state.retryCount]);

  // Build reconnect headers with Last-Event-ID
  const reconnectHeaders = useCallback((): Record<string, string> => {
    if (state.lastEventId) {
      return { "Last-Event-ID": state.lastEventId };
    }
    return {};
  }, [state.lastEventId]);

  return {
    lastEventId: state.lastEventId,
    degradedLevel: state.degradedLevel,
    retryCount: state.retryCount,
    trackEventId,
    onDisconnect,
    onReconnected,
    reconnectHeaders,
  };
}
