/**
 * SSE reconnection with Last-Event-ID support (protocol §5).
 *
 * Tracks the last event ID from the SSE stream. On disconnect, sends
 * Last-Event-ID header for ring buffer replay.
 *
 * writeResumeState is debounced: in-memory update is immediate,
 * disk flush happens after 2s of inactivity (max 1 write/2sec vs 15-30/sec).
 */

import { useState, useCallback, useRef, useEffect } from "react";
import { existsSync, readFileSync, writeFileSync, mkdirSync } from "fs";
import { join, dirname } from "path";
import { homedir } from "os";

export type DegradedLevel = "none" | "reconnecting" | "lost" | "exit";

interface SSEReconnectState {
  lastEventId: string | null;
  degradedLevel: DegradedLevel;
  retryCount: number;
}

interface ResumeEntry {
  session_id: string;
  last_event_id: string | null;
  last_message_index: number;
  updated_at: string;
}

const SESSIONS_PATH = join(homedir(), ".config", "omics-os", "sessions.json");
const DEBOUNCE_MS = 2000;

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
function writeResumeStateToDisk(sessionId: string, entry: ResumeEntry): void {
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
  });

  // Debounce disk writes: store pending entry in ref, flush after 2s idle
  const pendingEntryRef = useRef<ResumeEntry | null>(null);
  const flushTimerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  const flush = useCallback(() => {
    if (pendingEntryRef.current && sessionId) {
      writeResumeStateToDisk(sessionId, pendingEntryRef.current);
      pendingEntryRef.current = null;
    }
  }, [sessionId]);

  // Flush on unmount
  useEffect(() => {
    return () => {
      if (flushTimerRef.current) clearTimeout(flushTimerRef.current);
      flush();
    };
  }, [flush]);

  // Track an SSE event ID from the stream (in-memory immediate, disk debounced)
  const trackEventId = useCallback(
    (eventId: string) => {
      setState((prev) => ({ ...prev, lastEventId: eventId }));
      if (sessionId) {
        pendingEntryRef.current = {
          session_id: sessionId,
          last_event_id: eventId,
          last_message_index: 0,
          updated_at: new Date().toISOString(),
        };
        if (flushTimerRef.current) clearTimeout(flushTimerRef.current);
        flushTimerRef.current = setTimeout(flush, DEBOUNCE_MS);
      }
    },
    [sessionId, flush],
  );

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
    reconnectHeaders,
  };
}
