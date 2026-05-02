/**
 * Command history with persistence and draft restore.
 * Arrow-up/down cycles through previous inputs.
 * Persists to ~/.config/lobster/history (max 500 entries).
 */

import { useState, useCallback, useEffect, useRef } from "react";
import { existsSync, readFileSync, writeFileSync, mkdirSync } from "fs";
import { join, dirname } from "path";
import { homedir } from "os";

const HISTORY_PATH = join(homedir(), ".config", "lobster", "history");
const MAX_ENTRIES = 500;

function loadHistory(): string[] {
  try {
    if (!existsSync(HISTORY_PATH)) return [];
    return readFileSync(HISTORY_PATH, "utf-8")
      .split("\n")
      .filter(Boolean);
  } catch {
    return [];
  }
}

const SENSITIVE_PATTERNS = [/^omk_/, /^eyJ/, /^sk-/, /^Bearer\s/i];

export function isLikelySensitive(value: string): boolean {
  return SENSITIVE_PATTERNS.some((re) => re.test(value.trim()));
}

function saveHistory(entries: string[]) {
  try {
    const dir = dirname(HISTORY_PATH);
    if (!existsSync(dir)) {
      mkdirSync(dir, { recursive: true });
    }
    const safe = entries.filter((e) => !isLikelySensitive(e));
    const trimmed = safe.slice(-MAX_ENTRIES);
    writeFileSync(HISTORY_PATH, trimmed.join("\n") + "\n");
  } catch {
    // Silently fail on write errors
  }
}

export function useHistory() {
  const [entries, setEntries] = useState<string[]>(loadHistory);
  const [index, setIndex] = useState(-1);
  const draftRef = useRef("");

  // Save to disk when entries change
  useEffect(() => {
    if (entries.length > 0) {
      saveHistory(entries);
    }
  }, [entries]);

  /** Add a new entry to history. */
  const push = useCallback((value: string) => {
    const trimmed = value.trim();
    if (!trimmed) return;
    setEntries((prev) => {
      // Deduplicate consecutive entries
      if (prev[prev.length - 1] === trimmed) return prev;
      return [...prev, trimmed];
    });
    setIndex(-1);
    draftRef.current = "";
  }, []);

  /** Navigate up (older). Returns the history entry or undefined. */
  const up = useCallback(
    (currentDraft: string): string | undefined => {
      if (entries.length === 0) return undefined;

      const newIndex = index === -1 ? entries.length - 1 : Math.max(0, index - 1);

      // Save draft on first up press
      if (index === -1) {
        draftRef.current = currentDraft;
      }

      setIndex(newIndex);
      return entries[newIndex];
    },
    [entries, index],
  );

  /** Navigate down (newer). Returns the history entry, draft, or undefined. */
  const down = useCallback((): string | undefined => {
    if (index === -1) return undefined;

    const newIndex = index + 1;
    if (newIndex >= entries.length) {
      // Back to draft
      setIndex(-1);
      return draftRef.current;
    }

    setIndex(newIndex);
    return entries[newIndex];
  }, [entries, index]);

  /** Reset navigation (e.g., when user submits). */
  const reset = useCallback(() => {
    setIndex(-1);
    draftRef.current = "";
  }, []);

  return { push, up, down, reset, entries };
}
