/**
 * Hook for 'o' key to open plots/tables in browser.
 * Tracks the last openable URL/path and opens on keypress.
 */

import { useState, useCallback } from "react";
import { useInput } from "ink";
import { openInBrowser } from "../utils/openBrowser.js";

export function useBrowserOpen() {
  const [lastTarget, setLastTarget] = useState<string | undefined>();
  const [lastOpened, setLastOpened] = useState<string | undefined>();

  useInput((input) => {
    if (input === "o" && lastTarget) {
      openInBrowser(lastTarget);
      setLastOpened(lastTarget);
    }
  });

  const setOpenable = useCallback((target: string) => {
    setLastTarget(target);
  }, []);

  const clearOpenable = useCallback(() => {
    setLastTarget(undefined);
  }, []);

  return { setOpenable, clearOpenable, lastOpened, hasOpenable: !!lastTarget };
}
