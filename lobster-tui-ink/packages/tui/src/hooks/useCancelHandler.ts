import { useState, useEffect, useRef, useCallback } from "react";
import { useInput } from "ink";

interface CancelState {
  isCanceling: boolean;
  showWarning: boolean;
}

/**
 * Two-phase Ctrl+C cancellation matching Go TUI behavior:
 * 1st Ctrl+C → show "Press Ctrl+C again to cancel", start 2s timer
 * 2nd Ctrl+C within 2s → call onCancel()
 * Timer expiry → reset
 */
export function useCancelHandler(onCancel: () => void) {
  const [state, setState] = useState<CancelState>({
    isCanceling: false,
    showWarning: false,
  });
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const reset = useCallback(() => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    setState({ isCanceling: false, showWarning: false });
  }, []);

  useInput((_input, key) => {
    // Ctrl+C comes as key.ctrl with input 'c'
    if (!key.ctrl || _input !== "c") return;

    if (state.isCanceling) {
      // Second press within window → actually cancel
      reset();
      onCancel();
    } else {
      // First press → enter canceling state
      setState({ isCanceling: true, showWarning: true });
      timerRef.current = setTimeout(reset, 2000);
    }
  });

  // Cleanup timer on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  return state;
}
