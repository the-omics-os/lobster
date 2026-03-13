/**
 * 4-layer layout engine matching Go Charm TUI height management.
 *
 * Layers:
 *   Header     — fixed 3 rows (border + content + border)
 *   Viewport   — flex, fills remaining space (min 5 rows)
 *   Input      — dynamic 1-8 rows based on content + completion menu
 *   Footer     — fixed: StatusBar (3 rows) + ActivityFeed (0-5 rows)
 */

import { useMemo } from "react";
import { useTerminalSize } from "./useTerminalSize.js";

export interface LayoutDimensions {
  /** Terminal rows */
  rows: number;
  /** Terminal columns */
  columns: number;
  /** Header height (fixed) */
  headerRows: number;
  /** Viewport height (flex, min 5) */
  viewportRows: number;
  /** Input area height (dynamic) */
  inputRows: number;
  /** Footer height (StatusBar + ActivityFeed) */
  footerRows: number;
}

interface UseLayoutOptions {
  /** Number of lines in the composer text input (1-8). Default: 1 */
  inputLineCount?: number;
  /** Whether the completion menu is open (adds up to 5 rows). Default: false */
  completionMenuOpen?: boolean;
  /** Number of visible completion items (0-5). Default: 5 when menu open */
  completionItemCount?: number;
  /** Number of activity feed events currently displayed (0-5). Default: 0 */
  activityEventCount?: number;
}

/** Fixed heights */
const HEADER_ROWS = 3; // border-top + content + border-bottom
const STATUSBAR_ROWS = 3; // border-top + content + border-bottom
const COMPOSER_BORDER_ROWS = 2; // border-top + border-bottom (from Box borderStyle="single")
const MIN_VIEWPORT_ROWS = 5;
const MAX_INPUT_CONTENT_ROWS = 8;
const MAX_COMPLETION_ROWS = 5;

export function useLayout(options: UseLayoutOptions = {}): LayoutDimensions {
  const { rows, columns } = useTerminalSize();

  const {
    inputLineCount = 1,
    completionMenuOpen = false,
    completionItemCount = MAX_COMPLETION_ROWS,
    activityEventCount = 0,
  } = options;

  return useMemo(() => {
    const headerRows = HEADER_ROWS;

    // Input: content lines (clamped 1-8) + border (2) + optional completion menu
    const contentLines = Math.max(1, Math.min(inputLineCount, MAX_INPUT_CONTENT_ROWS));
    const completionRows = completionMenuOpen
      ? Math.min(completionItemCount, MAX_COMPLETION_ROWS) + 1 // +1 for overflow indicator line
      : 0;
    const inputRows = contentLines + COMPOSER_BORDER_ROWS + completionRows;

    // Footer: StatusBar (3 always) + ActivityFeed (0-5, only when events exist)
    const activityRows = Math.min(activityEventCount, 5);
    const footerRows = STATUSBAR_ROWS + activityRows;

    // Viewport: whatever remains, clamped to min 5
    const available = rows - headerRows - inputRows - footerRows;
    const viewportRows = Math.max(MIN_VIEWPORT_ROWS, available);

    return {
      rows,
      columns,
      headerRows,
      viewportRows,
      inputRows,
      footerRows,
    };
  }, [rows, columns, inputLineCount, completionMenuOpen, completionItemCount, activityEventCount]);
}
