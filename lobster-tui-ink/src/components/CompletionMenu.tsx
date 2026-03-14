/**
 * Styled completion overlay for slash commands and @mentions.
 *
 * Renders above the Composer as a floating list with keyboard navigation
 * managed by the parent composer.
 */

import React from "react";
import { Box, Text } from "ink";
import { useTheme } from "../hooks/useTheme.js";

export interface CompletionItem {
  /** Display label */
  label: string;
  /** Value to insert on accept */
  value: string;
  /** Optional description shown dimmed after the label */
  description?: string;
}

export interface CompletionMenuProps {
  /** Items to display */
  items: CompletionItem[];
  /** Selected item index */
  selectedIndex: number;
  /** Whether the menu is visible */
  visible: boolean;
  /** Horizontal anchor aligned to the active completion token */
  anchorColumn?: number;
  /** Max visible items before overflow indicators appear. Default: 5 */
  maxVisible?: number;
}

export const MAX_VISIBLE_COMPLETION_ITEMS = 5;

function clampIndex(index: number, count: number) {
  if (count <= 0) return 0;
  if (index < 0) return 0;
  if (index >= count) return count - 1;
  return index;
}

export function getCompletionWindow(
  selectedIndex: number,
  itemCount: number,
  maxVisible = MAX_VISIBLE_COMPLETION_ITEMS,
) {
  if (itemCount <= 0) {
    return { start: 0, end: 0 };
  }

  const limit = Math.min(itemCount, maxVisible);
  const selected = clampIndex(selectedIndex, itemCount);
  let start = selected - limit + 1;
  if (start < 0) {
    start = 0;
  }

  let end = start + limit;
  if (end > itemCount) {
    end = itemCount;
    start = Math.max(0, end - limit);
  }

  return { start, end };
}

export function getCompletionMenuRowCount(
  itemCount: number,
  selectedIndex: number,
  maxVisible = MAX_VISIBLE_COMPLETION_ITEMS,
) {
  if (itemCount <= 0) {
    return 0;
  }

  const { start, end } = getCompletionWindow(selectedIndex, itemCount, maxVisible);
  const hiddenAbove = itemCount - end;
  const hiddenBelow = start;

  return (end - start) + (hiddenAbove > 0 ? 1 : 0) + (hiddenBelow > 0 ? 1 : 0);
}

export function CompletionMenu({
  items,
  selectedIndex,
  visible,
  anchorColumn = 0,
  maxVisible = MAX_VISIBLE_COMPLETION_ITEMS,
}: CompletionMenuProps) {
  const theme = useTheme();

  if (!visible || items.length === 0) return null;

  const total = items.length;
  const selected = clampIndex(selectedIndex, total);
  const { start, end } = getCompletionWindow(selected, total, maxVisible);
  const hiddenAbove = total - end;
  const hiddenBelow = start;
  const paddingLeft = Math.max(0, anchorColumn);

  return (
    <Box flexDirection="column">
      {hiddenAbove > 0 && (
        <Box paddingLeft={paddingLeft}>
          <Text dimColor>{`  +${hiddenAbove} more`}</Text>
        </Box>
      )}
      {Array.from({ length: end - start }, (_, offset) => end - 1 - offset).map((itemIndex) => {
        const item = items[itemIndex]!;
        const isSelected = itemIndex === selected;
        return (
          <Box key={`${item.value}:${itemIndex}`} paddingLeft={paddingLeft}>
            <Text color={isSelected ? theme.primary : theme.textMuted} bold={isSelected}>
              {isSelected ? "\u203A " : "\u00B7 "}
            </Text>
            <Text color={isSelected ? theme.primary : undefined} bold={isSelected}>
              {item.label}
            </Text>
            {item.description && (
              <Text dimColor>{`  ${item.description}`}</Text>
            )}
          </Box>
        );
      })}
      {hiddenBelow > 0 && (
        <Box paddingLeft={paddingLeft}>
          <Text dimColor>{`  +${hiddenBelow} earlier`}</Text>
        </Box>
      )}
    </Box>
  );
}
