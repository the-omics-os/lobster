/**
 * Styled completion overlay for slash commands and @mentions.
 *
 * Renders above the Composer as a floating list with keyboard navigation.
 * Max 5 visible items with overflow indicators.
 */

import React, { useState, useCallback, useEffect } from "react";
import { Box, Text, useInput } from "ink";

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
  /** Called when an item is accepted (Tab/Enter) */
  onAccept: (item: CompletionItem) => void;
  /** Called when the menu is dismissed (Esc) */
  onDismiss: () => void;
  /** Whether the menu is visible */
  visible: boolean;
  /** Max visible items before overflow indicators appear. Default: 5 */
  maxVisible?: number;
}

const MAX_DEFAULT = 5;

export function CompletionMenu({
  items,
  onAccept,
  onDismiss,
  visible,
  maxVisible = MAX_DEFAULT,
}: CompletionMenuProps) {
  const [selectedIndex, setSelectedIndex] = useState(0);

  // Reset selection when items change
  useEffect(() => {
    setSelectedIndex(0);
  }, [items]);

  useInput(
    (input, key) => {
      if (!visible || items.length === 0) return;

      if (key.upArrow) {
        setSelectedIndex((prev) => (prev > 0 ? prev - 1 : items.length - 1));
      } else if (key.downArrow) {
        setSelectedIndex((prev) => (prev < items.length - 1 ? prev + 1 : 0));
      } else if (key.tab || key.return) {
        const item = items[selectedIndex];
        if (item) onAccept(item);
      } else if (key.escape) {
        onDismiss();
      }
    },
    { isActive: visible },
  );

  if (!visible || items.length === 0) return null;

  // Calculate visible window
  const total = items.length;
  const windowSize = Math.min(total, maxVisible);

  // Center the window around the selected index
  let windowStart: number;
  if (total <= maxVisible) {
    windowStart = 0;
  } else {
    const half = Math.floor(windowSize / 2);
    windowStart = Math.max(0, Math.min(selectedIndex - half, total - windowSize));
  }
  const windowEnd = windowStart + windowSize;

  const earlierCount = windowStart;
  const laterCount = total - windowEnd;

  return (
    <Box flexDirection="column" paddingX={1}>
      {earlierCount > 0 && (
        <Text dimColor>{`  +${earlierCount} earlier`}</Text>
      )}
      {items.slice(windowStart, windowEnd).map((item, i) => {
        const globalIndex = windowStart + i;
        const isSelected = globalIndex === selectedIndex;
        return (
          <Box key={item.value + globalIndex}>
            <Text color={isSelected ? "red" : undefined} bold={isSelected}>
              {isSelected ? "\u203A " : "\u00B7 "}
            </Text>
            <Text color={isSelected ? "red" : undefined} bold={isSelected}>
              {item.label}
            </Text>
            {item.description && (
              <Text dimColor>{`  ${item.description}`}</Text>
            )}
          </Box>
        );
      })}
      {laterCount > 0 && (
        <Text dimColor>{`  +${laterCount} more`}</Text>
      )}
    </Box>
  );
}
