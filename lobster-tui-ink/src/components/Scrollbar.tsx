/**
 * Vertical scrollbar rendered alongside the ChatViewport.
 *
 * Track: dimmed dots spanning the viewport height.
 * Thumb: solid block, proportional to content/viewport ratio.
 * Appears only when content exceeds the viewport.
 */

import React from "react";
import { Box, Text } from "ink";

export interface ScrollbarProps {
  /** Visible viewport height in rows */
  viewportHeight: number;
  /** Total content height in rows */
  contentHeight: number;
  /** Current scroll offset from top (0 = top) */
  scrollOffset: number;
}

const TRACK_CHAR = "\u00B7"; // middle dot
const THUMB_CHAR = "\u2588"; // full block

export function Scrollbar({
  viewportHeight,
  contentHeight,
  scrollOffset,
}: ScrollbarProps) {
  // Don't render if content fits in viewport
  if (contentHeight <= viewportHeight || viewportHeight < 1) {
    return (
      <Box flexDirection="column" width={1}>
        {Array.from({ length: viewportHeight }, (_, i) => (
          <Text key={i} dimColor>
            {" "}
          </Text>
        ))}
      </Box>
    );
  }

  // Calculate thumb size (minimum 1 row)
  const ratio = viewportHeight / contentHeight;
  const thumbSize = Math.max(1, Math.round(ratio * viewportHeight));

  // Calculate thumb position
  const maxScroll = contentHeight - viewportHeight;
  const scrollFraction = maxScroll > 0 ? scrollOffset / maxScroll : 0;
  const maxThumbTop = viewportHeight - thumbSize;
  const thumbTop = Math.round(scrollFraction * maxThumbTop);

  // Build the column
  const rows: React.ReactNode[] = [];
  for (let i = 0; i < viewportHeight; i++) {
    const isThumb = i >= thumbTop && i < thumbTop + thumbSize;
    rows.push(
      <Text key={i} {...(isThumb ? { color: "red" } : { dimColor: true })}>
        {isThumb ? THUMB_CHAR : TRACK_CHAR}
      </Text>,
    );
  }

  return <Box flexDirection="column" width={1}>{rows}</Box>;
}
