import React, { useRef, useCallback, useEffect, useState } from "react";
import { Box, useInput, useStdout } from "ink";
import { ScrollView, type ScrollViewRef } from "ink-scroll-view";
import { Scrollbar } from "./Scrollbar.js";

const PAGE_SIZE = 10;

export interface ChatViewportProps {
  children: React.ReactNode;
  /** Viewport height from useLayout. Falls back to terminal-based calc. */
  viewportHeight?: number;
}

export function ChatViewport({ children, viewportHeight: heightProp }: ChatViewportProps) {
  const scrollRef = useRef<ScrollViewRef>(null);
  const { stdout } = useStdout();

  // Fallback height calculation when layout hook isn't wired yet
  const fallbackHeight = (stdout?.rows ?? 24) - 6;
  const viewportHeight = heightProp ?? fallbackHeight;

  // Track scroll state for scrollbar
  const [scrollOffset, setScrollOffset] = useState(0);
  const [contentHeight, setContentHeight] = useState(0);

  useInput((input, key) => {
    if (key.pageUp) {
      scrollRef.current?.scrollBy(-PAGE_SIZE);
    } else if (key.pageDown) {
      scrollRef.current?.scrollBy(PAGE_SIZE);
    }
  });

  // Auto-scroll to bottom when content changes
  const handleContentHeightChange = useCallback(
    (height: number) => {
      setContentHeight(height);
      scrollRef.current?.scrollToBottom();
    },
    [],
  );

  const handleScroll = useCallback((offset: number) => {
    setScrollOffset(offset);
  }, []);

  // Handle terminal resize
  useEffect(() => {
    const onResize = () => scrollRef.current?.remeasure();
    stdout?.on("resize", onResize);
    return () => {
      stdout?.off("resize", onResize);
    };
  }, [stdout]);

  return (
    <Box height={viewportHeight} flexDirection="row">
      <Box flexGrow={1} flexDirection="column">
        <ScrollView
          ref={scrollRef}
          flexGrow={1}
          onContentHeightChange={handleContentHeightChange}
          onScroll={handleScroll}
        >
          {children}
        </ScrollView>
      </Box>
      <Scrollbar
        viewportHeight={viewportHeight}
        contentHeight={contentHeight}
        scrollOffset={scrollOffset}
      />
    </Box>
  );
}
