import React, { useRef, useCallback, useEffect } from "react";
import { Box, useInput, useStdout } from "ink";
import { ScrollView, type ScrollViewRef } from "ink-scroll-view";

const PAGE_SIZE = 10;

export function ChatViewport({ children }: { children: React.ReactNode }) {
  const scrollRef = useRef<ScrollViewRef>(null);
  const { stdout } = useStdout();

  // Reserve space for header (2 lines) and composer (3 lines)
  const headerHeight = 0; // will be added in Step 1.6
  const composerHeight = 3;
  const viewportHeight = (stdout?.rows ?? 24) - headerHeight - composerHeight;

  useInput((input, key) => {
    if (key.pageUp) {
      scrollRef.current?.scrollBy(-PAGE_SIZE);
    } else if (key.pageDown) {
      scrollRef.current?.scrollBy(PAGE_SIZE);
    }
  });

  // Auto-scroll to bottom when content changes
  const handleContentHeightChange = useCallback(() => {
    scrollRef.current?.scrollToBottom();
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
    <Box height={viewportHeight} flexDirection="column">
      <ScrollView
        ref={scrollRef}
        flexGrow={1}
        onContentHeightChange={handleContentHeightChange}
      >
        {children}
      </ScrollView>
    </Box>
  );
}
