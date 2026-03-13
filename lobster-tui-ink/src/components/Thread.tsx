import React from "react";
import { Box, Text } from "ink";
import { ThreadRoot, ThreadEmpty, ThreadMessages } from "@assistant-ui/react-ink";
import { ChatViewport } from "./ChatViewport.js";
import { UserMessage } from "./UserMessage.js";
import { AssistantMessage } from "./AssistantMessage.js";

interface ThreadProps {
  /** Viewport height from layout engine. Falls back to terminal-based calc. */
  viewportHeight?: number;
}

export function Thread({ viewportHeight }: ThreadProps) {
  return (
    <ThreadRoot>
      <ChatViewport viewportHeight={viewportHeight}>
        <ThreadEmpty>
          <Box>
            <Text dimColor>No messages yet. Start a conversation!</Text>
          </Box>
        </ThreadEmpty>
        <ThreadMessages
          components={{
            UserMessage,
            AssistantMessage,
          }}
        />
      </ChatViewport>
    </ThreadRoot>
  );
}
