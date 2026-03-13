import React from "react";
import { Box, Text } from "ink";
import { ThreadPrimitive } from "@assistant-ui/react-ink";
import { ChatViewport } from "./ChatViewport.js";
import { UserMessage } from "./UserMessage.js";
import { AssistantMessage } from "./AssistantMessage.js";

export function Thread() {
  return (
    <ThreadPrimitive.Root>
      <ChatViewport>
        <ThreadPrimitive.Empty>
          <Box>
            <Text dimColor>No messages yet. Start a conversation!</Text>
          </Box>
        </ThreadPrimitive.Empty>
        <ThreadPrimitive.Messages
          components={{
            UserMessage,
            AssistantMessage,
          }}
        />
      </ChatViewport>
    </ThreadPrimitive.Root>
  );
}
