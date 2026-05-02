import React from "react";
import { Box } from "ink";
import { ThreadPrimitive } from "@assistant-ui/react-ink";
import { useAuiState } from "@assistant-ui/store";
import { UserMessage } from "./UserMessage.js";
import { AssistantMessage } from "./AssistantMessage.js";
import { WelcomeAnimation } from "./WelcomeAnimation.js";

const THREAD_COMPONENTS = {
  UserMessage,
  AssistantMessage,
};

export function Thread() {
  const messages = useAuiState((s) => s.thread.messages);

  if (messages.length === 0) {
    return (
      <Box flexDirection="column">
        <WelcomeAnimation animate={false} idleSpark />
      </Box>
    );
  }

  return <ThreadPrimitive.Messages components={THREAD_COMPONENTS} />;
}
