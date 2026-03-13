import React from "react";
import type { ToolCallMessagePartProps } from "@assistant-ui/react-ink";
import { HandoffRenderer, isHandoffTool } from "./HandoffRenderer.js";
import { ToolCallRenderer } from "./ToolCallRenderer.js";

/**
 * Routes tool calls to specialized renderers based on tool name patterns.
 * Used as the tools.Fallback in ChainOfThought.Parts.
 */
export function ToolRouter(props: ToolCallMessagePartProps) {
  if (isHandoffTool(props.toolName)) {
    return <HandoffRenderer {...props} />;
  }
  return <ToolCallRenderer part={props} index={0} />;
}
