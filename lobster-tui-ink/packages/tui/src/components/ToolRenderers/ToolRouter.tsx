import React from "react";
import type { ToolCallMessagePartProps } from "@assistant-ui/react-ink";
import { HandoffRenderer, isHandoffTool } from "./HandoffRenderer.js";
import { ModalityRenderer } from "./ModalityRenderer.js";
import { TodoRenderer } from "./TodoRenderer.js";
import { DownloadRenderer } from "./DownloadRenderer.js";
import { PlotSummaryRenderer } from "./PlotSummaryRenderer.js";
import { DataTablePreviewRenderer } from "./DataTablePreviewRenderer.js";
import { ToolCallRenderer } from "./ToolCallRenderer.js";

const MODALITY_TOOLS = ["load_modality", "get_modality_info"];
const TODO_TOOLS = ["write_todos"];
const DOWNLOAD_TOOLS = ["execute_download_from_queue", "download_file"];
const PLOT_TOOLS = ["create_plot", "generate_plot", "create_figure"];
const TABLE_TOOLS = ["get_content_from_workspace", "query_data", "get_data_summary"];

/**
 * Routes tool calls to specialized renderers based on tool name patterns.
 * Used as the tools.Fallback in ChainOfThought.Parts.
 */
export function ToolRouter(props: ToolCallMessagePartProps) {
  const { toolName } = props;

  if (isHandoffTool(toolName)) {
    return <HandoffRenderer {...props} />;
  }
  if (MODALITY_TOOLS.includes(toolName)) {
    return <ModalityRenderer {...props} />;
  }
  if (TODO_TOOLS.includes(toolName)) {
    return <TodoRenderer {...props} />;
  }
  if (DOWNLOAD_TOOLS.includes(toolName)) {
    return <DownloadRenderer {...props} />;
  }
  if (PLOT_TOOLS.includes(toolName)) {
    return <PlotSummaryRenderer {...props} />;
  }
  if (TABLE_TOOLS.includes(toolName)) {
    return <DataTablePreviewRenderer {...props} />;
  }
  return <ToolCallRenderer part={props} index={0} />;
}
