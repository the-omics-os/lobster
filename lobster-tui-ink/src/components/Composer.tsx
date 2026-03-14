import React, { useState, useCallback, useMemo, useEffect, useLayoutEffect, useRef } from "react";
import { Box, Text, useInput } from "ink";
import { useAssistantRuntime, useAuiState } from "@assistant-ui/react-ink";
import { useStore } from "zustand";
import {
  getCommandCompletions,
  getNestedCommandCompletions,
  getSubcommandCompletions,
  hasNestedCompletions,
  isGroupedParityCommand,
} from "../commands/dispatcher.js";
import { useHistory } from "../hooks/useHistory.js";
import { useTerminalSize } from "../hooks/useTerminalSize.js";
import { filterResources, type Resource } from "../api/resources.js";
import type { AppState } from "../utils/stateHandlers.js";
import type { AppStateStore } from "../utils/appStateStore.js";
import { clearRunActivity } from "../utils/appStateStore.js";
import {
  resetFooterState,
  setFooterState,
  type FooterStateStore,
} from "../utils/footerStateStore.js";
import { useTheme } from "../hooks/useTheme.js";
import {
  CompletionMenu,
  type CompletionItem,
  getCompletionMenuRowCount,
} from "./CompletionMenu.js";

interface ComposerProps {
  /** Intercept input before sending to runtime. Return true if handled. */
  onIntercept?: (input: string) => boolean;
  /** Called when a normal message is submitted to the runtime. */
  onSubmit?: (input: string) => void;
  /** Resources catalog for @mention autocomplete. */
  resources?: Resource[];
  /** App state store for dynamic slash-command completions. */
  appStateStore: AppStateStore;
  /** Footer interaction state store for contextual status hints. */
  footerStateStore: FooterStateStore;
  /** Notify parent layout engine when composer height changes. */
  onLayoutChange?: (state: ComposerLayoutState) => void;
}

export interface ComposerLayoutState {
  inputLineCount: number;
  completionMenuRows: number;
}

interface CompletionState {
  items: CompletionItem[];
  anchorColumn: number;
}

type CompletionSourceState = Pick<AppState, "modalities" | "plots">;

const MAX_INPUT_LINES = 8;
const PROMPT = "> ";
const BORDER_COLUMNS = 2;

function clampLineCount(value: number) {
  return Math.max(1, Math.min(value, MAX_INPUT_LINES));
}

function normalizeInput(value: string) {
  return value.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
}

function countWrappedLines(value: string, width: number) {
  const normalized = normalizeInput(value);
  const lines = normalized.split("\n");
  let visualLines = 0;

  for (const line of lines) {
    const lineWidth = Math.max(Array.from(line).length, 1);
    visualLines += Math.ceil(lineWidth / width);
  }

  return clampLineCount(visualLines);
}

function extractNamedValues(
  items: unknown[] | undefined,
  keys: string[],
): string[] {
  if (!items?.length) return [];

  const values = items
    .map((item) => {
      if (!item || typeof item !== "object") return null;
      const record = item as Record<string, unknown>;
      for (const key of keys) {
        const value = record[key];
        if (typeof value === "string" && value.trim().length > 0) {
          return value;
        }
      }
      return null;
    })
    .filter((value): value is string => Boolean(value));

  return Array.from(new Set(values));
}

function buildCommandCompletionState(
  trimmedLine: string,
  leadingWhitespace: string,
  appState: CompletionSourceState | undefined,
): CompletionState | null {
  const withoutSlash = trimmedLine.slice(1);
  const hasTrailingSpace = trimmedLine.endsWith(" ");
  const tokens = withoutSlash.split(/\s+/).filter(Boolean);
  const topLevel = tokens[0]?.toLowerCase();
  if (!topLevel && trimmedLine !== "/") return null;

  if (tokens.length <= 1 && !hasTrailingSpace) {
    const partial = topLevel ?? "";
    const items = getCommandCompletions()
      .filter((command) => command.name.startsWith(partial))
      .map<CompletionItem>((command) => ({
        label: `/${command.name}`,
        value: `${leadingWhitespace}/${command.name}`,
        description: command.description,
      }));

    return {
      items,
      anchorColumn: leadingWhitespace.length,
    };
  }

  if (topLevel && isGroupedParityCommand(topLevel)) {
    if ((tokens.length === 1 && hasTrailingSpace) || (tokens.length === 2 && !hasTrailingSpace)) {
      const partial = tokens[1]?.toLowerCase() ?? "";
      const items = getSubcommandCompletions(topLevel)
        .filter((command) => command.name.startsWith(partial))
        .map<CompletionItem>((command) => ({
          label: command.name,
          value: `${leadingWhitespace}/${topLevel} ${command.name} `,
          description: command.description,
        }));

      return {
        items,
        anchorColumn: leadingWhitespace.length + `/${topLevel} `.length,
      };
    }

    const subcommand = tokens[1]?.toLowerCase();
    if (
      subcommand &&
      hasNestedCompletions(topLevel, subcommand) &&
      ((tokens.length === 2 && hasTrailingSpace) || (tokens.length === 3 && !hasTrailingSpace))
    ) {
      const partial = tokens[2]?.toLowerCase() ?? "";
      const items = getNestedCommandCompletions(`${topLevel} ${subcommand}`)
        .filter((command) => command.name.startsWith(partial))
        .map<CompletionItem>((command) => ({
          label: command.name,
          value: `${leadingWhitespace}/${topLevel} ${subcommand} ${command.name} `,
          description: command.description,
        }));

      return {
        items,
        anchorColumn:
          leadingWhitespace.length + `/${topLevel} ${subcommand} `.length,
      };
    }
  }

  if (topLevel === "save" && ((tokens.length === 1 && hasTrailingSpace) || (tokens.length === 2 && !hasTrailingSpace))) {
    const partial = tokens[1]?.toLowerCase() ?? "";
    const items = getNestedCommandCompletions("save")
      .filter((command) => command.name.startsWith(partial))
      .map<CompletionItem>((command) => ({
        label: command.name,
        value: `${leadingWhitespace}/save ${command.name}`,
        description: command.description,
      }));

    return {
      items,
      anchorColumn: leadingWhitespace.length + "/save ".length,
    };
  }

  if (topLevel === "describe" && ((tokens.length === 1 && hasTrailingSpace) || (tokens.length === 2 && !hasTrailingSpace))) {
    const partial = tokens[1]?.toLowerCase() ?? "";
    const items = extractNamedValues(appState?.modalities, ["name", "type", "id"])
      .filter((value) => value.toLowerCase().includes(partial))
      .map<CompletionItem>((value) => ({
        label: value,
        value: `${leadingWhitespace}/describe ${value}`,
      }));

    return {
      items,
      anchorColumn: leadingWhitespace.length + "/describe ".length,
    };
  }

  if (topLevel === "plot" && ((tokens.length === 1 && hasTrailingSpace) || (tokens.length === 2 && !hasTrailingSpace))) {
    const partial = tokens[1]?.toLowerCase() ?? "";
    const items = extractNamedValues(appState?.plots, ["title", "name", "id", "file_path"])
      .filter((value) => value.toLowerCase().includes(partial))
      .map<CompletionItem>((value) => ({
        label: value,
        value: `${leadingWhitespace}/plot ${value}`,
      }));

    return {
      items,
      anchorColumn: leadingWhitespace.length + "/plot ".length,
    };
  }

  return null;
}

function buildCompletionState(
  value: string,
  resources: Resource[] | undefined,
  appState: CompletionSourceState | undefined,
): CompletionState | null {
  const normalized = normalizeInput(value);
  const currentLine = normalized.split("\n").at(-1) ?? "";
  const lineStart = normalized.lastIndexOf("\n") + 1;

  const leadingWhitespace = currentLine.match(/^[ \t]*/)?.[0] ?? "";
  const trimmedLine = currentLine.slice(leadingWhitespace.length);
  if (!normalized.includes("\n") && trimmedLine.startsWith("/")) {
    const commandCompletion = buildCommandCompletionState(trimmedLine, leadingWhitespace, appState);
    if (commandCompletion && commandCompletion.items.length > 0) {
      return commandCompletion;
    }
  }

  const atIndex = normalized.lastIndexOf("@");
  if (atIndex >= 0 && resources && resources.length > 0) {
    const mentionPrefix = normalized.slice(atIndex + 1);
    if (!/[ \t\n]/.test(mentionPrefix)) {
      const matches = filterResources(resources, mentionPrefix);
      const items = matches.slice(0, 10).map<CompletionItem>((resource) => ({
        label: `@${resource.id}`,
        value: `${normalized.slice(0, atIndex)}@${resource.id} `,
        description: resource.label !== resource.id ? resource.label : resource.description,
      }));

      return {
        items,
        anchorColumn: atIndex - lineStart,
      };
    }
  }

  return null;
}

export function Composer({
  onIntercept,
  onSubmit,
  resources,
  appStateStore,
  footerStateStore,
  onLayoutChange,
}: ComposerProps) {
  const theme = useTheme();
  const runtime = useAssistantRuntime();
  const threadRuntime = runtime.thread;
  const { columns } = useTerminalSize();
  const modalities = useStore(appStateStore, (state) => state.modalities);
  const plots = useStore(appStateStore, (state) => state.plots);
  const inputBlocked = useAuiState(
    (s) =>
      s.thread.isDisabled ||
      s.thread.messages.some((message) => message.status?.type === "requires-action"),
  );
  const [value, setValue] = useState("");
  const [cursorPos, setCursorPos] = useState(0);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [dismissedCompletionValue, setDismissedCompletionValue] = useState<string | null>(null);
  const history = useHistory();
  const valueRef = useRef("");
  const cursorPosRef = useRef(0);

  useEffect(() => {
    valueRef.current = value;
  }, [value]);

  useEffect(() => {
    cursorPosRef.current = cursorPos;
  }, [cursorPos]);

  const completion = useMemo(
    () => buildCompletionState(value, resources, { modalities, plots }),
    [modalities, plots, value, resources],
  );
  const completionVisible =
    !inputBlocked &&
    !!completion &&
    completion.items.length > 0 &&
    dismissedCompletionValue !== value;
  const completionSignature = completion?.items.map((item) => item.value).join("\u0000") ?? "";

  useEffect(() => {
    setSelectedIndex(0);
  }, [completionSignature]);

  useEffect(() => {
    if (dismissedCompletionValue && dismissedCompletionValue !== value) {
      setDismissedCompletionValue(null);
    }
  }, [dismissedCompletionValue, value]);

  const inputWidth = Math.max(1, columns - BORDER_COLUMNS - PROMPT.length - 1);
  const inputLineCount = useMemo(
    () => countWrappedLines(value, inputWidth),
    [value, inputWidth],
  );
  const completionMenuRows = completionVisible
    ? getCompletionMenuRowCount(completion.items.length, selectedIndex)
    : 0;

  useEffect(() => {
    onLayoutChange?.({
      inputLineCount,
      completionMenuRows,
    });
  }, [completionMenuRows, inputLineCount, onLayoutChange]);

  useLayoutEffect(() => {
    setFooterState(footerStateStore, {
      completionVisible,
      inputBlocked,
      multiline: normalizeInput(value).includes("\n"),
    });
  }, [completionVisible, footerStateStore, inputBlocked, value]);

  useEffect(() => {
    return () => {
      resetFooterState(footerStateStore);
    };
  }, [footerStateStore]);

  const setCursorPosition = useCallback((nextPos: number) => {
    cursorPosRef.current = nextPos;
    setCursorPos(nextPos);
  }, []);

  const setValueAndCursor = useCallback((newValue: string, newPos?: number) => {
    valueRef.current = newValue;
    const nextCursorPos = newPos ?? newValue.length;
    cursorPosRef.current = nextCursorPos;
    setValue(newValue);
    setCursorPos(nextCursorPos);
  }, []);

  const clearValue = useCallback(() => {
    history.reset();
    setValueAndCursor("");
    setSelectedIndex(0);
    setDismissedCompletionValue(null);
  }, [history, setValueAndCursor]);

  const insertText = useCallback((text: string) => {
    if (!text) return;
    history.reset();
    const liveValue = valueRef.current;
    const liveCursorPos = cursorPosRef.current;
    const normalized = normalizeInput(text);
    const newValue =
      liveValue.slice(0, liveCursorPos) +
      normalized +
      liveValue.slice(liveCursorPos);
    setValueAndCursor(newValue, liveCursorPos + normalized.length);
  }, [history, setValueAndCursor]);

  const insertNewline = useCallback(() => {
    insertText("\n");
  }, [insertText]);

  const deleteBackward = useCallback(() => {
    const liveValue = valueRef.current;
    const liveCursorPos = cursorPosRef.current;
    if (liveCursorPos === 0) return;
    history.reset();
    const newValue =
      liveValue.slice(0, liveCursorPos - 1) +
      liveValue.slice(liveCursorPos);
    setValueAndCursor(newValue, liveCursorPos - 1);
  }, [history, setValueAndCursor]);

  const acceptCompletion = useCallback(() => {
    if (!completionVisible || !completion) return;
    const item = completion.items[selectedIndex];
    if (!item) return;
    history.reset();
    setValueAndCursor(item.value);
    setDismissedCompletionValue(item.value);
  }, [completion, completionVisible, history, selectedIndex, setValueAndCursor]);

  const handleSubmit = useCallback(
    (nextValue: string) => {
      const trimmed = nextValue.trim();
      if (!trimmed) return;

      history.push(trimmed);
      clearRunActivity(appStateStore);

      if (onIntercept && onIntercept(trimmed)) {
        clearValue();
        return;
      }

      onSubmit?.(trimmed);
      threadRuntime.append({
        role: "user",
        content: [{ type: "text", text: trimmed }],
      });
      clearValue();
    },
    [appStateStore, clearValue, onIntercept, onSubmit, threadRuntime, history],
  );

  const handleBufferedInput = useCallback((input: string, key: { meta?: boolean; shift?: boolean; ctrl?: boolean }) => {
    if (!input || key.ctrl || key.meta) return false;

    const normalized = normalizeInput(input);
    const containsControlChars = /[\n\b\u007f]/.test(normalized);
    if (!containsControlChars) {
      return false;
    }

    const newlineCount = Array.from(normalized).filter((char) => char === "\n").length;
    const hasTrailingSubmit =
      newlineCount === 1 &&
      normalized.endsWith("\n") &&
      !normalized.slice(0, -1).includes("\n") &&
      !/[\b\u007f]/.test(normalized);

    if (hasTrailingSubmit) {
      const text = normalized.slice(0, -1);
      if (text) {
        insertText(text);
      }
      if (key.shift) {
        insertNewline();
      } else {
        handleSubmit(valueRef.current);
      }
      return true;
    }

    let buffer = "";
    for (const char of Array.from(normalized)) {
      if (char === "\n") {
        if (buffer) {
          insertText(buffer);
          buffer = "";
        }
        insertNewline();
        continue;
      }

      if (char === "\b" || char === "\u007f") {
        if (buffer) {
          insertText(buffer);
          buffer = "";
        }
        deleteBackward();
        continue;
      }

      buffer += char;
    }

    if (buffer) {
      insertText(buffer);
    }

    return true;
  }, [deleteBackward, handleSubmit, insertNewline, insertText]);

  useInput(
    (input, key) => {
      if (inputBlocked) {
        return;
      }

      const liveValue = valueRef.current;
      const liveCursorPos = cursorPosRef.current;

      if (completionVisible && key.upArrow) {
        setSelectedIndex((prev) =>
          prev > 0 ? prev - 1 : (completion?.items.length ?? 1) - 1,
        );
        return;
      }

      if (completionVisible && key.downArrow) {
        setSelectedIndex((prev) => {
          const count = completion?.items.length ?? 0;
          if (count === 0) return 0;
          return prev < count - 1 ? prev + 1 : 0;
        });
        return;
      }

      // Cursor movement: left arrow
      if (key.leftArrow) {
        setCursorPosition(Math.max(0, liveCursorPos - 1));
        return;
      }

      // Cursor movement: right arrow
      if (key.rightArrow) {
        setCursorPosition(Math.min(liveValue.length, liveCursorPos + 1));
        return;
      }

      if (key.upArrow) {
        if (liveValue.includes("\n")) return;
        const entry = history.up(liveValue);
        if (entry !== undefined) {
          setValueAndCursor(entry);
        }
        return;
      }

      if (key.downArrow) {
        if (liveValue.includes("\n")) return;
        const entry = history.down();
        if (entry !== undefined) {
          setValueAndCursor(entry);
        }
        return;
      }

      if (key.escape) {
        if (completionVisible) {
          setDismissedCompletionValue(liveValue);
        }
        return;
      }

      if (key.tab) {
        if (completionVisible) {
          acceptCompletion();
        }
        return;
      }

      if (key.return) {
        if (completionVisible && !key.meta && !key.shift) {
          acceptCompletion();
          return;
        }

        if (key.meta || key.shift) {
          history.reset();
          const newValue = liveValue.slice(0, liveCursorPos) + "\n" + liveValue.slice(liveCursorPos);
          setValueAndCursor(newValue, liveCursorPos + 1);
          return;
        }

        handleSubmit(liveValue);
        return;
      }

      if (key.backspace || key.delete) {
        deleteBackward();
        return;
      }

      // Ctrl+A = Home, Ctrl+E = End
      if (key.ctrl && input === "a") {
        setCursorPosition(0);
        return;
      }
      if (key.ctrl && input === "e") {
        setCursorPosition(liveValue.length);
        return;
      }

      if (handleBufferedInput(input, key)) {
        return;
      }

      if (input && !key.ctrl && !key.meta) {
        insertText(input);
      }
    },
    { isActive: !inputBlocked },
  );

  const lines = normalizeInput(value).split("\n");
  const displayedLines = lines.length > 0 ? lines : [""];
  const placeholderVisible = value.length === 0;

  // Calculate cursor line and column for rendering
  const textBeforeCursor = value.slice(0, cursorPos);
  const cursorLineIndex = textBeforeCursor.split("\n").length - 1;
  const lastNewline = textBeforeCursor.lastIndexOf("\n");
  const cursorCol = lastNewline === -1 ? cursorPos : cursorPos - lastNewline - 1;

  return (
    <Box flexDirection="column">
      <CompletionMenu
        items={completion?.items ?? []}
        selectedIndex={selectedIndex}
        visible={completionVisible}
        anchorColumn={(completion?.anchorColumn ?? 0) + PROMPT.length}
      />
      {displayedLines.map((line, index) => {
        const isFirstLine = index === 0;
        const prompt = isFirstLine ? PROMPT : " ".repeat(PROMPT.length);
        const isCursorLine = index === cursorLineIndex;
        return (
          <Box key={index}>
            <Text color={theme.primary} bold>
              {prompt}
            </Text>
            {placeholderVisible && index === 0 ? (
              <>
                <Text dimColor>
                  {inputBlocked ? "Tool input active above..." : "Type a message or /help..."}
                </Text>
                {!inputBlocked && <Text color={theme.primary}>█</Text>}
              </>
            ) : isCursorLine && !inputBlocked ? (
              <>
                <Text>{line.slice(0, cursorCol)}</Text>
                <Text color={theme.primary}>█</Text>
                <Text>{line.slice(cursorCol)}</Text>
              </>
            ) : (
              <Text>{line}</Text>
            )}
          </Box>
        );
      })}
    </Box>
  );
}
