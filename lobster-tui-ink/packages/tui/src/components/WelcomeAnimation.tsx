import React, { useEffect, useRef, useState } from "react";
import { Box, Text } from "ink";
import { useTheme } from "../hooks/useTheme.js";
import { useTerminalSize } from "../hooks/useTerminalSize.js";

interface WelcomeAnimationProps {
  animate?: boolean;
  idleSpark?: boolean;
  onDone?: () => void;
}

const TITLE = "Lobster AI";
const TITLE_ASCII = [
  "▗▖    ▗▄▖ ▗▄▄▖  ▗▄▄▖▗▄▄▄▖▗▄▄▄▖▗▄▄▖      ▗▄▖ ▗▄▄▄▖",
  "▐▌   ▐▌ ▐▌▐▌ ▐▌▐▌     █  ▐▌   ▐▌ ▐▌    ▐▌ ▐▌  █  ",
  "▐▌   ▐▌ ▐▌▐▛▀▚▖ ▝▀▚▖  █  ▐▛▀▀▘▐▛▀▚▖    ▐▛▀▜▌  █  ",
  "▐▙▄▄▖▝▚▄▞▘▐▙▄▞▘▗▄▄▞▘  █  ▐▙▄▄▖▐▌ ▐▌    ▐▌ ▐▌▗▄█▄▖",
];
const NUCLEOTIDES = ["A", "T", "C", "G"];
const TAGLINE = "The self-evolving agentic framework for bioinformatics";
const SUBTAGLINE = "on-prem • python native • open-source";
const FADE_STEPS = 3;
const FADE_STEP_MS = 200;
const SCRAMBLE_FRAMES = 24;
const SCRAMBLE_INTERVAL_MS = 50;
const IDLE_SPARK_MIN_INTERVAL_MS = 2_000;
const IDLE_SPARK_JITTER_MS = 2_000;
const IDLE_SPARK_CHANCE = 0.7;
const IDLE_SPARK_FRAME_INTERVAL_MS = 100;
const IDLE_SPARK_FRAMES = 4;
const TOTAL_DURATION_MS = FADE_STEPS * FADE_STEP_MS + SCRAMBLE_FRAMES * SCRAMBLE_INTERVAL_MS;

interface IdleSparkState {
  cell: number;
  tick: number;
  frame: number;
  nextSparkAt: number;
}

function nextIdleSparkDelay() {
  return IDLE_SPARK_MIN_INTERVAL_MS + Math.floor(Math.random() * IDLE_SPARK_JITTER_MS);
}

function lineWidth(value: string) {
  return Array.from(value).length;
}

function centerLine(value: string, columns: number) {
  const pad = Math.max(0, Math.floor((columns - lineWidth(value)) / 2));
  return `${" ".repeat(pad)}${value}`;
}

function countVisibleCells(lines: string[]) {
  return lines.reduce((count, line) => {
    return count + Array.from(line).filter((char) => char !== " ").length;
  }, 0);
}

function titleLinesForWidth(columns: number) {
  const asciiWidth = Math.max(...TITLE_ASCII.map(lineWidth));
  return columns >= asciiWidth + 4 ? TITLE_ASCII : [TITLE];
}

function fadeColorForElapsed(elapsedMs: number, theme: ReturnType<typeof useTheme>) {
  if (elapsedMs < FADE_STEP_MS) {
    return theme.textDim;
  }
  if (elapsedMs < FADE_STEP_MS * 2) {
    return theme.textMuted;
  }
  return theme.text;
}

function nucleotideColor(nucleotide: string) {
  switch (nucleotide) {
    case "A":
      return "#22c55e";
    case "T":
      return "#ef4444";
    case "C":
      return "#3b82f6";
    case "G":
      return "#eab308";
    default:
      return undefined;
  }
}

export function WelcomeAnimation({
  animate = true,
  idleSpark = false,
  onDone,
}: WelcomeAnimationProps) {
  const theme = useTheme();
  const { columns } = useTerminalSize();
  const [elapsedMs, setElapsedMs] = useState(0);
  const [idleSparkState, setIdleSparkState] = useState<IdleSparkState>(() => ({
    cell: -1,
    tick: 0,
    frame: 0,
    nextSparkAt: Date.now() + nextIdleSparkDelay(),
  }));
  const doneRef = useRef(false);

  useEffect(() => {
    if (!animate) {
      if (!doneRef.current) {
        doneRef.current = true;
        onDone?.();
      }
      return undefined;
    }

    if (elapsedMs >= TOTAL_DURATION_MS) {
      if (!doneRef.current) {
        doneRef.current = true;
        onDone?.();
      }
      return undefined;
    }

    const timer = setTimeout(() => {
      setElapsedMs((current) => Math.min(current + SCRAMBLE_INTERVAL_MS, TOTAL_DURATION_MS));
    }, SCRAMBLE_INTERVAL_MS);

    return () => clearTimeout(timer);
  }, [animate, elapsedMs, onDone]);

  const titleLines = titleLinesForWidth(columns);
  const totalCells = countVisibleCells(titleLines);
  useEffect(() => {
    if (animate || !idleSpark || totalCells === 0) {
      return undefined;
    }

    if (idleSparkState.cell >= 0) {
      const timer = setTimeout(() => {
        setIdleSparkState((current) => {
          const nextTick = current.tick + 1;
          if (nextTick >= IDLE_SPARK_FRAMES) {
            return {
              cell: -1,
              tick: 0,
              frame: current.frame + 1,
              nextSparkAt: Date.now() + nextIdleSparkDelay(),
            };
          }

          return {
            ...current,
            tick: nextTick,
            frame: current.frame + 1,
          };
        });
      }, IDLE_SPARK_FRAME_INTERVAL_MS);

      return () => clearTimeout(timer);
    }

    const delay = Math.max(0, idleSparkState.nextSparkAt - Date.now());
    const timer = setTimeout(() => {
      setIdleSparkState((current) => {
        if (Math.random() >= IDLE_SPARK_CHANCE) {
          return {
            ...current,
            nextSparkAt: Date.now() + nextIdleSparkDelay(),
          };
        }

        return {
          cell: Math.floor(Math.random() * totalCells),
          tick: 0,
          frame: current.frame + 1,
          nextSparkAt: current.nextSparkAt,
        };
      });
    }, delay);

    return () => clearTimeout(timer);
  }, [animate, idleSpark, idleSparkState, totalCells]);

  const isScrambling = animate && elapsedMs >= FADE_STEPS * FADE_STEP_MS && elapsedMs < TOTAL_DURATION_MS;
  const scrambleElapsed = Math.max(0, elapsedMs - FADE_STEPS * FADE_STEP_MS);
  const scrambleFrame = Math.min(
    SCRAMBLE_FRAMES,
    Math.floor(scrambleElapsed / SCRAMBLE_INTERVAL_MS) + 1,
  );
  const lockedCount =
    !animate || elapsedMs < FADE_STEPS * FADE_STEP_MS
      ? totalCells
      : totalCells === 0
        ? 0
        : Math.floor((scrambleFrame / SCRAMBLE_FRAMES) * totalCells);
  const titleColor = animate ? fadeColorForElapsed(elapsedMs, theme) : theme.text;
  const compactMeta = columns < 70;
  const sparkCell =
    !animate && idleSpark && totalCells > 0 && idleSparkState.cell >= 0
      ? idleSparkState.cell % totalCells
      : -1;

  return (
    <Box flexDirection="column" marginTop={1}>
      {titleLines.map((line, lineIndex) => {
        const prefix = centerLine("", columns - lineWidth(line));
        let cellIndex = titleLines
          .slice(0, lineIndex)
          .reduce((count, current) => count + Array.from(current).filter((char) => char !== " ").length, 0);

        return (
          <Text key={`welcome:${lineIndex}`}>
            {prefix}
            {Array.from(line).map((char, charIndex) => {
              if (char === " ") {
                return <Text key={`welcome:${lineIndex}:${charIndex}`}> </Text>;
              }

              const locked = !isScrambling || cellIndex < lockedCount;
              const sparkActive = sparkCell === cellIndex;
              const display =
                !locked
                  ? NUCLEOTIDES[(scrambleFrame + cellIndex) % NUCLEOTIDES.length]!
                  : sparkActive
                    ? NUCLEOTIDES[
                      (idleSparkState.frame + idleSparkState.tick + cellIndex) % NUCLEOTIDES.length
                    ]!
                    : char;
              const color = !locked || sparkActive ? nucleotideColor(display) : titleColor;
              cellIndex += 1;

              return (
                <Text key={`welcome:${lineIndex}:${charIndex}`} bold color={color}>
                  {display}
                </Text>
              );
            })}
          </Text>
        );
      })}
      <Text color={theme.textMuted}>{centerLine(TAGLINE, columns)}</Text>
      <Text color={theme.textDim}>
        {centerLine(compactMeta ? "on-prem | python native | open-source" : SUBTAGLINE, columns)}
      </Text>
    </Box>
  );
}
