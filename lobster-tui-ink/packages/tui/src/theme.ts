/**
 * Semantic theme tokens for the Lobster AI Ink TUI.
 *
 * Ink only uses foreground colors here. Structural tokens are kept so the
 * React CLI and the Go TUI share the same semantic vocabulary.
 */

export interface Theme {
  name: "lobster-dark" | "lobster-light";
  primary: string;
  accent1: string;
  accent2: string;
  accent3: string;
  secondary: string;
  success: string;
  warning: string;
  error: string;
  info: string;
  text: string;
  textMuted: string;
  textDim: string;
  surface: string;
  overlay: string;
  background: string;
}

export const lobsterDark: Theme = {
  name: "lobster-dark",
  primary: "#e45c47",
  accent1: "#e45c47",
  accent2: "#ff875c",
  accent3: "#4caf50",
  secondary: "#cc2c18",
  success: "#28a745",
  warning: "#ffc107",
  error: "#dc3545",
  info: "#17a2b8",
  text: "#f5f2eb",
  textMuted: "#a3a3a3",
  textDim: "#737373",
  surface: "#1f1a17",
  overlay: "#4b4b4b",
  background: "#130f0d",
};

export const lobsterLight: Theme = {
  name: "lobster-light",
  primary: "#c7482d",
  accent1: "#c7482d",
  accent2: "#ef6a47",
  accent3: "#2f8f46",
  secondary: "#9f2a1b",
  success: "#1f7a33",
  warning: "#b77900",
  error: "#b42318",
  info: "#0f6c84",
  text: "#171412",
  textMuted: "#666666",
  textDim: "#8f8f8f",
  surface: "#f7f2ec",
  overlay: "#b8a999",
  background: "#fffaf5",
};

export type ThemeName = Theme["name"];

function isThemeName(value: string): value is ThemeName {
  return value === "lobster-dark" || value === "lobster-light";
}

function parseColorFgBg(value: string | undefined): number | null {
  if (!value) return null;

  const parts = value.split(";");
  const rawBackground = parts.at(-1);
  if (!rawBackground) return null;

  const parsed = Number.parseInt(rawBackground, 10);
  return Number.isNaN(parsed) ? null : parsed;
}

export function detectColorScheme(): "dark" | "light" {
  const bg = parseColorFgBg(process.env.COLORFGBG);
  if (bg !== null) {
    return bg > 8 ? "light" : "dark";
  }
  return "dark";
}

export function resolveThemeName(): ThemeName {
  const override = process.env.LOBSTER_TUI_THEME?.trim().toLowerCase();
  if (override && isThemeName(override)) {
    return override;
  }

  if (override === "lobster-default") {
    return detectColorScheme() === "light" ? "lobster-light" : "lobster-dark";
  }

  return detectColorScheme() === "light" ? "lobster-light" : "lobster-dark";
}

export function resolveTheme(): Theme {
  return resolveThemeName() === "lobster-light" ? lobsterLight : lobsterDark;
}

export const theme = resolveTheme();

/** Alert level color mapping. */
export const alertColors: Record<string, string> = {
  success: theme.success,
  warning: theme.warning,
  error: theme.error,
  info: theme.info,
};
