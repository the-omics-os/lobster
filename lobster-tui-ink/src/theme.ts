/**
 * Foreground-only color theme for Lobster AI Ink TUI.
 *
 * Ported from the Go Charm TUI palette. These colors are chosen to
 * be legible on BOTH dark and light terminal backgrounds.
 * NEVER set background colors on any Text element.
 */

// Primary brand colors
export const theme = {
  primary: '#e45c47',      // Lobster orange
  secondary: '#CC2C18',    // Darker orange
  accent1: '#e45c47',      // Same as primary
  accent2: '#FF6B4A',      // Lighter orange
  accent3: '#4CAF50',      // Green

  // Semantic colors
  success: '#28a745',
  warning: '#ffc107',
  error: '#dc3545',
  info: '#17a2b8',

  // Text colors (ANSI 256 equivalents)
  textMuted: 'gray',       // ANSI 245
  textDim: 'grey',         // ANSI 242
} as const;

/** Alert level color mapping. */
export const alertColors: Record<string, string> = {
  success: theme.success,
  warning: theme.warning,
  error: theme.error,
  info: theme.info,
};
