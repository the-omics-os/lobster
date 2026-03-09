// Package theme — Lobster-branded built-in themes.
//
// Text/Background colors are nil (terminal default) so the TUI inherits the
// user's terminal foreground/background — always readable, zero detection.
// Brand accent colors use hardcoded hex since they are foreground-only
// decorations that are readable on any background.
package theme

import "charm.land/lipgloss/v2"

// LobsterDefault is the single adaptive theme. nil colors mean "use the
// terminal's default", which guarantees contrast on any background.
var LobsterDefault = &Theme{
	ID:          "lobster-default",
	Name:        "Lobster",
	Description: "Adaptive Lobster AI theme (works on light and dark terminals)",
	Author:      "Omics-OS",
	Version:     "2.0.0",
	Colors: Colors{
		// Brand accents — hex, foreground-only (visible on any background).
		Primary:   lipgloss.Color("#e45c47"), // Lobster orange
		Secondary: lipgloss.Color("#CC2C18"),
		Accent1:   lipgloss.Color("#e45c47"),
		Accent2:   lipgloss.Color("#FF6B4A"),
		Accent3:   lipgloss.Color("#4CAF50"),

		// Structural — nil = terminal default (always contrasts with bg).
		Text:       nil,                      // Terminal's default foreground
		TextMuted:  lipgloss.ANSIColor(245),  // Mid-gray (#8a8a8a) — visible on both light & dark
		TextDim:    lipgloss.ANSIColor(242),  // Darker gray (#6c6c6c) — subtle but readable
		Background: nil,                      // Terminal's own background
		Surface:    nil,                      // No forced surface color
		Overlay:    lipgloss.ANSIColor(245),  // Mid-gray for borders/dividers

		// Semantic status — hex, foreground-only.
		Success: lipgloss.Color("#28a745"),
		Warning: lipgloss.Color("#ffc107"),
		Error:   lipgloss.Color("#dc3545"),
		Info:    lipgloss.Color("#17a2b8"),
	},
}

// Legacy aliases for backward compatibility with --theme flag and env var.
var LobsterDark = LobsterDefault
var LobsterLight = LobsterDefault

// init registers the theme and sets it as current.
func init() {
	Register(LobsterDefault)
	// Also register legacy IDs so --theme lobster-dark / lobster-light still work.
	Available["lobster-dark"] = LobsterDefault
	Available["lobster-light"] = LobsterDefault
	Current = LobsterDefault
}
