// Package theme — Lobster-branded built-in themes.
package theme

import "github.com/charmbracelet/lipgloss"

// LobsterDark is the official Lobster AI dark theme.
var LobsterDark = &Theme{
	ID:          "lobster-dark",
	Name:        "Lobster Dark",
	Description: "Official Lobster AI dark theme",
	Author:      "Omics-OS",
	Version:     "1.0.0",
	Colors: Colors{
		Primary:    lipgloss.Color("#e45c47"), // Lobster orange
		Secondary:  lipgloss.Color("#CC2C18"),
		Background: lipgloss.Color("#1a1a2e"),
		Surface:    lipgloss.Color("#252538"),
		Overlay:    lipgloss.Color("#2a2a40"),
		Text:       lipgloss.Color("#FAFAFA"),
		TextMuted:  lipgloss.Color("#888888"),
		TextDim:    lipgloss.Color("#555555"),
		Success:    lipgloss.Color("#28a745"),
		Warning:    lipgloss.Color("#ffc107"),
		Error:      lipgloss.Color("#dc3545"),
		Info:       lipgloss.Color("#17a2b8"),
		Accent1:    lipgloss.Color("#e45c47"),
		Accent2:    lipgloss.Color("#FF6B4A"),
		Accent3:    lipgloss.Color("#4CAF50"),
	},
}

// LobsterLight is the official Lobster AI light theme.
var LobsterLight = &Theme{
	ID:          "lobster-light",
	Name:        "Lobster Light",
	Description: "Official Lobster AI light theme",
	Author:      "Omics-OS",
	Version:     "1.0.0",
	Colors: Colors{
		Primary:    lipgloss.Color("#CC2C18"), // Lobster red (darker for readability on white)
		Secondary:  lipgloss.Color("#e45c47"),
		Background: lipgloss.Color("#FFFFFF"),
		Surface:    lipgloss.Color("#F5F5F5"),
		Overlay:    lipgloss.Color("#EBEBEB"),
		Text:       lipgloss.Color("#1a1a2e"),
		TextMuted:  lipgloss.Color("#555555"),
		TextDim:    lipgloss.Color("#999999"),
		Success:    lipgloss.Color("#1e7e34"),
		Warning:    lipgloss.Color("#856404"),
		Error:      lipgloss.Color("#721c24"),
		Info:       lipgloss.Color("#0c5460"),
		Accent1:    lipgloss.Color("#CC2C18"),
		Accent2:    lipgloss.Color("#e45c47"),
		Accent3:    lipgloss.Color("#2d6a4f"),
	},
}

// init registers the built-in themes and activates the dark theme by default.
func init() {
	Register(LobsterDark)
	Register(LobsterLight)
	// Default to dark theme; overridden at startup via LOBSTER_TUI_THEME or
	// the user's saved preference.
	Current = LobsterDark
}
