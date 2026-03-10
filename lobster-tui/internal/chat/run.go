package chat

import (
	"fmt"
	"os"
	"strings"

	tea "charm.land/bubbletea/v2"
	"golang.org/x/term"

	"github.com/the-omics-os/lobster-tui/internal/protocol"
	"github.com/the-omics-os/lobster-tui/internal/theme"
)

// Run starts the interactive chat session using the given file descriptor pair
// for protocol communication and the named theme for styling.
//
// fdIn and fdOut are file descriptor numbers for the JSON Lines IPC channel
// (typically inherited from the parent Python process).
func Run(fdIn, fdOut int, themeName string, version string, inline bool) error {
	// ---- Protocol setup -----------------------------------------------------
	protoIn := os.NewFile(uintptr(fdIn), "proto-in")
	if protoIn == nil {
		return fmt.Errorf("invalid proto-fd-in: %d", fdIn)
	}
	protoOut := os.NewFile(uintptr(fdOut), "proto-out")
	if protoOut == nil {
		return fmt.Errorf("invalid proto-fd-out: %d", fdOut)
	}

	handler := protocol.NewHandler(protoIn, protoOut)
	defer handler.Close()

	// ---- Terminal dimensions ------------------------------------------------
	width, height, err := term.GetSize(int(os.Stdout.Fd()))
	if err != nil {
		// Sensible fallback for non-interactive terminals.
		width = 80
		height = 24
	}

	// ---- Handshake ----------------------------------------------------------
	if err := handler.SendTyped(protocol.TypeHandshake, protocol.HandshakePayload{
		ClientVersion:   version,
		ProtocolVersion: protocol.Version,
		TerminalWidth:   width,
		TerminalHeight:  height,
	}, ""); err != nil {
		return fmt.Errorf("handshake failed: %w", err)
	}

	// ---- Theme --------------------------------------------------------------
	if themeName != "" {
		if err := theme.SetTheme(themeName); err != nil {
			// Non-fatal: fall back to whatever is current.
			fmt.Fprintf(os.Stderr, "warning: %v\n", err)
		}
	}
	activeTheme := theme.Current
	if activeTheme == nil {
		// Shouldn't happen — init() always sets Current — but guard anyway.
		_ = theme.SetTheme(theme.AutoDetect())
		activeTheme = theme.Current
	}

	// ---- BubbleTea model ----------------------------------------------------
	styles := activeTheme.Styles
	if inline {
		// Inline mode uses a minimal, less boxed style profile that better
		// matches the classic Lobster CLI aesthetic.
		styles = theme.BuildCleanStyles(activeTheme.Colors)
	}
	mouseCapture := shouldEnableMouseCapture(inline)
	model := NewModel(handler, styles, width, height, inline, mouseCapture, version)

	// In v2, AltScreen and MouseMode are set via View() fields on the Model,
	// not as Program options. See Model.View() for the field assignments.
	p := tea.NewProgram(model)
	if _, err := p.Run(); err != nil {
		return fmt.Errorf("chat session: %w", err)
	}

	return nil
}

// shouldEnableMouseCapture returns whether Bubble Tea mouse capture should be
// active for the chat session.
//
// Defaults:
// - inline mode: disabled (prioritize terminal text selection/copy)
// - fullscreen mode: enabled (preserve existing mouse scroll UX)
//
// Override with LOBSTER_TUI_MOUSE_CAPTURE:
// - true values: 1, true, yes, on
// - false values: 0, false, no, off
func shouldEnableMouseCapture(inline bool) bool {
	mode := strings.ToLower(strings.TrimSpace(os.Getenv("LOBSTER_TUI_MOUSE_CAPTURE")))
	switch mode {
	case "1", "true", "yes", "on":
		return true
	case "0", "false", "no", "off":
		return false
	default:
		return !inline
	}
}

// shouldShowInlineIntro returns whether the inline welcome block should render.
//
// Defaults:
// - inline mode: enabled
// - fullscreen mode: disabled
//
// Override with LOBSTER_TUI_NO_INTRO:
// - true values: 1, true, yes, on
// - false values: 0, false, no, off
func shouldShowInlineIntro(inline bool) bool {
	if !inline {
		return false
	}

	mode := strings.ToLower(strings.TrimSpace(os.Getenv("LOBSTER_TUI_NO_INTRO")))
	switch mode {
	case "1", "true", "yes", "on":
		return false
	case "0", "false", "no", "off", "":
		return true
	default:
		return true
	}
}
