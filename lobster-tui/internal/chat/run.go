package chat

import (
	"fmt"
	"os"

	tea "github.com/charmbracelet/bubbletea"
	"golang.org/x/term"

	"github.com/the-omics-os/lobster-tui/internal/protocol"
	"github.com/the-omics-os/lobster-tui/internal/theme"
)

// Run starts the interactive chat session using the given file descriptor pair
// for protocol communication and the named theme for styling.
//
// fdIn and fdOut are file descriptor numbers for the JSON Lines IPC channel
// (typically inherited from the parent Python process).
func Run(fdIn, fdOut int, themeName string, version string) error {
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
		activeTheme = theme.LobsterDark
	}

	// ---- BubbleTea model ----------------------------------------------------
	model := NewModel(handler, activeTheme.Styles, width, height)

	p := tea.NewProgram(model, tea.WithAltScreen())
	if _, err := p.Run(); err != nil {
		return fmt.Errorf("chat session: %w", err)
	}

	return nil
}
