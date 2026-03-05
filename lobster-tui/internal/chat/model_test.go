package chat

import (
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/the-omics-os/lobster-tui/internal/protocol"
)

func TestHandleConfirmKeyLocalExitConfirmYesQuits(t *testing.T) {
	m := Model{
		pendingConfirm: &protocol.ConfirmPayload{
			Message: "Exit Lobster session?",
		},
		pendingConfirmID: localExitConfirmID,
	}

	updated, cmd := m.handleConfirmKey(tea.KeyMsg{
		Type:  tea.KeyRunes,
		Runes: []rune{'y'},
	})

	got := updated.(Model)
	if !got.quitting {
		t.Fatal("expected model to enter quitting state")
	}
	if cmd == nil {
		t.Fatal("expected quit command")
	}
	if got.pendingConfirm != nil || got.pendingConfirmID != "" {
		t.Fatal("expected local confirm state to be cleared")
	}
}

func TestHandleConfirmKeyLocalExitConfirmNoCancels(t *testing.T) {
	m := Model{
		pendingConfirm: &protocol.ConfirmPayload{
			Message: "Exit Lobster session?",
		},
		pendingConfirmID: localExitConfirmID,
		messages:         make([]ChatMessage, 0, 1),
		streamBuf:        &strings.Builder{},
	}

	updated, cmd := m.handleConfirmKey(tea.KeyMsg{
		Type:  tea.KeyRunes,
		Runes: []rune{'n'},
	})

	got := updated.(Model)
	if got.quitting {
		t.Fatal("expected model to remain active")
	}
	if cmd != nil {
		t.Fatal("expected no quit command on cancel")
	}
	if len(got.messages) == 0 || got.messages[len(got.messages)-1].Content != "Exit cancelled." {
		t.Fatal("expected cancellation message")
	}
}
