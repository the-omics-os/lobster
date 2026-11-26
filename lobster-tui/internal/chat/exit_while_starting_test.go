package chat

import (
	"bytes"
	"strings"
	"testing"

	tea "charm.land/bubbletea/v2"

	"github.com/the-omics-os/lobster-tui/internal/protocol"
)

// Startup can take a while; /exit typed during "Initializing agents" must
// still leave, not be dropped with the rest of the early input.
func TestExitWhileStartingQuits(t *testing.T) {
	for _, typed := range []string{"/exit", "/quit", " /EXIT "} {
		var out bytes.Buffer
		m := newTestModel()
		m.handler = protocol.NewHandler(strings.NewReader(""), &out)
		m.ready = false
		m.spinnerActive = true
		m.input.SetValue(typed)

		updated, cmd := m.handleKey(tea.KeyPressMsg{Code: tea.KeyEnter})
		got := updated.(Model)

		if !got.quitting || cmd == nil {
			t.Fatalf("%q while starting: expected to quit", typed)
		}
		if msg := readSingleProtocolMessage(t, &out); msg.Type != protocol.TypeQuit {
			t.Fatalf("%q while starting: expected %q, got %q", typed, protocol.TypeQuit, msg.Type)
		}
	}
}

func TestOtherInputWhileStartingStaysInTheBox(t *testing.T) {
	var out bytes.Buffer
	m := newTestModel()
	m.handler = protocol.NewHandler(strings.NewReader(""), &out)
	m.ready = false
	m.input.SetValue("hello")

	updated, _ := m.handleKey(tea.KeyPressMsg{Code: tea.KeyEnter})
	got := updated.(Model)

	if got.quitting || got.input.Value() != "hello" || out.Len() != 0 {
		t.Fatalf("expected the input kept and nothing sent, got quitting=%v input=%q sent=%q",
			got.quitting, got.input.Value(), out.String())
	}
}
