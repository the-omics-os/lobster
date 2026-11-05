package chat

import (
	"bytes"
	"testing"

	"github.com/the-omics-os/lobster-tui/internal/protocol"
)

// A question screen must never stay drawn once Python stopped waiting for
// it (v3 L-01): its reply would be dropped and nobody would know.

func withScreen(t *testing.T, id string) (Model, *bytes.Buffer) {
	t.Helper()
	h, out := newCapturingHandler()
	m := newTestModelWithHandler(h)
	m.activeComponent = &ActiveComponent{Component: &mockComponent{name: "choice", mode: "overlay"}, MsgID: id}
	return m, out
}

func lastAlert(m Model) (string, string) {
	for i := len(m.messages) - 1; i >= 0; i-- {
		for _, b := range m.messages[i].Blocks {
			if a, ok := b.(BlockAlert); ok {
				return a.Level, a.Message
			}
		}
	}
	return "", ""
}

func TestComponentCloseRemovesTheScreenAndSaysWhy(t *testing.T) {
	m, out := withScreen(t, "render-1")
	msg := testProtocolMsg(t, protocol.TypeComponentClose, protocol.ComponentClosePayload{
		Message: "This question was closed (no answer after 30 minutes); nothing was sent.",
	})
	msg.Message.ID = "render-1"
	updated, _ := m.handleProtocol(msg)
	m = updated.(Model)
	if m.activeComponent != nil {
		t.Fatal("the screen is still drawn")
	}
	level, text := lastAlert(m)
	if level != "warning" || text != "This question was closed (no answer after 30 minutes); nothing was sent." {
		t.Fatalf("alert = %q %q", level, text)
	}
	if sent := parseCapturedMessages(out); len(sent) != 0 {
		t.Fatalf("closing must send nothing back, sent %+v", sent)
	}
}

func TestComponentCloseForAnotherScreenKeepsThisOne(t *testing.T) {
	m, _ := withScreen(t, "render-2")
	msg := testProtocolMsg(t, protocol.TypeComponentClose, protocol.ComponentClosePayload{Message: "closed"})
	msg.Message.ID = "render-1"
	updated, _ := m.handleProtocol(msg)
	if updated.(Model).activeComponent == nil {
		t.Fatal("a close for an earlier screen removed the current one")
	}
}

func TestTurnEndRemovesALeftoverScreen(t *testing.T) {
	for _, summary := range []string{"cancelled", "error", ""} {
		m, out := withScreen(t, "render-1")
		updated, _ := m.handleProtocol(testProtocolMsg(t, protocol.TypeDone, protocol.DonePayload{Summary: summary}))
		m = updated.(Model)
		if m.activeComponent != nil {
			t.Fatalf("done %q left the screen drawn", summary)
		}
		if _, text := lastAlert(m); text != staleComponentNotice {
			t.Fatalf("done %q: alert = %q", summary, text)
		}
		if sent := parseCapturedMessages(out); len(sent) != 0 {
			t.Fatalf("done %q sent %+v", summary, sent)
		}
	}
}

func TestInterruptPauseKeepsTheScreen(t *testing.T) {
	m, _ := withScreen(t, "render-1")
	updated, _ := m.handleProtocol(testProtocolMsg(t, protocol.TypeDone, protocol.DonePayload{Summary: "interrupt"}))
	m = updated.(Model)
	if m.activeComponent == nil {
		t.Fatal(`done "interrupt" (sent before a question is drawn) removed the screen`)
	}
	if _, text := lastAlert(m); text != "" {
		t.Fatalf("unexpected alert %q", text)
	}
}
