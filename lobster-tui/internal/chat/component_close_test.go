package chat

import (
	"bytes"
	"strings"
	"testing"

	tea "charm.land/bubbletea/v2"
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
		Message: "This question was closed (an error (KeyError)); nothing was sent.",
	})
	msg.Message.ID = "render-1"
	updated, _ := m.handleProtocol(msg)
	m = updated.(Model)
	if m.activeComponent != nil {
		t.Fatal("the screen is still drawn")
	}
	level, text := lastAlert(m)
	if level != "warning" || text != "This question was closed (an error (KeyError)); nothing was sent." {
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

// ctrl+c on a question cancels only that question: the turn goes on so the
// supervisor can reply to the cancel. A second ctrl+c cancels the turn.
func TestCtrlCOnAQuestionCancelsOnlyTheQuestion(t *testing.T) {
	for _, busy := range []string{"streaming", "spinner"} {
		m, out := withScreen(t, "render-1")
		m.isStreaming = busy == "streaming"
		m.spinnerActive = busy == "spinner"
		updated, _ := m.handleKey(tea.KeyPressMsg{Code: 'c', Mod: tea.ModCtrl})
		m = updated.(Model)
		if m.activeComponent != nil {
			t.Fatalf("%s: the screen is still drawn", busy)
		}
		if m.quitting || m.isCanceling {
			t.Fatalf("%s: the first ctrl+c also cancelled the turn or quit", busy)
		}
		if _, text := lastAlert(m); text != questionCancelledNotice {
			t.Fatalf("%s: alert = %q", busy, text)
		}
		sent := parseCapturedMessages(out)
		if len(sent) != 1 || sent[0].Type != protocol.TypeComponentResponse {
			t.Fatalf("%s: want only the question's cancel, sent %+v", busy, sent)
		}

		out.Reset()
		updated, _ = m.handleKey(tea.KeyPressMsg{Code: 'c', Mod: tea.ModCtrl})
		m = updated.(Model)
		sent = parseCapturedMessages(out)
		if len(sent) != 1 || sent[0].Type != protocol.TypeCancel || !m.isCanceling || m.quitting {
			t.Fatalf("%s: the second ctrl+c must cancel the turn, sent %+v", busy, sent)
		}
	}
}

// While a question is on screen the chat box is hidden and gets nothing: a
// paste meant for the question used to land in it too, and be sent later.
func TestAQuestionHidesTheChatBoxAndKeepsPastesOutOfIt(t *testing.T) {
	m, _ := withScreen(t, "render-1")
	m.inline = true
	m.width, m.height = 100, 40
	composer := m.renderComposer()
	if strings.Contains(m.View().Content, composer) {
		t.Fatal("the chat box is drawn under the question")
	}
	updated, _ := m.Update(tea.PasteMsg{Content: "smudged, unreadable"})
	m = updated.(Model)
	if v := m.input.Value(); v != "" {
		t.Fatalf("the paste reached the chat box: %q", v)
	}
	m.dropComponent()
	if !strings.Contains(m.View().Content, m.renderComposer()) {
		t.Fatal("the chat box should come back when the question is gone")
	}
}
