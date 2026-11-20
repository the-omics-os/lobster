package chat

import (
	"testing"

	"github.com/the-omics-os/lobster-tui/internal/protocol"
)

// A command that names what it will do, then asks before doing it (the files a
// clear would delete, then "Delete them?"), must show the list before the
// question: the answer is given on what is in scrollback.

func (h *transcriptHarness) commandListing(parts ...string) {
	h.t.Helper()
	h.user("/metadata clear exports")
	for _, p := range parts {
		h.text(p)
	}
}

func TestTextBeforeAConfirmIsShownBeforeTheQuestion(t *testing.T) {
	h := newTranscriptHarness(t)
	h.commandListing("FILE-1\n", "FILE-2\n", "FILE-3\n")
	h.proto(protocol.TypeConfirm, protocol.ConfirmPayload{Message: "Delete them?"})

	if h.m.activeComponent == nil {
		t.Fatal("the confirm must be open")
	}
	h.expectOnce("FILE-1", "FILE-2", "FILE-3")
	h.order("FILE-1", "FILE-2", "FILE-3")

	h.done()
	h.settle()
	h.expectOnce("FILE-1", "FILE-2", "FILE-3")
	h.statsBalanced()
}

func TestTextBeforeASelectIsShownBeforeTheQuestion(t *testing.T) {
	h := newTranscriptHarness(t)
	h.commandListing("CHOICE-CONTEXT\n")
	h.proto(protocol.TypeSelect, protocol.SelectPayload{Message: "Which?", Options: []string{"a", "b"}})

	if h.m.activeComponent == nil {
		t.Fatal("the select must be open")
	}
	h.expectOnce("CHOICE-CONTEXT")
}

func TestTextBeforeAFormIsShownBeforeTheForm(t *testing.T) {
	h := newTranscriptHarness(t)
	h.commandListing("FORM-CONTEXT\n")
	h.proto(protocol.TypeForm, protocol.FormPayload{
		Title:  "Settings",
		Fields: []protocol.FormField{{Key: "name", Label: "Name", Type: "text"}},
	})

	if h.m.activeForm == nil {
		t.Fatal("the form must be open")
	}
	h.expectOnce("FORM-CONTEXT")
}

func TestATableBeforeAConfirmIsShownBeforeTheQuestion(t *testing.T) {
	h := newTranscriptHarness(t)
	h.commandListing("LEAD-TEXT\n")
	h.proto(protocol.TypeTable, protocol.TablePayload{Headers: []string{"H"}, Rows: [][]string{{"ROW-CELL"}}})
	h.proto(protocol.TypeConfirm, protocol.ConfirmPayload{Message: "Go on?"})

	h.expectOnce("LEAD-TEXT", "ROW-CELL")
	h.order("LEAD-TEXT", "ROW-CELL")
}

func TestLiveChatTextBeforeAQuestionIsPrintedOnce(t *testing.T) {
	h := newTranscriptHarness(t)
	h.user("analyze")
	h.text("ASKED-BECAUSE")
	h.proto(protocol.TypeConfirm, protocol.ConfirmPayload{Message: "Proceed?"})
	h.settle()

	h.expectOnce("ASKED-BECAUSE")
	h.done()
	h.settle()
	h.expectOnce("ASKED-BECAUSE")
	h.statsBalanced()
}
