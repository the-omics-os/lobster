package chat

import (
	"bytes"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/the-omics-os/lobster-tui/internal/protocol"
)

// readErr returns the first error the handler reports for input.
func readErr(t *testing.T, input string) error {
	t.Helper()
	h := protocol.NewHandler(strings.NewReader(input), &bytes.Buffer{})
	h.StartReadLoop()
	t.Cleanup(h.Close)
	select {
	case err := <-h.Errs():
		return err
	case <-time.After(10 * time.Second):
		t.Fatal("no read error")
		return nil
	}
}

func alertsOf(m Model) []string {
	var out []string
	for _, msg := range m.messages {
		if b := findBlock[BlockAlert](msg.Blocks); b != nil {
			out = append(out, b.Level+": "+b.Message)
		}
	}
	return out
}

// A message too large to read is skipped; the user is told, not left to
// wonder where it went.
func TestAMessageSkippedForItsSizeIsTold(t *testing.T) {
	long := `{"version":1,"type":"text","payload":{"content":"` +
		strings.Repeat("x", 5*1024*1024) + `"}}` + "\n"
	m := newTestModelWithHandler(protocol.NewHandler(strings.NewReader(""), &bytes.Buffer{}))

	updated, _ := m.Update(protocolErr{err: readErr(t, long)})

	got := alertsOf(updated.(Model))
	want := "warning: A message from Lobster was too large to show and was skipped."
	if len(got) != 1 || got[0] != want {
		t.Fatalf("alerts = %q, want [%q]", got, want)
	}
}

// Other protocol errors stay out of the chat, as before.
func TestOtherProtocolErrorsAreNotShown(t *testing.T) {
	m := newTestModelWithHandler(protocol.NewHandler(strings.NewReader(""), &bytes.Buffer{}))

	for _, err := range []error{
		readErr(t, "not json\n"),
		errors.New("protocol read: broken pipe"),
	} {
		updated, _ := m.Update(protocolErr{err: err})
		m = updated.(Model)
	}

	if got := alertsOf(m); len(got) != 0 {
		t.Fatalf("alerts = %q, want none", got)
	}
}
