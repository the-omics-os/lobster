package chat

import (
	"bytes"
	"reflect"
	"regexp"
	"strings"
	"sync"
	"testing"

	tea "charm.land/bubbletea/v2"

	"github.com/the-omics-os/lobster-tui/internal/protocol"
)

// transcriptHarness drives the real Model.Update loop and executes every
// returned tea.Cmd, recording the bodies the program would write to terminal
// scrollback (tea.Println) in execution order. Assertions on this ordered
// transcript replace the earlier checks that only looked at message history or
// at "a command was returned".
//
// What it proves: which text reaches the renderer's scrollback, how many times
// and in what order. What it cannot prove: that a terminal painted it (that is
// covered by the real-PTY test against the built binary).
type transcriptHarness struct {
	t       *testing.T
	m       Model
	prints  []string
	held    []inlinePrintReady
	holdTic bool
	quit    bool
	// quitAfter is len(prints) when tea.Quit was observed.
	quitAfter int
}

type lockedBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (b *lockedBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Write(p)
}

var ansiPattern = regexp.MustCompile(`\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)`)

func newTranscriptHarness(t *testing.T) *transcriptHarness {
	t.Helper()
	m := newTestModel()
	m.inline = true
	m.inlineFlow = true
	m.ready = true
	m.showIntro = false
	m.inlineBannerPrinted = true // keep transcripts free of banner chrome
	// A handler whose input is already at EOF: protocol wait commands resolve
	// immediately (the harness ignores them) and Go->Python writes are captured.
	m.handler = protocol.NewHandler(strings.NewReader(""), &lockedBuffer{})
	m.handler.StartReadLoop()
	return &transcriptHarness{t: t, m: m}
}

// send delivers one message to Update and executes the resulting commands.
func (h *transcriptHarness) send(msg tea.Msg) {
	h.t.Helper()
	next, cmd := h.m.Update(msg)
	h.m = next.(Model)
	h.run(cmd)
}

func (h *transcriptHarness) proto(msgType string, payload any) {
	h.t.Helper()
	h.send(testProtocolMsg(h.t, msgType, payload))
}

func (h *transcriptHarness) text(s string) {
	h.proto(protocol.TypeText, protocol.TextPayload{Content: s, Markdown: true})
}
func (h *transcriptHarness) done() { h.proto(protocol.TypeDone, protocol.DonePayload{}) }
func (h *transcriptHarness) doneWith(summary string) {
	h.proto(protocol.TypeDone, protocol.DonePayload{Summary: summary})
}
func (h *transcriptHarness) alert(level, msg string) {
	h.proto(protocol.TypeAlert, protocol.AlertPayload{Level: protocol.AlertLevel(level), Message: msg})
}

// user types a prompt and presses Enter through the real key handler.
func (h *transcriptHarness) user(text string) {
	h.t.Helper()
	h.m.input.SetValue(text)
	h.send(tea.KeyPressMsg{Code: tea.KeyEnter})
}

// run executes a command tree synchronously, feeding results back to Update.
func (h *transcriptHarness) run(cmd tea.Cmd) {
	if cmd == nil {
		return
	}
	h.handle(cmd())
}

func (h *transcriptHarness) handle(msg tea.Msg) {
	if msg == nil {
		return
	}
	switch v := msg.(type) {
	case tea.BatchMsg:
		for _, c := range v {
			h.run(c)
		}
		return
	case protocolEOF:
		return // wait command resolving against the drained test handler
	case tea.QuitMsg:
		h.quit = true
		h.quitAfter = len(h.prints)
		return
	case inlinePrintReady:
		if h.holdTic {
			h.held = append(h.held, v)
			return
		}
		h.send(v)
		return
	case inlinePrintComplete, inlinePrintReset:
		h.send(v)
		return
	}
	rv := reflect.ValueOf(msg)
	switch rv.Type().String() {
	case "tea.sequenceMsg":
		for i := 0; i < rv.Len(); i++ {
			c, ok := rv.Index(i).Interface().(tea.Cmd)
			if !ok {
				h.t.Fatalf("unexpected element in sequence: %T", rv.Index(i).Interface())
			}
			h.run(c)
		}
	case "tea.printLineMessage":
		h.prints = append(h.prints, plain(rv.Field(0).String()))
	}
}

// release delivers held deferral ticks (oldest first).
func (h *transcriptHarness) release() {
	h.t.Helper()
	held := h.held
	h.held = nil
	for _, r := range held {
		h.send(r)
	}
}

// settle lets every deferral tick fire, then returns.
func (h *transcriptHarness) settle() {
	h.t.Helper()
	h.holdTic = false
	h.release()
}

func plain(s string) string { return ansiPattern.ReplaceAllString(s, "") }

func (h *transcriptHarness) joined() string { return strings.Join(h.prints, "\n<>\n") }

func (h *transcriptHarness) count(needle string) int {
	return strings.Count(h.joined(), needle)
}

// order asserts the needles each appear, in the given order, in the transcript.
func (h *transcriptHarness) order(needles ...string) {
	h.t.Helper()
	all := h.joined()
	at := 0
	for _, n := range needles {
		i := strings.Index(all[at:], n)
		if i < 0 {
			h.t.Fatalf("expected %q after offset %d in transcript:\n%s", n, at, all)
		}
		at += i + len(n)
	}
}

func (h *transcriptHarness) expectOnce(needles ...string) {
	h.t.Helper()
	for _, n := range needles {
		if c := h.count(n); c != 1 {
			h.t.Fatalf("expected %q printed exactly once, got %d in transcript:\n%s", n, c, h.joined())
		}
	}
}

func (h *transcriptHarness) expectNever(needles ...string) {
	h.t.Helper()
	for _, n := range needles {
		if c := h.count(n); c != 0 {
			h.t.Fatalf("expected %q never printed, got %d in transcript:\n%s", n, c, h.joined())
		}
	}
}

// statsBalanced checks the outbox bookkeeping invariant: every enqueued
// segment is queued, dispatched, or dropped; acknowledgements never exceed
// dispatches.
func (h *transcriptHarness) statsBalanced() {
	h.t.Helper()
	s := h.m.printStats
	if s.enqueued != s.dispatched+s.dropped+uint64(len(h.m.outbox)) {
		h.t.Fatalf("outbox accounting broken: %+v queued=%d", s, len(h.m.outbox))
	}
	if s.committed > s.dispatched {
		h.t.Fatalf("more acknowledgements than dispatches: %+v", s)
	}
}
