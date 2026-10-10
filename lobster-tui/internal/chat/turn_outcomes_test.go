package chat

import (
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/the-omics-os/lobster-tui/internal/protocol"
)

// Replays wire traffic recorded from the Python bridge
// (tests/unit/cli/test_go_tui_turn_outcomes.py regenerates the fixture and
// fails if the emitter drifts from it) through the real TUI model, and checks
// what reaches scrollback and that the turn is always closed afterwards.

type wireEvent struct {
	Type    string          `json:"type"`
	Payload json.RawMessage `json:"payload"`
}

func loadTurnOutcomes(t *testing.T) map[string][]wireEvent {
	t.Helper()
	raw, err := os.ReadFile("testdata/turn_outcomes.json")
	if err != nil {
		t.Fatalf("read golden wire fixture: %v", err)
	}
	var out map[string][]wireEvent
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatalf("decode golden wire fixture: %v", err)
	}
	return out
}

func (h *transcriptHarness) replay(events []wireEvent) {
	h.t.Helper()
	for _, ev := range events {
		if ev.Type == protocol.TypeComponentRender {
			continue // widget lifecycle is covered by component tests
		}
		var payload any
		if len(ev.Payload) > 0 {
			payload = json.RawMessage(ev.Payload)
		}
		h.proto(ev.Type, payload)
	}
	h.settle()
}

func TestRecordedPythonTurnOutcomesRenderCorrectly(t *testing.T) {
	const incomplete = "Response incomplete"
	cases := map[string]struct {
		once  []string // printed exactly once, in this order
		never []string
		// incomplete notice expected exactly once (true) or never (false)
		marked bool
	}{
		"complete":              {once: []string{"ANSWER-ONE"}},
		"error_after_partial":   {once: []string{"PARTIAL-ANSWER", incomplete}, marked: true},
		"exception_mid_stream":  {once: []string{"PARTIAL-ANSWER", incomplete}, marked: true},
		"ends_without_complete": {once: []string{"PARTIAL-ANSWER", incomplete}, marked: true},
		"silent_cancel":         {never: []string{"PARTIAL-ANSWER"}},
		"hitl_resume_complete":  {once: []string{"QUESTION-PART", "RESUMED-PART"}},
		"hitl_resume_error":     {once: []string{"QUESTION-PART", "RESUMED-PART", incomplete}, marked: true},
		"hitl_abandoned":        {once: []string{"QUESTION-PART"}},
	}

	traffic := loadTurnOutcomes(t)
	if len(traffic) != len(cases) {
		t.Fatalf("fixture has %d scenarios, test covers %d", len(traffic), len(cases))
	}

	for name, want := range cases {
		events, ok := traffic[name]
		if !ok {
			t.Fatalf("scenario %q missing from fixture", name)
		}
		t.Run(name, func(t *testing.T) {
			h := newTranscriptHarness(t)
			h.user("question")
			h.replay(events)

			h.expectOnce(want.once...)
			h.order(append([]string{"question"}, want.once...)...)
			h.expectNever(want.never...)
			if !want.marked {
				h.expectNever(incomplete)
			}
			if h.m.activeTurn != activeTurnNone || h.m.isStreaming || h.m.isCanceling || h.m.streamBuf.Len() != 0 {
				t.Fatalf("turn must be closed: turn=%v streaming=%v canceling=%v buf=%q",
					h.m.activeTurn, h.m.isStreaming, h.m.isCanceling, h.m.streamBuf.String())
			}

			// The next turn is independent: printed once and not merged into
			// whatever the previous turn left behind.
			h.user("next question")
			h.text("NEXT-ANSWER")
			h.done()
			h.settle()
			h.expectOnce("NEXT-ANSWER")
			for _, prior := range want.once {
				if prior == incomplete {
					continue
				}
				if strings.Contains(h.joined(), prior+"NEXT-ANSWER") {
					t.Fatalf("next answer was merged into %q", prior)
				}
				if h.count(prior) != 1 {
					t.Fatalf("%q reprinted by the next turn", prior)
				}
			}
			h.statsBalanced()
		})
	}
}
