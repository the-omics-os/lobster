package sequenceview

import (
	"encoding/json"
	"strings"
	"testing"

	tea "charm.land/bubbletea/v2"
	"github.com/charmbracelet/x/ansi"

	"github.com/the-omics-os/lobster-tui/internal/biocomp"
)

const letters = "ACGTACGTACGGGTTTAACCGGTT"

func newSeq(t *testing.T, extra map[string]any) *Component {
	t.Helper()
	d := map[string]any{
		"question": "Which guide?", "name": "construct_1", "kind": "dna",
		"sequence": letters,
		"options": []map[string]any{
			{"id": "g1", "label": "guide 1", "start": 1, "end": 5},
			{"id": "g2", "label": "guide 2", "start": 10, "end": 16},
			{"id": "g3", "label": "guide 3", "start": 18, "end": 22, "unavailable": "off-target hit"},
		},
	}
	for k, v := range extra {
		d[k] = v
	}
	raw, _ := json.Marshal(d)
	c := &Component{}
	if err := c.Init(raw); err != nil {
		t.Fatalf("Init: %v", err)
	}
	return c
}

func press(c *Component, keys ...string) *biocomp.ComponentResult {
	var result *biocomp.ComponentResult
	for _, k := range keys {
		var msg tea.KeyPressMsg
		switch k {
		case "down":
			msg = tea.KeyPressMsg{Code: tea.KeyDown}
		case "up":
			msg = tea.KeyPressMsg{Code: tea.KeyUp}
		case "enter":
			msg = tea.KeyPressMsg{Code: tea.KeyEnter}
		case "esc":
			msg = tea.KeyPressMsg{Code: tea.KeyEscape}
		case "pgdown":
			msg = tea.KeyPressMsg{Code: tea.KeyPgDown}
		case "ctrl+d":
			msg = tea.KeyPressMsg{Code: 'd', Mod: tea.ModCtrl}
		}
		result, _ = c.HandleMsg(msg)
	}
	return result
}

func TestPicksTheHighlightedRegion(t *testing.T) {
	c := newSeq(t, nil)
	r := press(c, "down", "enter")
	if r == nil || r.Action != "submit" {
		t.Fatalf("expected a submit, got %+v", r)
	}
	if got := r.Data["selected"].([]map[string]any)[0]["id"]; got != "g2" {
		t.Fatalf("picked %v, want g2", got)
	}
}

func TestShowsTheRegionLettersAndSpan(t *testing.T) {
	c := newSeq(t, nil)
	press(c, "down")
	view := ansi.Strip(c.View(90, 30))
	for _, want := range []string{"guide 2", "10–16", "CGGGTTT", "unavailable: off-target hit"} {
		if !strings.Contains(view, want) {
			t.Fatalf("view lacks %q:\n%s", want, view)
		}
	}
}

func TestUnavailableRegionIsRefused(t *testing.T) {
	c := newSeq(t, nil)
	if r := press(c, "down", "down", "enter"); r != nil {
		t.Fatalf("an unavailable region was picked: %+v", r)
	}
	if !strings.Contains(c.problem, "can't be picked") {
		t.Fatalf("no reason given: %q", c.problem)
	}
}

func TestDefaultStartsUnderTheCursor(t *testing.T) {
	c := newSeq(t, map[string]any{"default": "g2"})
	r := press(c, "enter")
	if got := r.Data["selected"].([]map[string]any)[0]["id"]; got != "g2" {
		t.Fatalf("picked %v, want the default g2", got)
	}
}

func TestScrollKeysDontMoveTheRegion(t *testing.T) {
	c := newSeq(t, nil)
	c.View(90, 30)
	press(c, "pgdown")
	if c.cursor != 0 {
		t.Fatalf("pgdown moved the region cursor to %d", c.cursor)
	}
}

func TestEscTwiceCancelsAndCtrlDDefers(t *testing.T) {
	c := newSeq(t, nil)
	if r := press(c, "esc"); r != nil {
		t.Fatal("one esc should only warn")
	}
	if r := press(c, "esc"); r == nil || r.Action != "cancel" {
		t.Fatalf("second esc should cancel, got %+v", r)
	}
	if r := press(newSeq(t, nil), "ctrl+d"); r == nil || r.Action != "defer" {
		t.Fatalf("ctrl+d should defer, got %+v", r)
	}
}

func TestRefusesARegionOutsideTheSequence(t *testing.T) {
	raw := `{"question":"q","sequence":"ACGT","options":[{"id":"g1","start":2,"end":9}]}`
	if err := (&Component{}).Init(json.RawMessage(raw)); err == nil {
		t.Fatal("a region past the end should be refused")
	}
}

func TestALongRegionRowWrapsInsteadOfBeingCut(t *testing.T) {
	reason := "excluded by the study policy because this guide has two off-target hits in the genome"
	c := newSeq(t, map[string]any{"options": []map[string]any{
		{"id": "g1", "label": "guide 1", "start": 1, "end": 5},
		{"id": "g3", "label": "guide 3", "start": 18, "end": 22, "unavailable": reason},
	}})
	view := ansi.Strip(c.View(60, 40))
	flat := strings.Join(strings.Fields(view), " ")
	if !strings.Contains(flat, "two off-target hits in the genome") {
		t.Fatalf("the reason was cut:\n%s", view)
	}
	for _, line := range strings.Split(view, "\n") {
		if w := len([]rune(line)); w > 60 {
			t.Fatalf("line wider than 60 (%d): %q", w, line)
		}
	}
}
