package dashboard

import (
	"encoding/json"
	"strings"
	"testing"

	tea "charm.land/bubbletea/v2"

	"github.com/the-omics-os/lobster-tui/internal/biocomp"
	_ "github.com/the-omics-os/lobster-tui/internal/biocomp/choice"
	_ "github.com/the-omics-os/lobster-tui/internal/biocomp/sampleplate"
	_ "github.com/the-omics-os/lobster-tui/internal/biocomp/textinput"
	_ "github.com/the-omics-os/lobster-tui/internal/biocomp/threshold"
)

// Synthetic sections, shaped exactly as the Python adapter sends them:
// source -> groups (plate) -> mito, plus notes with no dependencies.
const synthetic = `{
  "question": "Set up treated vs control",
  "sections": [
    {"key": "source", "title": "Count source", "depends_on": [], "component": "choice",
     "data": {"question": "Count source?", "min": 1, "max": 1, "options": [
       {"id": "SRC-A", "label": "Author matrix"}, {"id": "SRC-B", "label": "SRC-B"},
       {"id": "SRC-C", "label": "SRC-C", "unavailable": "Normalized; build unknown"}]}},
    {"key": "groups", "title": "Sample groups", "depends_on": ["source"], "component": "plate_map",
     "data": {"question": "Group of each sample?", "title": "groups", "format": 96,
       "choices": ["control", "treated"],
       "wells": [{"row": 0, "col": 0, "sample_id": "S01", "condition": "control"},
                 {"row": 0, "col": 1, "sample_id": "S05", "condition": "treated"}]}},
    {"key": "mito", "title": "Mito cutoff", "depends_on": ["groups"], "component": "threshold_slider",
     "data": {"label": "Mito cutoff?", "min": 0, "max": 0.5, "default": 0.2, "unit": "fraction"}},
    {"key": "notes", "title": "Notes", "depends_on": [], "component": "text_input",
     "data": {"question": "Anything to note?"}}
  ]
}`

func keyMsg(k string) tea.KeyPressMsg {
	switch k {
	case "enter":
		return tea.KeyPressMsg{Code: tea.KeyEnter}
	case "esc":
		return tea.KeyPressMsg{Code: tea.KeyEscape}
	case "down":
		return tea.KeyPressMsg{Code: tea.KeyDown}
	case "tab":
		return tea.KeyPressMsg{Code: tea.KeyTab}
	case "shift+tab":
		return tea.KeyPressMsg{Code: tea.KeyTab, Mod: tea.ModShift}
	case "ctrl+s":
		return tea.KeyPressMsg{Code: 's', Mod: tea.ModCtrl}
	case "ctrl+d":
		return tea.KeyPressMsg{Code: 'd', Mod: tea.ModCtrl}
	}
	return tea.KeyPressMsg{Code: rune(k[0]), Text: k}
}

func press(t *testing.T, c *Component, keys ...string) *biocomp.ComponentResult {
	t.Helper()
	var result *biocomp.ComponentResult
	for _, k := range keys {
		result, _ = c.HandleMsg(keyMsg(k))
	}
	return result
}

func newDashboard(t *testing.T) *Component {
	t.Helper()
	c := &Component{}
	if err := c.Init(json.RawMessage(synthetic)); err != nil {
		t.Fatal(err)
	}
	return c
}

// answerAll answers the four sections in order with their defaults.
func answerAll(t *testing.T, c *Component) {
	t.Helper()
	press(t, c, "enter") // source: SRC-A (cursor starts on the first option)
	press(t, c, "enter") // groups: every well already has a condition
	press(t, c, "enter") // mito: the default 0.2
	press(t, c, "o", "k", "enter")
}

func statuses(c *Component) string {
	var out []string
	for _, s := range c.sections {
		out = append(out, map[status]string{todo: "todo", answered: "done", stale: "stale"}[s.status])
	}
	return strings.Join(out, ",")
}

func TestRegistered(t *testing.T) {
	if biocomp.Get("dashboard") == nil {
		t.Fatal(`"dashboard" is not registered`)
	}
}

func TestInitRejectsBadSections(t *testing.T) {
	cases := map[string]string{
		"one section":        `{"question": "q", "sections": [{"key": "a", "component": "text_input", "data": {"question": "a"}}]}`,
		"unknown component":  strings.Replace(synthetic, `"component": "text_input"`, `"component": "protein_viewer"`, 1),
		"nested dashboard":   strings.Replace(synthetic, `"component": "text_input"`, `"component": "dashboard"`, 1),
		"bad section data":   strings.Replace(synthetic, `"question": "Anything to note?"`, `"question": ""`, 1),
		"unknown dependency": strings.Replace(synthetic, `"depends_on": ["groups"]`, `"depends_on": ["nope"]`, 1),
		"duplicate key":      strings.Replace(synthetic, `"key": "notes"`, `"key": "source"`, 1),
	}
	for name, raw := range cases {
		if err := (&Component{}).Init(json.RawMessage(raw)); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
}

func TestAnsweringMovesToTheNextSectionAndSubmitsEverything(t *testing.T) {
	c := newDashboard(t)
	press(t, c, "enter")
	if c.active != 1 || statuses(c) != "done,todo,todo,todo" {
		t.Fatalf("after source: active=%d statuses=%s", c.active, statuses(c))
	}
	if r := press(t, c, "ctrl+s"); r != nil {
		t.Fatalf("submitted with unanswered sections: %+v", r)
	}
	if !strings.Contains(c.message, `"Sample groups" is not answered`) {
		t.Fatalf("blocker not explained: %q", c.message)
	}
	press(t, c, "enter", "enter", "o", "k", "enter")
	r := press(t, c, "ctrl+s")
	if r == nil || r.Action != "submit" {
		t.Fatalf("expected submit, got %+v", r)
	}
	answers := r.Data["answers"].(map[string]any)
	if ids := selectedIDs(answers["source"].(map[string]any)); len(ids) != 1 || ids[0] != "SRC-A" {
		t.Fatalf("source = %v", answers["source"])
	}
	if got := answers["mito"].(map[string]any)["value"]; got != 0.2 {
		t.Fatalf("mito = %v", got)
	}
	if got := answers["notes"].(map[string]any)["answer"]; got != "ok" {
		t.Fatalf("notes = %v", got)
	}
	assignments := answers["groups"].(map[string]any)["assignments"].(map[string]any)
	if assignments["S01"] != "control" || assignments["S05"] != "treated" {
		t.Fatalf("groups = %v", assignments)
	}
}

func TestChangingUpstreamMakesEveryDependentStale(t *testing.T) {
	c := newDashboard(t)
	answerAll(t, c)
	press(t, c, "tab") // wraps from notes to source
	if c.active != 0 {
		t.Fatalf("active = %d, want the source section", c.active)
	}
	press(t, c, "down", "enter") // SRC-A -> SRC-B

	// groups depends on source; mito depends on groups; notes on nothing.
	if got := statuses(c); got != "done,stale,stale,done" {
		t.Fatalf("statuses = %s", got)
	}
	why := `"Count source" changed: Author matrix (SRC-A) → SRC-B`
	if c.sections[2].staleWhy != why {
		t.Fatalf("mito stale reason = %q", c.sections[2].staleWhy)
	}
	if view := c.View(100, 40); !strings.Contains(view, "⚠ Re-check: "+why) {
		t.Fatalf("view doesn't explain the stale section:\n%s", view)
	}

	if r := press(t, c, "ctrl+s"); r != nil {
		t.Fatalf("submitted with stale sections: %+v", r)
	}
	if !strings.Contains(c.message, `"Sample groups" needs re-checking`) {
		t.Fatalf("blocker not explained: %q", c.message)
	}

	// Re-checking groups with the same answer clears it but not mito: mito
	// was made stale by the source change, not by groups.
	if c.active != 1 {
		t.Fatalf("after the change, active = %d, want groups", c.active)
	}
	press(t, c, "enter")
	if got := statuses(c); got != "done,done,stale,done" {
		t.Fatalf("after re-checking groups: %s", got)
	}
	press(t, c, "enter")
	if r := press(t, c, "ctrl+s"); r == nil || r.Action != "submit" {
		t.Fatalf("expected submit after re-checking, got %+v (message %q)", r, c.message)
	}
}

func TestResubmittingTheSameAnswerChangesNothing(t *testing.T) {
	c := newDashboard(t)
	answerAll(t, c)
	press(t, c, "tab", "enter") // source again: still SRC-A (pre-filled)
	if got := statuses(c); got != "done,done,done,done" {
		t.Fatalf("statuses = %s", got)
	}
}

func TestRevisitedSectionIsPrefilled(t *testing.T) {
	c := newDashboard(t)
	press(t, c, "down", "enter") // source: SRC-B
	press(t, c, "shift+tab")     // back to source
	press(t, c, "enter")         // Enter keeps the pre-filled pick
	if ids := selectedIDs(c.sections[0].answer); len(ids) != 1 || ids[0] != "SRC-B" {
		t.Fatalf("revisit lost the answer: %v", ids)
	}
}

func TestAnsweringUpstreamAfterDownstreamAlsoMarksStale(t *testing.T) {
	c := newDashboard(t)
	press(t, c, "tab", "enter") // answer groups first
	press(t, c, "shift+tab", "shift+tab", "enter")
	if c.sections[1].status != stale || !strings.Contains(c.sections[1].staleWhy, `"Count source" was answered`) {
		t.Fatalf("groups = %v (%q)", c.sections[1].status, c.sections[1].staleWhy)
	}
}

func TestDeferAndCancel(t *testing.T) {
	if r := press(t, newDashboard(t), "ctrl+d"); r == nil || r.Action != "defer" {
		t.Fatalf("ctrl+d: got %+v", r)
	}
	if r := press(t, newDashboard(t), "esc"); r == nil || r.Action != "cancel" {
		t.Fatalf("esc: got %+v", r)
	}
}

func TestUnavailableOptionStillRefusedInsideTheDashboard(t *testing.T) {
	c := newDashboard(t)
	press(t, c, "down", "down", "enter") // SRC-C
	if c.sections[0].status != todo {
		t.Fatal("an unavailable option was accepted")
	}
	if view := c.View(100, 40); !strings.Contains(view, "Normalized; build unknown") {
		t.Fatalf("reason not shown:\n%s", view)
	}
}
