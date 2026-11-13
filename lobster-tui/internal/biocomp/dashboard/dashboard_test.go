package dashboard

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"testing"

	tea "charm.land/bubbletea/v2"
	"github.com/charmbracelet/x/ansi"

	"github.com/the-omics-os/lobster-tui/internal/biocomp"
	_ "github.com/the-omics-os/lobster-tui/internal/biocomp/choice"
	_ "github.com/the-omics-os/lobster-tui/internal/biocomp/choicelist"
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

// submitViaReview presses ctrl+s (opens the review screen) and enter.
func submitViaReview(t *testing.T, c *Component) *biocomp.ComponentResult {
	t.Helper()
	if r := press(t, c, "ctrl+s"); r != nil {
		t.Fatalf("ctrl+s should open the review screen, got %+v", r)
	}
	if !c.reviewing {
		t.Fatalf("ctrl+s did not open the review screen (message %q)", c.message)
	}
	return press(t, c, "enter")
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
	r := submitViaReview(t, c)
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
	// notes is a text section, so tab types there; shift+tab moves back.
	press(t, c, "shift+tab", "shift+tab", "shift+tab")
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
	if view := plainView(c, 100, 40); !strings.Contains(view, "⚠ Re-check: "+why) {
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
	if r := submitViaReview(t, c); r == nil || r.Action != "submit" {
		t.Fatalf("expected submit after re-checking, got %+v (message %q)", r, c.message)
	}
}

func TestResubmittingTheSameAnswerChangesNothing(t *testing.T) {
	c := newDashboard(t)
	answerAll(t, c)
	press(t, c, "shift+tab", "shift+tab", "shift+tab", "enter") // source again: still SRC-A
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
	if r := press(t, newDashboard(t), "esc", "esc"); r == nil || r.Action != "cancel" {
		t.Fatalf("esc esc: got %+v", r)
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

// withSections builds dashboard data from the synthetic sections, changed by
// edit (which may add keys such as "problems" or "answer").
func withSections(t *testing.T, edit func(d map[string]any, sections []map[string]any)) *Component {
	t.Helper()
	var d map[string]any
	if err := json.Unmarshal([]byte(synthetic), &d); err != nil {
		t.Fatal(err)
	}
	var sections []map[string]any
	for _, s := range d["sections"].([]any) {
		sections = append(sections, s.(map[string]any))
	}
	edit(d, sections)
	raw, _ := json.Marshal(d)
	c := &Component{}
	if err := c.Init(raw); err != nil {
		t.Fatal(err)
	}
	return c
}

// qcSection is a choice_list section: retain or exclude (with a reason).
var qcSection = map[string]any{
	"key": "qc", "title": "QC review", "depends_on": []any{"groups"}, "component": "choice_list",
	"data": map[string]any{
		"question": "Keep each flagged sample?",
		"items":    []any{map[string]any{"id": "S01", "label": "Donor A"}, map[string]any{"id": "S05", "label": "Donor E"}},
		"options": []any{
			map[string]any{"id": "retain", "label": "Retain"},
			map[string]any{"id": "exclude", "label": "Exclude", "reason": map[string]any{"key": "reason", "question": "Why?"}},
		},
	},
}

func withQC(t *testing.T) *Component {
	return withSections(t, func(d map[string]any, s []map[string]any) {
		d["sections"] = []any{s[0], s[1], qcSection, s[3]}
	})
}

func plainView(c *Component, w, h int) string { return ansi.Strip(c.View(w, h)) }

func TestDraftsSurviveSwitchingSections(t *testing.T) {
	c := newDashboard(t)
	press(t, c, "shift+tab", "o", "k")                         // notes (text): type without saving
	press(t, c, "shift+tab", "shift+tab", "shift+tab", "down") // source: highlight SRC-B
	press(t, c, "shift+tab")                                   // back to notes
	if view := plainView(c, 100, 40); !strings.Contains(view, "ok") {
		t.Fatalf("typed text lost on switching:\n%s", view)
	}
	press(t, c, "tab") // tab types in a text section: still on notes
	if c.active != 3 {
		t.Fatalf("tab left the text section: active = %d", c.active)
	}
	press(t, c, "enter") // notes recorded; the first open section is source
	if c.active != 0 {
		t.Fatalf("after notes, active = %d, want source", c.active)
	}
	press(t, c, "enter") // the highlighted SRC-B was kept while away
	if ids := selectedIDs(c.sections[0].answer); len(ids) != 1 || ids[0] != "SRC-B" {
		t.Fatalf("source draft lost: %v", ids)
	}
}

func TestUnsavedDraftBlocksSubmit(t *testing.T) {
	c := newDashboard(t)
	answerAll(t, c)
	press(t, c, "shift+tab", "shift+tab", "shift+tab", "down") // source: SRC-B on screen, SRC-A recorded
	if r := press(t, c, "ctrl+s"); r != nil || c.reviewing {
		t.Fatalf("submitted with an unsaved draft: %+v (reviewing %v)", r, c.reviewing)
	}
	want := `unsaved changes in "Count source": press enter in that section to keep them`
	if !strings.Contains(c.message, want) {
		t.Fatalf("message = %q", c.message)
	}
	press(t, c, "up") // back to what was recorded: no longer a blocker
	if r := submitViaReview(t, c); r == nil || r.Action != "submit" {
		t.Fatalf("got %+v (%q)", r, c.message)
	}
}

func TestEscWhileTypingDoesNotCancel(t *testing.T) {
	c := withQC(t)
	press(t, c, "tab", "tab") // qc
	press(t, c, "2")          // S01: exclude -> reason box
	if r := press(t, c, "tab"); r != nil || c.active != 2 {
		t.Fatalf("tab left the reason box: active %d, %+v", c.active, r)
	}
	if r := press(t, c, "esc", "esc"); r != nil {
		t.Fatalf("esc while typing a reason cancelled the dashboard: %+v", r)
	}
	// In a text_input section esc also goes to the text box.
	press(t, c, "shift+tab", "shift+tab", "shift+tab")
	if c.active != 3 {
		t.Fatalf("active = %d, want notes", c.active)
	}
	if r := press(t, c, "esc", "esc"); r != nil {
		t.Fatalf("esc in a text section cancelled the dashboard: %+v", r)
	}
}

func TestEscAsksOnceBeforeCancelling(t *testing.T) {
	c := newDashboard(t)
	if r := press(t, c, "esc"); r != nil {
		t.Fatalf("a single esc cancelled: %+v", r)
	}
	if !strings.Contains(c.message, "press esc again to cancel everything; nothing will be applied") {
		t.Fatalf("message = %q", c.message)
	}
	press(t, c, "down") // any other key clears the prompt
	if c.message != "" {
		t.Fatalf("prompt not cleared: %q", c.message)
	}
	if r := press(t, c, "esc"); r != nil {
		t.Fatalf("esc after another key cancelled at once: %+v", r)
	}
	if r := press(t, c, "esc"); r == nil || r.Action != "cancel" {
		t.Fatalf("second esc: %+v", r)
	}
}

// leaves collects every non-empty string and number in a JSON value.
func leaves(v any, out *[]string) {
	switch x := v.(type) {
	case map[string]any:
		for _, e := range x {
			leaves(e, out)
		}
	case []any:
		for _, e := range x {
			leaves(e, out)
		}
	case string:
		if x != "" {
			*out = append(*out, x)
		}
	case float64:
		*out = append(*out, fmt.Sprint(x))
	}
}

func TestReviewScreenShowsExactlyTheReplyAndEnterConfirms(t *testing.T) {
	c := withQC(t)
	press(t, c, "enter", "enter")   // source, groups
	press(t, c, "2")                // qc S01: exclude
	for _, r := range "low depth" { // type the reason
		c.HandleMsg(tea.KeyPressMsg{Code: r, Text: string(r)})
	}
	press(t, c, "enter", "down", "1", "enter") // save reason, S05 retain, submit qc
	press(t, c, "n", "o", "t", "e", "enter")
	if r := press(t, c, "ctrl+s"); r != nil || !c.reviewing {
		t.Fatalf("ctrl+s should open the review: %+v (%q)", r, c.message)
	}
	view := plainView(c, 60, 200)
	for _, s := range c.sections {
		if !strings.Contains(view, s.Title) {
			t.Fatalf("review is missing section %q:\n%s", s.Title, view)
		}
	}
	var want []string
	raw, _ := json.Marshal(c.pending)
	var reply any
	_ = json.Unmarshal(raw, &reply)
	leaves(reply, &want)
	for _, w := range want {
		if !strings.Contains(view, w) {
			t.Fatalf("review is missing %q from the reply:\n%s", w, view)
		}
	}
	for _, w := range []string{"Author matrix (SRC-A)", "control (1): S01", "treated (1): S05",
		"Donor A (S01) → Exclude (exclude) · reason: low depth", "note"} {
		if !strings.Contains(view, w) {
			t.Fatalf("review is missing %q:\n%s", w, view)
		}
	}
	shown := c.pending
	r := press(t, c, "enter")
	if r == nil || r.Action != "submit" || !reflect.DeepEqual(r.Data, shown) {
		t.Fatalf("enter should send exactly what was shown: %+v", r)
	}
}

func TestReviewBackKeysReturnToTheSections(t *testing.T) {
	for _, back := range []string{"b", "esc", "shift+tab"} {
		c := newDashboard(t)
		answerAll(t, c)
		press(t, c, "ctrl+s")
		if r := press(t, c, back); r != nil || c.reviewing {
			t.Fatalf("%s: still reviewing (%v) or finished: %+v", back, c.reviewing, r)
		}
		if r := press(t, c, "enter"); r != nil && r.Action == "submit" {
			t.Fatalf("%s: enter on the sections submitted the dashboard", back)
		}
	}
}

func TestReviewScrolls(t *testing.T) {
	c := newDashboard(t)
	answerAll(t, c)
	press(t, c, "ctrl+s")
	if full := plainView(c, 60, 200); !strings.Contains(full, "0.2 fraction") || !strings.Contains(full, "Author matrix (SRC-A)") {
		t.Fatalf("review is missing the slider or the choice:\n%s", full)
	}
	first := plainView(c, 60, 8)
	if n := len(strings.Split(first, "\n")); n > 8 {
		t.Fatalf("review is %d lines for height 8", n)
	}
	for i := 0; i < 30; i++ {
		press(t, c, "down")
	}
	last := plainView(c, 60, 8)
	if first == last || !strings.Contains(last, "ok") {
		t.Fatalf("review did not scroll to the end:\n%s", last)
	}
}

func TestProblemsAreShownAndThoseSectionsStartUnanswered(t *testing.T) {
	c := withSections(t, func(d map[string]any, s []map[string]any) {
		d["problems"] = []any{"'control' has no items left after removing S01"}
		s[1]["problems"] = []any{"'control' has no items left after removing S01"}
		s[1]["answer"] = map[string]any{"assignments": map[string]any{"S01": "treated", "S05": "treated"}}
		s[0]["answer"] = map[string]any{"selected": []any{map[string]any{"id": "SRC-B"}}}
	})
	if got := statuses(c); got != "done,todo,todo,todo" {
		t.Fatalf("statuses = %s", got)
	}
	if c.active != 1 {
		t.Fatalf("should open on the first unanswered section, active = %d", c.active)
	}
	view := plainView(c, 100, 50)
	if strings.Count(view, "⚠ 'control' has no items left after removing S01") != 2 {
		t.Fatalf("top and section problems not both shown:\n%s", view)
	}
	if top, sec := strings.Index(view, "⚠"), strings.Index(view, "Count source"); top > sec {
		t.Fatalf("top-level problems should be above the sections:\n%s", view)
	}
}

func TestSectionsWithAnAnswerStartAnsweredAndAppearOnReview(t *testing.T) {
	prev := map[string]any{
		"source": map[string]any{"selected": []any{map[string]any{"id": "SRC-B"}}},
		"groups": map[string]any{"assignments": map[string]any{"S01": "treated", "S05": "control"},
			"changes": []any{
				map[string]any{"sample_id": "S01", "supplied": "control", "assigned": "treated"},
				map[string]any{"sample_id": "S05", "supplied": "treated", "assigned": "control"}},
			"reviewed_flags": []any{}},
		"qc": map[string]any{"answers": map[string]any{
			"S01": map[string]any{"id": "exclude", "fields": map[string]any{"reason": "low depth"}},
			"S05": map[string]any{"id": "retain"}}},
		"notes": map[string]any{"answer": "earlier note"},
	}
	build := func() *Component {
		return withSections(t, func(d map[string]any, s []map[string]any) {
			qc := map[string]any{}
			for k, v := range qcSection {
				qc[k] = v
			}
			qc["answer"] = prev["qc"]
			d["sections"] = []any{s[0], s[1], qc, s[3]}
			s[0]["answer"], s[1]["answer"], s[3]["answer"] = prev["source"], prev["groups"], prev["notes"]
		})
	}
	c := build()
	if got := statuses(c); got != "done,done,done,done" {
		t.Fatalf("statuses = %s", got)
	}
	if view := plainView(c, 100, 50); !strings.Contains(view, "Retain (1) · Exclude (1)") {
		t.Fatalf("choice_list summary missing:\n%s", view)
	}
	// Children are pre-filled: no unsaved-draft blocker, review opens at once.
	r := submitViaReview(t, c)
	if r == nil || r.Action != "submit" {
		t.Fatalf("got %+v (%q)", r, c.message)
	}
	raw, _ := json.Marshal(r.Data["answers"])
	var got map[string]any
	_ = json.Unmarshal(raw, &got)
	if !reflect.DeepEqual(got, prev) {
		t.Fatalf("reply changed:\n got %v\nwant %v", got, prev)
	}
	// Re-recording a pre-filled section keeps the same answer: nothing goes stale.
	c = build()
	press(t, c, "tab", "enter") // groups
	if got := statuses(c); got != "done,done,done,done" {
		t.Fatalf("re-recording the pre-filled plate changed it: %s", got)
	}
}

func TestChoiceListSectionSummary(t *testing.T) {
	c := withQC(t)
	press(t, c, "enter", "enter") // source, groups
	press(t, c, "1", "down", "1", "enter")
	if c.sections[2].status != answered || !strings.Contains(c.sections[2].summary, "Retain (2) · Exclude (0)") {
		t.Fatalf("qc = %v %q", c.sections[2].status, c.sections[2].summary)
	}
}

func TestLongTextWrapsAndNothingIsLost(t *testing.T) {
	long := "this sentence is deliberately much longer than forty columns so it must wrap"
	c := withSections(t, func(d map[string]any, s []map[string]any) {
		d["question"] = "dashboard question " + long
		d["problems"] = []any{"top problem " + long}
		s[0]["title"] = "source title " + long
		s[2]["problems"] = []any{"section problem " + long}
	})
	press(t, c, "enter", "enter") // source, groups
	press(t, c, "shift+tab", "shift+tab", "down", "enter")
	press(t, c, "tab") // to mito: the Crust plate itself needs more than 40 columns
	view := plainView(c, 40, 200)
	for _, line := range strings.Split(view, "\n") {
		if w := ansi.StringWidth(line); w > 40 {
			t.Fatalf("line wider than 40 (%d): %q", w, line)
		}
	}
	words := strings.Fields(view)
	for _, text := range []string{c.question, c.problems[0], c.sections[0].Title,
		c.sections[2].Problems[0], c.sections[1].staleWhy} {
		for _, w := range strings.Fields(text) {
			found := false
			for _, x := range words {
				found = found || strings.Contains(x, w)
			}
			if !found {
				t.Fatalf("word %q missing:\n%s", w, view)
			}
		}
	}
}

// reask builds the synthetic dashboard as a re-ask: every section carries its
// previous answer, and "groups" also carries a problem.
func reask(t *testing.T) *Component {
	t.Helper()
	return withSections(t, func(d map[string]any, s []map[string]any) {
		s[0]["answer"] = map[string]any{"selected": []any{map[string]any{"id": "SRC-A"}}}
		s[1]["answer"] = map[string]any{"assignments": map[string]any{"S01": "control", "S05": "treated"},
			"changes": []any{}, "reviewed_flags": []any{}}
		s[1]["problems"] = []any{"'control' has too few items"}
		s[2]["answer"] = map[string]any{"value": 0.2}
		s[3]["answer"] = map[string]any{"answer": "ok"}
	})
}

func TestProblemSectionRecordedUnchangedMakesNothingStale(t *testing.T) {
	c := reask(t)
	if got := statuses(c); got != "done,todo,done,done" || c.active != 1 {
		t.Fatalf("statuses = %s, active = %d", got, c.active)
	}
	press(t, c, "enter") // groups: the pre-filled answer, unchanged
	if got := statuses(c); got != "done,done,done,done" {
		t.Fatalf("recording the same answer made something stale: %s", got)
	}
	if r := submitViaReview(t, c); r == nil || r.Action != "submit" {
		t.Fatalf("got %+v (%q)", r, c.message)
	}
}

func TestProblemSectionRecordedChangedMakesDependantsStale(t *testing.T) {
	c := reask(t)
	press(t, c, "2", "enter") // groups: S01 (cursor) -> treated
	if got := statuses(c); got != "done,done,stale,done" {
		t.Fatalf("statuses = %s", got)
	}
	why := `"Sample groups" changed: control (1) · treated (1) → treated (2)`
	if c.sections[2].staleWhy != why {
		t.Fatalf("mito stale reason = %q, want %q", c.sections[2].staleWhy, why)
	}
}

// pythonPlateAnswer is the plate answer as the Python adapter sends it
// (_as_reply): "assignments" only, without the "changes" and
// "reviewed_flags" the plate adds to its own reply (E2E_REPORT #1).
func pythonPlateAnswer(s01, s05 string) map[string]any {
	return map[string]any{"assignments": map[string]any{"S01": s01, "S05": s05}}
}

func TestReconfirmingAPlateWithAnAssignmentsOnlyBaselineMakesNothingStale(t *testing.T) {
	c := withSections(t, func(d map[string]any, s []map[string]any) {
		s[0]["answer"] = map[string]any{"selected": []any{map[string]any{"id": "SRC-A"}}}
		s[1]["answer"] = pythonPlateAnswer("control", "treated")
		s[1]["problems"] = []any{"'control' has too few items"}
		s[2]["answer"] = map[string]any{"value": 0.2}
		s[3]["answer"] = map[string]any{"answer": "ok"}
	})
	if got := statuses(c); got != "done,todo,done,done" || c.active != 1 {
		t.Fatalf("statuses = %s, active = %d", got, c.active)
	}
	press(t, c, "enter") // groups: the pre-filled answer, unchanged
	if got := statuses(c); got != "done,done,done,done" {
		t.Fatalf("re-confirming the unchanged plate made something stale: %s (%q)",
			got, c.sections[2].staleWhy)
	}
	// The recorded answer is the plate's full reply, as sent to Python.
	if _, ok := c.sections[1].answer["changes"]; !ok {
		t.Fatalf("recorded answer lacks the plate's changes: %v", c.sections[1].answer)
	}
}

func TestChangingAPlateWithAnAssignmentsOnlyBaselineMakesDependantsStale(t *testing.T) {
	c := withSections(t, func(d map[string]any, s []map[string]any) {
		s[0]["answer"] = map[string]any{"selected": []any{map[string]any{"id": "SRC-A"}}}
		s[1]["answer"] = pythonPlateAnswer("control", "treated")
		s[1]["problems"] = []any{"'control' has too few items"}
		s[2]["answer"] = map[string]any{"value": 0.2}
	})
	press(t, c, "2", "enter") // groups: S01 (cursor) -> treated
	if got := statuses(c); got != "done,done,stale,todo" {
		t.Fatalf("statuses = %s", got)
	}
}

func TestReconfirmingAnAnsweredPlateWithAnAssignmentsOnlyAnswerMakesNothingStale(t *testing.T) {
	c := withSections(t, func(d map[string]any, s []map[string]any) {
		s[0]["answer"] = map[string]any{"selected": []any{map[string]any{"id": "SRC-B"}}}
		s[1]["answer"] = pythonPlateAnswer("treated", "control")
		s[2]["answer"] = map[string]any{"value": 0.2}
		s[3]["answer"] = map[string]any{"answer": "ok"}
	})
	if got := statuses(c); got != "done,done,done,done" {
		t.Fatalf("statuses = %s", got)
	}
	press(t, c, "tab", "enter") // groups, unchanged
	if got := statuses(c); got != "done,done,done,done" {
		t.Fatalf("re-confirming the answered plate made something stale: %s (%q)",
			got, c.sections[2].staleWhy)
	}
}

func TestSameAnswerComparesTheEarlierAnswersKeys(t *testing.T) {
	full := map[string]any{"assignments": map[string]any{"S01": "control"}, "changes": []any{}}
	cases := []struct {
		name   string
		before map[string]any
		want   bool
	}{
		{"no earlier answer", nil, false},
		{"empty earlier answer", map[string]any{}, false},
		{"subset, same value", map[string]any{"assignments": map[string]any{"S01": "control"}}, true},
		{"subset, other value", map[string]any{"assignments": map[string]any{"S01": "treated"}}, false},
		{"key the reply lacks", map[string]any{"reviewed_flags": []any{}}, false},
		{"full, same", full, true},
	}
	for _, tc := range cases {
		if got := sameAnswer(tc.before, full); got != tc.want {
			t.Errorf("%s: sameAnswer = %v, want %v", tc.name, got, tc.want)
		}
	}
}

func TestPlainNumberNeverUsesExponentForm(t *testing.T) {
	cases := map[any]string{
		2e6:       "2,000,000",
		2000519.0: "2,000,519",
		0.31:      "0.31",
		-1500.0:   "-1,500",
		7:         "7",
		0.0001:    "0.0001",
		"n/a":     "n/a",
	}
	for in, want := range cases {
		if got := plainNumber(in); got != want {
			t.Errorf("plainNumber(%v) = %q, want %q", in, got, want)
		}
	}
}

// A moved slider or edited text after its section was recorded is unsaved
// too (v3 C-01): only choices reported a draft, so these were sent as the
// old value without a word.
func TestMovedSliderIsUnsaved(t *testing.T) {
	c := newDashboard(t)
	answerAll(t, c)
	press(t, c, "shift+tab") // mito: 0.2 recorded
	c.HandleMsg(tea.KeyPressMsg{Code: tea.KeyRight})
	if r := press(t, c, "ctrl+s"); r != nil || c.reviewing {
		t.Fatalf("submitted with a moved slider: %+v (reviewing %v)", r, c.reviewing)
	}
	if want := `unsaved changes in "Mito cutoff"`; !strings.Contains(c.message, want) {
		t.Fatalf("message = %q", c.message)
	}
	c.HandleMsg(tea.KeyPressMsg{Code: tea.KeyLeft}) // back to what was recorded
	if r := submitViaReview(t, c); r == nil || r.Action != "submit" {
		t.Fatalf("got %+v (%q)", r, c.message)
	}
}

func TestEditedTextIsUnsaved(t *testing.T) {
	c := newDashboard(t)
	answerAll(t, c) // notes: "ok" recorded, still the active section
	press(t, c, "!")
	if r := press(t, c, "ctrl+s"); r != nil || c.reviewing {
		t.Fatalf("submitted with edited text: %+v (reviewing %v)", r, c.reviewing)
	}
	if want := `unsaved changes in "Notes"`; !strings.Contains(c.message, want) {
		t.Fatalf("message = %q", c.message)
	}
	c.HandleMsg(tea.KeyPressMsg{Code: tea.KeyBackspace})
	if r := submitViaReview(t, c); r == nil || r.Action != "submit" {
		t.Fatalf("got %+v (%q)", r, c.message)
	}
}

// The dashboard lists ctrl+d once, as its own key: a child's is left out.
func TestCtrlDIsListedOnce(t *testing.T) {
	c := newDashboard(t)
	n := 0
	for _, b := range c.KeyBindings() {
		if b.Help().Key == "ctrl+d" {
			n++
		}
	}
	if n != 1 {
		t.Fatalf("ctrl+d listed %d times", n)
	}
}

func TestTheChoiceDotFollowsThePickNotTheProposal(t *testing.T) {
	// Proposed SRC-A, picked SRC-B: back on the section, (●) marks SRC-B.
	raw := strings.Replace(synthetic, `"min": 1, "max": 1, "options"`,
		`"min": 1, "max": 1, "default": ["SRC-A"], "options"`, 1)
	c := &Component{}
	if err := c.Init(json.RawMessage(raw)); err != nil {
		t.Fatal(err)
	}
	press(t, c, "down", "enter")
	press(t, c, "shift+tab")
	view := ansi.Strip(c.View(100, 40))
	if !strings.Contains(view, "(●) SRC-B") || strings.Contains(view, "(●) Author matrix") {
		t.Fatalf("the dot should be on SRC-B only:\n%s", view)
	}
}
