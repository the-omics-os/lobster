package sampleplate_test

import (
	"encoding/json"
	"strings"
	"testing"

	tea "charm.land/bubbletea/v2"
	"github.com/charmbracelet/x/ansi"

	"github.com/the-omics-os/lobster-tui/internal/biocomp/dashboard"
	_ "github.com/the-omics-os/lobster-tui/internal/biocomp/sampleplate"
)

// A plate and a choice_list as dashboard sections: the dashboard's own esc
// (asks once, then cancels) is unchanged, so neither section adds escs.
const twoSections = `{
  "question": "Set up the comparison",
  "sections": [
    {"key": "groups", "title": "Groups", "depends_on": [], "component": "plate_map",
     "data": {"question": "Group of each sample?", "choices": ["control", "treated"],
       "wells": [{"row": 0, "col": 0, "sample_id": "S01", "condition": "control"}]}},
    {"key": "qc", "title": "QC", "depends_on": [], "component": "choice_list",
     "data": {"question": "Keep each sample?", "items": [{"id": "S01"}],
       "options": [{"id": "retain"}, {"id": "exclude"}]}}
  ]
}`

func TestInADashboardEscEscStillCancelsEverySection(t *testing.T) {
	key := func(k string) tea.KeyPressMsg {
		if k == "tab" {
			return tea.KeyPressMsg{Code: tea.KeyTab}
		}
		return tea.KeyPressMsg{Code: tea.KeyEscape}
	}
	for tabs, section := range []string{"plate_map", "choice_list"} {
		d := &dashboard.Component{}
		if err := d.Init(json.RawMessage(twoSections)); err != nil {
			t.Fatal(err)
		}
		for i := 0; i < tabs; i++ {
			d.HandleMsg(key("tab"))
		}
		// The active section's own keys are in the help: check which it is.
		own := map[string]string{"plate_map": "1-9/0", "choice_list": "←/→/1-9"}[section]
		found := false
		for _, b := range d.KeyBindings() {
			found = found || b.Help().Key == own
		}
		if !found {
			t.Fatalf("%s is not the active section", section)
		}
		if r, _ := d.HandleMsg(key("esc")); r != nil {
			t.Fatalf("%s: a single esc cancelled: %+v", section, r)
		}
		view := d.View(100, 40)
		if !strings.Contains(view, "press esc again to cancel everything") ||
			strings.Contains(view, "cancel this question") {
			t.Fatalf("%s: expected only the dashboard's prompt:\n%s", section, view)
		}
		if r, _ := d.HandleMsg(key("esc")); r == nil || r.Action != "cancel" {
			t.Fatalf("%s: esc esc should cancel the dashboard, got %+v", section, r)
		}
	}
}

// While the plate's reason box is open, the dashboard gives esc to the box:
// it puts the well back instead of arming the dashboard's cancel.
func TestInADashboardEscInTheReasonBoxUndoesTheWell(t *testing.T) {
	data := strings.Replace(twoSections, `"choices": ["control", "treated"]`,
		`"options": [{"id": "control"}, {"id": "leave_out", "reason": {"key": "reason"}}]`, 1)
	d := &dashboard.Component{}
	if err := d.Init(json.RawMessage(data)); err != nil {
		t.Fatal(err)
	}
	d.HandleMsg(tea.KeyPressMsg{Code: '2', Text: "2"}) // S01 -> leave_out: the box opens
	if view := ansi.Strip(d.View(100, 40)); !strings.Contains(view, "S01 → leave_out · Reason: █") {
		t.Fatalf("reason box not open:\n%s", view)
	}
	if r, _ := d.HandleMsg(tea.KeyPressMsg{Code: tea.KeyEscape}); r != nil {
		t.Fatalf("esc in the box replied: %+v", r)
	}
	view := ansi.Strip(d.View(100, 40))
	if strings.Contains(view, "press esc again") || !strings.Contains(view, "S01 back to control.") {
		t.Fatalf("esc should undo the well, not arm the dashboard:\n%s", view)
	}
}
