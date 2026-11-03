package dashboard

import (
	"encoding/json"
	"os"
	"reflect"
	"strings"
	"testing"

	tea "charm.land/bubbletea/v2"
)

// TestPythonFixtureRoundTrip loads the dashboard data written by the Python
// adapter (tests/unit/cli/test_go_tui_components.py), plays a scientist's
// keys, and checks the reply against testdata/reply_from_go.json, which the
// Python test then validates against the original request. Regenerate with
// LOBSTER_UPDATE_DASHBOARD_FIXTURE=1 (see the Python test for the order).
func TestPythonFixtureRoundTrip(t *testing.T) {
	raw, err := os.ReadFile("testdata/from_python.json")
	if err != nil {
		t.Fatal(err)
	}
	c := &Component{}
	if err := c.Init(raw); err != nil {
		t.Fatalf("Go can't load what Python sends: %v", err)
	}

	press(t, c, "enter") // source: SRC-A
	press(t, c, "enter") // groups: S08 is unresolved, so the plate refuses
	if c.sections[1].status != todo {
		t.Fatal("the plate accepted an unresolved sample")
	}
	if view := c.View(120, 50); !strings.Contains(view, "S08") {
		t.Fatalf("the plate doesn't name the unresolved sample:\n%s", view)
	}
	for i := 0; i < 7; i++ { // the cursor starts on S01 (A1); S08 is A8
		c.HandleMsg(tea.KeyPressMsg{Code: tea.KeyRight})
	}
	press(t, c, "2", "enter") // S08 -> the 2nd choice, "treated"
	press(t, c, "enter")      // direction: treated compared with control
	if c.sections[c.active].Component != "choice_list" {
		t.Fatalf("QC should be one choice_list section, got %q", c.sections[c.active].Component)
	}
	if view := c.View(120, 50); !strings.Contains(view, "Low library size") {
		t.Fatalf("the QC list doesn't show the flagged sample's warning:\n%s", view)
	}
	press(t, c, "1")    // QC S02: retain
	press(t, c, "down") // QC S06
	press(t, c, "2")    // exclude: opens the reason box
	c.HandleMsg(tea.PasteMsg{Content: "high mito fraction"})
	press(t, c, "enter") // saves the reason
	press(t, c, "enter") // records the QC section

	// ctrl+s opens the review screen.
	if r := press(t, c, "ctrl+s"); r != nil || !c.reviewing {
		t.Fatalf("ctrl+s should open the review screen, got %+v (message %q)", r, c.message)
	}
	r := press(t, c, "enter") // confirms what the review shows
	if r == nil || r.Action != "submit" {
		t.Fatalf("expected submit, got %+v (message %q)", r, c.message)
	}

	compareFixture(t, r.Data, "testdata/reply_from_go.json")
}

// TestPythonReaskFixtureRoundTrip loads a dashboard the Python adapter re-asks
// after refusing an answer (testdata/reask_from_python.json): sections with
// problems start unanswered and pre-filled, the others start answered. The
// scientist fixes the groups, re-checks what became stale and submits; the
// reply (testdata/reask_reply_from_go.json) is checked by the Python test.
func TestPythonReaskFixtureRoundTrip(t *testing.T) {
	raw, err := os.ReadFile("testdata/reask_from_python.json")
	if err != nil {
		t.Fatal(err)
	}
	c := &Component{}
	if err := c.Init(raw); err != nil {
		t.Fatalf("Go can't load the re-ask Python sends: %v", err)
	}
	if got := statuses(c); got != "done,todo,done,todo" {
		t.Fatalf("sections with problems should start unanswered, the rest answered: %s", got)
	}
	if view := c.View(120, 50); !strings.Contains(view, "has no items left") {
		t.Fatalf("the re-ask doesn't show the problem:\n%s", view)
	}
	if c.sections[c.active].Key != "groups" {
		t.Fatalf("the first section to fix should be active, got %q", c.sections[c.active].Key)
	}

	press(t, c, "1", "enter") // S01 (A1, under the cursor) back to control
	if got := statuses(c); got != "done,done,stale,todo" || c.sections[c.active].Key != "direction" {
		t.Fatalf("changing the groups should make the direction stale: %s, active %q",
			got, c.sections[c.active].Key)
	}
	press(t, c, "enter") // re-checks the direction
	press(t, c, "enter") // QC: pre-filled with the previous answer, kept
	if got := statuses(c); got != "done,done,done,done" {
		t.Fatalf("every section should be answered now: %s (message %q)", got, c.message)
	}
	if r := press(t, c, "ctrl+s"); r != nil || !c.reviewing {
		t.Fatalf("ctrl+s should open the review screen, got %+v (message %q)", r, c.message)
	}
	r := press(t, c, "enter")
	if r == nil || r.Action != "submit" {
		t.Fatalf("expected submit, got %+v (message %q)", r, c.message)
	}
	compareFixture(t, r.Data, "testdata/reask_reply_from_go.json")
}

// compareFixture writes the reply with LOBSTER_UPDATE_DASHBOARD_FIXTURE=1 and
// checks it against the recorded file.
func compareFixture(t *testing.T, data map[string]any, path string) {
	t.Helper()
	produced, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if os.Getenv("LOBSTER_UPDATE_DASHBOARD_FIXTURE") == "1" {
		if err := os.WriteFile(path, append(produced, '\n'), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	recorded, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var got, want any
	_ = json.Unmarshal(produced, &got)
	if err := json.Unmarshal(recorded, &want); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("reply changed; regenerate the fixture if intended.\ngot:  %s", produced)
	}
}
