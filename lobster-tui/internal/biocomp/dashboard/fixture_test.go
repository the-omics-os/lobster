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
	press(t, c, "enter")      // mito: default 0.2
	press(t, c, "o", "k", "enter")
	r := press(t, c, "ctrl+s")
	if r == nil || r.Action != "submit" {
		t.Fatalf("expected submit, got %+v (message %q)", r, c.message)
	}

	produced, err := json.MarshalIndent(r.Data, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if os.Getenv("LOBSTER_UPDATE_DASHBOARD_FIXTURE") == "1" {
		if err := os.WriteFile("testdata/reply_from_go.json", append(produced, '\n'), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	recorded, err := os.ReadFile("testdata/reply_from_go.json")
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
