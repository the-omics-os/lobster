package sampleplate

import (
	"strings"
	"testing"

	tea "charm.land/bubbletea/v2"
)

// Look-alike samples for a synthetic example.
const lookAlike = `{
  "question": "Assign each sample to control or treated, or leave it out.",
  "title": "Sample conditions (synthetic)",
  "format": 96,
  "choices": ["control", "treated", "leave_out"],
  "wells": [
    {"row": 0, "col": 0, "sample_id": "X01", "label": "Line 1", "condition": "control", "batch": "B1", "warning": ""},
    {"row": 0, "col": 1, "sample_id": "X02", "label": "Line 2", "condition": "treated", "batch": "B1", "warning": "Low read depth"},
    {"row": 0, "col": 2, "sample_id": "X03", "label": "Line 3", "condition": "unknown", "batch": "B2", "warning": "Condition missing"}
  ]
}`

func press(s string) tea.KeyPressMsg {
	switch s {
	case "enter":
		return tea.KeyPressMsg{Code: tea.KeyEnter}
	case "esc":
		return tea.KeyPressMsg{Code: tea.KeyEscape}
	case "right":
		return tea.KeyPressMsg{Code: tea.KeyRight}
	}
	return tea.KeyPressMsg{Code: rune(s[0]), Text: s}
}

func newComp(t *testing.T) *SamplePlateComponent {
	t.Helper()
	c := &SamplePlateComponent{}
	if err := c.Init([]byte(lookAlike)); err != nil {
		t.Fatalf("Init: %v", err)
	}
	return c
}

func TestInitRejectsBadData(t *testing.T) {
	cases := map[string]string{
		"no wells":      `{"wells": []}`,
		"bad format":    `{"format": 100, "wells": [{"row":0,"col":0,"sample_id":"X01"}]}`,
		"off plate":     `{"wells": [{"row":8,"col":0,"sample_id":"X01"}]}`,
		"duplicate id":  `{"wells": [{"row":0,"col":0,"sample_id":"X01"},{"row":0,"col":1,"sample_id":"X01"}]}`,
		"shared well":   `{"wells": [{"row":0,"col":0,"sample_id":"X01"},{"row":0,"col":0,"sample_id":"X02"}]}`,
		"missing id":    `{"wells": [{"row":0,"col":0}]}`,
		"not an object": `"X01"`,
	}
	for name, data := range cases {
		if err := (&SamplePlateComponent{}).Init([]byte(data)); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
}

func TestSnakeCaseFieldsArePopulated(t *testing.T) {
	c := newComp(t)
	s := c.samples[2]
	if s.SampleID != "X03" || s.Label != "Line 3" || s.Batch != "B2" || s.Warning != "Condition missing" {
		t.Fatalf("fields not populated: %+v", s)
	}
	if c.assignments["X03"] != unknown {
		t.Fatalf("unknown condition should start unresolved, got %q", c.assignments["X03"])
	}
}

func TestEnterRefusedWhileUnresolved(t *testing.T) {
	c := newComp(t)
	if res, _ := c.HandleMsg(press("enter")); res != nil {
		t.Fatalf("enter should be refused, got %+v", res)
	}
	if !strings.Contains(c.message, "X03") {
		t.Fatalf("message should name the unresolved sample, got %q", c.message)
	}
}

func TestAssignAndSubmit(t *testing.T) {
	c := newComp(t)
	c.HandleMsg(press("right"))
	c.HandleMsg(press("right")) // cursor on X03 (A3)
	c.HandleMsg(press("2"))     // 2 = the second choice, "treated"
	if c.assignments["X03"] != "treated" {
		t.Fatalf("X03 = %q, want treated", c.assignments["X03"])
	}

	res, _ := c.HandleMsg(press("enter"))
	if res == nil || res.Action != "submit" {
		t.Fatalf("expected submit, got %+v", res)
	}
	got := res.Data["assignments"].(map[string]any)
	want := map[string]string{"X01": "control", "X02": "treated", "X03": "treated"}
	for id, v := range want {
		if got[id] != v {
			t.Errorf("%s = %v, want %s", id, got[id], v)
		}
	}
	changes := res.Data["changes"].([]any)
	if len(changes) != 1 || changes[0].(map[string]any)["sample_id"] != "X03" {
		t.Fatalf("changes should list only X03, got %v", changes)
	}
}

func TestEscCancels(t *testing.T) {
	c := newComp(t)
	res, _ := c.HandleMsg(press("esc"))
	if res == nil || res.Action != "cancel" {
		t.Fatalf("expected cancel, got %+v", res)
	}
}

func TestViewShowsFocusAndPlate(t *testing.T) {
	c := newComp(t)
	out := c.View(90, 18)
	for _, want := range []string{"X01", "Line 1", "unresolved 1", "96-Well Plate"} {
		if !strings.Contains(out, want) {
			t.Errorf("view missing %q", want)
		}
	}
}

func TestNumberKeysFollowTheChoices(t *testing.T) {
	c := newComp(t) // choices: control, treated, leave_out; cursor on X01
	for key, want := range map[string]string{"3": "leave_out", "1": "control", "0": unknown} {
		c.HandleMsg(press(key))
		if c.assignments["X01"] != want {
			t.Fatalf("key %s: X01 = %q, want %q", key, c.assignments["X01"], want)
		}
	}
	c.HandleMsg(press("1"))
	c.HandleMsg(press("4")) // no fourth choice: ignored
	if c.assignments["X01"] != "control" {
		t.Fatalf("key 4 changed X01 to %q", c.assignments["X01"])
	}
	if out := c.View(90, 18); !strings.Contains(out, "1 control · 2 treated · 3 leave_out · 0 unresolved") {
		t.Fatalf("legend missing:\n%s", out)
	}
}

func TestInitNeedsOneToNineChoices(t *testing.T) {
	for _, choices := range []string{`[]`, `["1","2","3","4","5","6","7","8","9","10"]`} {
		raw := strings.Replace(lookAlike, `["control", "treated", "leave_out"]`, choices, 1)
		if err := (&SamplePlateComponent{}).Init([]byte(raw)); err == nil {
			t.Errorf("choices %s: expected an error", choices)
		}
	}
}
