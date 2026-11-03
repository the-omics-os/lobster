package choice

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	tea "charm.land/bubbletea/v2"

	"github.com/the-omics-os/lobster-tui/internal/biocomp"
)

var sources = []map[string]any{
	{"id": "SRC-A", "label": "Author matrix", "details": "Raw counts · Build-A"},
	{"id": "SRC-B", "label": "Repository matrix"},
	{"id": "SRC-C", "label": "Author summary", "unavailable": "Normalized; build unknown"},
}

var donors = []map[string]any{
	{"id": "S01", "label": "Donor A"},
	{"id": "S02", "label": "Donor B"},
	{"id": "S05", "label": "Donor A"},
}

func data(min, max int, options []map[string]any, extra map[string]any) json.RawMessage {
	d := map[string]any{"question": "Pick", "min": min, "max": max, "options": options}
	for k, v := range extra {
		d[k] = v
	}
	raw, _ := json.Marshal(d)
	return raw
}

func newChoice(t *testing.T, raw json.RawMessage) *Component {
	t.Helper()
	c := &Component{}
	if err := c.Init(raw); err != nil {
		t.Fatal(err)
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
		case "space":
			msg = tea.KeyPressMsg{Code: tea.KeySpace, Text: " "}
		case "enter":
			msg = tea.KeyPressMsg{Code: tea.KeyEnter}
		case "esc":
			msg = tea.KeyPressMsg{Code: tea.KeyEscape}
		default:
			msg = tea.KeyPressMsg{Code: rune(k[0]), Text: k}
		}
		result, _ = c.HandleMsg(msg)
	}
	return result
}

func selectedIDs(t *testing.T, r *biocomp.ComponentResult) []string {
	t.Helper()
	if r == nil || r.Action != "submit" {
		t.Fatalf("expected submit, got %+v", r)
	}
	var ids []string
	for _, s := range r.Data["selected"].([]map[string]any) {
		ids = append(ids, s["id"].(string))
	}
	return ids
}

func TestRegistered(t *testing.T) {
	if biocomp.Get("choice") == nil {
		t.Fatal(`"choice" is not registered`)
	}
}

func TestInitRejectsUnanswerableData(t *testing.T) {
	cases := map[string]json.RawMessage{
		"no options":      data(1, 1, nil, nil),
		"no question":     data(1, 1, sources, map[string]any{"question": ""}),
		"duplicate id":    data(1, 1, []map[string]any{{"id": "a"}, {"id": "a"}}, nil),
		"min above max":   data(2, 1, donors, nil),
		"min above avail": data(3, 3, sources, nil),
		"not json":        json.RawMessage(`{"options": "S01"}`),
	}
	for name, raw := range cases {
		if err := (&Component{}).Init(raw); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
}

func TestSinglePickReturnsTheID(t *testing.T) {
	c := newChoice(t, data(1, 1, sources, nil))
	if got := selectedIDs(t, press(c, "down", "enter")); !reflect.DeepEqual(got, []string{"SRC-B"}) {
		t.Fatalf("got %v", got)
	}
	c = newChoice(t, data(1, 1, sources, nil))
	if got := selectedIDs(t, press(c, "2")); !reflect.DeepEqual(got, []string{"SRC-B"}) {
		t.Fatalf("number key: got %v", got)
	}
}

func TestUnavailableOptionCannotBePicked(t *testing.T) {
	c := newChoice(t, data(1, 1, sources, nil))
	if r := press(c, "down", "down", "enter"); r != nil {
		t.Fatalf("picked an unavailable option: %+v", r)
	}
	if !strings.Contains(c.problem, "Normalized; build unknown") {
		t.Fatalf("reason not shown: %q", c.problem)
	}
	if r := press(c, "3"); r != nil {
		t.Fatalf("number key picked an unavailable option: %+v", r)
	}
	view := c.View(100, 20)
	if !strings.Contains(view, "unavailable: Normalized; build unknown") {
		t.Fatalf("view hides the reason:\n%s", view)
	}
}

func TestCursorStartsOnFirstAvailableOption(t *testing.T) {
	opts := []map[string]any{{"id": "x", "unavailable": "no"}, {"id": "y"}}
	c := newChoice(t, data(1, 1, opts, nil))
	if got := selectedIDs(t, press(c, "enter")); !reflect.DeepEqual(got, []string{"y"}) {
		t.Fatalf("got %v", got)
	}
}

func TestDefaultPreselects(t *testing.T) {
	c := newChoice(t, data(1, 1, sources, map[string]any{"default": []string{"SRC-B"}}))
	if got := selectedIDs(t, press(c, "enter")); !reflect.DeepEqual(got, []string{"SRC-B"}) {
		t.Fatalf("single default: got %v", got)
	}
	c = newChoice(t, data(1, 3, donors, map[string]any{"default": []string{"S01", "S05"}}))
	if got := selectedIDs(t, press(c, "enter")); !reflect.DeepEqual(got, []string{"S01", "S05"}) {
		t.Fatalf("multi default: got %v", got)
	}
	c = newChoice(t, data(1, 1, sources, map[string]any{"default": []string{"SRC-C"}}))
	if got := selectedIDs(t, press(c, "enter")); !reflect.DeepEqual(got, []string{"SRC-A"}) {
		t.Fatalf("unavailable default should be ignored: got %v", got)
	}
}

func TestMultiPickEnforcesLimitsAndKeepsOrder(t *testing.T) {
	c := newChoice(t, data(2, 2, donors, nil))
	if r := press(c, "down", "down", "space", "enter"); r != nil {
		t.Fatalf("submitted with too few picks: %+v", r)
	}
	press(c, "up", "space", "up", "space")
	if c.count() != 2 || !strings.Contains(c.problem, "at most 2") {
		t.Fatalf("max not enforced: count=%d problem=%q", c.count(), c.problem)
	}
	if got := selectedIDs(t, press(c, "enter")); !reflect.DeepEqual(got, []string{"S02", "S05"}) {
		t.Fatalf("got %v", got)
	}
}

func TestRepeatedLabelsShowTheirIDs(t *testing.T) {
	c := newChoice(t, data(1, 3, donors, nil))
	view := c.View(100, 20)
	if !strings.Contains(view, "S01") || !strings.Contains(view, "S05") {
		t.Fatalf("ids missing for repeated labels:\n%s", view)
	}
}

func TestEscCancels(t *testing.T) {
	c := newChoice(t, data(1, 1, sources, nil))
	if r := press(c, "esc"); r == nil || r.Action != "cancel" {
		t.Fatalf("got %+v", r)
	}
}
