package ontologyterm

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	tea "charm.land/bubbletea/v2"
	"github.com/charmbracelet/x/ansi"

	"github.com/the-omics-os/lobster-tui/internal/biocomp"
)

// A made-up ontology, shaped as Python sends it: each term names its parent.
var terms = []map[string]any{
	{"id": "T:1", "label": "anatomical entity"},
	{"id": "T:2", "label": "organ", "parent": "T:1"},
	{"id": "T:3", "label": "liver", "parent": "T:2", "details": "suggested · An organ."},
	{"id": "T:4", "label": "lung", "parent": "T:2", "details": "suggested"},
	{"id": "T:5", "label": "liver lobule", "parent": "T:3"},
}

func newTerm(t *testing.T, extra map[string]any) *Component {
	t.Helper()
	d := map[string]any{
		"question": "S01: Which tissue?", "ontology": "toy",
		"options": terms, "suggested": []string{"T:3", "T:4"},
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
		case "right":
			msg = tea.KeyPressMsg{Code: tea.KeyRight}
		case "space":
			msg = tea.KeyPressMsg{Code: tea.KeySpace, Text: " "}
		case "enter":
			msg = tea.KeyPressMsg{Code: tea.KeyEnter}
		case "esc":
			msg = tea.KeyPressMsg{Code: tea.KeyEscape}
		case "tab":
			msg = tea.KeyPressMsg{Code: tea.KeyTab}
		case "backspace":
			msg = tea.KeyPressMsg{Code: tea.KeyBackspace}
		case "ctrl+d":
			msg = tea.KeyPressMsg{Code: 'd', Mod: tea.ModCtrl}
		default:
			msg = tea.KeyPressMsg{Code: rune(k[0]), Text: k}
		}
		result, _ = c.HandleMsg(msg)
	}
	return result
}

func picked(t *testing.T, r *biocomp.ComponentResult) string {
	t.Helper()
	if r == nil || r.Action != "submit" {
		t.Fatalf("expected a submit, got %+v", r)
	}
	sel := r.Data["selected"].([]map[string]any)
	return sel[0]["id"].(string)
}

func TestOpensOnTheSuggestionsAndPicksOne(t *testing.T) {
	c := newTerm(t, nil)
	if c.browsing {
		t.Fatal("should open on the suggested terms")
	}
	if got := picked(t, press(c, "down", "enter")); got != "T:4" {
		t.Fatalf("picked %s, want T:4 (lung)", got)
	}
}

func TestDefaultIsMarkedAndUnderTheCursor(t *testing.T) {
	c := newTerm(t, map[string]any{"default": "T:4"})
	view := ansi.Strip(c.View(80, 30))
	if !strings.Contains(view, "(●) lung") {
		t.Fatalf("default not marked:\n%s", view)
	}
	if got := picked(t, press(c, "enter")); got != "T:4" {
		t.Fatalf("picked %s, want the default T:4", got)
	}
}

func TestBrowsePicksABranchWithSpace(t *testing.T) {
	c := newTerm(t, nil)
	press(c, "down", "down", "enter") // "Browse nearby toy terms"
	if !c.browsing {
		t.Fatal("enter on Browse should open the browser")
	}
	view := ansi.Strip(c.View(90, 30))
	if !strings.Contains(view, "anatomical entity") {
		t.Fatalf("browser not drawn:\n%s", view)
	}
	// It opens on the last suggestion highlighted (lung); up is liver
	// lobule, then liver, then organ (every branch is open).
	if node := c.browser.Selected(); node == nil || node.ID != "T:4" {
		t.Fatalf("browser opened on %+v, want T:4 (lung)", node)
	}
	press(c, "up", "up", "up")
	if got := picked(t, press(c, "space")); got != "T:2" {
		t.Fatalf("space picked %s, want the branch T:2 (organ)", got)
	}
}

func TestEscInTheBrowserGoesBackToTheList(t *testing.T) {
	c := newTerm(t, nil)
	press(c, "down", "down", "enter")
	if r := press(c, "esc"); r != nil {
		t.Fatalf("esc in the browser should not end the question: %+v", r)
	}
	if c.browsing {
		t.Fatal("esc should go back to the list")
	}
}

func TestSearchTypesSpaceInsteadOfPicking(t *testing.T) {
	c := newTerm(t, nil)
	press(c, "down", "down", "enter", "tab")
	if r := press(c, "l", "i", "v", "e", "r", "space"); r != nil {
		t.Fatalf("space in the search box should type, not pick: %+v", r)
	}
	press(c, "backspace")
	if got := picked(t, press(c, "enter")); got != "T:3" && got != "T:5" {
		t.Fatalf("enter in search should pick a liver match, got %s", got)
	}
}

func TestEscTwiceCancelsAndCtrlDDefers(t *testing.T) {
	c := newTerm(t, nil)
	if r := press(c, "esc"); r != nil {
		t.Fatal("one esc should only warn")
	}
	if r := press(c, "esc"); r == nil || r.Action != "cancel" {
		t.Fatalf("second esc should cancel, got %+v", r)
	}
	c = newTerm(t, nil)
	if r := press(c, "ctrl+d"); r == nil || r.Action != "defer" {
		t.Fatalf("ctrl+d should defer, got %+v", r)
	}
}

func TestSuggestedTermsAreStarredInTheTree(t *testing.T) {
	c := newTerm(t, nil)
	var names []string
	for _, root := range c.tree() {
		names = append(names, root.Name)
		for _, child := range root.Children {
			names = append(names, child.Name)
			for _, grand := range child.Children {
				names = append(names, grand.Name)
			}
		}
	}
	want := []string{"anatomical entity", "organ", "★ liver", "★ lung"}
	if !reflect.DeepEqual(names, want) {
		t.Fatalf("tree names %v, want %v", names, want)
	}
}

func TestRefusesBadData(t *testing.T) {
	for name, raw := range map[string]string{
		"no question": `{"options":[{"id":"T:1"}]}`,
		"no terms":    `{"question":"q","options":[]}`,
		"duplicate":   `{"question":"q","options":[{"id":"T:1"},{"id":"T:1"}]}`,
	} {
		if err := (&Component{}).Init(json.RawMessage(raw)); err == nil {
			t.Errorf("%s: Init should fail", name)
		}
	}
}
