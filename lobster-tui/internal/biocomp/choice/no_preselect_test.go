package choice

import (
	"strings"
	"testing"

	tea "charm.land/bubbletea/v2"
	"github.com/charmbracelet/x/ansi"
)

// Nothing pre-selected looks unselected: the cursor is the > pointer, and
// (●) marks only a real default.
func TestASingleChoiceShowsADotOnlyForItsDefault(t *testing.T) {
	c := &Component{}
	if err := c.Init([]byte(`{"question": "Source?", "options": [{"id": "A"}, {"id": "B"}], "min": 1, "max": 1}`)); err != nil {
		t.Fatal(err)
	}
	view := ansi.Strip(c.View(80, 20))
	if strings.Contains(view, "(●)") || !strings.Contains(view, "> ( ) A") {
		t.Fatalf("nothing is pre-selected, but:\n%s", view)
	}
	c.HandleMsg(tea.KeyPressMsg{Code: tea.KeyDown})
	if view := ansi.Strip(c.View(80, 20)); strings.Contains(view, "(●)") {
		t.Fatalf("moving the cursor drew a dot:\n%s", view)
	}

	d := &Component{}
	_ = d.Init([]byte(`{"question": "Source?", "options": [{"id": "A"}, {"id": "B"}], "min": 1, "max": 1, "default": ["B"]}`))
	if view := ansi.Strip(d.View(80, 20)); !strings.Contains(view, "(●) B") || strings.Contains(view, "(●) A") {
		t.Fatalf("the default should carry the dot:\n%s", view)
	}
}
