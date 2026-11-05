package choicelist

import (
	"fmt"
	"strings"
	"testing"

	tea "charm.land/bubbletea/v2"
	"github.com/charmbracelet/x/ansi"
)

// Rows scrolled out of view are counted above and below the window, so
// samples above it are never silently hidden.
func TestHiddenRowsAreCountedAboveAndBelow(t *testing.T) {
	var items []string
	for i := 1; i <= 30; i++ {
		items = append(items, fmt.Sprintf(`{"id": "S%02d"}`, i))
	}
	raw := `{"question": "Keep?", "items": [` + strings.Join(items, ",") +
		`], "options": [{"id": "retain"}, {"id": "drop"}]}`
	c := &Component{}
	if err := c.Init([]byte(raw)); err != nil {
		t.Fatal(err)
	}
	view := ansi.Strip(c.View(80, 30))
	if strings.Contains(view, "↑") || !strings.Contains(view, "more") || !strings.Contains(view, "↓") {
		t.Fatalf("at the top only ↓ should show:\n%s", view)
	}
	for i := 0; i < 20; i++ {
		c.HandleMsg(tea.KeyPressMsg{Code: tea.KeyDown})
	}
	view = ansi.Strip(c.View(80, 30))
	if !strings.Contains(view, "↑ ") || !strings.Contains(view, "↓ ") {
		t.Fatalf("in the middle both should show:\n%s", view)
	}
	if lines := strings.Count(view, "\n") + 1; lines > 30 {
		t.Fatalf("view is %d lines, given 30", lines)
	}
	short := &Component{}
	_ = short.Init([]byte(`{"question": "Keep?", "items": [{"id": "S01"}], "options": [{"id": "retain"}]}`))
	if v := ansi.Strip(short.View(80, 30)); strings.Contains(v, "more") {
		t.Fatalf("nothing hidden, nothing said:\n%s", v)
	}
}
