package threshold

import (
	"testing"

	tea "charm.land/bubbletea/v2"
)

// A 0.50-0.99 slider with a 0.01 step lands on 0.75 exactly, and steps
// never drift.
func TestValuesLandOnTheStepGrid(t *testing.T) {
	c := &ThresholdSliderComponent{}
	if err := c.Init([]byte(`{"label": "Cutoff", "min": 0.5, "max": 0.99, "step": 0.01, "default": 0.5}`)); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 25; i++ {
		c.HandleMsg(tea.KeyPressMsg{Code: tea.KeyRight})
	}
	if c.value != 0.75 {
		t.Fatalf("value = %v, want 0.75", c.value)
	}
	r, _ := c.HandleMsg(tea.KeyPressMsg{Code: tea.KeyEnter})
	if r == nil || r.Data["value"] != 0.75 {
		t.Fatalf("reply = %+v", r)
	}
	off := &ThresholdSliderComponent{}
	_ = off.Init([]byte(`{"min": 0.5, "max": 0.99, "step": 0.01, "default": 0.7531}`))
	if off.value != 0.75 {
		t.Fatalf("a default off the grid should snap to 0.75, got %v", off.value)
	}
}
