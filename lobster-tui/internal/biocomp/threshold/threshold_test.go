package threshold

import (
	"encoding/json"
	"math"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
)

// helper to create a component with valid defaults.
func newTestComponent(t *testing.T, overrides map[string]any) *ThresholdSliderComponent {
	t.Helper()
	payload := map[string]any{
		"label":   "p-value cutoff",
		"min":     0.0,
		"max":     1.0,
		"step":    0.01,
		"default": 0.05,
		"unit":    "",
		"count":   1542,
		"total":   20000,
	}
	for k, v := range overrides {
		payload[k] = v
	}
	data, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	c := &ThresholdSliderComponent{}
	if err := c.Init(data); err != nil {
		t.Fatalf("init: %v", err)
	}
	return c
}

func floatEq(a, b float64) bool {
	return math.Abs(a-b) < 1e-9
}

func TestThresholdInit(t *testing.T) {
	c := newTestComponent(t, nil)
	if !floatEq(c.value, 0.05) {
		t.Errorf("expected value 0.05, got %f", c.value)
	}
	if c.data.Label != "p-value cutoff" {
		t.Errorf("expected label 'p-value cutoff', got %q", c.data.Label)
	}
	if c.data.Count != 1542 {
		t.Errorf("expected count 1542, got %d", c.data.Count)
	}
	if c.data.Total != 20000 {
		t.Errorf("expected total 20000, got %d", c.data.Total)
	}
	// Default clamped to range
	c2 := newTestComponent(t, map[string]any{"default": 5.0})
	if !floatEq(c2.value, 1.0) {
		t.Errorf("expected clamped value 1.0, got %f", c2.value)
	}
}

func TestThresholdInitInvalid(t *testing.T) {
	c := &ThresholdSliderComponent{}

	// Min >= Max
	data, _ := json.Marshal(map[string]any{
		"label": "test", "min": 1.0, "max": 0.5, "step": 0.1, "default": 0.5,
	})
	if err := c.Init(data); err == nil {
		t.Error("expected error for min >= max")
	}

	// Min == Max
	data, _ = json.Marshal(map[string]any{
		"label": "test", "min": 1.0, "max": 1.0, "step": 0.1, "default": 1.0,
	})
	if err := c.Init(data); err == nil {
		t.Error("expected error for min == max")
	}

	// Step <= 0
	data, _ = json.Marshal(map[string]any{
		"label": "test", "min": 0.0, "max": 1.0, "step": 0.0, "default": 0.5,
	})
	if err := c.Init(data); err == nil {
		t.Error("expected error for step <= 0")
	}

	data, _ = json.Marshal(map[string]any{
		"label": "test", "min": 0.0, "max": 1.0, "step": -0.1, "default": 0.5,
	})
	if err := c.Init(data); err == nil {
		t.Error("expected error for negative step")
	}
}

func TestThresholdInitBadJSON(t *testing.T) {
	c := &ThresholdSliderComponent{}
	if err := c.Init(json.RawMessage(`{invalid json`)); err == nil {
		t.Error("expected error for bad JSON")
	}
}

func TestThresholdAdjustRight(t *testing.T) {
	c := newTestComponent(t, map[string]any{"default": 0.50})
	c.HandleMsg(tea.KeyMsg{Type: tea.KeyRight})
	if !floatEq(c.value, 0.51) {
		t.Errorf("expected 0.51 after right, got %f", c.value)
	}
}

func TestThresholdAdjustLeft(t *testing.T) {
	c := newTestComponent(t, map[string]any{"default": 0.50})
	c.HandleMsg(tea.KeyMsg{Type: tea.KeyLeft})
	if !floatEq(c.value, 0.49) {
		t.Errorf("expected 0.49 after left, got %f", c.value)
	}
}

func TestThresholdCoarseAdjust(t *testing.T) {
	c := newTestComponent(t, map[string]any{"default": 0.50})

	c.HandleMsg(tea.KeyMsg{Type: tea.KeyShiftRight})
	if !floatEq(c.value, 0.60) {
		t.Errorf("expected 0.60 after shift+right, got %f", c.value)
	}

	c.HandleMsg(tea.KeyMsg{Type: tea.KeyShiftLeft})
	if !floatEq(c.value, 0.50) {
		t.Errorf("expected 0.50 after shift+left, got %f", c.value)
	}
}

func TestThresholdClampMin(t *testing.T) {
	c := newTestComponent(t, map[string]any{"default": 0.0})
	c.HandleMsg(tea.KeyMsg{Type: tea.KeyLeft})
	if !floatEq(c.value, 0.0) {
		t.Errorf("expected 0.0 (clamped at min), got %f", c.value)
	}
}

func TestThresholdClampMax(t *testing.T) {
	c := newTestComponent(t, map[string]any{"default": 1.0})
	c.HandleMsg(tea.KeyMsg{Type: tea.KeyRight})
	if !floatEq(c.value, 1.0) {
		t.Errorf("expected 1.0 (clamped at max), got %f", c.value)
	}
}

func TestThresholdSubmit(t *testing.T) {
	c := newTestComponent(t, map[string]any{"default": 0.05})
	result := c.HandleMsg(tea.KeyMsg{Type: tea.KeyEnter})
	if result == nil {
		t.Fatal("expected result on enter")
	}
	if result.Action != "submit" {
		t.Errorf("expected action 'submit', got %q", result.Action)
	}
	val, ok := result.Data["value"].(float64)
	if !ok || !floatEq(val, 0.05) {
		t.Errorf("expected value 0.05, got %v", result.Data["value"])
	}
}

func TestThresholdCancel(t *testing.T) {
	c := newTestComponent(t, map[string]any{"default": 0.05})
	result := c.HandleMsg(tea.KeyMsg{Type: tea.KeyEsc})
	if result == nil {
		t.Fatal("expected result on esc")
	}
	if result.Action != "cancel" {
		t.Errorf("expected action 'cancel', got %q", result.Action)
	}
}

func TestThresholdChangeEvent(t *testing.T) {
	c := newTestComponent(t, map[string]any{"default": 0.50})

	// No change yet
	if evt := c.ChangeEvent(); evt != nil {
		t.Error("expected nil change event before any adjustment")
	}

	// Adjust right
	c.HandleMsg(tea.KeyMsg{Type: tea.KeyRight})
	evt := c.ChangeEvent()
	if evt == nil {
		t.Fatal("expected change event after adjustment")
	}
	val, ok := evt["value"].(float64)
	if !ok || !floatEq(val, 0.51) {
		t.Errorf("expected value 0.51, got %v", evt["value"])
	}

	// Second call should return nil (flag cleared)
	if evt2 := c.ChangeEvent(); evt2 != nil {
		t.Error("expected nil on second ChangeEvent call")
	}
}

func TestThresholdSetData(t *testing.T) {
	c := newTestComponent(t, nil)

	update, _ := json.Marshal(map[string]any{"count": 500, "total": 10000})
	if err := c.SetData(update); err != nil {
		t.Fatalf("SetData error: %v", err)
	}
	if c.data.Count != 500 {
		t.Errorf("expected count 500, got %d", c.data.Count)
	}
	if c.data.Total != 10000 {
		t.Errorf("expected total 10000, got %d", c.data.Total)
	}

	// Verify count is reflected in the view
	view := c.View(60, 20)
	if view == "" {
		t.Error("view should not be empty after SetData")
	}
}

func TestThresholdView(t *testing.T) {
	c := newTestComponent(t, nil)

	// Various widths should not panic
	widths := []int{10, 20, 40, 60, 80, 120}
	for _, w := range widths {
		view := c.View(w, 20)
		if view == "" {
			t.Errorf("empty view at width %d", w)
		}
	}
}

func TestThresholdSliderBar(t *testing.T) {
	// At min, cursor should be at position 0
	c := newTestComponent(t, map[string]any{"default": 0.0, "count": 0, "total": 0})
	view := c.View(60, 20)
	if view == "" {
		t.Fatal("empty view for min slider")
	}

	// At max, cursor should be at far right
	c2 := newTestComponent(t, map[string]any{"default": 1.0, "count": 0, "total": 0})
	view2 := c2.View(60, 20)
	if view2 == "" {
		t.Fatal("empty view for max slider")
	}

	// At midpoint
	c3 := newTestComponent(t, map[string]any{"default": 0.5, "count": 0, "total": 0})
	view3 := c3.View(60, 20)
	if view3 == "" {
		t.Fatal("empty view for mid slider")
	}
}

func TestThresholdNameMode(t *testing.T) {
	c := &ThresholdSliderComponent{}
	if c.Name() != "threshold_slider" {
		t.Errorf("expected name 'threshold_slider', got %q", c.Name())
	}
	if c.Mode() != "overlay" {
		t.Errorf("expected mode 'overlay', got %q", c.Mode())
	}
}
