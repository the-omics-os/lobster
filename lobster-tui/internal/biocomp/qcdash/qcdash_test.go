package qcdash

import (
	"encoding/json"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
)

// sampleMetrics returns a 3-metric payload for testing.
func sampleMetrics() QCDashboardData {
	return QCDashboardData{
		Title: "QC Summary",
		Metrics: []QCMetric{
			{Name: "Reads", Value: 82, Min: 0, Max: 100, Unit: "%", Status: "pass"},
			{Name: "Genes", Value: 65, Min: 0, Max: 100, Unit: "%", Status: "warn"},
			{Name: "Mito %", Value: 3.2, Min: 0, Max: 20, Unit: "%", Status: "pass"},
		},
	}
}

func mustMarshal(t *testing.T, v any) json.RawMessage {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal failed: %v", err)
	}
	return b
}

func TestQCDashInit(t *testing.T) {
	c := &QCDashboardComponent{}
	data := mustMarshal(t, sampleMetrics())
	if err := c.Init(data); err != nil {
		t.Fatalf("Init with valid data should succeed, got: %v", err)
	}
	if len(c.data.Metrics) != 3 {
		t.Fatalf("expected 3 metrics, got %d", len(c.data.Metrics))
	}
	if c.data.Title != "QC Summary" {
		t.Fatalf("expected title 'QC Summary', got %q", c.data.Title)
	}
}

func TestQCDashInitEmpty(t *testing.T) {
	c := &QCDashboardComponent{}
	data := mustMarshal(t, QCDashboardData{Metrics: []QCMetric{}})
	err := c.Init(data)
	if err == nil {
		t.Fatal("Init with empty metrics should return error")
	}
	if !strings.Contains(err.Error(), "at least 1 metric") {
		t.Fatalf("unexpected error message: %v", err)
	}
}

func TestQCDashInitInvalid(t *testing.T) {
	c := &QCDashboardComponent{}
	err := c.Init(json.RawMessage(`{not valid json`))
	if err == nil {
		t.Fatal("Init with invalid JSON should return error")
	}
}

func TestQCDashHandleMsgNoop(t *testing.T) {
	c := &QCDashboardComponent{}
	_ = c.Init(mustMarshal(t, sampleMetrics()))

	// Key messages should always return nil (non-interactive).
	result := c.HandleMsg(tea.KeyMsg{Type: tea.KeyEnter})
	if result != nil {
		t.Fatal("HandleMsg should always return nil for inline component")
	}
	result = c.HandleMsg(tea.KeyMsg{Type: tea.KeyEsc})
	if result != nil {
		t.Fatal("HandleMsg should return nil for Esc too")
	}
	result = c.HandleMsg(tea.KeyMsg{Type: tea.KeyLeft})
	if result != nil {
		t.Fatal("HandleMsg should return nil for arrow keys")
	}
}

func TestQCDashViewRendering(t *testing.T) {
	c := &QCDashboardComponent{}
	_ = c.Init(mustMarshal(t, sampleMetrics()))

	widths := []int{40, 60, 80, 120}
	for _, w := range widths {
		// Should not panic at any width.
		v := c.View(w, 20)
		if v == "" {
			t.Fatalf("View at width %d returned empty string", w)
		}
	}
}

func TestQCDashViewContainsMetricNames(t *testing.T) {
	c := &QCDashboardComponent{}
	_ = c.Init(mustMarshal(t, sampleMetrics()))

	v := c.View(80, 20)
	for _, name := range []string{"Reads", "Genes", "Mito %"} {
		if !strings.Contains(v, name) {
			t.Fatalf("View should contain metric name %q, got:\n%s", name, v)
		}
	}
}

func TestQCDashViewStatusColors(t *testing.T) {
	data := QCDashboardData{
		Title: "Status Test",
		Metrics: []QCMetric{
			{Name: "PassMetric", Value: 90, Min: 0, Max: 100, Status: "pass"},
			{Name: "WarnMetric", Value: 50, Min: 0, Max: 100, Status: "warn"},
			{Name: "FailMetric", Value: 10, Min: 0, Max: 100, Status: "fail"},
		},
	}

	c := &QCDashboardComponent{}
	_ = c.Init(mustMarshal(t, data))

	v := c.View(80, 20)

	// Status text should appear in uppercase.
	if !strings.Contains(v, "PASS") {
		t.Fatal("View should contain 'PASS' status text")
	}
	if !strings.Contains(v, "WARN") {
		t.Fatal("View should contain 'WARN' status text")
	}
	if !strings.Contains(v, "FAIL") {
		t.Fatal("View should contain 'FAIL' status text")
	}
}

func TestQCDashSetData(t *testing.T) {
	c := &QCDashboardComponent{}
	_ = c.Init(mustMarshal(t, sampleMetrics()))

	// Update with new metrics.
	updated := QCDashboardData{
		Title: "Updated QC",
		Metrics: []QCMetric{
			{Name: "Doublets", Value: 1.1, Min: 0, Max: 10, Unit: "%", Status: "pass"},
		},
	}
	err := c.SetData(mustMarshal(t, updated))
	if err != nil {
		t.Fatalf("SetData should succeed, got: %v", err)
	}
	if len(c.data.Metrics) != 1 {
		t.Fatalf("expected 1 metric after SetData, got %d", len(c.data.Metrics))
	}
	if c.data.Metrics[0].Name != "Doublets" {
		t.Fatalf("expected metric name 'Doublets', got %q", c.data.Metrics[0].Name)
	}

	// Verify View reflects the update.
	v := c.View(80, 20)
	if !strings.Contains(v, "Doublets") {
		t.Fatal("View should contain updated metric name 'Doublets'")
	}
	if !strings.Contains(v, "Updated QC") {
		t.Fatal("View should contain updated title 'Updated QC'")
	}
}

func TestQCDashNameMode(t *testing.T) {
	c := &QCDashboardComponent{}
	_ = c.Init(mustMarshal(t, sampleMetrics()))

	if c.Name() != "qc_dashboard" {
		t.Fatalf("expected Name() = 'qc_dashboard', got %q", c.Name())
	}
	if c.Mode() != "inline" {
		t.Fatalf("expected Mode() = 'inline', got %q", c.Mode())
	}
}

func TestQCDashKeyBindings(t *testing.T) {
	c := &QCDashboardComponent{}
	_ = c.Init(mustMarshal(t, sampleMetrics()))

	bindings := c.KeyBindings()
	if bindings != nil && len(bindings) != 0 {
		t.Fatalf("expected empty KeyBindings for inline component, got %d", len(bindings))
	}
}
