package celltype

import (
	"encoding/json"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
)

func makeClusters(clusters []ClusterInfo) json.RawMessage {
	d, _ := json.Marshal(CellTypeSelectorData{Clusters: clusters})
	return d
}

func sampleClusters() []ClusterInfo {
	return []ClusterInfo{
		{ID: 0, Size: 1204, Markers: []string{"CD3D", "CD8A", "GZMB", "PRF1"}, Label: ""},
		{ID: 1, Size: 892, Markers: []string{"CD14", "LYZ", "S100A8"}, Label: "Monocytes"},
		{ID: 2, Size: 567, Markers: []string{"MS4A1", "CD79A", "CD19"}, Label: ""},
	}
}

func initComponent(t *testing.T, clusters []ClusterInfo) *CellTypeSelectorComponent {
	t.Helper()
	c := &CellTypeSelectorComponent{}
	if err := c.Init(makeClusters(clusters)); err != nil {
		t.Fatalf("unexpected init error: %v", err)
	}
	return c
}

func TestCellTypeInit(t *testing.T) {
	t.Parallel()
	c := initComponent(t, sampleClusters())

	if len(c.clusters) != 3 {
		t.Fatalf("expected 3 clusters, got %d", len(c.clusters))
	}
	if len(c.inputs) != 3 {
		t.Fatalf("expected 3 inputs, got %d", len(c.inputs))
	}
	if c.cursor != 0 {
		t.Fatalf("expected cursor at 0, got %d", c.cursor)
	}
	if c.editingIndex != -1 {
		t.Fatalf("expected editingIndex at -1, got %d", c.editingIndex)
	}
}

func TestCellTypeInitEmpty(t *testing.T) {
	t.Parallel()
	c := &CellTypeSelectorComponent{}
	err := c.Init(makeClusters([]ClusterInfo{}))
	if err == nil {
		t.Fatal("expected error for empty clusters")
	}
}

func TestCellTypeInitInvalid(t *testing.T) {
	t.Parallel()
	c := &CellTypeSelectorComponent{}
	err := c.Init(json.RawMessage(`{bad json`))
	if err == nil {
		t.Fatal("expected error for invalid JSON")
	}
}

func TestCellTypeNavigation(t *testing.T) {
	t.Parallel()
	c := initComponent(t, sampleClusters())

	// Down twice.
	result := c.HandleMsg(tea.KeyMsg{Type: tea.KeyDown})
	if result != nil {
		t.Fatal("expected nil result during navigation")
	}
	result = c.HandleMsg(tea.KeyMsg{Type: tea.KeyDown})
	if result != nil {
		t.Fatal("expected nil result during navigation")
	}
	if c.cursor != 2 {
		t.Fatalf("expected cursor at 2, got %d", c.cursor)
	}

	// Up once.
	result = c.HandleMsg(tea.KeyMsg{Type: tea.KeyUp})
	if result != nil {
		t.Fatal("expected nil result during navigation")
	}
	if c.cursor != 1 {
		t.Fatalf("expected cursor at 1, got %d", c.cursor)
	}

	// j/k navigation.
	c.HandleMsg(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'j'}})
	if c.cursor != 2 {
		t.Fatalf("expected cursor at 2 after j, got %d", c.cursor)
	}
	c.HandleMsg(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'k'}})
	if c.cursor != 1 {
		t.Fatalf("expected cursor at 1 after k, got %d", c.cursor)
	}
}

func TestCellTypeEditMode(t *testing.T) {
	t.Parallel()
	c := initComponent(t, sampleClusters())

	// Enter edit mode on cluster 0.
	result := c.HandleMsg(tea.KeyMsg{Type: tea.KeyEnter})
	if result != nil {
		t.Fatal("expected nil result when entering edit mode")
	}
	if c.editingIndex != 0 {
		t.Fatalf("expected editingIndex 0, got %d", c.editingIndex)
	}

	// Type "T cells".
	for _, r := range "T cells" {
		c.HandleMsg(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{r}})
	}
	if c.inputs[0].Value() != "T cells" {
		t.Fatalf("expected input value 'T cells', got %q", c.inputs[0].Value())
	}

	// Esc exits edit mode.
	result = c.HandleMsg(tea.KeyMsg{Type: tea.KeyEsc})
	if result != nil {
		t.Fatal("expected nil result when exiting edit mode via Esc")
	}
	if c.editingIndex != -1 {
		t.Fatalf("expected editingIndex -1 after Esc, got %d", c.editingIndex)
	}
	// Text should be preserved.
	if c.inputs[0].Value() != "T cells" {
		t.Fatalf("expected preserved value 'T cells', got %q", c.inputs[0].Value())
	}
}

func TestCellTypeTabAcceptNext(t *testing.T) {
	t.Parallel()
	c := initComponent(t, sampleClusters())

	// Enter edit mode on cluster 0.
	c.HandleMsg(tea.KeyMsg{Type: tea.KeyEnter})
	if c.editingIndex != 0 {
		t.Fatalf("expected editingIndex 0, got %d", c.editingIndex)
	}

	// Type a label.
	for _, r := range "NK cells" {
		c.HandleMsg(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{r}})
	}

	// Tab moves to next cluster and enters edit mode there.
	result := c.HandleMsg(tea.KeyMsg{Type: tea.KeyTab})
	if result != nil {
		t.Fatal("expected nil result on Tab")
	}
	if c.cursor != 1 {
		t.Fatalf("expected cursor at 1 after Tab, got %d", c.cursor)
	}
	if c.editingIndex != 1 {
		t.Fatalf("expected editingIndex 1 after Tab, got %d", c.editingIndex)
	}
	// Previous input should still have the value.
	if c.inputs[0].Value() != "NK cells" {
		t.Fatalf("expected preserved value 'NK cells', got %q", c.inputs[0].Value())
	}
}

func TestCellTypeSubmit(t *testing.T) {
	t.Parallel()
	c := initComponent(t, sampleClusters())

	// Type labels in cluster 0 and 2.
	c.HandleMsg(tea.KeyMsg{Type: tea.KeyEnter}) // edit cluster 0
	for _, r := range "CD8+ T cells" {
		c.HandleMsg(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{r}})
	}
	c.HandleMsg(tea.KeyMsg{Type: tea.KeyEsc}) // exit edit

	// Cluster 1 already has prefilled "Monocytes".

	// Submit via Ctrl+S.
	result := c.HandleMsg(tea.KeyMsg{Type: tea.KeyCtrlS})
	if result == nil {
		t.Fatal("expected non-nil result on Ctrl+S")
	}
	if result.Action != "submit" {
		t.Fatalf("expected action 'submit', got %q", result.Action)
	}
	assignments, ok := result.Data["assignments"].(map[string]any)
	if !ok {
		t.Fatal("expected assignments map in result data")
	}
	if assignments["0"] != "CD8+ T cells" {
		t.Fatalf("expected cluster 0 = 'CD8+ T cells', got %v", assignments["0"])
	}
	if assignments["1"] != "Monocytes" {
		t.Fatalf("expected cluster 1 = 'Monocytes', got %v", assignments["1"])
	}
	// Cluster 2 has no label, should not be in assignments.
	if _, exists := assignments["2"]; exists {
		t.Fatal("expected cluster 2 not in assignments (empty label)")
	}
}

func TestCellTypeCancel(t *testing.T) {
	t.Parallel()
	c := initComponent(t, sampleClusters())

	// Esc when not editing should cancel.
	result := c.HandleMsg(tea.KeyMsg{Type: tea.KeyEsc})
	if result == nil {
		t.Fatal("expected non-nil result on Esc")
	}
	if result.Action != "cancel" {
		t.Fatalf("expected action 'cancel', got %q", result.Action)
	}
}

func TestCellTypePrefilledLabels(t *testing.T) {
	t.Parallel()
	c := initComponent(t, sampleClusters())

	// Cluster 1 was pre-filled with "Monocytes".
	if c.inputs[1].Value() != "Monocytes" {
		t.Fatalf("expected prefilled value 'Monocytes', got %q", c.inputs[1].Value())
	}
	// Cluster 0 should be empty.
	if c.inputs[0].Value() != "" {
		t.Fatalf("expected empty value for cluster 0, got %q", c.inputs[0].Value())
	}
}

func TestCellTypeViewRendering(t *testing.T) {
	t.Parallel()
	c := initComponent(t, sampleClusters())

	// Verify View does not panic at various widths.
	for _, w := range []int{60, 80, 120} {
		v := c.View(w, 30)
		if v == "" {
			t.Fatalf("expected non-empty view at width %d", w)
		}
	}
}

func TestCellTypeProgress(t *testing.T) {
	t.Parallel()
	c := initComponent(t, sampleClusters())

	// Initially 1 labeled (cluster 1 pre-filled with "Monocytes").
	if c.labeledCount() != 1 {
		t.Fatalf("expected 1 labeled initially, got %d", c.labeledCount())
	}

	// Label cluster 0.
	c.HandleMsg(tea.KeyMsg{Type: tea.KeyEnter}) // edit cluster 0
	for _, r := range "T cells" {
		c.HandleMsg(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{r}})
	}
	c.HandleMsg(tea.KeyMsg{Type: tea.KeyEsc}) // exit edit

	if c.labeledCount() != 2 {
		t.Fatalf("expected 2 labeled after typing, got %d", c.labeledCount())
	}
}

func TestCellTypeNameMode(t *testing.T) {
	t.Parallel()
	c := &CellTypeSelectorComponent{}
	if c.Name() != "cell_type_selector" {
		t.Fatalf("expected name 'cell_type_selector', got %q", c.Name())
	}
	if c.Mode() != "overlay" {
		t.Fatalf("expected mode 'overlay', got %q", c.Mode())
	}
}
