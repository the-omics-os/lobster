// Package celltype implements the BioComp cell type annotation selector.
//
// This component lets users assign cell type labels to scRNA-seq clusters
// using marker gene context. It is the showcase domain-specific BioCharm
// component for Lobster AI's Go TUI.
package celltype

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/charmbracelet/bubbles/key"
	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/the-omics-os/lobster-tui/internal/biocomp"
)

func init() {
	biocomp.Register("cell_type_selector", func() biocomp.BioComponent {
		return &CellTypeSelectorComponent{}
	})
}

// CellTypeSelectorComponent displays a scrollable list of scRNA-seq clusters
// with marker genes, allowing the user to type a cell type label for each.
type CellTypeSelectorComponent struct {
	clusters     []ClusterInfo
	inputs       []textinput.Model
	cursor       int // currently highlighted cluster row
	editingIndex int // which cluster's textinput is focused (-1 = none)
	offset       int // scroll offset for long cluster lists
}

func (c *CellTypeSelectorComponent) Init(data json.RawMessage) error {
	var d CellTypeSelectorData
	if err := json.Unmarshal(data, &d); err != nil {
		return fmt.Errorf("cell_type_selector: invalid data: %w", err)
	}
	if len(d.Clusters) == 0 {
		return fmt.Errorf("cell_type_selector: at least one cluster is required")
	}

	c.clusters = d.Clusters
	c.inputs = make([]textinput.Model, len(d.Clusters))
	for i, cl := range d.Clusters {
		ti := textinput.New()
		ti.Placeholder = "cell type..."
		ti.CharLimit = 128
		ti.Width = 30
		if cl.Label != "" {
			ti.SetValue(cl.Label)
		}
		c.inputs[i] = ti
	}
	c.cursor = 0
	c.editingIndex = -1
	c.offset = 0

	return nil
}

func (c *CellTypeSelectorComponent) HandleMsg(msg tea.Msg) *biocomp.ComponentResult {
	km, ok := msg.(tea.KeyMsg)
	if !ok {
		// Forward non-key messages to active textinput if editing.
		if c.editingIndex >= 0 && c.editingIndex < len(c.inputs) {
			var cmd tea.Cmd
			c.inputs[c.editingIndex], cmd = c.inputs[c.editingIndex].Update(msg)
			_ = cmd
		}
		return nil
	}

	// When editing a textinput, most keys go to the input.
	if c.editingIndex >= 0 {
		return c.handleEditMode(km)
	}
	return c.handleNavMode(km)
}

// handleNavMode processes keys when not editing any textinput.
func (c *CellTypeSelectorComponent) handleNavMode(km tea.KeyMsg) *biocomp.ComponentResult {
	switch km.Type {
	case tea.KeyUp:
		c.moveUp()
		return nil
	case tea.KeyDown:
		c.moveDown()
		return nil
	case tea.KeyEnter:
		c.enterEditMode()
		return nil
	case tea.KeyEsc:
		return &biocomp.ComponentResult{
			Action: "cancel",
			Data:   map[string]any{},
		}
	case tea.KeyCtrlS:
		return c.submitAll()
	case tea.KeyRunes:
		switch km.String() {
		case "k":
			c.moveUp()
			return nil
		case "j":
			c.moveDown()
			return nil
		}
	}
	return nil
}

// handleEditMode processes keys when a textinput is focused.
func (c *CellTypeSelectorComponent) handleEditMode(km tea.KeyMsg) *biocomp.ComponentResult {
	switch km.Type {
	case tea.KeyEsc:
		// Exit edit mode, keep text.
		c.exitEditMode()
		return nil
	case tea.KeyTab:
		// Accept current input and move to next cluster.
		c.exitEditMode()
		if c.cursor < len(c.clusters)-1 {
			c.cursor++
			c.adjustOffset()
		}
		c.enterEditMode()
		return nil
	case tea.KeyCtrlS:
		c.exitEditMode()
		return c.submitAll()
	case tea.KeyEnter:
		// Accept current input and exit edit mode.
		c.exitEditMode()
		return nil
	default:
		// Forward to the active textinput.
		var cmd tea.Cmd
		c.inputs[c.editingIndex], cmd = c.inputs[c.editingIndex].Update(km)
		_ = cmd
		return nil
	}
}

func (c *CellTypeSelectorComponent) enterEditMode() {
	if c.cursor >= 0 && c.cursor < len(c.inputs) {
		c.editingIndex = c.cursor
		c.inputs[c.editingIndex].Focus()
	}
}

func (c *CellTypeSelectorComponent) exitEditMode() {
	if c.editingIndex >= 0 && c.editingIndex < len(c.inputs) {
		c.inputs[c.editingIndex].Blur()
	}
	c.editingIndex = -1
}

func (c *CellTypeSelectorComponent) moveUp() {
	if c.cursor > 0 {
		c.cursor--
		if c.cursor < c.offset {
			c.offset = c.cursor
		}
	}
}

func (c *CellTypeSelectorComponent) moveDown() {
	if c.cursor < len(c.clusters)-1 {
		c.cursor++
	}
}

func (c *CellTypeSelectorComponent) adjustOffset() {
	// Called after cursor changes to ensure visibility.
	// The actual offset adjustment happens in View based on available height.
}

func (c *CellTypeSelectorComponent) submitAll() *biocomp.ComponentResult {
	assignments := make(map[string]any)
	for i, inp := range c.inputs {
		val := inp.Value()
		if val != "" {
			assignments[strconv.Itoa(c.clusters[i].ID)] = val
		}
	}
	return &biocomp.ComponentResult{
		Action: "submit",
		Data:   map[string]any{"assignments": assignments},
	}
}

// labeledCount returns the number of clusters with non-empty labels.
func (c *CellTypeSelectorComponent) labeledCount() int {
	count := 0
	for _, inp := range c.inputs {
		if inp.Value() != "" {
			count++
		}
	}
	return count
}

func (c *CellTypeSelectorComponent) View(width, height int) string {
	// Styles.
	headerStyle := lipgloss.NewStyle().
		Bold(true).
		Foreground(lipgloss.Color("63"))

	markerStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color("240"))

	selectedStyle := lipgloss.NewStyle().
		Bold(true).
		Foreground(lipgloss.Color("63"))

	normalStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color("252"))

	dimStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color("240"))

	// Each cluster row takes 3 lines: header, markers, label input.
	const linesPerCluster = 3

	// Calculate visible clusters based on available height.
	// Reserve 2 lines for the status/help bar at the bottom.
	contentH := height - 2
	if contentH < linesPerCluster {
		contentH = linesPerCluster
	}
	visibleClusters := contentH / linesPerCluster
	if visibleClusters < 1 {
		visibleClusters = 1
	}
	if visibleClusters > len(c.clusters) {
		visibleClusters = len(c.clusters)
	}

	// Adjust offset so cursor is always visible.
	if c.cursor >= c.offset+visibleClusters {
		c.offset = c.cursor - visibleClusters + 1
	}
	if c.cursor < c.offset {
		c.offset = c.cursor
	}

	end := c.offset + visibleClusters
	if end > len(c.clusters) {
		end = len(c.clusters)
	}

	// Adjust textinput widths.
	inputW := width - 12 // "  Label: " prefix + padding
	if inputW < 10 {
		inputW = 10
	}
	for i := range c.inputs {
		c.inputs[i].Width = inputW
	}

	var rows []string
	for i := c.offset; i < end; i++ {
		cl := c.clusters[i]
		isCursor := i == c.cursor
		isEditing := i == c.editingIndex

		// Cluster header line.
		prefix := "  "
		hStyle := normalStyle
		if isCursor {
			prefix = "> "
			hStyle = selectedStyle
		}
		header := hStyle.Render(fmt.Sprintf(
			"%sCluster %d (%d cells)",
			prefix, cl.ID, cl.Size,
		))

		// Markers line.
		markersStr := markerStyle.Render(
			fmt.Sprintf("  Markers: %s", strings.Join(cl.Markers, ", ")),
		)

		// Label input line.
		var labelLine string
		if isEditing {
			labelLine = headerStyle.Render("  Label: ") + c.inputs[i].View()
		} else {
			val := c.inputs[i].Value()
			if val == "" {
				labelLine = dimStyle.Render("  Label: [empty]")
			} else {
				labelLine = dimStyle.Render("  Label: ") + normalStyle.Render(val)
			}
		}

		rows = append(rows, header, markersStr, labelLine)
	}

	content := strings.Join(rows, "\n")

	// Status bar.
	labeled := c.labeledCount()
	total := len(c.clusters)
	statusStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color("240")).
		Width(width)
	status := statusStyle.Render(fmt.Sprintf(
		"  %d/%d labeled", labeled, total,
	))

	return lipgloss.JoinVertical(lipgloss.Left, content, "", status)
}

func (c *CellTypeSelectorComponent) SetData(data json.RawMessage) error {
	var d CellTypeSelectorData
	if err := json.Unmarshal(data, &d); err != nil {
		return fmt.Errorf("cell_type_selector: invalid update data: %w", err)
	}
	if len(d.Clusters) > 0 {
		// Re-initialize with new cluster data.
		return c.Init(data)
	}
	return nil
}

func (c *CellTypeSelectorComponent) Name() string { return "cell_type_selector" }
func (c *CellTypeSelectorComponent) Mode() string { return "overlay" }

func (c *CellTypeSelectorComponent) KeyBindings() []key.Binding {
	return []key.Binding{
		key.NewBinding(key.WithKeys("up", "down"), key.WithHelp("^/v", "navigate")),
		key.NewBinding(key.WithKeys("enter"), key.WithHelp("enter", "edit")),
		key.NewBinding(key.WithKeys("tab"), key.WithHelp("tab", "accept+next")),
		key.NewBinding(key.WithKeys("ctrl+s"), key.WithHelp("C-s", "submit all")),
		key.NewBinding(key.WithKeys("esc"), key.WithHelp("esc", "cancel/exit edit")),
	}
}

func (c *CellTypeSelectorComponent) ChangeEvent() map[string]any { return nil }
