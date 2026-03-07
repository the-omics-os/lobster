// Package qcdash implements the BioComp QC Dashboard inline component.
//
// QCDashboard displays quality-control metrics as colored status bars.
// It is non-interactive (inline mode) — Python streams updated metrics
// via SetData, and the component re-renders automatically.
package qcdash

import (
	"encoding/json"
	"fmt"
	"image/color"
	"math"
	"strings"

	"charm.land/bubbles/v2/key"
	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"

	"github.com/the-omics-os/lobster-tui/internal/biocomp"
)

func init() {
	biocomp.Register("qc_dashboard", func() biocomp.BioComponent {
		return &QCDashboardComponent{}
	})
}

// Status color constants.
var (
	colorPass = lipgloss.Color("42")  // green
	colorWarn = lipgloss.Color("214") // yellow/orange
	colorFail = lipgloss.Color("196") // red
	colorDim  = lipgloss.Color("240") // dim gray for borders
)

// QCDashboardComponent is the inline QC metrics dashboard.
type QCDashboardComponent struct {
	data QCDashboardData
}

func (c *QCDashboardComponent) Init(data json.RawMessage) error {
	var d QCDashboardData
	if err := json.Unmarshal(data, &d); err != nil {
		return fmt.Errorf("qc_dashboard: invalid data: %w", err)
	}
	if len(d.Metrics) == 0 {
		return fmt.Errorf("qc_dashboard: at least 1 metric required")
	}
	c.data = d
	return nil
}

// HandleMsg always returns nil — QCDashboard is non-interactive.
func (c *QCDashboardComponent) HandleMsg(msg tea.Msg) *biocomp.ComponentResult {
	return nil
}

// View renders the QC dashboard as a compact bordered block with colored bars.
func (c *QCDashboardComponent) View(width, height int) string {
	if width < 20 {
		width = 20
	}

	title := c.data.Title
	if title == "" {
		title = "QC Summary"
	}

	// Styles.
	borderStyle := lipgloss.NewStyle().Foreground(colorDim)

	// Calculate inner width (subtract border chars: "+--" prefix and trailing "--+").
	innerW := width - 4 // "| " prefix + " |" suffix
	if innerW < 16 {
		innerW = 16
	}

	var lines []string

	// Top border with title.
	titleText := " " + title + " "
	dashCount := innerW - len(titleText)
	if dashCount < 2 {
		dashCount = 2
	}
	leftDash := 2
	rightDash := dashCount - leftDash
	if rightDash < 0 {
		rightDash = 0
	}
	topBorder := borderStyle.Render("+" + strings.Repeat("-", leftDash) + titleText + strings.Repeat("-", rightDash) + "+")
	lines = append(lines, topBorder)

	// Metric lines.
	for _, m := range c.data.Metrics {
		line := c.renderMetricLine(m, innerW)
		lines = append(lines, borderStyle.Render("| ")+line+borderStyle.Render(" |"))
	}

	// Bottom border.
	bottomBorder := borderStyle.Render("+" + strings.Repeat("-", innerW) + "+")
	lines = append(lines, bottomBorder)

	return lipgloss.JoinVertical(lipgloss.Left, lines...)
}

// renderMetricLine renders a single metric as: Name  ######  Value  STATUS
func (c *QCDashboardComponent) renderMetricLine(m QCMetric, innerW int) string {
	// Name column: max 12 chars, left-aligned.
	name := m.Name
	if len(name) > 12 {
		name = name[:12]
	}
	namePadded := fmt.Sprintf("%-12s", name)

	// Status text.
	statusText := strings.ToUpper(m.Status)
	if statusText == "" {
		statusText = "    "
	}
	statusStyled := c.styledStatus(statusText, m.Status)

	// Value string.
	valStr := formatValue(m.Value, m.Unit)

	// Calculate bar width: innerW - name(12) - spaces(2) - value(~8) - spaces(2) - status(~4) - spaces(2)
	valDisplayLen := len(valStr)
	if valDisplayLen < 6 {
		valDisplayLen = 6
	}
	statusDisplayLen := len(statusText)
	if statusDisplayLen < 4 {
		statusDisplayLen = 4
	}
	// Layout: name(12) + " " + bar + " " + value + "  " + status
	overhead := 12 + 1 + 1 + valDisplayLen + 2 + statusDisplayLen
	barWidth := innerW - overhead
	if barWidth < 2 {
		barWidth = 2
	}
	if barWidth > 30 {
		barWidth = 30
	}

	// Calculate fill ratio.
	fillRatio := 0.0
	rangeVal := m.Max - m.Min
	if rangeVal > 0 {
		fillRatio = (m.Value - m.Min) / rangeVal
	}
	fillRatio = math.Max(0, math.Min(1, fillRatio))
	fillCount := int(math.Round(fillRatio * float64(barWidth)))

	// Build bar.
	barColor := c.statusColor(m.Status)
	barStyle := lipgloss.NewStyle().Foreground(barColor)
	emptyStyle := lipgloss.NewStyle().Foreground(colorDim)

	bar := barStyle.Render(strings.Repeat("#", fillCount)) +
		emptyStyle.Render(strings.Repeat(" ", barWidth-fillCount))

	// Right-align value string.
	valPadded := fmt.Sprintf("%-*s", valDisplayLen, valStr)

	return namePadded + " " + bar + " " + valPadded + "  " + statusStyled
}

// styledStatus returns the status text with appropriate color.
func (c *QCDashboardComponent) styledStatus(text, status string) string {
	color := c.statusColor(status)
	return lipgloss.NewStyle().Bold(true).Foreground(color).Render(text)
}

// statusColor returns the color for a status string.
func (c *QCDashboardComponent) statusColor(status string) color.Color {
	switch strings.ToLower(status) {
	case "pass":
		return colorPass
	case "warn":
		return colorWarn
	case "fail":
		return colorFail
	default:
		return colorDim
	}
}

// SetData replaces metrics entirely with new data from Python.
func (c *QCDashboardComponent) SetData(data json.RawMessage) error {
	var d QCDashboardData
	if err := json.Unmarshal(data, &d); err != nil {
		return fmt.Errorf("qc_dashboard: invalid update data: %w", err)
	}
	if len(d.Metrics) == 0 {
		return fmt.Errorf("qc_dashboard: at least 1 metric required")
	}
	c.data = d
	return nil
}

func (c *QCDashboardComponent) Name() string                { return "qc_dashboard" }
func (c *QCDashboardComponent) Mode() string                { return "inline" }
func (c *QCDashboardComponent) KeyBindings() []key.Binding  { return nil }
func (c *QCDashboardComponent) ChangeEvent() map[string]any { return nil }

// formatValue formats a float value with its unit for display.
func formatValue(value float64, unit string) string {
	var s string
	if value == math.Trunc(value) {
		s = fmt.Sprintf("%.0f", value)
	} else {
		s = fmt.Sprintf("%.1f", value)
	}
	if unit != "" {
		s += unit
	}
	return s
}
