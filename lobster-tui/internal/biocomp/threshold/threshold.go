// Package threshold implements the BioComp threshold slider component.
//
// ThresholdSlider allows the user to adjust a numeric threshold with live
// preview of how many items pass. It supports bidirectional streaming:
// value changes are emitted via ChangeEvent, and Python can push updated
// counts via SetData.
package threshold

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"

	"charm.land/bubbles/v2/key"
	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"

	"github.com/the-omics-os/lobster-tui/internal/biocomp"
)

func init() {
	biocomp.Register("threshold_slider", func() biocomp.BioComponent {
		return &ThresholdSliderComponent{}
	})
}

// ThresholdSliderComponent is the interactive threshold slider.
type ThresholdSliderComponent struct {
	data          ThresholdSliderData
	value         float64
	pendingChange bool
	precision     int // decimal places derived from step
}

func (c *ThresholdSliderComponent) Init(data json.RawMessage) error {
	var d ThresholdSliderData
	if err := json.Unmarshal(data, &d); err != nil {
		return fmt.Errorf("threshold_slider: invalid data: %w", err)
	}
	if d.Min >= d.Max {
		return fmt.Errorf("threshold_slider: min (%g) must be less than max (%g)", d.Min, d.Max)
	}
	if d.Step <= 0 {
		return fmt.Errorf("threshold_slider: step must be positive, got %g", d.Step)
	}
	c.data = d
	c.value = clamp(d.Default, d.Min, d.Max)
	c.precision = decimalPlaces(d.Step)
	c.pendingChange = false
	return nil
}

func (c *ThresholdSliderComponent) HandleMsg(msg tea.Msg) *biocomp.ComponentResult {
	km, ok := msg.(tea.KeyPressMsg)
	if !ok {
		return nil
	}

	switch km.String() {
	case "right":
		c.adjustValue(c.data.Step)
		return nil
	case "left":
		c.adjustValue(-c.data.Step)
		return nil
	case "shift+right":
		c.adjustValue(10 * c.data.Step)
		return nil
	case "shift+left":
		c.adjustValue(-10 * c.data.Step)
		return nil
	case "enter":
		return &biocomp.ComponentResult{
			Action: "submit",
			Data:   map[string]any{"value": roundTo(c.value, c.precision)},
		}
	case "esc":
		return &biocomp.ComponentResult{
			Action: "cancel",
			Data:   map[string]any{"value": roundTo(c.value, c.precision)},
		}
	}
	return nil
}

func (c *ThresholdSliderComponent) View(width, height int) string {
	if width < 20 {
		width = 20
	}

	// Styles
	titleStyle := lipgloss.NewStyle().Bold(true).Width(width)
	valueStyle := lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("63"))
	countStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("245"))
	filledStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("63"))
	emptyStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("240"))
	cursorStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("15")).Bold(true)

	// Label
	label := c.data.Label
	if label == "" {
		label = "Threshold"
	}

	// Value string
	valStr := fmt.Sprintf("%.*f", c.precision, c.value)
	if c.data.Unit != "" {
		valStr += " " + c.data.Unit
	}
	renderedVal := valueStyle.Render(valStr)

	// Slider bar
	// Reserve space for: "  [" + bar + "]  " + value display
	valDisplayWidth := lipgloss.Width(renderedVal)
	barOverhead := 6 + valDisplayWidth // "  [" (3) + "]  " (3) + value
	barWidth := width - barOverhead
	if barWidth < 10 {
		barWidth = 10
	}

	fillRatio := 0.0
	rangeVal := c.data.Max - c.data.Min
	if rangeVal > 0 {
		fillRatio = (c.value - c.data.Min) / rangeVal
	}
	fillRatio = clamp(fillRatio, 0, 1)

	cursorPos := int(math.Round(fillRatio * float64(barWidth-1)))
	if cursorPos < 0 {
		cursorPos = 0
	}
	if cursorPos >= barWidth {
		cursorPos = barWidth - 1
	}

	var bar strings.Builder
	for i := 0; i < barWidth; i++ {
		if i == cursorPos {
			bar.WriteString(cursorStyle.Render("|"))
		} else if i < cursorPos {
			bar.WriteString(filledStyle.Render("="))
		} else {
			bar.WriteString(emptyStyle.Render("-"))
		}
	}

	sliderLine := fmt.Sprintf("  [%s]  %s", bar.String(), renderedVal)

	// Count line (only if Total > 0)
	var countLine string
	if c.data.Total > 0 {
		pct := float64(c.data.Count) / float64(c.data.Total) * 100
		countLine = countStyle.Render(fmt.Sprintf(
			"%d / %d items passing (%.1f%%)",
			c.data.Count, c.data.Total, pct,
		))
	}

	// Title
	title := titleStyle.Render(label)

	// Assemble
	parts := []string{"", title, "", sliderLine, ""}
	if countLine != "" {
		parts = append(parts, countLine, "")
	}

	return lipgloss.JoinVertical(lipgloss.Left, parts...)
}

func (c *ThresholdSliderComponent) SetData(data json.RawMessage) error {
	var update ThresholdSliderData
	if err := json.Unmarshal(data, &update); err != nil {
		return fmt.Errorf("threshold_slider: invalid update data: %w", err)
	}
	// Update count/total from Python recalculation
	c.data.Count = update.Count
	c.data.Total = update.Total
	return nil
}

func (c *ThresholdSliderComponent) Name() string { return "threshold_slider" }
func (c *ThresholdSliderComponent) Mode() string { return "overlay" }

func (c *ThresholdSliderComponent) KeyBindings() []key.Binding {
	return []key.Binding{
		key.NewBinding(key.WithKeys("left", "right"), key.WithHelp("</>", "adjust")),
		key.NewBinding(key.WithKeys("shift+left", "shift+right"), key.WithHelp("S-</>", "coarse")),
		key.NewBinding(key.WithKeys("enter"), key.WithHelp("enter", "go")),
		key.NewBinding(key.WithKeys("esc"), key.WithHelp("esc", "cancel")),
	}
}

func (c *ThresholdSliderComponent) ChangeEvent() map[string]any {
	if !c.pendingChange {
		return nil
	}
	c.pendingChange = false
	return map[string]any{"value": roundTo(c.value, c.precision)}
}

// adjustValue changes the current value by delta, clamped to [min, max].
func (c *ThresholdSliderComponent) adjustValue(delta float64) {
	c.value = clamp(c.value+delta, c.data.Min, c.data.Max)
	c.pendingChange = true
}

// clamp restricts v to [lo, hi].
func clamp(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// decimalPlaces returns the number of decimal places in a float step value.
func decimalPlaces(step float64) int {
	s := fmt.Sprintf("%g", step)
	idx := strings.IndexByte(s, '.')
	if idx < 0 {
		return 0
	}
	return len(s) - idx - 1
}

// roundTo rounds v to the given number of decimal places.
func roundTo(v float64, places int) float64 {
	pow := math.Pow(10, float64(places))
	return math.Round(v*pow) / pow
}
