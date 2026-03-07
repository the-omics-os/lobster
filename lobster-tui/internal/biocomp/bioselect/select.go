// Package bioselect implements the BioComp selection list component.
// The package name avoids conflict with Go's select keyword.
package bioselect

import (
	"encoding/json"
	"fmt"
	"strings"

	"charm.land/bubbles/v2/key"
	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"

	"github.com/the-omics-os/lobster-tui/internal/biocomp"
)

func init() {
	biocomp.Register("select", func() biocomp.BioComponent {
		return &SelectComponent{}
	})
}

type selectData struct {
	Question string   `json:"question"`
	Options  []string `json:"options"`
	Default  int      `json:"default,omitempty"`
}

// SelectComponent displays a vertical list of options for the user to pick from.
type SelectComponent struct {
	question string
	options  []string
	cursor   int
	offset   int // scroll offset for long lists
}

func (s *SelectComponent) Init(data json.RawMessage) error {
	var d selectData
	if err := json.Unmarshal(data, &d); err != nil {
		return fmt.Errorf("select: invalid data: %w", err)
	}
	if len(d.Options) == 0 {
		return fmt.Errorf("select: at least one option is required")
	}
	if d.Question == "" {
		return fmt.Errorf("select: question is required")
	}
	s.question = d.Question
	s.options = d.Options
	s.cursor = d.Default
	if s.cursor < 0 || s.cursor >= len(s.options) {
		s.cursor = 0
	}
	return nil
}

func (s *SelectComponent) HandleMsg(msg tea.Msg) *biocomp.ComponentResult {
	km, ok := msg.(tea.KeyPressMsg)
	if !ok {
		return nil
	}

	switch km.String() {
	case "up", "k":
		s.moveUp()
		return nil
	case "down", "j":
		s.moveDown()
		return nil
	case "enter":
		return &biocomp.ComponentResult{
			Action: "submit",
			Data: map[string]any{
				"selected": s.options[s.cursor],
				"index":    s.cursor,
			},
		}
	case "esc":
		return &biocomp.ComponentResult{
			Action: "cancel",
			Data:   map[string]any{},
		}
	default:
		// Number keys 1-9 for direct selection.
		if km.Text != "" && len([]rune(km.Text)) == 1 {
			r := []rune(km.Text)[0]
			if r >= '1' && r <= '9' && len(s.options) <= 9 {
				idx := int(r - '1')
				if idx < len(s.options) {
					s.cursor = idx
					return &biocomp.ComponentResult{
						Action: "submit",
						Data: map[string]any{
							"selected": s.options[idx],
							"index":    idx,
						},
					}
				}
			}
		}
	}
	return nil
}

func (s *SelectComponent) moveUp() {
	if s.cursor > 0 {
		s.cursor--
		if s.cursor < s.offset {
			s.offset = s.cursor
		}
	}
}

func (s *SelectComponent) moveDown() {
	if s.cursor < len(s.options)-1 {
		s.cursor++
	}
}

func (s *SelectComponent) View(width, height int) string {
	questionStyle := lipgloss.NewStyle().
		Bold(true).
		Width(width).
		MarginBottom(1)

	selectedStyle := lipgloss.NewStyle().
		Bold(true).
		Foreground(lipgloss.Color("63"))

	normalStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color("240"))

	q := questionStyle.Render(s.question)

	// Calculate visible window for scrolling.
	visibleH := height - 3 // question + margin + bottom padding
	if visibleH < 1 {
		visibleH = len(s.options)
	}

	// Adjust offset so cursor is always visible.
	if s.cursor >= s.offset+visibleH {
		s.offset = s.cursor - visibleH + 1
	}
	if s.cursor < s.offset {
		s.offset = s.cursor
	}

	end := s.offset + visibleH
	if end > len(s.options) {
		end = len(s.options)
	}

	var lines []string
	for i := s.offset; i < end; i++ {
		prefix := "  "
		style := normalStyle
		if i == s.cursor {
			prefix = "> "
			style = selectedStyle
		}
		numHint := ""
		if len(s.options) <= 9 {
			numHint = fmt.Sprintf("%d. ", i+1)
		}
		line := style.Render(prefix + numHint + s.options[i])
		lines = append(lines, line)
	}

	optionList := strings.Join(lines, "\n")
	return lipgloss.JoinVertical(lipgloss.Left, q, optionList)
}

func (s *SelectComponent) SetData(data json.RawMessage) error {
	var d selectData
	if err := json.Unmarshal(data, &d); err != nil {
		return fmt.Errorf("select: invalid update data: %w", err)
	}
	if len(d.Options) > 0 {
		s.options = d.Options
		s.cursor = 0
		s.offset = 0
	}
	if d.Question != "" {
		s.question = d.Question
	}
	return nil
}

func (s *SelectComponent) Name() string { return "select" }
func (s *SelectComponent) Mode() string { return "overlay" }

func (s *SelectComponent) KeyBindings() []key.Binding {
	return []key.Binding{
		key.NewBinding(key.WithKeys("up", "down"), key.WithHelp("↑/↓", "navigate")),
		key.NewBinding(key.WithKeys("enter"), key.WithHelp("enter", "select")),
		key.NewBinding(key.WithKeys("1", "2", "3"), key.WithHelp("1-9", "direct")),
		key.NewBinding(key.WithKeys("esc"), key.WithHelp("esc", "cancel")),
	}
}

func (s *SelectComponent) ChangeEvent() map[string]any { return nil }
