// Package confirm implements the BioComp confirm dialog component.
package confirm

import (
	"encoding/json"
	"fmt"

	"github.com/charmbracelet/bubbles/key"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/the-omics-os/lobster-tui/internal/biocomp"
)

func init() {
	biocomp.Register("confirm", func() biocomp.BioComponent {
		return &ConfirmComponent{}
	})
}

type confirmData struct {
	Question string `json:"question"`
	Default  bool   `json:"default"`
}

// ConfirmComponent displays a yes/no confirmation dialog.
type ConfirmComponent struct {
	question string
	selected bool // true = Yes, false = No
}

func (c *ConfirmComponent) Init(data json.RawMessage) error {
	var d confirmData
	if err := json.Unmarshal(data, &d); err != nil {
		return fmt.Errorf("confirm: invalid data: %w", err)
	}
	if d.Question == "" {
		return fmt.Errorf("confirm: question is required")
	}
	c.question = d.Question
	c.selected = d.Default
	return nil
}

func (c *ConfirmComponent) HandleMsg(msg tea.Msg) *biocomp.ComponentResult {
	km, ok := msg.(tea.KeyMsg)
	if !ok {
		return nil
	}

	switch km.Type {
	case tea.KeyLeft, tea.KeyRight, tea.KeyTab:
		c.selected = !c.selected
		return nil
	case tea.KeyEnter:
		return &biocomp.ComponentResult{
			Action: "submit",
			Data:   map[string]any{"confirmed": c.selected},
		}
	case tea.KeyEsc:
		return &biocomp.ComponentResult{
			Action: "cancel",
			Data:   map[string]any{"confirmed": false},
		}
	case tea.KeyRunes:
		switch km.String() {
		case "y", "Y":
			return &biocomp.ComponentResult{
				Action: "submit",
				Data:   map[string]any{"confirmed": true},
			}
		case "n", "N":
			return &biocomp.ComponentResult{
				Action: "submit",
				Data:   map[string]any{"confirmed": false},
			}
		}
	}
	return nil
}

func (c *ConfirmComponent) View(width, height int) string {
	questionStyle := lipgloss.NewStyle().
		Bold(true).
		Width(width).
		MarginBottom(1)

	activeBtn := lipgloss.NewStyle().
		Bold(true).
		Foreground(lipgloss.Color("15")).
		Background(lipgloss.Color("63")).
		Padding(0, 2)

	inactiveBtn := lipgloss.NewStyle().
		Foreground(lipgloss.Color("240")).
		Padding(0, 2)

	q := questionStyle.Render(c.question)

	var yesBtn, noBtn string
	if c.selected {
		yesBtn = activeBtn.Render("Yes")
		noBtn = inactiveBtn.Render("No")
	} else {
		yesBtn = inactiveBtn.Render("Yes")
		noBtn = activeBtn.Render("No")
	}

	buttons := lipgloss.JoinHorizontal(lipgloss.Center, yesBtn, "  ", noBtn)

	return lipgloss.JoinVertical(lipgloss.Left, q, buttons)
}

func (c *ConfirmComponent) SetData(data json.RawMessage) error {
	var d confirmData
	if err := json.Unmarshal(data, &d); err != nil {
		return fmt.Errorf("confirm: invalid update data: %w", err)
	}
	if d.Question != "" {
		c.question = d.Question
	}
	return nil
}

func (c *ConfirmComponent) Name() string { return "confirm" }
func (c *ConfirmComponent) Mode() string { return "overlay" }

func (c *ConfirmComponent) KeyBindings() []key.Binding {
	return []key.Binding{
		key.NewBinding(key.WithKeys("left", "right"), key.WithHelp("←/→", "choose")),
		key.NewBinding(key.WithKeys("enter"), key.WithHelp("enter", "confirm")),
		key.NewBinding(key.WithKeys("y", "n"), key.WithHelp("y/n", "direct")),
		key.NewBinding(key.WithKeys("esc"), key.WithHelp("esc", "cancel")),
	}
}

func (c *ConfirmComponent) ChangeEvent() map[string]any { return nil }
