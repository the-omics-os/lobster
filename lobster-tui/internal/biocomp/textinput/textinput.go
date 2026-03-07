// Package textinput implements the BioComp text input component.
package textinput

import (
	"encoding/json"
	"fmt"

	"charm.land/bubbles/v2/key"
	"charm.land/bubbles/v2/textinput"
	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"

	"github.com/the-omics-os/lobster-tui/internal/biocomp"
)

func init() {
	biocomp.Register("text_input", func() biocomp.BioComponent {
		return &TextInputComponent{}
	})
}

type textInputData struct {
	Question    string `json:"question"`
	Placeholder string `json:"placeholder,omitempty"`
	Multiline   bool   `json:"multiline,omitempty"`
}

// TextInputComponent provides a text input field for free-form user input.
type TextInputComponent struct {
	question  string
	multiline bool
	input     textinput.Model
}

func (t *TextInputComponent) Init(data json.RawMessage) error {
	var d textInputData
	if err := json.Unmarshal(data, &d); err != nil {
		return fmt.Errorf("text_input: invalid data: %w", err)
	}
	if d.Question == "" {
		return fmt.Errorf("text_input: question is required")
	}
	t.question = d.Question
	t.multiline = d.Multiline

	t.input = textinput.New()
	t.input.Placeholder = d.Placeholder
	t.input.Focus()
	t.input.CharLimit = 1024
	t.input.SetWidth(40) // default, adjusted in View

	return nil
}

func (t *TextInputComponent) HandleMsg(msg tea.Msg) *biocomp.ComponentResult {
	km, ok := msg.(tea.KeyPressMsg)
	if ok {
		switch km.String() {
		case "enter":
			return &biocomp.ComponentResult{
				Action: "submit",
				Data:   map[string]any{"answer": t.input.Value()},
			}
		case "esc":
			return &biocomp.ComponentResult{
				Action: "cancel",
				Data:   map[string]any{},
			}
		}
	}

	// Forward all other messages to the bubbles textinput model.
	var cmd tea.Cmd
	t.input, cmd = t.input.Update(msg)
	// We discard cmd here since BioComponent.HandleMsg does not return tea.Cmd.
	// The textinput blink cursor will not animate, which is acceptable for overlay use.
	_ = cmd
	return nil
}

func (t *TextInputComponent) View(width, height int) string {
	questionStyle := lipgloss.NewStyle().
		Bold(true).
		Width(width).
		MarginBottom(1)

	q := questionStyle.Render(t.question)

	// Adjust input width to fit available space.
	inputW := width - 2
	if inputW < 10 {
		inputW = 10
	}
	t.input.SetWidth(inputW)

	return lipgloss.JoinVertical(lipgloss.Left, q, t.input.View())
}

func (t *TextInputComponent) SetData(data json.RawMessage) error {
	var d textInputData
	if err := json.Unmarshal(data, &d); err != nil {
		return fmt.Errorf("text_input: invalid update data: %w", err)
	}
	if d.Question != "" {
		t.question = d.Question
	}
	if d.Placeholder != "" {
		t.input.Placeholder = d.Placeholder
	}
	return nil
}

func (t *TextInputComponent) Name() string { return "text_input" }
func (t *TextInputComponent) Mode() string { return "overlay" }

func (t *TextInputComponent) KeyBindings() []key.Binding {
	return []key.Binding{
		key.NewBinding(key.WithKeys("enter"), key.WithHelp("enter", "submit")),
		key.NewBinding(key.WithKeys("esc"), key.WithHelp("esc", "cancel")),
	}
}

func (t *TextInputComponent) ChangeEvent() map[string]any { return nil }
