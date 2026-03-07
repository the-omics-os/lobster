package chat

import (
	"fmt"
	"strings"

	"charm.land/bubbles/v2/textinput"
	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"

	"github.com/the-omics-os/lobster-tui/internal/protocol"
)

// formResult is returned after the inline form completes.
type formResult struct {
	id     string
	values map[string]string
	err    error
}

// ---------------------------------------------------------------------------
// formFieldModel — wraps a single form field
// ---------------------------------------------------------------------------

type formFieldKind int

const (
	formFieldText formFieldKind = iota
	formFieldPassword
	formFieldSelect
	formFieldConfirm
)

type formFieldModel struct {
	key         string
	label       string
	description string
	kind        formFieldKind

	// For text/password fields.
	textInput textinput.Model

	// For select fields.
	options []string
	cursor  int

	// For confirm fields.
	confirmed bool
}

func newFormFieldModel(field protocol.FormField) formFieldModel {
	f := formFieldModel{
		key:         field.Key,
		label:       field.Label,
		description: field.Description,
	}

	switch field.Type {
	case "password":
		f.kind = formFieldPassword
		ti := textinput.New()
		ti.Placeholder = field.Default
		ti.EchoMode = textinput.EchoPassword
		f.textInput = ti

	case "select":
		f.kind = formFieldSelect
		f.options = field.Options
		// Set cursor to default if it matches an option.
		for i, opt := range field.Options {
			if opt == field.Default {
				f.cursor = i
				break
			}
		}

	case "confirm":
		f.kind = formFieldConfirm
		f.confirmed = field.Default == "true"

	default: // "text" or ""
		f.kind = formFieldText
		ti := textinput.New()
		ti.Placeholder = field.Default
		if field.Default != "" {
			ti.SetValue(field.Default)
		}
		f.textInput = ti
	}

	return f
}

func (f *formFieldModel) Focus() {
	switch f.kind {
	case formFieldText, formFieldPassword:
		f.textInput.Focus()
	}
}

func (f *formFieldModel) Blur() {
	switch f.kind {
	case formFieldText, formFieldPassword:
		f.textInput.Blur()
	}
}

func (f formFieldModel) Value() string {
	switch f.kind {
	case formFieldText, formFieldPassword:
		return f.textInput.Value()
	case formFieldSelect:
		if f.cursor >= 0 && f.cursor < len(f.options) {
			return f.options[f.cursor]
		}
		return ""
	case formFieldConfirm:
		if f.confirmed {
			return "true"
		}
		return "false"
	}
	return ""
}

// ---------------------------------------------------------------------------
// inlineFormModel — handles an entire protocol form inline
// ---------------------------------------------------------------------------

type inlineFormModel struct {
	id        string
	title     string
	fields    []formFieldModel
	focusIdx  int
	submitted bool
	cancelled bool
}

func newInlineFormModel(payload protocol.FormPayload, msgID string) inlineFormModel {
	fields := make([]formFieldModel, len(payload.Fields))
	for i, f := range payload.Fields {
		fields[i] = newFormFieldModel(f)
	}

	fm := inlineFormModel{
		id:     msgID,
		title:  payload.Title,
		fields: fields,
	}

	// Focus the first field.
	if len(fm.fields) > 0 {
		fm.fields[0].Focus()
	}

	return fm
}

func (f *inlineFormModel) Update(msg tea.Msg) tea.Cmd {
	km, ok := msg.(tea.KeyPressMsg)
	if !ok {
		// Forward non-key messages to active text field.
		if f.focusIdx < len(f.fields) {
			field := &f.fields[f.focusIdx]
			if field.kind == formFieldText || field.kind == formFieldPassword {
				var cmd tea.Cmd
				field.textInput, cmd = field.textInput.Update(msg)
				return cmd
			}
		}
		return nil
	}

	switch km.String() {
	case "esc":
		f.cancelled = true
		return nil
	case "enter":
		// On the last field, submit. Otherwise advance.
		if f.focusIdx >= len(f.fields)-1 {
			f.submitted = true
			return nil
		}
		return f.advanceFocus()
	case "tab":
		if f.focusIdx < len(f.fields)-1 {
			return f.advanceFocus()
		}
		return nil
	case "shift+tab":
		if f.focusIdx > 0 {
			return f.retreatFocus()
		}
		return nil
	}

	// Delegate to the current field.
	if f.focusIdx < len(f.fields) {
		field := &f.fields[f.focusIdx]
		switch field.kind {
		case formFieldText, formFieldPassword:
			var cmd tea.Cmd
			field.textInput, cmd = field.textInput.Update(msg)
			return cmd
		case formFieldSelect:
			switch km.String() {
			case "up", "k":
				if field.cursor > 0 {
					field.cursor--
				}
			case "down", "j":
				if field.cursor < len(field.options)-1 {
					field.cursor++
				}
			}
		case formFieldConfirm:
			switch km.String() {
			case "y", "Y":
				field.confirmed = true
			case "n", "N":
				field.confirmed = false
			case "left", "h":
				field.confirmed = true
			case "right", "l":
				field.confirmed = false
			}
		}
	}
	return nil
}

func (f *inlineFormModel) advanceFocus() tea.Cmd {
	f.fields[f.focusIdx].Blur()
	f.focusIdx++
	f.fields[f.focusIdx].Focus()
	if f.fields[f.focusIdx].kind == formFieldText || f.fields[f.focusIdx].kind == formFieldPassword {
		return textinput.Blink
	}
	return nil
}

func (f *inlineFormModel) retreatFocus() tea.Cmd {
	f.fields[f.focusIdx].Blur()
	f.focusIdx--
	f.fields[f.focusIdx].Focus()
	if f.fields[f.focusIdx].kind == formFieldText || f.fields[f.focusIdx].kind == formFieldPassword {
		return textinput.Blink
	}
	return nil
}

func (f inlineFormModel) View() string {
	titleStyle := lipgloss.NewStyle().Bold(true).MarginBottom(1)
	labelStyle := lipgloss.NewStyle().Bold(true)
	descStyle := lipgloss.NewStyle().Faint(true)
	cursorStyle := lipgloss.NewStyle().Bold(true)

	var sb strings.Builder

	if f.title != "" {
		sb.WriteString(titleStyle.Render(f.title) + "\n\n")
	}

	for i, field := range f.fields {
		indicator := "  "
		if i == f.focusIdx {
			indicator = cursorStyle.Render("> ")
		}

		sb.WriteString(indicator + labelStyle.Render(field.label) + "\n")
		if field.description != "" {
			sb.WriteString("  " + descStyle.Render(field.description) + "\n")
		}

		switch field.kind {
		case formFieldText, formFieldPassword:
			sb.WriteString("  " + field.textInput.View() + "\n")

		case formFieldSelect:
			for j, opt := range field.options {
				sel := "  "
				if j == field.cursor {
					sel = cursorStyle.Render("> ")
				}
				sb.WriteString("    " + sel + opt + "\n")
			}

		case formFieldConfirm:
			yes := "Yes"
			no := "No"
			if field.confirmed {
				yes = cursorStyle.Render("> Yes")
				no = "  No"
			} else {
				yes = "  Yes"
				no = cursorStyle.Render("> No")
			}
			sb.WriteString("  " + yes + "  " + no + "\n")
		}
		sb.WriteString("\n")
	}

	sb.WriteString(descStyle.Render("enter: submit  tab: next  esc: cancel"))
	return sb.String()
}

func (f inlineFormModel) collectValues() map[string]string {
	values := make(map[string]string, len(f.fields))
	for _, field := range f.fields {
		values[field.key] = field.Value()
	}
	return values
}

func (f inlineFormModel) toFormResult() formResult {
	if f.cancelled {
		return formResult{
			id:  f.id,
			err: fmt.Errorf("cancelled"),
		}
	}
	return formResult{
		id:     f.id,
		values: f.collectValues(),
	}
}

// ---------------------------------------------------------------------------
// Integration with chat Model
// ---------------------------------------------------------------------------

// handleFormMessage creates an inline form from a protocol form request.
// It sets the activeForm field and returns a blink command for text inputs.
func (m Model) handleFormMessage(payload protocol.FormPayload, msgID string) (Model, tea.Cmd) {
	fm := newInlineFormModel(payload, msgID)
	m.activeForm = &fm
	return m, textinput.Blink
}

// updateActiveForm processes messages for the inline form.
// Returns the updated model and whether the form is still active.
func (m Model) updateActiveForm(msg tea.Msg) (Model, tea.Cmd) {
	if m.activeForm == nil {
		return m, nil
	}

	cmd := m.activeForm.Update(msg)

	if m.activeForm.submitted || m.activeForm.cancelled {
		result := m.activeForm.toFormResult()
		m.activeForm = nil

		// Send form response back to Python.
		if result.err != nil {
			_ = m.handler.SendTyped(protocol.TypeFormResponse, protocol.FormResponsePayload{
				ID:     result.id,
				Values: map[string]string{},
			}, result.id)
		} else {
			_ = m.handler.SendTyped(protocol.TypeFormResponse, protocol.FormResponsePayload{
				ID:     result.id,
				Values: result.values,
			}, result.id)
		}
		return m, waitForProtocolMsg(m.handler)
	}

	return m, cmd
}
