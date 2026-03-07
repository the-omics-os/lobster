package chat

import (
	"fmt"
	"io"
	"os"

	"github.com/charmbracelet/huh"
	tea "charm.land/bubbletea/v2"

	"github.com/the-omics-os/lobster-tui/internal/protocol"
)

// formResult is returned after the suspended form execution completes.
type formResult struct {
	id     string
	values map[string]string
	err    error
}

// formExecCmd implements tea.ExecCommand to run a huh form in the raw terminal.
type formExecCmd struct {
	payload protocol.FormPayload
	msgID   string
	result  *formResult // shared pointer for result capture
}

func (f *formExecCmd) SetStdin(_ io.Reader)  {}
func (f *formExecCmd) SetStdout(_ io.Writer) {}
func (f *formExecCmd) SetStderr(_ io.Writer) {}

func (f *formExecCmd) Run() error {
	values := make(map[string]*string)
	var fields []huh.Field

	for _, field := range f.payload.Fields {
		val := field.Default
		values[field.Key] = &val

		switch field.Type {
		case "text", "":
			input := huh.NewInput().
				Title(field.Label).
				Value(&val)
			if field.Description != "" {
				input = input.Description(field.Description)
			}
			fields = append(fields, input)

		case "password":
			input := huh.NewInput().
				Title(field.Label).
				Value(&val).
				EchoMode(huh.EchoModePassword)
			if field.Description != "" {
				input = input.Description(field.Description)
			}
			fields = append(fields, input)

		case "select":
			opts := make([]huh.Option[string], len(field.Options))
			for i, opt := range field.Options {
				opts[i] = huh.NewOption(opt, opt)
			}
			sel := huh.NewSelect[string]().
				Title(field.Label).
				Options(opts...).
				Value(&val)
			if field.Description != "" {
				sel = sel.Description(field.Description)
			}
			fields = append(fields, sel)

		case "confirm":
			boolVal := field.Default == "true"
			confirm := huh.NewConfirm().
				Title(field.Label).
				Value(&boolVal)
			if field.Description != "" {
				confirm = confirm.Description(field.Description)
			}
			fields = append(fields, confirm)

		default:
			input := huh.NewInput().
				Title(field.Label).
				Value(&val)
			fields = append(fields, input)
		}
	}

	if f.payload.Title != "" {
		fmt.Fprintf(os.Stderr, "\n%s\n\n", f.payload.Title)
	}

	form := huh.NewForm(huh.NewGroup(fields...))
	if err := form.Run(); err != nil {
		f.result.err = err
		return err
	}

	collected := make(map[string]string, len(values))
	for k, v := range values {
		collected[k] = *v
	}
	f.result.values = collected
	return nil
}

// runFormSuspended returns a tea.Cmd that suspends the TUI, runs a huh form
// in the raw terminal, and returns the results as a formResult message.
func (m Model) runFormSuspended(payload protocol.FormPayload, msgID string) tea.Cmd {
	result := &formResult{id: msgID}
	cmd := &formExecCmd{
		payload: payload,
		msgID:   msgID,
		result:  result,
	}
	return tea.Exec(cmd, func(err error) tea.Msg {
		if err != nil {
			result.err = err
		}
		return *result
	})
}
