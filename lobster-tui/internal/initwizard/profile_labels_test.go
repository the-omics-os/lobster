package initwizard

import (
	"strings"
	"testing"

	tea "charm.land/bubbletea/v2"
)

// The profile does not choose a model (the model step does), so no screen after
// the model step may offer other models in its labels.
func TestScreenAfterModelSelectNamesNoOtherModel(t *testing.T) {
	m := NewWizardModel("", "")

	// Anthropic (index 1), a key, then the default model.
	result, _ := m.Update(makeKeyPress("down"))
	m = result.(WizardModel)
	result, _ = m.Update(makeKeyPress("enter"))
	m = result.(WizardModel)
	for _, ch := range "sk-test" {
		result, _ = m.Update(tea.KeyPressMsg{Code: ch, Text: string(ch)})
		m = result.(WizardModel)
	}
	result, _ = m.Update(makeKeyPress("enter"))
	m = result.(WizardModel)
	result, _ = m.Update(makeKeyPress("enter"))
	m = result.(WizardModel)

	screen := render(m)
	for _, other := range []string{"Sonnet 4", "Opus 4.5", "Haiku"} {
		if strings.Contains(screen, other) {
			t.Fatalf("the screen after the model step offers %q:\n%s", other, screen)
		}
	}
}
