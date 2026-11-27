package initwizard

import (
	"strings"
	"testing"

	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"
)

// The renderer cuts a line wider than the window, so a long question must wrap.
func assertFits(t *testing.T, view string, width int) {
	t.Helper()
	for _, line := range strings.Split(view, "\n") {
		if w := lipgloss.Width(line); w > width {
			t.Fatalf("a %d-column line in %d columns: %q", w, width, ansi.Strip(line))
		}
	}
	flat := strings.Join(strings.Fields(ansi.Strip(view)), " ")
	if !strings.Contains(flat, "not set up on this machine.") {
		t.Fatalf("the end of the LaminDB question is missing: %q", flat)
	}
}

func TestLaminQuestionWrapsToTheWindowWidth(t *testing.T) {
	m := press(t, laminModel(t, "0"), "n")
	model, _ := m.Update(tea.WindowSizeMsg{Width: 50, Height: 40})
	m = model.(WizardModel)
	if m.optionalSub != subStepLaminConfirm || m.done {
		t.Fatal("a window size message must not change the step")
	}
	assertFits(t, m.View().Content, 50)
}

func TestLaminQuestionWrapsBeforeAnyWindowSize(t *testing.T) {
	m := press(t, laminModel(t, "0"), "n")
	assertFits(t, m.View().Content, accessNoticeWidth)
}
