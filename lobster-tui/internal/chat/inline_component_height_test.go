package chat

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/the-omics-os/lobster-tui/internal/biocomp"
)

// tallChoiceList builds a choice_list with n rows, which fills whatever
// height it is given.
func tallChoiceList(t *testing.T, n int) biocomp.BioComponent {
	t.Helper()
	items := make([]map[string]string, n)
	for i := range items {
		items[i] = map[string]string{"id": fmt.Sprintf("S%02d", i+1), "label": fmt.Sprintf("Sample %02d", i+1)}
	}
	data, err := json.Marshal(map[string]any{
		"question": "Which group is each sample in?",
		"items":    items,
		"options": []map[string]string{
			{"id": "control", "label": "control"},
			{"id": "treated", "label": "treated"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	comp := biocomp.Get("choice_list")()
	if err := comp.Init(data); err != nil {
		t.Fatalf("init choice_list: %v", err)
	}
	return comp
}

// In inline mode (the launcher's default) a tall component used to get the
// whole terminal height, and the composer and status line were drawn under
// it, so the frame was taller than the terminal: the question scrolled off
// the top and the renderer repainted without pause.
func TestInlineTallComponentFrameFitsTheTerminal(t *testing.T) {
	for _, height := range []int{12, 20, 40} {
		h, _ := newCapturingHandler()
		m := newTestModelWithHandler(h)
		m.inline = true
		m.inlineBannerPrinted = true
		m.height = height
		m.messages = append(m.messages, ChatMessage{Role: "user"})
		m.activeComponent = &ActiveComponent{Component: tallChoiceList(t, 72), MsgID: "q1"}
		m.recalculateViewportHeight()

		frame := m.View().Content
		if got := strings.Count(frame, "\n") + 1; got > height {
			t.Errorf("height %d: inline frame is %d lines, want at most %d:\n%s", height, got, height, frame)
		}
		if !strings.Contains(frame, "Which group is each sample in?") {
			t.Errorf("height %d: question line missing from the inline frame:\n%s", height, frame)
		}
	}
}

// Components that do not fill the terminal keep the full height in inline mode.
func TestInlineComponentHeightUnchangedForOtherComponents(t *testing.T) {
	m := testModel(40, 80)
	m.inline = true
	m.activeComponent = &ActiveComponent{Component: &layoutMockComponent{name: "threshold_slider", mode: "inline"}}
	if got := m.inlineComponentHeight(); got != 40 {
		t.Errorf("threshold_slider: inline height = %d, want 40 (unchanged)", got)
	}
	m.activeComponent = &ActiveComponent{Component: &layoutMockComponent{name: "choice_list", mode: "overlay"}}
	if got, want := m.inlineComponentHeight(), m.componentFooterHeight(); got != want {
		t.Errorf("choice_list: inline height = %d, want the footer budget %d", got, want)
	}
}
