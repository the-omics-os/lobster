package confirm

import (
	"encoding/json"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
)

func makeData(question string, def bool) json.RawMessage {
	d, _ := json.Marshal(map[string]any{
		"question": question,
		"default":  def,
	})
	return d
}

func TestInitValid(t *testing.T) {
	t.Parallel()
	c := &ConfirmComponent{}
	if err := c.Init(makeData("Continue?", true)); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if c.question != "Continue?" {
		t.Fatalf("expected question 'Continue?', got %q", c.question)
	}
	if !c.selected {
		t.Fatal("expected default selected=true")
	}
}

func TestInitEmptyQuestion(t *testing.T) {
	t.Parallel()
	c := &ConfirmComponent{}
	err := c.Init(makeData("", false))
	if err == nil {
		t.Fatal("expected error for empty question")
	}
}

func TestInitInvalidJSON(t *testing.T) {
	t.Parallel()
	c := &ConfirmComponent{}
	err := c.Init(json.RawMessage(`{invalid`))
	if err == nil {
		t.Fatal("expected error for invalid JSON")
	}
}

func TestToggleLeftRight(t *testing.T) {
	t.Parallel()
	c := &ConfirmComponent{}
	_ = c.Init(makeData("Test?", true))

	// Toggle with left key.
	result := c.HandleMsg(tea.KeyMsg{Type: tea.KeyLeft})
	if result != nil {
		t.Fatal("expected nil result during navigation")
	}
	if c.selected != false {
		t.Fatal("expected selected to toggle to false")
	}

	// Toggle with right key.
	result = c.HandleMsg(tea.KeyMsg{Type: tea.KeyRight})
	if result != nil {
		t.Fatal("expected nil result during navigation")
	}
	if c.selected != true {
		t.Fatal("expected selected to toggle back to true")
	}

	// Toggle with tab.
	result = c.HandleMsg(tea.KeyMsg{Type: tea.KeyTab})
	if result != nil {
		t.Fatal("expected nil result during navigation")
	}
	if c.selected != false {
		t.Fatal("expected selected to toggle to false via tab")
	}
}

func TestSubmitEnter(t *testing.T) {
	t.Parallel()
	c := &ConfirmComponent{}
	_ = c.Init(makeData("Proceed?", true))

	result := c.HandleMsg(tea.KeyMsg{Type: tea.KeyEnter})
	if result == nil {
		t.Fatal("expected non-nil result on enter")
	}
	if result.Action != "submit" {
		t.Fatalf("expected action 'submit', got %q", result.Action)
	}
	confirmed, ok := result.Data["confirmed"].(bool)
	if !ok || !confirmed {
		t.Fatal("expected confirmed=true")
	}
}

func TestSubmitYKey(t *testing.T) {
	t.Parallel()
	c := &ConfirmComponent{}
	_ = c.Init(makeData("Continue?", false))

	result := c.HandleMsg(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'y'}})
	if result == nil {
		t.Fatal("expected non-nil result on y key")
	}
	if result.Action != "submit" {
		t.Fatalf("expected action 'submit', got %q", result.Action)
	}
	if confirmed, _ := result.Data["confirmed"].(bool); !confirmed {
		t.Fatal("expected confirmed=true for y key")
	}
}

func TestSubmitNKey(t *testing.T) {
	t.Parallel()
	c := &ConfirmComponent{}
	_ = c.Init(makeData("Continue?", true))

	result := c.HandleMsg(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'n'}})
	if result == nil {
		t.Fatal("expected non-nil result on n key")
	}
	if confirmed, _ := result.Data["confirmed"].(bool); confirmed {
		t.Fatal("expected confirmed=false for n key")
	}
}

func TestCancelEsc(t *testing.T) {
	t.Parallel()
	c := &ConfirmComponent{}
	_ = c.Init(makeData("Continue?", true))

	result := c.HandleMsg(tea.KeyMsg{Type: tea.KeyEsc})
	if result == nil {
		t.Fatal("expected non-nil result on esc")
	}
	if result.Action != "cancel" {
		t.Fatalf("expected action 'cancel', got %q", result.Action)
	}
}

func TestViewNonEmpty(t *testing.T) {
	t.Parallel()
	c := &ConfirmComponent{}
	_ = c.Init(makeData("Continue?", true))
	v := c.View(60, 10)
	if v == "" {
		t.Fatal("expected non-empty view")
	}
}

func TestNameAndMode(t *testing.T) {
	t.Parallel()
	c := &ConfirmComponent{}
	if c.Name() != "confirm" {
		t.Fatalf("expected name 'confirm', got %q", c.Name())
	}
	if c.Mode() != "overlay" {
		t.Fatalf("expected mode 'overlay', got %q", c.Mode())
	}
}

func TestKeyBindingsNotEmpty(t *testing.T) {
	t.Parallel()
	c := &ConfirmComponent{}
	if len(c.KeyBindings()) == 0 {
		t.Fatal("expected non-empty key bindings")
	}
}

func TestChangeEventNil(t *testing.T) {
	t.Parallel()
	c := &ConfirmComponent{}
	if c.ChangeEvent() != nil {
		t.Fatal("expected nil change event")
	}
}
