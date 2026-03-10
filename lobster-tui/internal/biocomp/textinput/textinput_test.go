package textinput

import (
	"encoding/json"
	"testing"

	tea "charm.land/bubbletea/v2"
)

func makeData(question, placeholder string) json.RawMessage {
	d, _ := json.Marshal(map[string]any{
		"question":    question,
		"placeholder": placeholder,
	})
	return d
}

func TestInitValid(t *testing.T) {
	t.Parallel()
	c := &TextInputComponent{}
	err := c.Init(makeData("Your name?", "Enter name..."))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if c.question != "Your name?" {
		t.Fatalf("expected question 'Your name?', got %q", c.question)
	}
}

func TestInitEmptyQuestion(t *testing.T) {
	t.Parallel()
	c := &TextInputComponent{}
	err := c.Init(makeData("", "placeholder"))
	if err == nil {
		t.Fatal("expected error for empty question")
	}
}

func TestInitInvalidJSON(t *testing.T) {
	t.Parallel()
	c := &TextInputComponent{}
	err := c.Init(json.RawMessage(`{bad`))
	if err == nil {
		t.Fatal("expected error for invalid JSON")
	}
}

func TestSubmitEnter(t *testing.T) {
	t.Parallel()
	c := &TextInputComponent{}
	_ = c.Init(makeData("Name?", ""))

	// Type some characters via rune key messages.
	c.HandleMsg(tea.KeyPressMsg{Code: 'H', Text: "H"})
	c.HandleMsg(tea.KeyPressMsg{Code: 'i', Text: "i"})

	result, _ := c.HandleMsg(tea.KeyPressMsg{Code: tea.KeyEnter})
	if result == nil {
		t.Fatal("expected non-nil result on enter")
	}
	if result.Action != "submit" {
		t.Fatalf("expected action 'submit', got %q", result.Action)
	}
	// The answer should contain what was typed.
	answer, ok := result.Data["answer"].(string)
	if !ok {
		t.Fatal("expected answer to be a string")
	}
	if answer != "Hi" {
		t.Fatalf("expected answer 'Hi', got %q", answer)
	}
}

func TestCancelEsc(t *testing.T) {
	t.Parallel()
	c := &TextInputComponent{}
	_ = c.Init(makeData("Name?", ""))

	result, _ := c.HandleMsg(tea.KeyPressMsg{Code: tea.KeyEscape})
	if result == nil {
		t.Fatal("expected non-nil result on esc")
	}
	if result.Action != "cancel" {
		t.Fatalf("expected action 'cancel', got %q", result.Action)
	}
}

func TestViewNonEmpty(t *testing.T) {
	t.Parallel()
	c := &TextInputComponent{}
	_ = c.Init(makeData("Name?", "Enter name"))
	v := c.View(60, 10)
	if v == "" {
		t.Fatal("expected non-empty view")
	}
}

func TestNameAndMode(t *testing.T) {
	t.Parallel()
	c := &TextInputComponent{}
	if c.Name() != "text_input" {
		t.Fatalf("expected name 'text_input', got %q", c.Name())
	}
	if c.Mode() != "overlay" {
		t.Fatalf("expected mode 'overlay', got %q", c.Mode())
	}
}

func TestKeyBindingsNotEmpty(t *testing.T) {
	t.Parallel()
	c := &TextInputComponent{}
	if len(c.KeyBindings()) == 0 {
		t.Fatal("expected non-empty key bindings")
	}
}

func TestChangeEventNil(t *testing.T) {
	t.Parallel()
	c := &TextInputComponent{}
	if c.ChangeEvent() != nil {
		t.Fatal("expected nil change event")
	}
}

func TestSetDataUpdatesQuestion(t *testing.T) {
	t.Parallel()
	c := &TextInputComponent{}
	_ = c.Init(makeData("Name?", ""))

	newData, _ := json.Marshal(map[string]any{
		"question":    "Full name?",
		"placeholder": "First Last",
	})
	err := c.SetData(newData)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if c.question != "Full name?" {
		t.Fatalf("expected updated question, got %q", c.question)
	}
}
