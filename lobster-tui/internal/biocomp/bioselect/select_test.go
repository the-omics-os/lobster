package bioselect

import (
	"encoding/json"
	"testing"

	tea "charm.land/bubbletea/v2"
)

func makeData(question string, options []string, def int) json.RawMessage {
	d, _ := json.Marshal(map[string]any{
		"question": question,
		"options":  options,
		"default":  def,
	})
	return d
}

func TestInitValid(t *testing.T) {
	t.Parallel()
	s := &SelectComponent{}
	err := s.Init(makeData("Pick one:", []string{"A", "B", "C"}, 1))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if s.question != "Pick one:" {
		t.Fatalf("expected question 'Pick one:', got %q", s.question)
	}
	if s.cursor != 1 {
		t.Fatalf("expected cursor at 1, got %d", s.cursor)
	}
	if len(s.options) != 3 {
		t.Fatalf("expected 3 options, got %d", len(s.options))
	}
}

func TestInitEmptyOptions(t *testing.T) {
	t.Parallel()
	s := &SelectComponent{}
	err := s.Init(makeData("Pick:", []string{}, 0))
	if err == nil {
		t.Fatal("expected error for empty options")
	}
}

func TestInitEmptyQuestion(t *testing.T) {
	t.Parallel()
	s := &SelectComponent{}
	err := s.Init(makeData("", []string{"A"}, 0))
	if err == nil {
		t.Fatal("expected error for empty question")
	}
}

func TestInitInvalidJSON(t *testing.T) {
	t.Parallel()
	s := &SelectComponent{}
	err := s.Init(json.RawMessage(`not json`))
	if err == nil {
		t.Fatal("expected error for invalid JSON")
	}
}

func TestInitDefaultOutOfRange(t *testing.T) {
	t.Parallel()
	s := &SelectComponent{}
	err := s.Init(makeData("Pick:", []string{"A", "B"}, 99))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if s.cursor != 0 {
		t.Fatalf("expected cursor clamped to 0, got %d", s.cursor)
	}
}

func TestNavigateDown(t *testing.T) {
	t.Parallel()
	s := &SelectComponent{}
	_ = s.Init(makeData("Pick:", []string{"A", "B", "C"}, 0))

	result := s.HandleMsg(tea.KeyPressMsg{Code: tea.KeyDown})
	if result != nil {
		t.Fatal("expected nil result during navigation")
	}
	if s.cursor != 1 {
		t.Fatalf("expected cursor at 1, got %d", s.cursor)
	}

	// Navigate with j.
	result = s.HandleMsg(tea.KeyPressMsg{Code: 'j', Text: "j"})
	if result != nil {
		t.Fatal("expected nil result during j navigation")
	}
	if s.cursor != 2 {
		t.Fatalf("expected cursor at 2, got %d", s.cursor)
	}

	// Past end should stay at last.
	s.HandleMsg(tea.KeyPressMsg{Code: tea.KeyDown})
	if s.cursor != 2 {
		t.Fatalf("expected cursor to stay at 2, got %d", s.cursor)
	}
}

func TestNavigateUp(t *testing.T) {
	t.Parallel()
	s := &SelectComponent{}
	_ = s.Init(makeData("Pick:", []string{"A", "B", "C"}, 2))

	result := s.HandleMsg(tea.KeyPressMsg{Code: tea.KeyUp})
	if result != nil {
		t.Fatal("expected nil result during navigation")
	}
	if s.cursor != 1 {
		t.Fatalf("expected cursor at 1, got %d", s.cursor)
	}

	// Navigate with k.
	s.HandleMsg(tea.KeyPressMsg{Code: 'k', Text: "k"})
	if s.cursor != 0 {
		t.Fatalf("expected cursor at 0, got %d", s.cursor)
	}

	// Past beginning should stay at 0.
	s.HandleMsg(tea.KeyPressMsg{Code: tea.KeyUp})
	if s.cursor != 0 {
		t.Fatalf("expected cursor to stay at 0, got %d", s.cursor)
	}
}

func TestSubmitEnter(t *testing.T) {
	t.Parallel()
	s := &SelectComponent{}
	_ = s.Init(makeData("Pick:", []string{"Alpha", "Beta"}, 1))

	result := s.HandleMsg(tea.KeyPressMsg{Code: tea.KeyEnter})
	if result == nil {
		t.Fatal("expected non-nil result on enter")
	}
	if result.Action != "submit" {
		t.Fatalf("expected action 'submit', got %q", result.Action)
	}
	if sel, _ := result.Data["selected"].(string); sel != "Beta" {
		t.Fatalf("expected selected 'Beta', got %q", sel)
	}
	if idx, _ := result.Data["index"].(int); idx != 1 {
		t.Fatalf("expected index 1, got %v", result.Data["index"])
	}
}

func TestNumberKeyDirectSelect(t *testing.T) {
	t.Parallel()
	s := &SelectComponent{}
	_ = s.Init(makeData("Pick:", []string{"A", "B", "C"}, 0))

	result := s.HandleMsg(tea.KeyPressMsg{Code: '2', Text: "2"})
	if result == nil {
		t.Fatal("expected non-nil result on number key")
	}
	if sel, _ := result.Data["selected"].(string); sel != "B" {
		t.Fatalf("expected selected 'B', got %q", sel)
	}
	if idx, _ := result.Data["index"].(int); idx != 1 {
		t.Fatalf("expected index 1, got %v", result.Data["index"])
	}
}

func TestNumberKeyIgnoredWhenMoreThan9Options(t *testing.T) {
	t.Parallel()
	opts := make([]string, 10)
	for i := range opts {
		opts[i] = "option"
	}
	s := &SelectComponent{}
	_ = s.Init(makeData("Pick:", opts, 0))

	result := s.HandleMsg(tea.KeyPressMsg{Code: '1', Text: "1"})
	if result != nil {
		t.Fatal("expected nil result when >9 options")
	}
}

func TestCancelEsc(t *testing.T) {
	t.Parallel()
	s := &SelectComponent{}
	_ = s.Init(makeData("Pick:", []string{"A"}, 0))

	result := s.HandleMsg(tea.KeyPressMsg{Code: tea.KeyEscape})
	if result == nil {
		t.Fatal("expected non-nil result on esc")
	}
	if result.Action != "cancel" {
		t.Fatalf("expected action 'cancel', got %q", result.Action)
	}
}

func TestViewNonEmpty(t *testing.T) {
	t.Parallel()
	s := &SelectComponent{}
	_ = s.Init(makeData("Pick:", []string{"A", "B"}, 0))
	v := s.View(60, 20)
	if v == "" {
		t.Fatal("expected non-empty view")
	}
}

func TestViewNarrowWidth(t *testing.T) {
	t.Parallel()
	s := &SelectComponent{}
	_ = s.Init(makeData("Pick:", []string{"A", "B"}, 0))
	v := s.View(20, 10)
	if v == "" {
		t.Fatal("expected non-empty view at narrow width")
	}
}

func TestNameAndMode(t *testing.T) {
	t.Parallel()
	s := &SelectComponent{}
	if s.Name() != "select" {
		t.Fatalf("expected name 'select', got %q", s.Name())
	}
	if s.Mode() != "overlay" {
		t.Fatalf("expected mode 'overlay', got %q", s.Mode())
	}
}

func TestKeyBindingsNotEmpty(t *testing.T) {
	t.Parallel()
	s := &SelectComponent{}
	if len(s.KeyBindings()) == 0 {
		t.Fatal("expected non-empty key bindings")
	}
}

func TestSetDataUpdatesOptions(t *testing.T) {
	t.Parallel()
	s := &SelectComponent{}
	_ = s.Init(makeData("Pick:", []string{"A", "B"}, 0))

	newData, _ := json.Marshal(map[string]any{
		"question": "New question:",
		"options":  []string{"X", "Y", "Z"},
	})
	err := s.SetData(newData)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if s.question != "New question:" {
		t.Fatalf("expected updated question, got %q", s.question)
	}
	if len(s.options) != 3 {
		t.Fatalf("expected 3 options after update, got %d", len(s.options))
	}
}
