package biocomp

import (
	"encoding/json"
	"testing"

	"charm.land/bubbles/v2/key"
	tea "charm.land/bubbletea/v2"
)

// stubComponent is a minimal BioComponent for registry tests.
type stubComponent struct{ name string }

func (s *stubComponent) Init(json.RawMessage) error         { return nil }
func (s *stubComponent) HandleMsg(tea.Msg) *ComponentResult { return nil }
func (s *stubComponent) View(int, int) string               { return "" }
func (s *stubComponent) SetData(json.RawMessage) error      { return nil }
func (s *stubComponent) Name() string                       { return s.name }
func (s *stubComponent) Mode() string                       { return "overlay" }
func (s *stubComponent) KeyBindings() []key.Binding         { return nil }
func (s *stubComponent) ChangeEvent() map[string]any        { return nil }

func TestRegisterAndGet(t *testing.T) {
	t.Parallel()

	// Save and restore registry state.
	orig := registry
	registry = map[string]Factory{}
	defer func() { registry = orig }()

	Register("test_comp", func() BioComponent {
		return &stubComponent{name: "test_comp"}
	})

	f := Get("test_comp")
	if f == nil {
		t.Fatal("expected factory to be registered")
	}

	comp := f()
	if comp.Name() != "test_comp" {
		t.Fatalf("expected name 'test_comp', got %q", comp.Name())
	}
}

func TestGetUnknownReturnsNil(t *testing.T) {
	t.Parallel()
	if Get("nonexistent_xyz") != nil {
		t.Fatal("expected nil for unregistered component")
	}
}

func TestNames(t *testing.T) {
	orig := registry
	registry = map[string]Factory{}
	defer func() { registry = orig }()

	Register("alpha", func() BioComponent { return &stubComponent{name: "alpha"} })
	Register("beta", func() BioComponent { return &stubComponent{name: "beta"} })

	names := Names()
	if len(names) != 2 {
		t.Fatalf("expected 2 names, got %d", len(names))
	}

	found := map[string]bool{}
	for _, n := range names {
		found[n] = true
	}
	if !found["alpha"] || !found["beta"] {
		t.Fatalf("expected alpha and beta, got %v", names)
	}
}
