// Package spike validates Charm v2 API assumptions in isolation before
// committing to full migration. These tests run independently of production
// code and prove that v2 packages behave as documented.
//
// If any test fails, the migration approach must be revised before touching
// the 28 production files that currently use v1.
package spike

import (
	"image/color"
	"strings"
	"testing"

	tea "charm.land/bubbletea/v2"
	"charm.land/bubbles/v2/viewport"
	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/glamour"
)

// --- Test 1: View type ---

// minimalModel is a minimal tea.Model for v2 API validation.
type minimalModel struct {
	text string
}

func (m minimalModel) Init() tea.Cmd { return nil }

func (m minimalModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	return m, nil
}

func (m minimalModel) View() tea.View {
	return tea.NewView(m.text)
}

func TestV2ViewType(t *testing.T) {
	// Compile-time check: minimalModel satisfies tea.Model interface.
	var _ tea.Model = minimalModel{}

	m := minimalModel{text: "hello"}
	v := m.View()

	// View has a Content field (string), not a Body() method.
	if v.Content == "" {
		t.Fatal("View.Content is empty")
	}
	if v.Content != "hello" {
		t.Fatalf("View.Content = %q, want %q", v.Content, "hello")
	}
}

// --- Test 2: Color type ---

func TestV2ColorType(t *testing.T) {
	// lipgloss.Color() is now a function returning color.Color.
	c := lipgloss.Color("#e45c47")

	// Must be assignable to the standard library color.Color interface.
	var _ color.Color = c

	// Create a style with the color and render -- must not panic.
	s := lipgloss.NewStyle().Foreground(lipgloss.Color("#e45c47"))
	result := s.Render("test")
	if result == "" {
		t.Fatal("Style.Render returned empty string")
	}
}

// --- Test 3: KeyPressMsg ---

func TestV2KeyPressMsg(t *testing.T) {
	tests := []struct {
		name string
		msg  tea.KeyPressMsg
		want []string // accept any of these strings
	}{
		{
			name: "enter",
			msg:  tea.KeyPressMsg{Code: tea.KeyEnter},
			want: []string{"enter"},
		},
		{
			name: "ctrl+c",
			msg:  tea.KeyPressMsg{Code: 'c', Mod: tea.ModCtrl},
			want: []string{"ctrl+c"},
		},
		{
			name: "regular char a",
			msg:  tea.KeyPressMsg{Code: 'a', Text: "a"},
			want: []string{"a"},
		},
		{
			name: "escape",
			msg:  tea.KeyPressMsg{Code: tea.KeyEscape},
			want: []string{"escape", "esc"},
		},
		{
			name: "space",
			msg:  tea.KeyPressMsg{Code: tea.KeySpace},
			want: []string{"space", " "},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.msg.String()
			matched := false
			for _, w := range tc.want {
				if got == w {
					matched = true
					break
				}
			}
			if !matched {
				t.Errorf("KeyPressMsg.String() = %q, want one of %v", got, tc.want)
			}
			t.Logf("KeyPressMsg(%s).String() = %q", tc.name, got)
		})
	}
}

// --- Test 4: Viewport basic ---

func TestV2ViewportBasic(t *testing.T) {
	vp := viewport.New(viewport.WithWidth(40), viewport.WithHeight(10))

	content := strings.Repeat("line\n", 20)
	vp.SetContent(content)

	out := vp.View()
	if out == "" {
		t.Fatal("viewport.View() returned empty string")
	}
}

// --- Test 5: Glamour + Lipgloss v2 coexistence ---

func TestV2GlamourCoexistence(t *testing.T) {
	// glamour is v1 (github.com/charmbracelet/glamour), uses lipgloss v1 internally.
	// lipgloss v2 (charm.land/lipgloss/v2) is used directly by us.
	// Both must work in the same binary without module conflicts.

	// Render markdown with glamour (v1 dependency chain).
	r, err := glamour.NewTermRenderer(glamour.WithAutoStyle())
	if err != nil {
		t.Fatalf("glamour.NewTermRenderer failed: %v", err)
	}
	md, err := r.Render("# Hello\n\nworld")
	if err != nil {
		t.Fatalf("glamour.Render failed: %v", err)
	}
	if md == "" {
		t.Fatal("glamour rendered empty output")
	}

	// Create a lipgloss v2 style in the same test.
	s := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#ff0000")).
		Bold(true)
	result := s.Render("styled text")
	if result == "" {
		t.Fatal("lipgloss v2 Render returned empty string")
	}

	// Both worked -- no module conflict.
}

// --- Test 6: Program options moved to View fields ---

func TestV2ProgramOptions(t *testing.T) {
	// In v2, WithAltScreen and WithMouseCellMotion are replaced by View fields.
	// Verify tea.NewView supports AltScreen and MouseMode fields.
	v := tea.NewView("content")
	v.AltScreen = true
	v.MouseMode = tea.MouseModeCellMotion

	if !v.AltScreen {
		t.Fatal("AltScreen field not settable on View")
	}
	if v.MouseMode != tea.MouseModeCellMotion {
		t.Fatal("MouseMode field not settable on View")
	}
}
