package theme

import (
	"reflect"
	"testing"

	"charm.land/lipgloss/v2"
)

// isZeroStyle checks if a style is effectively the zero-value style by comparing
// rendered output of an empty string. lipgloss.Style contains slices so cannot
// be compared with ==.
func isZeroStyle(s lipgloss.Style) bool {
	return s.Render("x") == lipgloss.NewStyle().Render("x")
}

// TestStyleTokenCount verifies the Styles struct has at least 40 exported
// lipgloss.Style fields (semantic tokens).
func TestStyleTokenCount(t *testing.T) {
	rt := reflect.TypeOf(Styles{})
	styleType := reflect.TypeOf(lipgloss.Style{})
	count := 0
	for i := 0; i < rt.NumField(); i++ {
		f := rt.Field(i)
		if f.IsExported() && f.Type == styleType {
			count++
		}
	}
	if count < 40 {
		t.Errorf("Styles struct has %d exported lipgloss.Style fields, want >= 40", count)
	}
}

// TestTableStyles verifies that BuildStyles produces non-zero table tokens.
func TestTableStyles(t *testing.T) {
	s := BuildStyles(LobsterDark.Colors)

	for _, tc := range []struct {
		name  string
		style lipgloss.Style
	}{
		{"TableHeader", s.TableHeader},
		{"TableRowEven", s.TableRowEven},
		{"TableRowOdd", s.TableRowOdd},
		{"TableBorder", s.TableBorder},
	} {
		if isZeroStyle(tc.style) {
			t.Errorf("%s is zero-value style", tc.name)
		}
	}
}

// TestCodeStyles verifies CodeBlock and CodeLabel fields exist and are non-zero.
func TestCodeStyles(t *testing.T) {
	s := BuildStyles(LobsterDark.Colors)

	if isZeroStyle(s.CodeBlock) {
		t.Error("CodeBlock is zero-value style")
	}
	if isZeroStyle(s.CodeLabel) {
		t.Error("CodeLabel is zero-value style")
	}
}

// TestFooterStyles verifies footer group tokens exist and are non-zero.
func TestFooterStyles(t *testing.T) {
	s := BuildStyles(LobsterDark.Colors)

	for _, tc := range []struct {
		name  string
		style lipgloss.Style
	}{
		{"FooterStatus", s.FooterStatus},
		{"FooterToolFeed", s.FooterToolFeed},
		{"FooterComponentFrame", s.FooterComponentFrame},
	} {
		if isZeroStyle(tc.style) {
			t.Errorf("%s is zero-value style", tc.name)
		}
	}
}

// TestChatStyles verifies chat group tokens that use accent/brand colors are
// non-zero. MessageBody intentionally uses nil (terminal default) foreground
// so it is excluded from the non-zero check.
func TestChatStyles(t *testing.T) {
	s := BuildStyles(LobsterDark.Colors)

	for _, tc := range []struct {
		name  string
		style lipgloss.Style
	}{
		{"AgentName", s.AgentName},
		{"UserName", s.UserName},
		{"HandoffPrefix", s.HandoffPrefix},
	} {
		if isZeroStyle(tc.style) {
			t.Errorf("%s is zero-value style", tc.name)
		}
	}
}

// TestAlertStyles verifies AlertSuccess, AlertWarning, AlertError, AlertInfo
// style fields have non-zero foreground color (STYL-03).
func TestAlertStyles(t *testing.T) {
	s := BuildStyles(LobsterDark.Colors)

	for _, tc := range []struct {
		name  string
		style lipgloss.Style
	}{
		{"AlertSuccess", s.AlertSuccess},
		{"AlertWarning", s.AlertWarning},
		{"AlertError", s.AlertError},
		{"AlertInfo", s.AlertInfo},
	} {
		if isZeroStyle(tc.style) {
			t.Errorf("%s is zero-value style", tc.name)
		}
		// Verify each has a non-nil foreground color set
		fg := tc.style.GetForeground()
		if fg == nil {
			t.Errorf("%s has nil foreground color", tc.name)
		}
	}
}

// TestCrushStyleMessages verifies that UserMessage uses border-left (not full
// box border) and AssistantMessage has padding but no border at all.
func TestCrushStyleMessages(t *testing.T) {
	s := BuildStyles(LobsterDark.Colors)

	// UserMessage: should have left border but NOT top/right/bottom
	if !s.UserMessage.GetBorderLeft() {
		t.Error("UserMessage should have BorderLeft=true")
	}
	if s.UserMessage.GetBorderTop() {
		t.Error("UserMessage should NOT have BorderTop (crush-style)")
	}
	if s.UserMessage.GetBorderRight() {
		t.Error("UserMessage should NOT have BorderRight (crush-style)")
	}
	if s.UserMessage.GetBorderBottom() {
		t.Error("UserMessage should NOT have BorderBottom (crush-style)")
	}

	// AssistantMessage: no border at all, just padding
	if s.AssistantMessage.GetBorderTop() {
		t.Error("AssistantMessage should NOT have BorderTop (crush-style)")
	}
	if s.AssistantMessage.GetBorderLeft() {
		t.Error("AssistantMessage should NOT have BorderLeft (crush-style)")
	}
	if s.AssistantMessage.GetBorderRight() {
		t.Error("AssistantMessage should NOT have BorderRight (crush-style)")
	}
	if s.AssistantMessage.GetBorderBottom() {
		t.Error("AssistantMessage should NOT have BorderBottom (crush-style)")
	}
}
