package chat

import (
	"testing"

	"github.com/the-omics-os/lobster-tui/internal/theme"
)

// testModel returns a minimal Model with only layout-relevant fields set.
func testModel(height, width int) Model {
	colors := theme.Colors{
		Primary:    nil,
		Secondary:  nil,
		Background: nil,
		Surface:    nil,
		Overlay:    nil,
		Text:       nil,
		TextMuted:  nil,
		TextDim:    nil,
	}
	styles := theme.BuildStyles(colors)
	m := Model{
		height: height,
		width:  width,
		styles: styles,
	}
	return m
}

func TestComputeLayout_HeightsSum(t *testing.T) {
	m := testModel(40, 80)
	layout := m.computeLayout()
	sum := layout.HeaderHeight + layout.ViewportHeight + layout.InputHeight + layout.FooterHeight
	if sum != 40 {
		t.Errorf("heights sum = %d, want 40 (header=%d viewport=%d input=%d footer=%d)",
			sum, layout.HeaderHeight, layout.ViewportHeight, layout.InputHeight, layout.FooterHeight)
	}
}

func TestComputeLayout_FooterModeStatus(t *testing.T) {
	m := testModel(40, 80)
	mode := m.footerMode()
	if mode != FooterModeStatus {
		t.Errorf("footerMode() = %d, want FooterModeStatus (%d)", mode, FooterModeStatus)
	}
	fh := m.footerHeight()
	if fh != 1 {
		t.Errorf("footerHeight() = %d, want 1 for status mode", fh)
	}
}

func TestComputeLayout_FooterModeToolFeed(t *testing.T) {
	m := testModel(40, 80)
	m.toolFeed = []ToolFeedEntry{
		{Name: "tool1", Event: "start"},
		{Name: "tool2", Event: "finish"},
		{Name: "tool3", Event: "error"},
	}
	mode := m.footerMode()
	if mode != FooterModeToolFeed {
		t.Errorf("footerMode() = %d, want FooterModeToolFeed (%d)", mode, FooterModeToolFeed)
	}
	fh := m.footerHeight()
	// 3 entries + 1 "Tools" header + 1 status line = 5
	if fh != 5 {
		t.Errorf("footerHeight() = %d, want 5 for 3 tool feed entries", fh)
	}
}

func TestComputeLayout_FooterModeComponent(t *testing.T) {
	m := testModel(40, 80)
	m.activeComponent = &ActiveComponent{}
	mode := m.footerMode()
	if mode != FooterModeComponent {
		t.Errorf("footerMode() = %d, want FooterModeComponent (%d)", mode, FooterModeComponent)
	}
}

func TestComputeLayout_InlineEarlyReturn(t *testing.T) {
	m := testModel(40, 80)
	m.inline = true
	layout := m.computeLayout()
	zero := Layout{}
	if layout != zero {
		t.Errorf("computeLayout() with inline=true should return zero Layout, got %+v", layout)
	}
}

func TestComputeLayout_ViewportGreedy(t *testing.T) {
	// Base case: no tool feed
	m1 := testModel(40, 80)
	l1 := m1.computeLayout()

	// With tool feed: footer grows, viewport shrinks proportionally
	m2 := testModel(40, 80)
	m2.toolFeed = []ToolFeedEntry{
		{Name: "tool1", Event: "start"},
		{Name: "tool2", Event: "finish"},
	}
	l2 := m2.computeLayout()

	footerDelta := l2.FooterHeight - l1.FooterHeight
	viewportDelta := l1.ViewportHeight - l2.ViewportHeight
	if footerDelta != viewportDelta {
		t.Errorf("footer grew by %d but viewport shrank by %d; should be equal", footerDelta, viewportDelta)
	}
}

func TestComputeLayout_ViewportMinimum(t *testing.T) {
	// Very small terminal with large tool feed -- viewport should never go below 1.
	m := testModel(5, 80)
	m.toolFeed = make([]ToolFeedEntry, 20) // Way more than the terminal can hold
	for i := range m.toolFeed {
		m.toolFeed[i] = ToolFeedEntry{Name: "tool", Event: "start"}
	}
	layout := m.computeLayout()
	if layout.ViewportHeight < 1 {
		t.Errorf("ViewportHeight = %d, want >= 1", layout.ViewportHeight)
	}
}

func TestFooterStatusMode(t *testing.T) {
	m := testModel(40, 80)
	layout := m.computeLayout()
	result := m.renderStatusFooter(layout)
	if result == "" {
		t.Error("renderStatusFooter() returned empty string")
	}
}

func TestToolFeedInFooter(t *testing.T) {
	m := testModel(40, 80)
	m.toolFeed = []ToolFeedEntry{
		{Name: "search_pubmed", Event: "start", Summary: "searching..."},
		{Name: "load_data", Event: "finish", Summary: "loaded 500 genes"},
	}
	layout := m.computeLayout()
	result := m.renderToolFeedFooter(layout)
	if result == "" {
		t.Error("renderToolFeedFooter() returned empty string")
	}
}
