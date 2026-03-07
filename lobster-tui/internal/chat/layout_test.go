package chat

import (
	"encoding/json"
	"testing"

	"charm.land/bubbles/v2/key"
	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
	"github.com/the-omics-os/lobster-tui/internal/biocomp"
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

// layoutMockComponent is a minimal BioComponent for layout tests.
type layoutMockComponent struct {
	name string
	mode string
}

func (c *layoutMockComponent) Init(_ json.RawMessage) error                  { return nil }
func (c *layoutMockComponent) HandleMsg(_ tea.Msg) *biocomp.ComponentResult  { return nil }
func (c *layoutMockComponent) View(w, h int) string                          { return "mock content" }
func (c *layoutMockComponent) SetData(_ json.RawMessage) error               { return nil }
func (c *layoutMockComponent) Name() string                                  { return c.name }
func (c *layoutMockComponent) Mode() string                                  { return c.mode }
func (c *layoutMockComponent) KeyBindings() []key.Binding                    { return nil }
func (c *layoutMockComponent) ChangeEvent() map[string]any                   { return nil }

func TestFooterComponentExpand(t *testing.T) {
	m := testModel(40, 80)
	m.activeComponent = &ActiveComponent{
		Component: &layoutMockComponent{name: "threshold_slider", mode: "overlay"},
	}

	mode := m.footerMode()
	if mode != FooterModeComponent {
		t.Fatalf("footerMode() = %d, want FooterModeComponent", mode)
	}

	fh := m.footerHeight()
	// Component footer: base height + borders/help, clamped to min(height/2, 20)
	maxH := m.height / 2
	if maxH > 20 {
		maxH = 20
	}
	if fh > maxH {
		t.Errorf("footerHeight() = %d, exceeds max clamp %d", fh, maxH)
	}
	if fh < 5 {
		t.Errorf("footerHeight() = %d, want >= 5 minimum", fh)
	}

	// Heights must still sum correctly
	layout := m.computeLayout()
	sum := layout.HeaderHeight + layout.ViewportHeight + layout.InputHeight + layout.FooterHeight
	if sum != 40 {
		t.Errorf("heights sum = %d, want 40", sum)
	}

	// renderFooterRegion should dispatch to component footer (not empty)
	result := m.renderFooterRegion(layout)
	if result == "" {
		t.Error("renderFooterRegion() returned empty for component mode")
	}

	// renderComponentFooter should produce output with frame
	compResult := m.renderComponentFooter(layout)
	if compResult == "" {
		t.Error("renderComponentFooter() returned empty string")
	}
}

func TestFooterComponentContract(t *testing.T) {
	// When activeComponent is nil, footer should be status mode (1 line)
	m := testModel(40, 80)
	m.activeComponent = nil

	mode := m.footerMode()
	if mode != FooterModeStatus {
		t.Errorf("footerMode() = %d after component dismissed, want FooterModeStatus", mode)
	}
	fh := m.footerHeight()
	if fh != 1 {
		t.Errorf("footerHeight() = %d after component dismissed, want 1", fh)
	}
}

func TestLayoutResize(t *testing.T) {
	// Changing height should produce layouts that still sum correctly
	for _, height := range []int{20, 30, 40, 50, 60, 80} {
		m := testModel(height, 80)
		m.toolFeed = []ToolFeedEntry{
			{Name: "tool1", Event: "start"},
		}
		layout := m.computeLayout()
		sum := layout.HeaderHeight + layout.ViewportHeight + layout.InputHeight + layout.FooterHeight
		if sum != height {
			t.Errorf("height=%d: sum=%d (header=%d viewport=%d input=%d footer=%d)",
				height, sum, layout.HeaderHeight, layout.ViewportHeight, layout.InputHeight, layout.FooterHeight)
		}
		if layout.ViewportHeight < 1 {
			t.Errorf("height=%d: viewport=%d, want >= 1", height, layout.ViewportHeight)
		}
	}
}

func TestComponentFooterClamp(t *testing.T) {
	// Small terminal (height=15): component footer clamped to height/2 = 7
	m := testModel(15, 80)
	m.activeComponent = &ActiveComponent{
		Component: &layoutMockComponent{name: "cell_type_selector", mode: "overlay"},
	}

	fh := m.footerHeight()
	maxH := 15 / 2 // = 7
	if fh > maxH {
		t.Errorf("footerHeight() = %d on height=15, want <= %d (height/2)", fh, maxH)
	}

	layout := m.computeLayout()
	if layout.ViewportHeight < 1 {
		t.Errorf("ViewportHeight = %d, want >= 1 even with component on small terminal", layout.ViewportHeight)
	}
	sum := layout.HeaderHeight + layout.ViewportHeight + layout.InputHeight + layout.FooterHeight
	if sum != 15 {
		t.Errorf("heights sum = %d, want 15", sum)
	}
}

// ---------------------------------------------------------------------------
// Regression tests: width invariants (prevent JoinVertical overflow)
// ---------------------------------------------------------------------------

func TestRegionWidthInvariant(t *testing.T) {
	// Every rendered region must be <= m.width to prevent JoinVertical from
	// expanding the frame beyond the terminal, causing auto-wrap and ghosting.
	for _, width := range []int{30, 40, 60, 80, 120} {
		m := testModel(30, width)
		layout := m.computeLayout()

		header := m.renderHeaderRegion(layout)
		viewport := m.renderViewportRegion(layout)
		input := m.renderInputRegion(layout)
		footer := m.renderFooterRegion(layout)

		for name, region := range map[string]string{
			"header":   header,
			"viewport": viewport,
			"input":    input,
			"footer":   footer,
		} {
			if region == "" {
				continue
			}
			w := lipgloss.Width(region)
			if w > width {
				t.Errorf("width=%d: %s region width=%d exceeds terminal width", width, name, w)
			}
		}
	}
}

func TestRegionWidthWithToolFeed(t *testing.T) {
	// Tool feed footer must also respect width.
	for _, width := range []int{30, 50, 80} {
		m := testModel(30, width)
		m.toolFeed = []ToolFeedEntry{
			{Name: "search_pubmed", Event: "start", Summary: "searching PubMed for CRISPR gene editing papers"},
			{Name: "load_dataset", Event: "finish", Summary: "loaded 500 genes from GEO dataset GSE12345"},
		}
		layout := m.computeLayout()
		footer := m.renderFooterRegion(layout)
		w := lipgloss.Width(footer)
		if w > width {
			t.Errorf("width=%d: tool feed footer width=%d exceeds terminal width", width, w)
		}
	}
}

func TestHeightSumAcrossNarrowWidths(t *testing.T) {
	// Heights must sum to m.height at all widths, including narrow ones
	// where content might try to wrap.
	for _, width := range []int{20, 30, 40, 60, 80, 120} {
		for _, height := range []int{10, 20, 30, 50} {
			m := testModel(height, width)
			layout := m.computeLayout()
			sum := layout.HeaderHeight + layout.ViewportHeight + layout.InputHeight + layout.FooterHeight
			if sum != height {
				t.Errorf("width=%d height=%d: sum=%d (header=%d vp=%d input=%d footer=%d)",
					width, height, sum, layout.HeaderHeight, layout.ViewportHeight, layout.InputHeight, layout.FooterHeight)
			}
		}
	}
}

func TestGeometryFirstHeader(t *testing.T) {
	// Header height should be constant (2) regardless of width,
	// not dependent on rendering.
	for _, width := range []int{20, 40, 80, 120} {
		m := testModel(30, width)
		layout := m.computeLayout()
		if layout.HeaderHeight != 2 {
			t.Errorf("width=%d: HeaderHeight=%d, want 2 (geometry-first)", width, layout.HeaderHeight)
		}
	}
}
