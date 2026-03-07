package chat

// Layout holds the computed heights for the 4 vertical regions of the TUI.
// HeaderHeight + ViewportHeight + InputHeight + FooterHeight == Model.height
// (except when inline=true, where Layout is zero-valued).
type Layout struct {
	HeaderHeight   int
	ViewportHeight int
	InputHeight    int
	FooterHeight   int
}

// FooterMode classifies what the footer region currently displays.
type FooterMode int

const (
	// FooterModeStatus renders a single status line (spinner + agent + cost).
	FooterModeStatus FooterMode = iota
	// FooterModeToolFeed renders the tool execution ring buffer + status line.
	FooterModeToolFeed
	// FooterModeComponent renders a BioCharm component inside the footer.
	FooterModeComponent
)

// footerMode returns the current footer classification based on model state.
func (m Model) footerMode() FooterMode {
	if m.activeComponent != nil {
		return FooterModeComponent
	}
	if len(m.toolFeed) > 0 {
		return FooterModeToolFeed
	}
	return FooterModeStatus
}

// footerHeight returns the number of terminal rows the footer should occupy.
func (m Model) footerHeight() int {
	switch m.footerMode() {
	case FooterModeToolFeed:
		// feed entries + "Tools" header + status line
		return len(m.toolFeed) + 2
	case FooterModeComponent:
		return m.componentFooterHeight()
	default: // FooterModeStatus
		return 1
	}
}

// componentFooterHeight returns the footer height for an active BioCharm component.
// Height is based on component name, plus 3 for borders + help bar,
// clamped to min(m.height/2, 20) max and 5 min.
func (m Model) componentFooterHeight() int {
	if m.activeComponent == nil || m.activeComponent.Component == nil {
		return 1
	}

	// Base content height by component name.
	var base int
	switch m.activeComponent.Component.Name() {
	case "cell_type_selector", "ontology_browser":
		base = 15
	case "threshold_slider":
		base = 8
	default:
		base = 10
	}

	// Add 3 for top border + bottom border + help bar line.
	h := base + 3

	// Clamp to max half-terminal or 20, whichever is smaller.
	maxH := m.height / 2
	if maxH > 20 {
		maxH = 20
	}
	if h > maxH {
		h = maxH
	}

	// Minimum of 5 (borders + at least 2 content lines).
	if h < 5 {
		h = 5
	}

	return h
}

// computeLayout calculates the vertical region heights for the 4-layer layout.
// It is the single source of truth for height distribution.
func (m Model) computeLayout() Layout {
	if m.inline {
		return Layout{}
	}

	// Header: always 1 line in non-inline mode + 1 newline separator.
	// renderHeader() constrains to Width(m.width).MaxWidth(m.width) so it
	// never wraps. We use a constant instead of render-to-measure.
	header := 0
	if m.shouldRenderHeaderInFrame() {
		header = 2 // 1 header line + 1 separator newline
	}

	// Input: composer + newline, or confirm prompt, or 0 for overlay component.
	input := 0
	if m.activeComponent != nil && m.activeComponent.Component != nil && m.activeComponent.Component.Mode() == "overlay" {
		input = 0
	} else if m.pendingConfirm != nil {
		input = lineCount(renderConfirmPrompt(m.pendingConfirm, m.styles, m.width)) + 1
	} else {
		input = lineCount(m.renderComposer()) + 1
		if m.completionMenuVisible() {
			input += lineCount(m.renderCompletionMenu())
		}
		if m.activeForm != nil {
			input += lineCount(m.activeForm.View()) + 1
		}
		if m.progressActive {
			input += lineCount(renderProgressBar(m.progressLabel, m.progressCurrent, m.progressTotal, m.width, m.styles))
		}
	}

	// Footer height based on mode.
	footer := m.footerHeight()

	// Viewport absorbs remaining height; clamp to min 1.
	viewport := m.height - header - input - footer
	if viewport < 1 {
		viewport = 1
	}

	return Layout{
		HeaderHeight:   header,
		ViewportHeight: viewport,
		InputHeight:    input,
		FooterHeight:   footer,
	}
}

// layoutReservedRows returns the number of rows consumed by everything except
// the viewport. It delegates to computeLayout so existing callers work during
// the transition to the new layout engine.
func (m Model) layoutReservedRows() int {
	if m.inline {
		return m.layoutReservedRowsLegacy()
	}
	layout := m.computeLayout()
	return m.height - layout.ViewportHeight
}
