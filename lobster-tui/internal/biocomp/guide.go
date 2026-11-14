package biocomp

import (
	"fmt"
	"strings"

	"charm.land/lipgloss/v2"
)

// Hinter is a component that says, in a few words, what to do next on the
// screen it is showing right now (picking, typing a reason, reviewing...).
// The chat draws it under the component, above the key legend.
type Hinter interface {
	Hint() string
}

// The instructions screens share, so the same action always reads the same.
const (
	HintPickOne       = "Choose one option, then press enter."
	hintPickSome      = "Tick %s with space, then press enter."
	HintPickEach      = "Choose an option for every row, then press enter."
	HintPlate         = "Give every well a group with its number key, then press enter."
	HintReason        = "Type the reason, then press enter to keep it."
	HintSlider        = "Set the value with ←/→ (shift: bigger steps), then press enter."
	HintText          = "Type your answer, then press enter."
	HintYesNo         = "Answer yes or no."
	HintDashboard     = "Answer each section in turn; tab moves between them."
	HintDashboardDone = "Every section is answered: press ctrl+s to review and send."
	HintReview        = "Check your answers, then press enter to send them."
	HintSuggested     = "Choose a suggested term, or browse the terms around them."
	HintBrowse        = "Pick any term with space; / searches; esc goes back."
	HintRegion        = "Compare the regions with ↑/↓, then press enter to choose one."
)

// HintPickSome is the checklist instruction for picking min to max options.
func HintPickSome(low, high int) string {
	switch {
	case low == high:
		return fmt.Sprintf(hintPickSome, fmt.Sprintf("exactly %d", low))
	case low == 0:
		return fmt.Sprintf(hintPickSome, fmt.Sprintf("up to %d", high))
	default:
		return fmt.Sprintf(hintPickSome, fmt.Sprintf("%d to %d", low, high))
	}
}

// HintFor is a component's instruction: its own, or for the upstream
// components (which don't implement Hinter), one chosen by name.
func HintFor(c BioComponent) string {
	if h, ok := c.(Hinter); ok {
		return h.Hint()
	}
	switch c.Name() {
	case "threshold_slider":
		return HintSlider
	case "text_input":
		return HintText
	case "confirm":
		return HintYesNo
	case "select", "bioselect", "cell_type_selector":
		return HintPickOne
	}
	return ""
}

// The shared palette: one quiet meaning per colour, used the same way on
// every screen.
var (
	// StatStyle is for counts and status lines ("retain (0) · exclude (1)"):
	// the normal text colour.
	StatStyle = lipgloss.NewStyle()
	// HintStyle is for the instruction line: the grey of details.
	HintStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("245"))
	// DetailStyle is for a question's details: the same readable grey.
	DetailStyle = HintStyle
)

// CursorColor marks what the cursor is on (the option, row, well or section
// being answered): Lobster's brand red.
var CursorColor = lipgloss.Color("#e45c47")

// numberKeyStyle draws an option's number key as a small chip, like inline
// code: soft red on dark grey.
var numberKeyStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("210")).Background(lipgloss.Color("236"))

// NumberKey is option n's key as a chip (" 1 "), so it reads as a key to
// press, apart from the option's text.
func NumberKey(n int) string { return numberKeyStyle.Render(fmt.Sprintf(" %d ", n)) }

// roomyHeight is the height from which a screen has room for gaps and the
// instruction line; below it only the key legend is kept.
const roomyHeight = 20

// Roomy reports whether a screen of this height has room for gaps between
// its parts (a short terminal keeps every row for the content).
func Roomy(height int) bool { return height >= roomyHeight }

// Guide is what the chat draws under a component given height rows: a
// blank line, the instruction (if any) and the key legend; on a short
// terminal only the legend.
func Guide(c BioComponent, width, height int) string {
	lines := []string{""}
	if !Roomy(height) {
		lines = nil
	}
	if hint := HintFor(c); hint != "" && Roomy(height) {
		lines = append(lines, HintStyle.Width(max(width, 1)).Render(hint))
	}
	if legend := wrappedLegend(c, width); legend != "" {
		lines = append(lines, legend)
	}
	if len(lines) == 0 || (len(lines) == 1 && lines[0] == "") {
		return ""
	}
	return strings.Join(lines, "\n")
}

// wrappedLegend is the key legend: each key a chip like the option number
// keys, its words in normal text, wrapped between entries so no key is cut
// off at the edge.
func wrappedLegend(c BioComponent, width int) string {
	var lines []string
	line, used := "", 0
	for _, b := range c.KeyBindings() {
		k, d := b.Help().Key, b.Help().Desc
		if k == "" {
			continue
		}
		part := numberKeyStyle.Render(" "+k+" ") + " " + d
		size := lipgloss.Width(k) + 2 + 1 + lipgloss.Width(d)
		if used > 0 && used+2+size > width {
			lines = append(lines, line)
			line, used = "", 0
		}
		if used > 0 {
			line, used = line+"  ", used+2
		}
		line, used = line+part, used+size
	}
	if used > 0 {
		lines = append(lines, line)
	}
	return strings.Join(lines, "\n")
}
