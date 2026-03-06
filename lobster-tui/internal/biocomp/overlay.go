package biocomp

import (
	"strings"

	"github.com/charmbracelet/bubbles/key"
	"github.com/charmbracelet/lipgloss"
)

// OverlaySize returns overlay dimensions based on size hint and terminal size.
// Size hints: "small" (60%/50%), "medium" (70%/60%), "large" (80%/70%).
// Forces fullscreen when terminal is smaller than 77 wide or 20 tall.
func OverlaySize(termW, termH int, sizeHint string) (w, h int) {
	if termW < 77 || termH < 20 {
		return termW, termH
	}

	switch sizeHint {
	case "small":
		w = termW * 60 / 100
		h = termH * 50 / 100
	case "large":
		w = termW * 80 / 100
		h = termH * 70 / 100
	default: // "medium"
		w = termW * 70 / 100
		h = termH * 60 / 100
	}

	// Minimum useful size.
	if w < 40 {
		w = 40
	}
	if h < 10 {
		h = 10
	}
	// Clamp to terminal.
	if w > termW {
		w = termW
	}
	if h > termH {
		h = termH
	}
	return w, h
}

// RenderFrame renders a bordered overlay frame centered on the terminal.
// The frame includes a title bar at the top and a help bar at the bottom.
func RenderFrame(title, content, helpBar string, width, height int) string {
	// Build the border frame style.
	frameStyle := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color("63")).
		Padding(0, 1)

	// Reserve space for border (2 chars each side), title line, help line.
	innerW := width - 4 // 2 border + 2 padding
	if innerW < 1 {
		innerW = 1
	}
	innerH := height - 4 // 2 border + title + help
	if innerH < 1 {
		innerH = 1
	}

	// Title bar.
	titleStyle := lipgloss.NewStyle().
		Bold(true).
		Foreground(lipgloss.Color("63")).
		Width(innerW)
	titleLine := titleStyle.Render(title)

	// Separator.
	sep := lipgloss.NewStyle().
		Foreground(lipgloss.Color("240")).
		Render(strings.Repeat("─", innerW))

	// Content area — constrain to remaining height.
	contentH := innerH - 2 // minus title and separator
	if contentH < 1 {
		contentH = 1
	}
	contentStyle := lipgloss.NewStyle().
		Width(innerW).
		Height(contentH)
	contentBlock := contentStyle.Render(content)

	// Help bar at bottom.
	helpStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color("240")).
		Width(innerW)
	helpLine := helpStyle.Render(helpBar)

	// Assemble vertical stack.
	inner := lipgloss.JoinVertical(lipgloss.Left,
		titleLine,
		sep,
		contentBlock,
		helpLine,
	)

	framed := frameStyle.Render(inner)
	return framed
}

// RenderHelpBar renders key bindings as a compact help string.
func RenderHelpBar(bindings []key.Binding, width int) string {
	if len(bindings) == 0 {
		return ""
	}

	parts := make([]string, 0, len(bindings))
	keyStyle := lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("63"))
	descStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("240"))

	for _, b := range bindings {
		keys := b.Help().Key
		desc := b.Help().Desc
		if keys == "" {
			continue
		}
		part := keyStyle.Render(keys) + " " + descStyle.Render(desc)
		parts = append(parts, part)
	}

	joined := strings.Join(parts, "  ")
	return lipgloss.NewStyle().MaxWidth(width).Render(joined)
}
