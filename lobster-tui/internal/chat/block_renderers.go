package chat

import (
	"strings"

	"github.com/charmbracelet/glamour"
	"charm.land/lipgloss/v2"
	"charm.land/lipgloss/v2/table"

	"github.com/the-omics-os/lobster-tui/internal/theme"
)

// renderBlock dispatches rendering for a single ContentBlock.
func renderBlock(block ContentBlock, styles theme.Styles, width int, mdRenderer *glamour.TermRenderer) string {
	switch b := block.(type) {
	case BlockText:
		return renderBlockText(b, styles, width, mdRenderer)
	case BlockTable:
		return renderBlockTable(b, styles, width)
	case BlockCode:
		return renderBlockCode(b, styles, width)
	case BlockAlert:
		return renderBlockAlert(b, styles, width)
	case BlockHandoff:
		return renderBlockHandoff(b, styles, width)
	default:
		return ""
	}
}

// renderBlockText renders a text block, optionally through glamour for markdown.
func renderBlockText(b BlockText, styles theme.Styles, width int, mdRenderer *glamour.TermRenderer) string {
	if mdRenderer != nil {
		rendered, err := mdRenderer.Render(b.Text)
		if err == nil {
			return strings.TrimRight(rendered, "\n")
		}
	}
	return lipgloss.NewStyle().Width(width).Render(b.Text)
}

// renderBlockTable renders a table using lipgloss/table with themed styles.
func renderBlockTable(b BlockTable, styles theme.Styles, width int) string {
	if len(b.Headers) == 0 {
		return ""
	}

	t := table.New().
		Headers(b.Headers...).
		Border(lipgloss.RoundedBorder()).
		BorderStyle(styles.TableBorder).
		Width(width).
		StyleFunc(func(row, col int) lipgloss.Style {
			switch {
			case row == table.HeaderRow:
				return styles.TableHeader
			case row%2 == 0:
				return styles.TableRowEven
			default:
				return styles.TableRowOdd
			}
		})

	// Add rows individually since Rows takes variadic []string.
	for _, row := range b.Rows {
		t = t.Row(row...)
	}

	return t.String()
}

// renderBlockCode is a stub for code block rendering (Plan 02).
func renderBlockCode(b BlockCode, styles theme.Styles, width int) string {
	return "```" + b.Language + "\n" + b.Content + "\n```"
}

// renderBlockAlert is a stub for alert block rendering (Plan 02).
func renderBlockAlert(b BlockAlert, styles theme.Styles, width int) string {
	return b.Message
}

// renderBlockHandoff is a stub for handoff block rendering (Plan 02).
func renderBlockHandoff(b BlockHandoff, styles theme.Styles, width int) string {
	return b.Reason
}
