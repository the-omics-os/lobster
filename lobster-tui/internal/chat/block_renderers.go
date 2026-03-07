package chat

import (
	"strings"

	"github.com/alecthomas/chroma/v2"
	"github.com/alecthomas/chroma/v2/formatters"
	"github.com/alecthomas/chroma/v2/lexers"
	chromaStyles "github.com/alecthomas/chroma/v2/styles"
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

// renderBlockCode renders a code block with syntax highlighting via chroma.
func renderBlockCode(b BlockCode, styles theme.Styles, width int) string {
	var parts []string

	// Language label line
	if b.Language != "" {
		parts = append(parts, styles.CodeLabel.Render(b.Language))
	}

	// Syntax highlight the code content
	highlighted := highlightCode(b.Content, b.Language)

	// Wrap in CodeBlock style with width accounting for padding
	codeWidth := width - 4
	if codeWidth < 1 {
		codeWidth = 1
	}
	codeRendered := styles.CodeBlock.Width(codeWidth).Render(highlighted)
	parts = append(parts, codeRendered)

	return strings.Join(parts, "\n")
}

// highlightCode applies chroma syntax highlighting for terminal256 output.
// Falls back to plain text if the language is unknown or tokenisation fails.
func highlightCode(code, language string) string {
	lexer := lexers.Get(language)
	if lexer == nil {
		lexer = lexers.Fallback
	}
	lexer = chroma.Coalesce(lexer)

	formatter := formatters.Get("terminal256")
	if formatter == nil {
		return code
	}

	style := chromaStyles.Get("monokai")
	if style == nil {
		return code
	}

	iterator, err := lexer.Tokenise(nil, code)
	if err != nil {
		return code
	}

	var buf strings.Builder
	if err := formatter.Format(&buf, style, iterator); err != nil {
		return code
	}

	return strings.TrimRight(buf.String(), "\n")
}

// renderBlockAlert renders an alert with colored severity icon, chip, and body.
func renderBlockAlert(b BlockAlert, styles theme.Styles, width int) string {
	contentWidth := width - 8
	if contentWidth < 1 {
		contentWidth = 1
	}

	var style lipgloss.Style
	var icon string
	var chip string

	switch b.Level {
	case "error":
		style = styles.AlertError
		icon = "\u2716" // ✖
		chip = "ERROR"
	case "warning":
		style = styles.AlertWarning
		icon = "\u26a0" // ⚠
		chip = "WARNING"
	case "success":
		style = styles.AlertSuccess
		icon = "\u2713" // ✓
		chip = "SUCCESS"
	default: // "info" and anything else
		style = styles.AlertInfo
		icon = "\u2139" // ℹ
		chip = "INFO"
	}

	title, body := parseAlertContent(b.Message)
	if strings.EqualFold(title, chip) || strings.EqualFold(title, b.Level) {
		title = ""
	}
	if body == "" {
		body = strings.TrimSpace(b.Message)
	}

	chipStyle := lipgloss.NewStyle().
		Foreground(style.GetForeground()).
		Bold(true)
	header := chipStyle.Render(icon + " " + chip)
	if title != "" {
		header += " " + styles.Bold.Render(title)
	}
	if strings.TrimSpace(body) != "" {
		if title != "" {
			body = "  " + styles.Muted.Render(strings.TrimSpace(body))
			return style.Width(contentWidth).Render(header + "\n" + body)
		}
		header += ": " + strings.TrimSpace(body)
	}
	return style.Width(contentWidth).Render(header)
}

// renderBlockHandoff renders an agent handoff with from/to/reason formatting.
func renderBlockHandoff(b BlockHandoff, styles theme.Styles, width int) string {
	var parts []string

	// First line: [from] --> to
	var firstLine strings.Builder
	if b.From != "" {
		firstLine.WriteString(styles.Dimmed.Render(b.From))
		firstLine.WriteByte(' ')
	}
	firstLine.WriteString(styles.HandoffPrefix.Render("-->"))
	firstLine.WriteByte(' ')
	firstLine.WriteString(styles.AgentName.Render(b.To))
	parts = append(parts, firstLine.String())

	// Second line: reason (indented)
	if b.Reason != "" {
		parts = append(parts, "    "+styles.Muted.Render(b.Reason))
	}

	return strings.Join(parts, "\n")
}
