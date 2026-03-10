package chat

import (
	"strings"

	"charm.land/lipgloss/v2"
	"charm.land/lipgloss/v2/table"
	"github.com/alecthomas/chroma/v2"
	"github.com/alecthomas/chroma/v2/formatters"
	"github.com/alecthomas/chroma/v2/lexers"
	chromaStyles "github.com/alecthomas/chroma/v2/styles"
	"github.com/charmbracelet/glamour"
	"github.com/charmbracelet/x/ansi"
	"github.com/muesli/reflow/wordwrap"

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
	if b.Markdown && mdRenderer != nil {
		rendered, err := mdRenderer.Render(b.Text)
		if err == nil {
			return strings.TrimRight(rendered, "\n")
		}
	}
	return lipgloss.NewStyle().Width(width).Render(b.Text)
}

// renderBlockTable renders a table using lipgloss/table with themed styles.
func renderBlockTable(b BlockTable, styles theme.Styles, width int) string {
	headers, columns := normalizeTableColumns(b)
	if len(headers) == 0 {
		return ""
	}
	headers = formatTableHeaders(headers, columns)
	rows := formatTableRows(b.Rows, columns)

	t := table.New().
		Headers(headers...).
		Border(lipgloss.RoundedBorder()).
		BorderStyle(styles.TableBorder).
		Wrap(true).
		StyleFunc(func(row, col int) lipgloss.Style {
			var style lipgloss.Style
			switch {
			case row == table.HeaderRow:
				style = styles.TableHeader
			case row%2 == 0:
				style = styles.TableRowEven
			default:
				style = styles.TableRowOdd
			}
			if col < len(columns) {
				style = applyTableColumnStyle(style, columns[col])
			}
			return style
		})

	targetWidth := estimateTableWidth(headers, rows, columns)
	if targetWidth <= 0 || targetWidth > width {
		targetWidth = width
	}
	t = t.Width(targetWidth)

	// Add rows individually since Rows takes variadic []string.
	for _, row := range rows {
		t = t.Row(row...)
	}

	return t.String()
}

func normalizeTableColumns(b BlockTable) ([]string, []BlockTableColumn) {
	columnCount := len(b.Headers)
	if len(b.Columns) > columnCount {
		columnCount = len(b.Columns)
	}
	if columnCount == 0 {
		return nil, nil
	}

	headers := make([]string, columnCount)
	columns := make([]BlockTableColumn, columnCount)
	for i := 0; i < columnCount; i++ {
		if i < len(b.Headers) {
			headers[i] = b.Headers[i]
		}
		if i < len(b.Columns) {
			columns[i] = b.Columns[i]
		}
		if columns[i].Name == "" {
			columns[i].Name = headers[i]
		}
		if headers[i] == "" {
			headers[i] = columns[i].Name
		}
	}
	return headers, columns
}

func formatTableHeaders(headers []string, columns []BlockTableColumn) []string {
	formatted := make([]string, len(headers))
	for i, header := range headers {
		formatted[i] = formatTableHeader(header, columns[i])
	}
	return formatted
}

func formatTableRows(rows [][]string, columns []BlockTableColumn) [][]string {
	if len(columns) == 0 {
		return rows
	}

	formatted := make([][]string, 0, len(rows))
	for _, row := range rows {
		formattedRow := make([]string, len(columns))
		for colIdx := range columns {
			cell := ""
			if colIdx < len(row) {
				cell = row[colIdx]
			}
			formattedRow[colIdx] = formatTableCell(cell, columns[colIdx])
		}
		formatted = append(formatted, formattedRow)
	}
	return formatted
}

func formatTableHeader(header string, col BlockTableColumn) string {
	limit := columnRenderLimit(col)
	if limit <= 0 || lipgloss.Width(header) <= limit {
		return header
	}
	return ansi.Truncate(header, limit, "…")
}

func formatTableCell(cell string, col BlockTableColumn) string {
	if cell == "" {
		return cell
	}

	if col.Width > 0 && !col.NoWrap && col.Overflow != "crop" && col.Overflow != "ellipsis" {
		return cell
	}

	limit := columnRenderLimit(col)
	if limit <= 0 || widestLineWidth(cell) <= limit {
		return cell
	}

	if !col.NoWrap && col.Overflow != "crop" && col.Overflow != "ellipsis" {
		return wrapCell(cell, limit)
	}

	truncationToken := "…"
	if col.Overflow == "crop" {
		truncationToken = ""
	}
	return ansi.Truncate(cell, limit, truncationToken)
}

func wrapCell(cell string, width int) string {
	if width <= 0 {
		return cell
	}

	lines := strings.Split(cell, "\n")
	for i, line := range lines {
		if line == "" {
			continue
		}
		lines[i] = wordwrap.String(line, width)
	}
	return strings.Join(lines, "\n")
}

func applyTableColumnStyle(style lipgloss.Style, col BlockTableColumn) lipgloss.Style {
	if col.Width > 0 {
		style = style.Width(col.Width)
	}
	switch strings.ToLower(strings.TrimSpace(col.Justify)) {
	case "center":
		style = style.Align(lipgloss.Center)
	case "right":
		style = style.Align(lipgloss.Right)
	default:
		style = style.Align(lipgloss.Left)
	}
	return style
}

func columnRenderLimit(col BlockTableColumn) int {
	if col.Width > 0 {
		return col.Width
	}
	if col.MaxWidth > 0 {
		return col.MaxWidth
	}
	return 0
}

func estimateTableWidth(headers []string, rows [][]string, columns []BlockTableColumn) int {
	if len(headers) == 0 {
		return 0
	}

	widths := make([]int, len(headers))
	for i, header := range headers {
		widths[i] = widestLineWidth(header)
		if i < len(columns) {
			if columns[i].Width > 0 {
				widths[i] = columns[i].Width
				continue
			}
			if columns[i].MaxWidth > 0 && widths[i] > columns[i].MaxWidth {
				widths[i] = columns[i].MaxWidth
			}
		}
	}
	for _, row := range rows {
		for i, cell := range row {
			if i >= len(widths) {
				break
			}
			if columns[i].Width > 0 {
				continue
			}
			if w := widestLineWidth(cell); w > widths[i] {
				widths[i] = w
			}
			if columns[i].MaxWidth > 0 && widths[i] > columns[i].MaxWidth {
				widths[i] = columns[i].MaxWidth
			}
		}
	}

	total := 1 // left border
	for _, colWidth := range widths {
		total += colWidth + 3 // content + padding + column border
	}
	return total
}

func widestLineWidth(text string) int {
	maxWidth := 0
	for _, line := range strings.Split(text, "\n") {
		if w := lipgloss.Width(line); w > maxWidth {
			maxWidth = w
		}
	}
	return maxWidth
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

	style, icon, chip := alertStyleParts(b.Level, styles)
	return renderWrappedAlert(style, b.Level, icon, chip, b.Message, contentWidth)
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
