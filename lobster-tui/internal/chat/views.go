package chat

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/glamour"
	"github.com/charmbracelet/lipgloss"

	"github.com/the-omics-os/lobster-tui/internal/protocol"
	"github.com/the-omics-os/lobster-tui/internal/theme"
)

// renderHeader renders the top header line with logo, optional agent badge, and session ID.
func renderHeader(m Model) string {
	title := m.styles.Header.Render("Lobster AI")

	// Build right-side components.
	var rightParts []string
	if m.activeAgent != "" {
		rightParts = append(rightParts, m.styles.AgentBadge.Render(truncateMiddle(m.activeAgent, 24)))
	}
	if m.sessionID != "" {
		sid := truncateMiddle(m.sessionID, 28)
		rightParts = append(rightParts, m.styles.Dimmed.Render(sid))
	}

	if len(rightParts) == 0 {
		return lipgloss.NewStyle().Width(m.width).MaxWidth(m.width).Render(title)
	}

	right := strings.Join(rightParts, "  ")
	maxRightWidth := m.width - lipgloss.Width(title) - 2
	if maxRightWidth < 8 {
		return lipgloss.NewStyle().Width(m.width).MaxWidth(m.width).Render(title)
	}
	right = lipgloss.NewStyle().MaxWidth(maxRightWidth).Render(right)

	gap := m.width - lipgloss.Width(title) - lipgloss.Width(right)
	if gap < 1 {
		gap = 1
	}

	line := lipgloss.JoinHorizontal(
		lipgloss.Top,
		title,
		strings.Repeat(" ", gap),
		right,
	)
	return lipgloss.NewStyle().Width(m.width).MaxWidth(m.width).Render(line)
}

// renderMessage renders a single chat message with role-appropriate styling.
// For assistant messages, mdRenderer may be non-nil to enable glamour rendering.
func renderMessage(msg ChatMessage, styles theme.Styles, width int, mdRenderer *glamour.TermRenderer) string {
	// Constrain content width to leave room for borders/padding.
	contentWidth := width - 6
	if contentWidth < 20 {
		contentWidth = 20
	}

	switch msg.Role {
	case "user":
		body := lipgloss.NewStyle().Width(contentWidth).Render(msg.Content)
		header := styles.InputPrompt.Render("You")
		return styles.UserMessage.MaxWidth(width - 1).Render(header + "\n" + body)

	case "assistant":
		agent := "Assistant"
		if msg.Agent != "" {
			agent = msg.Agent
		}
		header := styles.Bold.Render(agent)

		// Render markdown through glamour if available.
		var body string
		if mdRenderer != nil {
			rendered, err := mdRenderer.Render(msg.Content)
			if err == nil {
				// Glamour adds trailing newlines — trim them.
				body = strings.TrimRight(rendered, "\n")
			} else {
				// Fallback to plain text on glamour error.
				body = lipgloss.NewStyle().Width(contentWidth).Render(msg.Content)
			}
		} else {
			body = lipgloss.NewStyle().Width(contentWidth).Render(msg.Content)
		}
		return styles.AssistantMessage.MaxWidth(width - 1).Render(header + "\n" + body)

	case "system":
		return styles.SystemMessage.Width(contentWidth).Render(msg.Content)
	case "alert_error":
		return renderAlert("error", msg.Content, styles, width)
	case "alert_warning":
		return renderAlert("warning", msg.Content, styles, width)
	case "alert_success":
		return renderAlert("success", msg.Content, styles, width)
	case "alert_info":
		return renderAlert("info", msg.Content, styles, width)

	default:
		return lipgloss.NewStyle().Width(contentWidth).Render(msg.Content)
	}
}

// renderToolFeed renders the tool execution ring buffer with status icons.
func renderToolFeed(feed []ToolFeedEntry, styles theme.Styles, width int) string {
	if len(feed) == 0 {
		return ""
	}

	contentWidth := width - 4
	if contentWidth < 20 {
		contentWidth = 20
	}

	lines := make([]string, 0, len(feed)+1)
	lines = append(lines, lipgloss.NewStyle().Width(contentWidth).Render("  "+styles.Dimmed.Render("Recent tools")))
	for _, entry := range feed {
		var icon string
		var style lipgloss.Style
		var suffix string

		switch entry.Event {
		case "start":
			icon = "●"
			style = styles.ToolRunning
			suffix = "(running)"
		case "finish":
			icon = "✓"
			style = styles.ToolSuccess
			suffix = ""
		case "error":
			icon = "✗"
			style = styles.ToolError
			suffix = "(error)"
		default:
			icon = "·"
			style = styles.ToolExecution
		}

		name := style.Render(icon + " " + entry.Name)
		if entry.Summary != "" {
			name += styles.Dimmed.Render(": " + entry.Summary)
		}
		if suffix != "" {
			name += " " + styles.Dimmed.Render(suffix)
		}

		line := lipgloss.NewStyle().Width(contentWidth).Render("  " + name)
		lines = append(lines, line)
	}

	return strings.Join(lines, "\n")
}

func truncateMiddle(s string, max int) string {
	if max <= 0 || len(s) <= max {
		return s
	}
	if max <= 3 {
		return s[:max]
	}
	left := (max - 1) / 2
	right := max - 1 - left
	return s[:left] + "…" + s[len(s)-right:]
}

// renderProgressBar renders a visual progress bar when active.
func renderProgressBar(label string, current, total, width int, styles theme.Styles) string {
	contentWidth := width - 4
	if contentWidth < 20 {
		contentWidth = 20
	}

	// Calculate bar dimensions — label + bar + percentage.
	barWidth := contentWidth - len(label) - 8 // space for "  label  ⣿⣿  XX%"
	if barWidth < 10 {
		barWidth = 10
	}

	var pct int
	if total > 0 {
		pct = current * 100 / total
	}
	if pct > 100 {
		pct = 100
	}

	filled := barWidth * pct / 100
	empty := barWidth - filled

	bar := strings.Repeat("⣿", filled) + strings.Repeat("⣀", empty)
	text := fmt.Sprintf("  %s  %s  %d%%", styles.ProgressBar.Render(bar), styles.Dimmed.Render(label), pct)
	return lipgloss.NewStyle().Width(contentWidth).Render(text)
}

// renderConfirmPrompt renders an inline confirm dialog.
func renderConfirmPrompt(p *protocol.ConfirmPayload, styles theme.Styles, width int) string {
	hint := "[Y/n]"
	if !p.Default {
		hint = "[y/N]"
	}
	msg := p.Message
	if p.Title != "" {
		msg = p.Title + ": " + msg
	}
	prompt := styles.Bold.Render("? ") + msg + "  " + styles.Dimmed.Render(hint)
	return lipgloss.NewStyle().Width(width - 4).Render(prompt)
}

// renderSelectPrompt renders an inline select dialog with arrow navigation.
func renderSelectPrompt(p *protocol.SelectPayload, selectedIdx int, styles theme.Styles, width int) string {
	contentWidth := width - 4
	if contentWidth < 20 {
		contentWidth = 20
	}

	var b strings.Builder
	msg := p.Message
	if p.Title != "" {
		msg = p.Title + ": " + msg
	}
	b.WriteString(styles.Bold.Render("? ") + msg + "\n")

	for i, opt := range p.Options {
		if i == selectedIdx {
			b.WriteString(styles.InputPrompt.Render("  > ") + styles.Bold.Render(opt))
		} else {
			b.WriteString("    " + styles.Dimmed.Render(opt))
		}
		if i < len(p.Options)-1 {
			b.WriteByte('\n')
		}
	}

	return lipgloss.NewStyle().Width(contentWidth).Render(b.String())
}

// renderStatusBar renders the bottom status bar spanning the full terminal width.
func renderStatusBar(text string, styles theme.Styles, width int) string {
	if text == "" {
		text = "ready"
	}
	return styles.StatusBar.Width(width).Render(text)
}

// renderAlert renders a colored alert box based on severity level.
func renderAlert(level string, message string, styles theme.Styles, width int) string {
	contentWidth := width - 8
	if contentWidth < 20 {
		contentWidth = 20
	}

	var style lipgloss.Style
	var prefix string

	switch level {
	case "error":
		style = styles.AlertError
		prefix = "ERROR"
	case "warning":
		style = styles.AlertWarning
		prefix = "WARNING"
	case "success":
		style = styles.AlertSuccess
		prefix = "OK"
	default: // "info" and anything else
		style = styles.AlertInfo
		prefix = "INFO"
	}

	content := fmt.Sprintf("[%s] %s", prefix, message)
	return style.Width(contentWidth).Render(content)
}
