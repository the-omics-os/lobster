package chat

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/lipgloss"

	"github.com/the-omics-os/lobster-tui/internal/theme"
)

// renderHeader renders the top header line with logo and optional agent badge.
func renderHeader(m Model) string {
	title := m.styles.Header.Render("Lobster AI")

	if m.activeAgent == "" {
		// Pad to full width.
		return lipgloss.NewStyle().Width(m.width).Render(title)
	}

	badge := m.styles.AgentBadge.Render(m.activeAgent)
	gap := m.width - lipgloss.Width(title) - lipgloss.Width(badge)
	if gap < 1 {
		gap = 1
	}

	return lipgloss.JoinHorizontal(
		lipgloss.Top,
		title,
		strings.Repeat(" ", gap),
		badge,
	)
}

// renderMessage renders a single chat message with role-appropriate styling.
func renderMessage(msg ChatMessage, styles theme.Styles, width int) string {
	// Constrain content width to leave room for borders/padding.
	contentWidth := width - 6
	if contentWidth < 20 {
		contentWidth = 20
	}

	switch msg.Role {
	case "user":
		prefix := styles.Bold.Render("You: ")
		body := lipgloss.NewStyle().Width(contentWidth).Render(msg.Content)
		return styles.UserMessage.Render(prefix + body)

	case "assistant":
		agent := "Assistant"
		if msg.Agent != "" {
			agent = msg.Agent
		}
		prefix := lipgloss.NewStyle().
			Foreground(styles.AssistantMessage.GetBorderBottomForeground()).
			Bold(true).
			Render(agent + ": ")
		body := lipgloss.NewStyle().Width(contentWidth).Render(msg.Content)
		return styles.AssistantMessage.Render(prefix + body)

	case "system":
		return styles.SystemMessage.Width(contentWidth).Render(msg.Content)

	default:
		return lipgloss.NewStyle().Width(contentWidth).Render(msg.Content)
	}
}

// renderToolFeed renders the tool execution ring buffer as dim italic lines.
func renderToolFeed(feed []string, styles theme.Styles, width int) string {
	if len(feed) == 0 {
		return ""
	}

	lines := make([]string, 0, len(feed))
	for _, entry := range feed {
		line := styles.ToolExecution.Width(width - 4).Render(fmt.Sprintf("  %s", entry))
		lines = append(lines, line)
	}

	return strings.Join(lines, "\n")
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
