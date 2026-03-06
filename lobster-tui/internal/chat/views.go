package chat

import (
	"fmt"
	"strings"
	"time"

	"github.com/charmbracelet/glamour"
	"github.com/charmbracelet/lipgloss"

	"github.com/the-omics-os/lobster-tui/internal/protocol"
	"github.com/the-omics-os/lobster-tui/internal/theme"
)

var welcomeTitleASCII = []string{
	"▗▖    ▗▄▖ ▗▄▄▖  ▗▄▄▖▗▄▄▄▖▗▄▄▄▖▗▄▄▖      ▗▄▖ ▗▄▄▄▖",
	"▐▌   ▐▌ ▐▌▐▌ ▐▌▐▌     █  ▐▌   ▐▌ ▐▌    ▐▌ ▐▌  █  ",
	"▐▌   ▐▌ ▐▌▐▛▀▚▖ ▝▀▚▖  █  ▐▛▀▀▘▐▛▀▚▖    ▐▛▀▜▌  █  ",
	"▐▙▄▄▖▝▚▄▞▘▐▙▄▞▘▗▄▄▞▘  █  ▐▙▄▄▖▐▌ ▐▌    ▐▌ ▐▌▗▄█▄▖",
}

var welcomeNucleotides = []rune{'A', 'T', 'C', 'G'}

const (
	welcomeTitleText          = "Lobster AI"
	welcomeTagline            = "The self-evolving agentic framework for bioinformatics"
	welcomeSubTagline         = "on-prem • python native • open-source"
	welcomeColorForegroundHex = "#f5f5f5"
	welcomeColorMutedHex      = "#888888"
	welcomeColorSubtleHex     = "#555555"
)

type welcomePhase int

const (
	welcomePhaseFade welcomePhase = iota
	welcomePhaseInitialScramble
	welcomePhaseIdle
	welcomePhaseSporadic
)

// renderHeader renders the top header line with logo, optional agent badge, and session ID.
func renderHeader(m Model) string {
	if m.inline {
		version := m.version
		if version == "" {
			version = "unknown"
		}
		left := m.styles.Header.Render("● lobster v" + version + " free │ semantic │ local │ /help")

		if m.sessionID == "" {
			return lipgloss.NewStyle().Width(m.width).MaxWidth(m.width).Render(left)
		}
		right := m.styles.Dimmed.Render(truncateMiddle(m.sessionID, 28))
		maxRightWidth := m.width - lipgloss.Width(left) - 2
		if maxRightWidth < 8 {
			return lipgloss.NewStyle().Width(m.width).MaxWidth(m.width).Render(left)
		}
		right = lipgloss.NewStyle().MaxWidth(maxRightWidth).Render(right)
		gap := m.width - lipgloss.Width(left) - lipgloss.Width(right)
		if gap < 1 {
			gap = 1
		}
		line := lipgloss.JoinHorizontal(
			lipgloss.Top,
			left,
			strings.Repeat(" ", gap),
			right,
		)
		return lipgloss.NewStyle().Width(m.width).MaxWidth(m.width).Render(line)
	}

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

func renderRuntimeSummary(m Model) string {
	if !m.inline {
		return ""
	}

	ram := "N/A"
	if m.totalRAMGB > 0 {
		ram = fmt.Sprintf("%dGB RAM", m.totalRAMGB)
	}
	compute := m.computeTarget
	if compute == "" {
		compute = "CPU"
	}

	storage := "N/A"
	if m.freeStorageGB >= 0 {
		storage = fmt.Sprintf("%dGB free", m.freeStorageGB)
	}

	provider := m.provider
	if provider == "" {
		provider = "auto"
	}
	icon := providerIcon(provider)

	lines := []string{
		m.styles.Muted.Render("└─ Compute: " + ram + " │ " + compute),
		m.styles.Dimmed.Render("   Storage: " + storage + " (workspace)"),
		m.styles.Muted.Render("   Provider: " + icon + " " + provider),
	}
	return strings.Join(lines, "\n")
}

func renderInlineIntro(m Model) string {
	if !m.inline || !m.showIntro || m.width <= 0 {
		return ""
	}

	titleLines := welcomeTitleLinesForWidth(m.width)
	title := renderAnimatedWelcomeTitle(m, titleLines)
	center := lipgloss.NewStyle().Width(m.width).MaxWidth(m.width).Align(lipgloss.Center)
	tagline := center.Render(lipgloss.NewStyle().Foreground(lipgloss.Color(welcomeColorForegroundHex)).Render(welcomeTagline))
	subTagline := center.Render(lipgloss.NewStyle().Foreground(lipgloss.Color(welcomeColorMutedHex)).Render(welcomeSubTagline))

	lines := []string{center.Render(title), "", tagline, subTagline}
	if m.width < 70 {
		lines = []string{
			center.Render(title),
			"",
			center.Render(lipgloss.NewStyle().Foreground(lipgloss.Color(welcomeColorMutedHex)).Render("on-prem | python native | open-source")),
		}
	}
	return strings.Join(lines, "\n")
}

func renderAnimatedWelcomeTitle(m Model, lines []string) string {
	if len(lines) == 0 {
		lines = []string{welcomeTitleText}
	}
	seq := m.welcomeDNA
	if strings.TrimSpace(seq) == "" {
		seq = makeDNASequence(256)
	}

	elapsed := time.Since(m.welcomeStart)
	if !m.welcomeActive {
		// Keep the intro visible after animation completes by pinning to static idle.
		elapsed = time.Duration(welcomeFadeSteps)*welcomeFadeStepDuration +
			time.Duration(welcomeInitialScrambleFrames)*welcomeAnimInterval +
			time.Millisecond
	}

	cellCount := countNonSpaceCells(lines)
	phase := welcomeAnimationPhase(elapsed)
	baseColor := welcomeBaseColor(elapsed)
	locked := 0
	if phase == welcomePhaseInitialScramble && cellCount > 0 {
		locked = welcomeLockedCells(elapsed, cellCount)
	}
	sporadicCell := -1
	sporadicActive := false
	if cellCount > 0 && m.welcomeSporadicCell >= 0 {
		if m.welcomeActive && phase == welcomePhaseSporadic && m.welcomeSporadicTick < welcomeSporadicFrames {
			sporadicActive = true
		}
		if !m.welcomeActive && !m.welcomeNextSporadic.IsZero() && m.welcomeSporadicTick < welcomePersistentSparkFrames {
			sporadicActive = true
		}
		if sporadicActive {
			sporadicCell = m.welcomeSporadicCell % cellCount
		}
	}

	var out strings.Builder
	cellIndex := 0
	for y, line := range lines {
		if y > 0 {
			out.WriteByte('\n')
		}
		for _, ch := range line {
			if ch == ' ' {
				out.WriteRune(ch)
				continue
			}

			display := ch
			color := baseColor
			if phase == welcomePhaseInitialScramble && cellIndex >= locked {
				display = welcomeNucleotideAt(seq, cellIndex, m.welcomeFrame)
				color = welcomeNucleotideColor(display)
			} else if sporadicActive && cellIndex == sporadicCell {
				display = welcomeNucleotideAt(seq, cellIndex, m.welcomeFrame+m.welcomeSporadicTick)
				color = welcomeNucleotideColor(display)
			}
			out.WriteString(lipgloss.NewStyle().Foreground(color).Bold(true).Render(string(display)))
			cellIndex++
		}
	}
	return out.String()
}

func welcomeTitleLinesForWidth(width int) []string {
	if width < maxLineWidth(welcomeTitleASCII) {
		return []string{welcomeTitleText}
	}
	return welcomeTitleASCII
}

func maxLineWidth(lines []string) int {
	max := 0
	for _, line := range lines {
		w := lipgloss.Width(line)
		if w > max {
			max = w
		}
	}
	return max
}

func welcomeTitleCellCount(width int) int {
	return countNonSpaceCells(welcomeTitleLinesForWidth(width))
}

func countNonSpaceCells(lines []string) int {
	count := 0
	for _, line := range lines {
		for _, ch := range line {
			if ch != ' ' {
				count++
			}
		}
	}
	return count
}

func welcomeAnimationPhase(elapsed time.Duration) welcomePhase {
	fadeEnd := time.Duration(welcomeFadeSteps) * welcomeFadeStepDuration
	initialEnd := fadeEnd + time.Duration(welcomeInitialScrambleFrames)*welcomeAnimInterval
	switch {
	case elapsed < fadeEnd:
		return welcomePhaseFade
	case elapsed < initialEnd:
		return welcomePhaseInitialScramble
	case elapsed < welcomeSporadicDelay:
		return welcomePhaseIdle
	default:
		return welcomePhaseSporadic
	}
}

func welcomeBaseColor(elapsed time.Duration) lipgloss.Color {
	switch {
	case elapsed < welcomeFadeStepDuration:
		return lipgloss.Color(welcomeColorSubtleHex)
	case elapsed < 2*welcomeFadeStepDuration:
		return lipgloss.Color(welcomeColorMutedHex)
	default:
		return lipgloss.Color(welcomeColorForegroundHex)
	}
}

func welcomeLockedCells(elapsed time.Duration, cellCount int) int {
	if cellCount <= 0 {
		return 0
	}
	fadeEnd := time.Duration(welcomeFadeSteps) * welcomeFadeStepDuration
	phaseElapsed := elapsed - fadeEnd
	if phaseElapsed < 0 {
		return 0
	}
	tick := int(phaseElapsed / welcomeAnimInterval)
	if tick < 0 {
		tick = 0
	}
	if tick >= welcomeInitialScrambleFrames {
		tick = welcomeInitialScrambleFrames - 1
	}
	progress := float64(tick+1) / float64(welcomeInitialScrambleFrames)
	locked := int(progress * float64(cellCount))
	if locked > cellCount {
		return cellCount
	}
	return locked
}

func welcomeNucleotideAt(seq string, cellIndex, frame int) rune {
	if strings.TrimSpace(seq) == "" {
		return welcomeNucleotides[(cellIndex+frame)%len(welcomeNucleotides)]
	}
	idx := (cellIndex*7 + frame*11) % len(seq)
	candidate := rune(seq[idx])
	switch candidate {
	case 'A', 'T', 'C', 'G':
		return candidate
	}
	return welcomeNucleotides[(cellIndex+frame)%len(welcomeNucleotides)]
}

func welcomeNucleotideColor(base rune) lipgloss.Color {
	switch base {
	case 'A':
		return lipgloss.Color("#22c55e")
	case 'T':
		return lipgloss.Color("#ef4444")
	case 'C':
		return lipgloss.Color("#3b82f6")
	case 'G':
		return lipgloss.Color("#eab308")
	default:
		return lipgloss.Color(welcomeColorForegroundHex)
	}
}

func clampRenderWidth(totalWidth, reserve int) int {
	if totalWidth <= 0 {
		return 1
	}
	w := totalWidth - reserve
	if w < 1 {
		w = 1
	}
	if w > totalWidth {
		w = totalWidth
	}
	return w
}

// renderMessage renders a single chat message with role-appropriate styling.
// For assistant messages, mdRenderer may be non-nil to enable glamour rendering.
func renderMessage(msg ChatMessage, styles theme.Styles, width int, mdRenderer *glamour.TermRenderer, inline bool) string {
	// Constrain content width to leave room for borders/padding.
	contentWidth := clampRenderWidth(width, 6)
	messageWidth := clampRenderWidth(width, 1)
	if inline {
		return renderInlineMessage(msg, styles, width, mdRenderer)
	}

	switch msg.Role {
	case "user":
		body := lipgloss.NewStyle().Width(contentWidth).Render(msg.Content)
		header := styles.InputPrompt.Render("You")
		return styles.UserMessage.MaxWidth(messageWidth).Render(header + "\n" + body)

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
		return styles.AssistantMessage.MaxWidth(messageWidth).Render(header + "\n" + body)

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

func renderInlineMessage(msg ChatMessage, styles theme.Styles, width int, mdRenderer *glamour.TermRenderer) string {
	contentWidth := clampRenderWidth(width, 2)

	switch msg.Role {
	case "user":
		return renderPrefixedBlock("You", msg.Content, styles.InputPrompt, lipgloss.NewStyle(), contentWidth)
	case "assistant":
		body := msg.Content
		if mdRenderer != nil {
			rendered, err := mdRenderer.Render(msg.Content)
			if err == nil {
				body = strings.Trim(rendered, "\n")
			}
		}
		body = trimSharedLeftPadding(body, 2)
		return styles.AssistantMessage.Width(contentWidth).Render(body)
	case "system":
		return styles.Dimmed.Render("• " + strings.TrimSpace(msg.Content))
	case "alert_error":
		return renderInlineAlert("error", msg.Content, styles, width)
	case "alert_warning":
		return renderInlineAlert("warning", msg.Content, styles, width)
	case "alert_success":
		return renderInlineAlert("success", msg.Content, styles, width)
	case "alert_info":
		return renderInlineAlert("info", msg.Content, styles, width)
	default:
		return lipgloss.NewStyle().Width(contentWidth).Render(strings.TrimSpace(msg.Content))
	}
}

func trimSharedLeftPadding(text string, maxTrim int) string {
	lines := strings.Split(text, "\n")
	minPad := -1
	for _, line := range lines {
		if strings.TrimSpace(line) == "" {
			continue
		}
		pad := 0
		for pad < len(line) && line[pad] == ' ' {
			pad++
		}
		if minPad == -1 || pad < minPad {
			minPad = pad
		}
	}
	if minPad <= 0 {
		return text
	}
	if minPad > maxTrim {
		minPad = maxTrim
	}
	for i, line := range lines {
		if len(line) >= minPad {
			lines[i] = line[minPad:]
		}
	}
	return strings.Join(lines, "\n")
}

func renderPrefixedBlock(label, body string, labelStyle, bodyStyle lipgloss.Style, width int) string {
	label = strings.TrimSpace(label)
	if label == "" {
		label = "Assistant"
	}
	prefixRaw := label + ":"
	prefix := labelStyle.Render(prefixRaw)

	bodyWidth := width - lipgloss.Width(prefixRaw) - 1
	if bodyWidth < 8 {
		bodyWidth = clampRenderWidth(width, 0)
		body = strings.TrimSpace(body)
		if body == "" {
			return prefix
		}
		renderedBody := bodyStyle.Width(bodyWidth).Render(body)
		renderedBody = strings.TrimRight(renderedBody, "\n")
		return prefix + "\n" + renderedBody
	}
	renderedBody := bodyStyle.Width(bodyWidth).Render(strings.TrimSpace(body))
	renderedBody = strings.TrimRight(renderedBody, "\n")
	lines := strings.Split(renderedBody, "\n")
	if len(lines) == 0 {
		lines = []string{""}
	}

	indent := strings.Repeat(" ", lipgloss.Width(prefixRaw)+1)
	var out strings.Builder
	out.WriteString(prefix)
	out.WriteByte(' ')
	out.WriteString(lines[0])
	for _, line := range lines[1:] {
		out.WriteByte('\n')
		out.WriteString(indent)
		out.WriteString(line)
	}
	return out.String()
}

// renderToolFeed renders the tool execution ring buffer with status icons.
func renderToolFeed(feed []ToolFeedEntry, styles theme.Styles, width int, inline bool) string {
	if len(feed) == 0 {
		return ""
	}

	contentWidth := clampRenderWidth(width, 4)
	const sectionIndent = "  "

	if inline {
		lines := make([]string, 0, len(feed)+1)
		lines = append(lines, styles.Dimmed.Render("Tools"))
		for _, entry := range feed {
			icon := "·"
			switch entry.Event {
			case "start":
				icon = "●"
			case "finish":
				icon = "✓"
			case "error":
				icon = "✗"
			}
			part := icon + " " + entry.Name
			if entry.Summary != "" {
				part += " " + entry.Summary
			}
			lines = append(lines, sectionIndent+styles.Muted.Render(part))
		}
		return lipgloss.NewStyle().Width(contentWidth).Render(strings.Join(lines, "\n"))
	}

	lines := make([]string, 0, len(feed)+1)
	lines = append(lines, lipgloss.NewStyle().Width(contentWidth).Render(sectionIndent+styles.Dimmed.Render("Tools")))
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

		line := lipgloss.NewStyle().Width(contentWidth).Render(sectionIndent + name)
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
	contentWidth := clampRenderWidth(width, 4)
	label = strings.TrimSpace(label)
	if label == "" {
		label = "Progress"
	}

	var pct int
	if total > 0 {
		pct = current * 100 / total
	}
	if pct > 100 {
		pct = 100
	}

	pctText := fmt.Sprintf("%d%%", pct)
	barWidth := contentWidth - lipgloss.Width(label) - len(pctText) - 6
	if barWidth < 6 {
		line := "  " + styles.Dimmed.Render(label) + " " + styles.ProgressBar.Render(pctText)
		return lipgloss.NewStyle().Width(contentWidth).Render(line)
	}

	filled := barWidth * pct / 100
	empty := barWidth - filled

	bar := strings.Repeat("⣿", filled) + strings.Repeat("⣀", empty)
	text := fmt.Sprintf("  %s %s %s", styles.Dimmed.Render(label), styles.ProgressBar.Render(bar), styles.Bold.Render(pctText))
	return lipgloss.NewStyle().Width(contentWidth).Render(text)
}

// renderConfirmPrompt renders an inline confirm dialog.
func renderConfirmPrompt(p *protocol.ConfirmPayload, styles theme.Styles, width int) string {
	contentWidth := clampRenderWidth(width, 4)
	hint := "[Y/n]"
	if !p.Default {
		hint = "[y/N]"
	}
	msg := strings.TrimSpace(p.Message)
	title := strings.TrimSpace(p.Title)
	var lines []string
	if title != "" {
		lines = append(lines, "  "+styles.Bold.Render(title))
	}
	if msg == "" {
		msg = "Confirm?"
	}
	prompt := "  " + styles.Bold.Render("?") + " " + msg + "  " + styles.Dimmed.Render(hint)
	lines = append(lines, prompt)
	return lipgloss.NewStyle().Width(contentWidth).Render(strings.Join(lines, "\n"))
}

// renderSelectPrompt renders an inline select dialog with arrow navigation.
func renderSelectPrompt(p *protocol.SelectPayload, selectedIdx int, styles theme.Styles, width int) string {
	contentWidth := clampRenderWidth(width, 4)

	var b strings.Builder
	msg := strings.TrimSpace(p.Message)
	title := strings.TrimSpace(p.Title)
	if title != "" {
		b.WriteString("  " + styles.Bold.Render(title) + "\n")
	}
	if msg == "" {
		msg = "Select an option"
	}
	b.WriteString("  " + styles.Bold.Render("?") + " " + msg + "\n")

	for i, opt := range p.Options {
		text := strings.TrimSpace(opt)
		if text == "" {
			text = "(blank)"
		}
		if i == selectedIdx {
			b.WriteString("  " + styles.InputPrompt.Render(">") + " " + styles.Bold.Render(text))
		} else {
			b.WriteString("    " + styles.Dimmed.Render(text))
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
	contentWidth := clampRenderWidth(width, 8)

	var style lipgloss.Style
	var icon string
	var chip string

	switch level {
	case "error":
		style = styles.AlertError
		icon = "✖"
		chip = "ERROR"
	case "warning":
		style = styles.AlertWarning
		icon = "⚠"
		chip = "WARNING"
	case "success":
		style = styles.AlertSuccess
		icon = "✓"
		chip = "SUCCESS"
	default: // "info" and anything else
		style = styles.AlertInfo
		icon = "ℹ"
		chip = "INFO"
	}

	title, body := parseAlertContent(message)
	if strings.EqualFold(title, chip) || strings.EqualFold(title, level) {
		title = ""
	}
	if body == "" {
		body = strings.TrimSpace(message)
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

func renderInlineAlert(level string, message string, styles theme.Styles, width int) string {
	contentWidth := clampRenderWidth(width, 2)
	var style lipgloss.Style
	var icon string
	var chip string

	switch level {
	case "error":
		style = styles.AlertError
		icon = "✖"
		chip = "ERROR"
	case "warning":
		style = styles.AlertWarning
		icon = "⚠"
		chip = "WARNING"
	case "success":
		style = styles.AlertSuccess
		icon = "✓"
		chip = "SUCCESS"
	default:
		style = styles.AlertInfo
		icon = "ℹ"
		chip = "INFO"
	}

	title, body := parseAlertContent(message)
	if strings.EqualFold(title, chip) || strings.EqualFold(title, level) {
		title = ""
	}
	if body == "" {
		body = strings.TrimSpace(message)
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
			return lipgloss.NewStyle().Width(contentWidth).Render(header + "\n  " + styles.Muted.Render(strings.TrimSpace(body)))
		}
		header += ": " + strings.TrimSpace(body)
	}
	return lipgloss.NewStyle().Width(contentWidth).Render(header)
}

func parseAlertContent(message string) (title, body string) {
	text := strings.TrimSpace(message)
	if text == "" {
		return "", ""
	}

	if strings.HasPrefix(text, "[") {
		if idx := strings.Index(text, "]"); idx > 1 {
			head := strings.TrimSpace(text[1:idx])
			rest := strings.TrimSpace(text[idx+1:])
			rest = strings.TrimLeft(rest, ":-| ")
			if looksLikeAlertTitle(head) && rest != "" {
				return head, rest
			}
		}
	}

	lines := strings.Split(text, "\n")
	if len(lines) > 1 {
		head := strings.TrimSpace(lines[0])
		rest := strings.TrimSpace(strings.Join(lines[1:], "\n"))
		if looksLikeAlertTitle(head) && rest != "" {
			return strings.Trim(head, "[]"), rest
		}
	}

	for _, sep := range []string{": ", " - ", " | ", " — "} {
		if idx := strings.Index(text, sep); idx > 0 {
			head := strings.TrimSpace(text[:idx])
			rest := strings.TrimSpace(text[idx+len(sep):])
			if looksLikeAlertTitle(head) && rest != "" {
				return strings.Trim(head, "[]"), rest
			}
		}
	}

	return "", text
}

func looksLikeAlertTitle(s string) bool {
	candidate := strings.TrimSpace(strings.Trim(s, "[]"))
	if candidate == "" {
		return false
	}
	if strings.Contains(candidate, "\n") || strings.Contains(candidate, "://") {
		return false
	}
	return lipgloss.Width(candidate) <= 48
}
