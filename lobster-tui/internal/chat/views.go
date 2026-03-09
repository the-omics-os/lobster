package chat

import (
	"fmt"
	"image/color"
	"strings"
	"time"

	"github.com/charmbracelet/glamour"
	"charm.land/lipgloss/v2"

	"github.com/the-omics-os/lobster-tui/internal/biocomp"
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
	// Welcome colors now use theme tokens — see welcomeThemeColor* helpers below.
)

type welcomePhase int

const (
	welcomePhaseFade welcomePhase = iota
	welcomePhaseInitialScramble
	welcomePhaseIdle
	welcomePhaseSporadic
)

// renderHeader renders the top header line with logo and session metadata.
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

	providerLine := "   Provider: " + icon + " " + provider
	if m.modelID != "" {
		providerLine += "  ·  " + m.modelID
	}

	lines := []string{
		m.styles.Muted.Render("└─ Compute: " + ram + " │ " + compute),
		m.styles.Dimmed.Render("   Storage: " + storage + " (workspace)"),
		m.styles.Muted.Render(providerLine),
	}
	return strings.Join(lines, "\n")
}

func renderInlineIntro(m Model) string {
	if !m.inline || !m.showIntro || m.width <= 0 {
		return ""
	}
	// Keep the large intro for startup/idle, but remove it once transcript
	// content exists so chat output doesn't crowd out the visible viewport.
	if m.ready && (len(m.messages) > 0 || m.streamBuf.Len() > 0) {
		return ""
	}

	titleLines := welcomeTitleLinesForWidth(m.width)
	title := renderAnimatedWelcomeTitle(m, titleLines)
	center := lipgloss.NewStyle().Width(m.width).MaxWidth(m.width).Align(lipgloss.Center)
	tagline := center.Render(lipgloss.NewStyle().Foreground(theme.Current.Colors.Text).Render(welcomeTagline))
	subTagline := center.Render(lipgloss.NewStyle().Foreground(theme.Current.Colors.TextMuted).Render(welcomeSubTagline))

	lines := []string{center.Render(title), "", tagline, subTagline}
	if m.width < 70 {
		lines = []string{
			center.Render(title),
			"",
			center.Render(lipgloss.NewStyle().Foreground(theme.Current.Colors.TextMuted).Render("on-prem | python native | open-source")),
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

func welcomeBaseColor(elapsed time.Duration) color.Color {
	switch {
	case elapsed < welcomeFadeStepDuration:
		return theme.Current.Colors.TextDim
	case elapsed < 2*welcomeFadeStepDuration:
		return theme.Current.Colors.TextMuted
	default:
		return theme.Current.Colors.Text
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

func welcomeNucleotideColor(base rune) color.Color {
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
		return theme.Current.Colors.Text
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
// Finalized messages (IsStreaming=false) are cached by width; streaming messages
// are never cached.
func renderMessage(msg ChatMessage, styles theme.Styles, width int, mdRenderer *glamour.TermRenderer, inline bool) string {
	// Check cache for finalized messages.
	if !msg.IsStreaming {
		if cached, _, ok := msg.cache.get(width); ok {
			return cached
		}
	}

	rendered := renderMessageUncached(msg, styles, width, mdRenderer, inline)

	// Populate cache for finalized messages only.
	if !msg.IsStreaming {
		height := strings.Count(rendered, "\n") + 1
		msg.cache.set(rendered, width, height)
	}

	return rendered
}

// renderMessageUncached performs the actual rendering without cache logic.
func renderMessageUncached(msg ChatMessage, styles theme.Styles, width int, mdRenderer *glamour.TermRenderer, inline bool) string {
	// Constrain content width to leave room for borders/padding.
	contentWidth := clampRenderWidth(width, 6)
	messageWidth := clampRenderWidth(width, 1)
	if inline {
		return renderInlineMessage(msg, styles, width, mdRenderer)
	}

	switch msg.Role {
	case "user":
		body := renderBlocksOrContent(msg, styles, contentWidth, nil)
		header := styles.UserName.Render("You")
		return styles.UserMessage.MaxWidth(messageWidth).Render(header + "\n" + body)

	case "assistant":
		agent := "Supervisor"
		if msg.Agent != "" {
			agent = msg.Agent
		}
		header := styles.AgentName.Render(agent)
		body := renderBlocksOrContent(msg, styles, contentWidth, mdRenderer)
		return styles.AssistantMessage.MaxWidth(messageWidth).Render(header + "\n" + body)

	case "handoff":
		if b := findBlock[BlockHandoff](msg.Blocks); b != nil {
			return renderBlockHandoff(*b, styles, contentWidth)
		}
		return renderBranchBlock("└─", msg.Content(), styles.AgentTransition, styles.Dimmed, contentWidth)
	case "system":
		return styles.SystemMessage.Width(contentWidth).Render(msg.Content())
	case "alert_error":
		if b := findBlock[BlockAlert](msg.Blocks); b != nil {
			return renderBlockAlert(*b, styles, width)
		}
		return renderAlert("error", msg.Content(), styles, width)
	case "alert_warning":
		if b := findBlock[BlockAlert](msg.Blocks); b != nil {
			return renderBlockAlert(*b, styles, width)
		}
		return renderAlert("warning", msg.Content(), styles, width)
	case "alert_success":
		if b := findBlock[BlockAlert](msg.Blocks); b != nil {
			return renderBlockAlert(*b, styles, width)
		}
		return renderAlert("success", msg.Content(), styles, width)
	case "alert_info":
		if b := findBlock[BlockAlert](msg.Blocks); b != nil {
			return renderBlockAlert(*b, styles, width)
		}
		return renderAlert("info", msg.Content(), styles, width)

	default:
		return lipgloss.NewStyle().Width(contentWidth).Render(msg.Content())
	}
}

// renderBlocksOrContent renders message blocks individually via renderBlock()
// if the message has typed blocks, otherwise falls back to Content() rendering.
func renderBlocksOrContent(msg ChatMessage, styles theme.Styles, width int, mdRenderer *glamour.TermRenderer) string {
	if len(msg.Blocks) == 0 {
		return ""
	}

	parts := make([]string, 0, len(msg.Blocks))
	for _, block := range msg.Blocks {
		rendered := renderBlock(block, styles, width, mdRenderer)
		if rendered != "" {
			parts = append(parts, rendered)
		}
	}
	return strings.Join(parts, "\n")
}

func renderInlineMessage(msg ChatMessage, styles theme.Styles, width int, mdRenderer *glamour.TermRenderer) string {
	contentWidth := clampRenderWidth(width, 2)

	switch msg.Role {
	case "user":
		return renderPrefixedBlock("You", msg.Content(), styles.UserName, lipgloss.NewStyle(), contentWidth)
	case "assistant":
		body := renderBlocksOrContent(msg, styles, contentWidth, mdRenderer)
		body = trimSharedLeftPadding(body, 2)
		return styles.AssistantMessage.Width(contentWidth).Render(body)
	case "handoff":
		if b := findBlock[BlockHandoff](msg.Blocks); b != nil {
			return renderBlockHandoff(*b, styles, contentWidth)
		}
		return renderBranchBlock("└─", msg.Content(), styles.AgentTransition, styles.Dimmed, contentWidth)
	case "system":
		return styles.Dimmed.Render("• " + strings.TrimSpace(msg.Content()))
	case "alert_error":
		if b := findBlock[BlockAlert](msg.Blocks); b != nil {
			return renderBlockAlert(*b, styles, width)
		}
		return renderInlineAlert("error", msg.Content(), styles, width)
	case "alert_warning":
		if b := findBlock[BlockAlert](msg.Blocks); b != nil {
			return renderBlockAlert(*b, styles, width)
		}
		return renderInlineAlert("warning", msg.Content(), styles, width)
	case "alert_success":
		if b := findBlock[BlockAlert](msg.Blocks); b != nil {
			return renderBlockAlert(*b, styles, width)
		}
		return renderInlineAlert("success", msg.Content(), styles, width)
	case "alert_info":
		if b := findBlock[BlockAlert](msg.Blocks); b != nil {
			return renderBlockAlert(*b, styles, width)
		}
		return renderInlineAlert("info", msg.Content(), styles, width)
	default:
		return lipgloss.NewStyle().Width(contentWidth).Render(strings.TrimSpace(msg.Content()))
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

func renderBranchBlock(prefix, body string, prefixStyle, bodyStyle lipgloss.Style, width int) string {
	prefix = strings.TrimSpace(prefix)
	body = strings.TrimSpace(body)
	if body == "" {
		return ""
	}
	if prefix == "" {
		return bodyStyle.Width(width).Render(body)
	}

	prefixRendered := prefixStyle.Render(prefix)
	bodyWidth := width - lipgloss.Width(prefix) - 1
	if bodyWidth < 8 {
		bodyWidth = clampRenderWidth(width, 0)
		renderedBody := bodyStyle.Width(bodyWidth).Render(body)
		renderedBody = strings.TrimRight(renderedBody, "\n")
		return prefixRendered + "\n" + renderedBody
	}

	renderedBody := bodyStyle.Width(bodyWidth).Render(body)
	renderedBody = strings.TrimRight(renderedBody, "\n")
	lines := strings.Split(renderedBody, "\n")
	if len(lines) == 0 {
		lines = []string{""}
	}

	indent := strings.Repeat(" ", lipgloss.Width(prefix)+1)
	var out strings.Builder
	out.WriteString(prefixRendered)
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

// ---------------------------------------------------------------------------
// Region rendering functions (called by View() with Layout struct)
// ---------------------------------------------------------------------------

// renderHeaderRegion renders the header row constrained to layout.HeaderHeight.
func (m Model) renderHeaderRegion(layout Layout) string {
	if layout.HeaderHeight == 0 {
		return ""
	}
	header := renderHeader(m)
	return lipgloss.NewStyle().
		Height(layout.HeaderHeight).
		MaxHeight(layout.HeaderHeight).
		Width(m.width).
		MaxWidth(m.width).
		Render(header)
}

// renderViewportRegion renders the scrollable message viewport constrained
// to exactly layout.ViewportHeight rows and m.width columns. The viewport is
// set to m.width-1 so the scrollbar column fits within the total m.width.
func (m Model) renderViewportRegion(layout Layout) string {
	vpView := m.viewport.View()
	vpView = renderViewportWithScrollbar(vpView, m.viewport, m.styles)
	return lipgloss.NewStyle().
		Height(layout.ViewportHeight).
		MaxHeight(layout.ViewportHeight).
		Width(m.width).
		MaxWidth(m.width).
		Render(vpView)
}

// renderInputRegion renders the composer, completion menu, confirm prompt,
// progress bar, and forms constrained to layout.InputHeight rows.
func (m Model) renderInputRegion(layout Layout) string {
	if layout.InputHeight == 0 {
		return ""
	}

	var b strings.Builder

	// Inline BioComp component (rendered as block above composer).
	if m.activeComponent != nil && m.activeComponent.Component != nil && m.activeComponent.Component.Mode() == "inline" {
		comp := m.activeComponent.Component
		inlineView, panicked := safeView(comp, m.width, m.height)
		if panicked {
			m.sendComponentResponse(m.activeComponent.MsgID, "error", map[string]any{"error": "view_panic"})
		} else if inlineView != "" {
			b.WriteString(inlineView)
			b.WriteByte('\n')
		}
	}

	// Progress bar.
	if m.progressActive {
		b.WriteString(renderProgressBar(m.progressLabel, m.progressCurrent, m.progressTotal, m.width, m.styles))
		b.WriteByte('\n')
	}

	// Inline form.
	if m.activeForm != nil {
		b.WriteString(m.activeForm.View())
		b.WriteByte('\n')
	}

	// Confirm prompt or composer.
	if m.pendingConfirm != nil {
		b.WriteString(renderConfirmPrompt(m.pendingConfirm, m.styles, m.width))
	} else if m.activeComponent == nil || m.activeComponent.Component == nil || m.activeComponent.Component.Mode() != "overlay" {
		completionView := m.renderCompletionMenu()
		if completionView != "" {
			b.WriteString(completionView)
			b.WriteByte('\n')
		}
		b.WriteString(m.renderComposer())
	}

	return lipgloss.NewStyle().
		Height(layout.InputHeight).
		MaxHeight(layout.InputHeight).
		Width(m.width).
		MaxWidth(m.width).
		Render(b.String())
}

// ---------------------------------------------------------------------------
// Footer rendering functions (dispatched by Layout engine)
// ---------------------------------------------------------------------------

// renderFooterRegion dispatches to the appropriate footer renderer based on
// footerMode(). Returns styled output constrained to layout.FooterHeight rows.
func (m Model) renderFooterRegion(layout Layout) string {
	switch m.footerMode() {
	case FooterModeToolFeed:
		return m.renderToolFeedFooter(layout)
	case FooterModeComponent:
		return m.renderComponentFooter(layout)
	default:
		return m.renderStatusFooter(layout)
	}
}

// renderStatusFooter renders the status line using FooterStatus style.
// The output is height- and width-constrained to prevent terminal wrapping.
func (m Model) renderStatusFooter(layout Layout) string {
	statusText := m.currentStatusLine()
	// Truncate to width to prevent wrapping (status is always 1 visual line).
	styled := m.styles.FooterStatus.
		Width(m.width).
		MaxWidth(m.width).
		Render(statusText)
	return lipgloss.NewStyle().
		Height(layout.FooterHeight).
		MaxHeight(layout.FooterHeight).
		Width(m.width).
		MaxWidth(m.width).
		Render(styled)
}

// renderToolFeedFooter renders the tool feed entries + status line together
// in the footer region with FooterToolFeed styling. Width is constrained once
// at the outer level to avoid double-wrapping.
func (m Model) renderToolFeedFooter(layout Layout) string {
	feed := renderToolFeed(m.toolFeed, m.styles, m.width, false)
	status := m.styles.FooterStatus.Render(m.currentStatusLine())
	var content string
	if feed == "" {
		content = status
	} else {
		content = feed + "\n" + status
	}
	return m.styles.FooterToolFeed.
		Width(m.width).
		MaxWidth(m.width).
		Height(layout.FooterHeight).
		MaxHeight(layout.FooterHeight).
		Render(content)
}

// renderComponentFooter renders a BioCharm component inside the footer frame
// with title bar and help bar. Falls back to status footer on panic.
func (m Model) renderComponentFooter(layout Layout) string {
	if m.activeComponent == nil || m.activeComponent.Component == nil {
		return m.renderStatusFooter(layout)
	}

	comp := m.activeComponent.Component
	contentW := m.width - 4 // frame padding/borders
	if contentW < 1 {
		contentW = 1
	}
	contentH := layout.FooterHeight - 3 // borders + help bar
	if contentH < 1 {
		contentH = 1
	}

	content, panicked := safeView(comp, contentW, contentH)
	if panicked {
		m.sendComponentResponse(m.activeComponent.MsgID, "error", map[string]any{"error": "view_panic"})
		return m.renderStatusFooter(layout)
	}

	helpBar := biocomp.RenderHelpBar(comp.KeyBindings(), contentW)
	frame := biocomp.RenderFrame(comp.Name(), content, helpBar, m.width, layout.FooterHeight)

	return m.styles.FooterComponentFrame.
		Width(m.width).
		MaxWidth(m.width).
		Height(layout.FooterHeight).
		MaxHeight(layout.FooterHeight).
		Render(frame)
}
