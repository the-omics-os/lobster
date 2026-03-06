package chat

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"math/rand"
	"strings"
	"testing"
	"time"

	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/the-omics-os/lobster-tui/internal/protocol"
	"github.com/the-omics-os/lobster-tui/internal/theme"
)

func TestHandleConfirmKeyLocalExitConfirmYesQuits(t *testing.T) {
	m := Model{
		pendingConfirm: &protocol.ConfirmPayload{
			Message: "Exit Lobster session?",
		},
		pendingConfirmID: localExitConfirmID,
	}

	updated, cmd := m.handleConfirmKey(tea.KeyMsg{
		Type:  tea.KeyRunes,
		Runes: []rune{'y'},
	})

	got := updated.(Model)
	if !got.quitting {
		t.Fatal("expected model to enter quitting state")
	}
	if cmd == nil {
		t.Fatal("expected quit command")
	}
	if got.pendingConfirm != nil || got.pendingConfirmID != "" {
		t.Fatal("expected local confirm state to be cleared")
	}
}

func TestHandleConfirmKeyLocalExitConfirmNoCancels(t *testing.T) {
	m := Model{
		pendingConfirm: &protocol.ConfirmPayload{
			Message: "Exit Lobster session?",
		},
		pendingConfirmID: localExitConfirmID,
		messages:         make([]ChatMessage, 0, 1),
		streamBuf:        &strings.Builder{},
	}

	updated, cmd := m.handleConfirmKey(tea.KeyMsg{
		Type:  tea.KeyRunes,
		Runes: []rune{'n'},
	})

	got := updated.(Model)
	if got.quitting {
		t.Fatal("expected model to remain active")
	}
	if cmd != nil {
		t.Fatal("expected no quit command on cancel")
	}
	if len(got.messages) == 0 || got.messages[len(got.messages)-1].Content != "Exit cancelled." {
		t.Fatal("expected cancellation message")
	}
}

func TestHandleProtocolStatusParsesSessionProviderAndCost(t *testing.T) {
	m := newTestModel()
	m.statusText = "unchanged"
	m.provider = "auto"

	updated, _ := m.handleProtocol(testProtocolMsg(t, protocol.TypeStatus, protocol.StatusPayload{
		Text: "  Session: sess-123  ",
	}))
	m = updated.(Model)
	if m.sessionID != "sess-123" {
		t.Fatalf("expected sessionID to be parsed, got %q", m.sessionID)
	}
	if m.statusText != "unchanged" {
		t.Fatalf("expected session status parse to not overwrite status text, got %q", m.statusText)
	}

	updated, _ = m.handleProtocol(testProtocolMsg(t, protocol.TypeStatus, protocol.StatusPayload{
		Text: "  Provider: OpenAI  ",
	}))
	m = updated.(Model)
	if m.provider != "openai" {
		t.Fatalf("expected provider normalization to openai, got %q", m.provider)
	}

	updated, _ = m.handleProtocol(testProtocolMsg(t, protocol.TypeStatus, protocol.StatusPayload{
		Text: "  Ready ($0.1234)  ",
	}))
	m = updated.(Model)
	if m.statusText != "Ready ($0.1234)" {
		t.Fatalf("expected trimmed status text, got %q", m.statusText)
	}
	if m.promptCostUSD != 0.1234 {
		t.Fatalf("expected parsed prompt cost 0.1234, got %f", m.promptCostUSD)
	}
}

func TestHandleProtocolSpinnerStartEnablesDefaultLabel(t *testing.T) {
	m := newTestModel()
	m.statusText = "existing"

	updated, _ := m.handleProtocol(testProtocolMsg(t, protocol.TypeSpinner, protocol.SpinnerPayload{
		Active: true,
	}))
	m = updated.(Model)

	if !m.spinnerActive {
		t.Fatal("expected spinner to become active")
	}
	if m.spinnerLabel != "thinking" {
		t.Fatalf("expected default spinner label 'thinking', got %q", m.spinnerLabel)
	}
	if m.spinnerFrame != 0 {
		t.Fatalf("expected spinner frame reset to 0, got %d", m.spinnerFrame)
	}
	if m.statusText != "existing" {
		t.Fatalf("expected status text to be preserved while spinner active, got %q", m.statusText)
	}
}

func TestQuietStartupSpinnerStaysStaticBeforeReady(t *testing.T) {
	m := newTestModel()
	m.ready = false
	m.showIntro = false
	m.quietStartup = true

	updated, _ := m.handleProtocol(testProtocolMsg(t, protocol.TypeSpinner, protocol.SpinnerPayload{
		Active: true,
		Label:  "Initializing agents",
	}))
	m = updated.(Model)

	if got := m.currentStatusLine(); strings.Contains(got, spinnerFrames[0]) {
		t.Fatalf("expected quiet startup status line without animated spinner frame, got %q", got)
	}
	if got := m.currentStatusLine(); strings.Contains(got, "💡") {
		t.Fatalf("expected quiet startup status line without rotating tips, got %q", got)
	}

	updatedModel, cmd := m.Update(spinnerTick{})
	got := updatedModel.(Model)
	if got.spinnerFrame != 0 {
		t.Fatalf("expected quiet startup spinner frame to stay static, got %d", got.spinnerFrame)
	}
	if cmd != nil {
		t.Fatal("expected quiet startup spinner tick to schedule no follow-up animation")
	}
}

func TestHandleProtocolSpinnerStopClearsStatusButPreservesParsedFields(t *testing.T) {
	m := newTestModel()
	m.statusText = "Ready"
	m.sessionID = "sess-1"
	m.provider = "openai"
	m.promptCostUSD = 0.0099

	updated, _ := m.handleProtocol(testProtocolMsg(t, protocol.TypeSpinner, protocol.SpinnerPayload{
		Active: true,
		Label:  "loading",
	}))
	m = updated.(Model)

	updated, _ = m.handleProtocol(testProtocolMsg(t, protocol.TypeSpinner, protocol.SpinnerPayload{
		Active: false,
	}))
	m = updated.(Model)

	if m.spinnerActive {
		t.Fatal("expected spinner to be inactive after stop")
	}
	if m.statusText != "" {
		t.Fatalf("expected stop to clear status text, got %q", m.statusText)
	}
	if m.sessionID != "sess-1" {
		t.Fatalf("expected sessionID to remain unchanged, got %q", m.sessionID)
	}
	if m.provider != "openai" {
		t.Fatalf("expected provider to remain unchanged, got %q", m.provider)
	}
	if m.promptCostUSD != 0.0099 {
		t.Fatalf("expected promptCostUSD to remain unchanged, got %f", m.promptCostUSD)
	}
}

func TestHandleProtocolSpinnerStopClearsWhenNoStatusExists(t *testing.T) {
	m := newTestModel()

	updated, _ := m.handleProtocol(testProtocolMsg(t, protocol.TypeSpinner, protocol.SpinnerPayload{
		Active: true,
	}))
	m = updated.(Model)

	updated, _ = m.handleProtocol(testProtocolMsg(t, protocol.TypeSpinner, protocol.SpinnerPayload{
		Active: false,
	}))
	m = updated.(Model)

	if m.spinnerActive {
		t.Fatal("expected spinner to be inactive after stop")
	}
	if m.statusText != "" {
		t.Fatalf("expected empty status text when no prior status exists, got %q", m.statusText)
	}
}

func TestRenderProtocolTableWrapsLongCellsAndKeepsLineWidthsAligned(t *testing.T) {
	rendered := renderProtocolTable(
		[]string{"Location", "Status", "Path"},
		[][]string{
			{
				"Workspace Config",
				"Not found",
				"/Users/tyo/Omics-OS/lobster-charm-ui/provider_config.json",
			},
		},
		52,
	)

	lines := strings.Split(rendered, "\n")
	if len(lines) < 4 {
		t.Fatalf("expected wrapped table output, got:\n%s", rendered)
	}
	if !strings.Contains(lines[1], "┼") {
		t.Fatalf("expected divider line, got %q", lines[1])
	}

	wantWidth := lipgloss.Width(lines[0])
	for _, line := range lines {
		if got := lipgloss.Width(line); got != wantWidth {
			t.Fatalf("expected aligned line width %d, got %d for %q", wantWidth, got, line)
		}
	}
}

func TestHandleProtocolTableAppendsPreformattedCodeBlock(t *testing.T) {
	m := newTestModel()
	m.streamBuf.WriteString("⚙️  Current Configuration\n")

	updated, _ := m.handleProtocol(testProtocolMsg(t, protocol.TypeTable, protocol.TablePayload{
		Headers: []string{"Setting", "Value", "Source"},
		Rows: [][]string{
			{"Provider", "ollama", "global user config (~/.config/lobster/)"},
			{"Profile", "production", "global user config"},
		},
	}))
	m = updated.(Model)

	got := m.streamBuf.String()
	if !strings.Contains(got, "```text\n") {
		t.Fatalf("expected table to be wrapped in a text code block, got:\n%s", got)
	}
	if strings.Contains(got, "| --- |") {
		t.Fatalf("expected markdown table fallback to be removed, got:\n%s", got)
	}
	if !strings.Contains(got, "Setting") || !strings.Contains(got, "Provider") {
		t.Fatalf("expected rendered table contents in stream buffer, got:\n%s", got)
	}
}

func TestHandleSelectKeyNavigationWraps(t *testing.T) {
	m := newTestModel()
	m.pendingSelect = &protocol.SelectPayload{
		Message: "Choose one",
		Options: []string{"a", "b", "c"},
	}
	m.selectIndex = 0

	updated, _ := m.handleSelectKey(tea.KeyMsg{Type: tea.KeyUp})
	m = updated.(Model)
	if m.selectIndex != 2 {
		t.Fatalf("expected up from index 0 to wrap to 2, got %d", m.selectIndex)
	}

	updated, _ = m.handleSelectKey(tea.KeyMsg{Type: tea.KeyDown})
	m = updated.(Model)
	if m.selectIndex != 0 {
		t.Fatalf("expected down from index 2 to wrap to 0, got %d", m.selectIndex)
	}
}

func TestHandleSelectKeyCtrlCReturnsCurrentIndex(t *testing.T) {
	var out bytes.Buffer
	m := newTestModel()
	m.handler = protocol.NewHandler(strings.NewReader(""), &out)
	m.pendingSelect = &protocol.SelectPayload{
		Message: "Choose one",
		Options: []string{"a", "b", "c"},
	}
	m.pendingSelectID = "sel-1"
	m.selectIndex = 2

	updated, _ := m.handleSelectKey(tea.KeyMsg{Type: tea.KeyCtrlC})
	m = updated.(Model)

	if m.pendingSelect != nil || m.pendingSelectID != "" {
		t.Fatal("expected pending select to be cleared after Ctrl+C")
	}

	msg := readSingleProtocolMessage(t, &out)
	if msg.Type != protocol.TypeSelectResponse {
		t.Fatalf("expected %q message, got %q", protocol.TypeSelectResponse, msg.Type)
	}

	var p protocol.SelectResponsePayload
	if err := protocol.DecodePayload(msg, &p); err != nil {
		t.Fatalf("decode select response payload: %v", err)
	}
	if p.Index != 2 {
		t.Fatalf("expected Ctrl+C to return current index 2, got %d", p.Index)
	}
	if p.Value != "c" {
		t.Fatalf("expected Ctrl+C to return current value 'c', got %q", p.Value)
	}
}

func TestClearParityLocalAndProtocolTargetsProduceEquivalentState(t *testing.T) {
	local := seededModelForClear()
	local.input.SetValue("/clear")
	updated, _ := local.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	local = updated.(Model)

	protocolOutput := seededModelForClear()
	updated, _ = protocolOutput.handleProtocol(testProtocolMsg(t, protocol.TypeClear, protocol.ClearPayload{
		Target: "output",
	}))
	protocolOutput = updated.(Model)

	protocolAll := seededModelForClear()
	updated, _ = protocolAll.handleProtocol(testProtocolMsg(t, protocol.TypeClear, protocol.ClearPayload{
		Target: "all",
	}))
	protocolAll = updated.(Model)

	assertEquivalentClearState(t, local, protocolOutput, "protocol target=output")
	assertEquivalentClearState(t, local, protocolAll, "protocol target=all")
}

func TestStreamingAppendDoesNotJumpWhenUserIsScrolledUp(t *testing.T) {
	m := seededScrollableModel()
	m.viewport.SetYOffset(0)
	if m.viewport.AtBottom() {
		t.Fatal("expected fixture to be scrolled away from bottom")
	}
	before := m.viewport.YOffset

	updated, _ := m.handleProtocol(testProtocolMsg(t, protocol.TypeText, protocol.TextPayload{
		Content: "stream chunk",
	}))
	m = updated.(Model)

	if m.viewport.YOffset != before {
		t.Fatalf("expected y offset to remain %d while scrolled up, got %d", before, m.viewport.YOffset)
	}
}

func TestStreamingAppendFollowsWhenAtBottom(t *testing.T) {
	m := seededScrollableModel()
	m.viewport.GotoBottom()
	if !m.viewport.AtBottom() {
		t.Fatal("expected fixture to start at bottom")
	}

	updated, _ := m.handleProtocol(testProtocolMsg(t, protocol.TypeText, protocol.TextPayload{
		Content: "stream chunk",
	}))
	m = updated.(Model)

	if !m.viewport.AtBottom() {
		t.Fatal("expected viewport to stay pinned to bottom while streaming")
	}
}

func TestInlineFlowModeKeepsHeaderVisibleWithinTerminalHeight(t *testing.T) {
	m := newTestModel()
	m.inline = true
	m.inlineFlow = true
	m.width = 80
	m.height = 16
	m.welcomeActive = false
	m.statusText = "ready"
	m.spinnerActive = false
	for i := 0; i < 60; i++ {
		m.messages = append(m.messages, ChatMessage{
			Role:    "assistant",
			Content: fmt.Sprintf("overflow line %d", i),
		})
	}
	m.rebuildViewport()

	if m.viewport.VisibleLineCount() >= m.viewport.TotalLineCount() {
		t.Fatalf("expected bounded inline flow viewport under heavy output: visible=%d total=%d", m.viewport.VisibleLineCount(), m.viewport.TotalLineCount())
	}

	rendered := m.View()
	if h := lipgloss.Height(rendered); h > m.height {
		t.Fatalf("expected inline flow view height <= terminal height, rendered=%d terminal=%d", h, m.height)
	}
	if !strings.Contains(rendered, "● lobster") {
		t.Fatalf("expected inline header to remain visible, got:\n%s", rendered)
	}
	if !strings.Contains(rendered, "overflow line 59") {
		t.Fatalf("expected latest transcript lines to remain visible, got:\n%s", rendered)
	}
}

func TestInlinePromptAndLongInputDoNotOverflowTerminalWidth(t *testing.T) {
	m := newTestModel()
	m.inline = true
	m.inlineFlow = true
	m.width = 40
	m.provider = "openai"
	m.promptCostUSD = 12.3456
	m.input.SetValue(strings.Repeat("paste-", 40))

	m.recalculateViewportHeight()

	for _, line := range strings.Split(m.renderComposer(), "\n") {
		if got := lipgloss.Width(line); got > m.width {
			t.Fatalf("expected each composer line width <= %d, got %d (inputWidth=%d prompt=%q line=%q)", m.width, got, m.input.Width(), m.inputPromptText(), line)
		}
	}
}

func TestApplyComposerStylesRemovesFocusedCursorBackground(t *testing.T) {
	m := newTestModel()

	bg := m.input.FocusedStyle.CursorLine.GetBackground()
	if _, isNoColor := bg.(lipgloss.NoColor); !isNoColor {
		t.Fatalf("expected transparent cursor-line background, got %T", bg)
	}
}

func TestKeyEnterSubmitsComposerInput(t *testing.T) {
	m := newTestModel()
	m.ready = true
	m.input.SetValue("hello world")

	updated, _ := m.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	m = updated.(Model)

	if got := m.input.Value(); got != "" {
		t.Fatalf("expected composer to clear after submit, got %q", got)
	}
	if len(m.messages) == 0 {
		t.Fatal("expected submitted user message to be appended")
	}
	last := m.messages[len(m.messages)-1]
	if last.Role != "user" || last.Content != "hello world" {
		t.Fatalf("unexpected submitted message: %#v", last)
	}
}

func TestKeyAltEnterInsertsComposerNewline(t *testing.T) {
	m := newTestModel()
	m.ready = true
	m.input.SetValue("hello")

	updated, _ := m.handleKey(tea.KeyMsg{Type: tea.KeyEnter, Alt: true})
	m = updated.(Model)

	if got := m.input.Value(); got != "hello\n" {
		t.Fatalf("expected alt+enter to insert newline, got %q", got)
	}
	if len(m.messages) != 0 {
		t.Fatalf("expected no submitted message, got %d", len(m.messages))
	}
}

func TestComposerHeightClampsAtMaxLines(t *testing.T) {
	m := newTestModel()
	m.width = 42
	m.inline = true
	m.inlineFlow = true
	m.input.SetValue(strings.Repeat("0123456789abcdef", 32))

	m.recalculateViewportHeight()

	if got := m.input.Height(); got != composerMaxHeight {
		t.Fatalf("expected composer height clamp at %d, got %d", composerMaxHeight, got)
	}
}

func TestInlineFlowStatusCopyEmphasizesTerminalScrollback(t *testing.T) {
	m := newTestModel()
	m.inline = true
	m.inlineFlow = true
	m.ready = true
	m.statusText = ""

	got := m.currentStatusLine()
	if !strings.Contains(got, "inline flow") || !strings.Contains(got, "terminal scrollback") {
		t.Fatalf("expected inline flow status guidance, got %q", got)
	}
}

func TestKeyTabAppliesSlashCompletionSuggestion(t *testing.T) {
	m := newTestModel()
	m.ready = true
	m.input.SetValue("/wo")
	m.refreshSuggestions()

	updated, _ := m.handleKey(tea.KeyMsg{Type: tea.KeyTab})
	m = updated.(Model)

	if got := m.input.Value(); got != "/workspace" {
		t.Fatalf("expected tab completion to apply /workspace, got %q", got)
	}
}

func TestKeyUpUsesHistoryEvenWhenViewportScrollable(t *testing.T) {
	m := seededScrollableModel()
	m.ready = true
	m.input.SetValue("")
	m.pushInputHistory("/status")
	m.pushInputHistory("/help")
	m.viewport.GotoBottom()
	startYOffset := m.viewport.YOffset
	if startYOffset <= 0 {
		t.Fatalf("expected seeded model to be scrollable, got yOffset=%d", startYOffset)
	}

	updated, _ := m.handleKey(tea.KeyMsg{Type: tea.KeyUp})
	m = updated.(Model)

	if m.viewport.YOffset != startYOffset {
		t.Fatalf("expected viewport offset to remain %d, got %d", startYOffset, m.viewport.YOffset)
	}
	if got := m.input.Value(); got != "/help" {
		t.Fatalf("expected history recall to set latest input '/help', got %q", got)
	}
}

func TestWelcomeTickKeepsAnimatingUntilReady(t *testing.T) {
	m := newTestModel()
	m.inline = true
	m.inlineFlow = true
	m.ready = false
	m.welcomeActive = true
	m.welcomeStart = time.Now().Add(-3 * time.Second)
	m.welcomeRNG = rand.New(rand.NewSource(42))
	m.welcomeSporadicCell = -1

	updated, cmd := m.Update(welcomeTick{})
	got := updated.(Model)

	if !got.welcomeActive {
		t.Fatal("expected welcome animation to remain active before ready")
	}
	if got.welcomeFrame != 1 {
		t.Fatalf("expected welcome frame increment to 1, got %d", got.welcomeFrame)
	}
	if cmd == nil {
		t.Fatal("expected next welcome tick command")
	}
}

func TestHandleProtocolReadyDisablesWelcomeAnimation(t *testing.T) {
	m := newTestModel()
	m.inline = true
	m.inlineFlow = true
	m.welcomeActive = true
	m.welcomeSporadicCell = 2
	m.welcomeSporadicTick = 4

	updated, _ := m.handleProtocol(testProtocolMsg(t, protocol.TypeReady, map[string]any{}))
	m = updated.(Model)

	if !m.ready {
		t.Fatal("expected model to be ready after ready event")
	}
	if m.welcomeActive {
		t.Fatal("expected ready event to disable welcome animation")
	}
	if m.welcomeSporadicCell != -1 || m.welcomeSporadicTick != 0 {
		t.Fatal("expected sporadic animation state to reset on ready")
	}
}

func TestWelcomeTickStartsPersistentSparkAfterReady(t *testing.T) {
	m := newTestModel()
	m.inline = true
	m.inlineFlow = true
	m.ready = true
	m.welcomeActive = false
	m.welcomeRNG = rand.New(rand.NewSource(42))
	m.welcomeSporadicCell = -1
	m.welcomeSporadicTick = 0
	m.welcomeNextSporadic = time.Now().Add(-20 * time.Millisecond)

	updated, cmd := m.Update(welcomeTick{})
	got := updated.(Model)

	if got.welcomeSporadicCell < 0 {
		t.Fatal("expected persistent spark to start when interval is due")
	}
	if got.welcomeSporadicTick != 0 {
		t.Fatalf("expected persistent spark tick reset to 0, got %d", got.welcomeSporadicTick)
	}
	if cmd == nil {
		t.Fatal("expected next welcome tick command for persistent spark frame")
	}
}

func TestWelcomeTickPersistentSparkCompletesAndReschedules(t *testing.T) {
	m := newTestModel()
	m.inline = true
	m.inlineFlow = true
	m.ready = true
	m.welcomeActive = false
	m.welcomeRNG = rand.New(rand.NewSource(7))
	m.welcomeSporadicCell = 3
	m.welcomeSporadicTick = welcomePersistentSparkFrames - 1

	updated, cmd := m.Update(welcomeTick{})
	got := updated.(Model)

	if got.welcomeSporadicCell != -1 || got.welcomeSporadicTick != 0 {
		t.Fatal("expected persistent spark state to clear after final frame")
	}
	if got.welcomeNextSporadic.IsZero() {
		t.Fatal("expected next spark time to be scheduled")
	}
	if cmd == nil {
		t.Fatal("expected command to schedule next spark interval")
	}
}

func TestRenderInlineIntroReturnsEmptyWhenDisabled(t *testing.T) {
	m := newTestModel()
	m.inline = true
	m.inlineFlow = true
	m.showIntro = false
	m.welcomeActive = false

	if got := renderInlineIntro(m); got != "" {
		t.Fatalf("expected no intro when disabled, got %q", got)
	}
}

func TestKeyPgUpScrollsViewportEvenWhenInputHasContent(t *testing.T) {
	m := seededScrollableModel()
	m.ready = true
	m.viewport.GotoBottom()
	startYOffset := m.viewport.YOffset
	if startYOffset <= 0 {
		t.Fatalf("expected seeded model to be scrollable, got yOffset=%d", startYOffset)
	}
	m.input.SetValue("/config model")

	updated, _ := m.handleKey(tea.KeyMsg{Type: tea.KeyPgUp})
	m = updated.(Model)

	if m.viewport.YOffset >= startYOffset {
		t.Fatalf("expected page-up to scroll viewport (from %d), got %d", startYOffset, m.viewport.YOffset)
	}
	if got := m.input.Value(); got != "/config model" {
		t.Fatalf("expected input draft to remain unchanged, got %q", got)
	}
}

func TestKeyUpDoesNotScrollViewportWhenHistoryEmpty(t *testing.T) {
	m := seededScrollableModel()
	m.ready = true
	m.input.SetValue("")
	m.viewport.GotoBottom()
	startYOffset := m.viewport.YOffset
	if startYOffset <= 0 {
		t.Fatalf("expected seeded model to be scrollable, got yOffset=%d", startYOffset)
	}

	updated, _ := m.handleKey(tea.KeyMsg{Type: tea.KeyUp})
	m = updated.(Model)

	if m.viewport.YOffset != startYOffset {
		t.Fatalf("expected viewport offset to remain %d, got %d", startYOffset, m.viewport.YOffset)
	}
	if got := m.input.Value(); got != "" {
		t.Fatalf("expected input to remain empty, got %q", got)
	}
}

func TestKeyCtrlGTogglesMouseCaptureOn(t *testing.T) {
	m := newTestModel()
	m.mouseCapture = false

	updated, cmd := m.handleKey(tea.KeyMsg{Type: tea.KeyCtrlG})
	m = updated.(Model)

	if !m.mouseCapture {
		t.Fatal("expected Ctrl+G to enable mouse capture")
	}
	if cmd == nil {
		t.Fatal("expected mouse enable command")
	}
	if got := fmt.Sprintf("%T", cmd()); got != "tea.enableMouseCellMotionMsg" {
		t.Fatalf("expected enable mouse command, got %s", got)
	}
}

func TestKeyCtrlGTogglesMouseCaptureOff(t *testing.T) {
	m := newTestModel()
	m.mouseCapture = true

	updated, cmd := m.handleKey(tea.KeyMsg{Type: tea.KeyCtrlG})
	m = updated.(Model)

	if m.mouseCapture {
		t.Fatal("expected Ctrl+G to disable mouse capture")
	}
	if cmd == nil {
		t.Fatal("expected mouse disable command")
	}
	if got := fmt.Sprintf("%T", cmd()); got != "tea.disableMouseMsg" {
		t.Fatalf("expected disable mouse command, got %s", got)
	}
}

func TestCurrentStatusLineFallsBackToMouseHint(t *testing.T) {
	m := newTestModel()
	m.statusText = ""
	m.mouseCapture = false

	got := m.currentStatusLine()

	if got != "mouse: select  ·  Ctrl+G toggles" {
		t.Fatalf("unexpected status hint: %q", got)
	}
}

func TestViewRendersScrollbarWhenViewportScrollable(t *testing.T) {
	m := seededScrollableModel()
	m.ready = true
	m.viewport.GotoBottom()

	view := m.View()
	if !strings.Contains(view, "█") || !strings.Contains(view, "│") {
		t.Fatalf("expected view to include scrollbar thumb/track, got: %q", view)
	}
}

func testProtocolMsg(t *testing.T, msgType string, payload any) protocolMsg {
	t.Helper()

	raw, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal payload for %s: %v", msgType, err)
	}

	return protocolMsg{
		Message: protocol.Message{
			Type:    msgType,
			Payload: raw,
		},
	}
}

func newTestModel() Model {
	vp := viewport.New(80, 8)
	vp.SetContent("")
	styles := theme.BuildCleanStyles(theme.LobsterDark.Colors)

	in := textarea.New()
	in.Focus()
	in.Prompt = ""
	in.ShowLineNumbers = false
	in.SetWidth(76)
	in.SetHeight(composerMinHeight)
	applyComposerStyles(&in, styles)

	return Model{
		viewport:                vp,
		input:                   in,
		messages:                make([]ChatMessage, 0, 16),
		toolFeed:                make([]ToolFeedEntry, 0, maxToolFeed),
		modalities:              make([]ModalityInfo, 0, 8),
		streamBuf:               &strings.Builder{},
		completionRequestInputs: make(map[string]string, 4),
		completionCache:         make(map[string][]string, 4),
		width:                   80,
		height:                  20,
		ready:                   true,
		inlineFlow:              true,
		showIntro:               true,
		welcomeSporadicCell:     -1,
		styles:                  styles,
	}
}

func seededModelForClear() Model {
	m := newTestModel()
	m.messages = append(m.messages, ChatMessage{Role: "assistant", Content: "one"})
	m.messages = append(m.messages, ChatMessage{Role: "assistant", Content: "two"})
	m.streamBuf.WriteString("partial stream")
	m.toolFeed = append(m.toolFeed, ToolFeedEntry{Name: "x", Event: protocol.ToolExecutionStart})
	m.modalities = append(m.modalities, ModalityInfo{Name: "rna", Shape: "10x10"})
	m.statusText = "persist"
	m.rebuildViewport()
	return m
}

func seededScrollableModel() Model {
	m := newTestModel()
	for i := 0; i < 24; i++ {
		m.messages = append(m.messages, ChatMessage{
			Role:    "assistant",
			Content: fmt.Sprintf("line %d", i),
		})
	}
	m.rebuildViewport()
	return m
}

func assertEquivalentClearState(t *testing.T, want Model, got Model, label string) {
	t.Helper()
	if len(got.messages) != len(want.messages) {
		t.Fatalf("%s: messages len mismatch: want %d got %d", label, len(want.messages), len(got.messages))
	}
	if got.streamBuf.Len() != want.streamBuf.Len() {
		t.Fatalf("%s: streamBuf len mismatch: want %d got %d", label, want.streamBuf.Len(), got.streamBuf.Len())
	}
	if len(got.toolFeed) != len(want.toolFeed) {
		t.Fatalf("%s: toolFeed len mismatch: want %d got %d", label, len(want.toolFeed), len(got.toolFeed))
	}
	if len(got.modalities) != len(want.modalities) {
		t.Fatalf("%s: modalities len mismatch: want %d got %d", label, len(want.modalities), len(got.modalities))
	}
	if got.statusText != want.statusText {
		t.Fatalf("%s: statusText mismatch: want %q got %q", label, want.statusText, got.statusText)
	}
}

func readSingleProtocolMessage(t *testing.T, buf *bytes.Buffer) protocol.Message {
	t.Helper()

	sc := bufio.NewScanner(buf)
	if !sc.Scan() {
		t.Fatal("expected protocol message, got none")
	}

	var msg protocol.Message
	if err := json.Unmarshal(sc.Bytes(), &msg); err != nil {
		t.Fatalf("unmarshal protocol message: %v", err)
	}
	return msg
}
