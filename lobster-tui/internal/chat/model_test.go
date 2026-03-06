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

func TestInlineFlowModeDoesNotRenderArchivedTranscriptInFrame(t *testing.T) {
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

	rendered := m.View()
	if h := lipgloss.Height(rendered); h > m.height {
		t.Fatalf("expected inline flow view height <= terminal height, rendered=%d terminal=%d", h, m.height)
	}
	if !strings.Contains(rendered, "● lobster") {
		t.Fatalf("expected inline header to remain visible, got:\n%s", rendered)
	}
	if strings.Contains(rendered, "overflow line 0") || strings.Contains(rendered, "overflow line 59") {
		t.Fatalf("expected archived transcript to be omitted from inline frame, got:\n%s", rendered)
	}
}

func TestInlineFlowViewportHidesActiveStream(t *testing.T) {
	m := newTestModel()
	m.inline = true
	m.inlineFlow = true
	m.messages = append(m.messages, ChatMessage{Role: "assistant", Content: "older output"})
	m.streamBuf.WriteString("live output")

	m.rebuildViewport()

	view := m.viewport.View()
	if strings.TrimSpace(view) != "" {
		t.Fatalf("expected inline viewport to suppress partial stream content, got:\n%s", view)
	}
}

func TestInlineFlowAgentTransitionDoesNotPrintSystemMessage(t *testing.T) {
	m := newTestModel()
	m.inline = true
	m.inlineFlow = true

	updated, cmd := m.handleProtocol(testProtocolMsg(t, protocol.TypeAgentTransition, protocol.AgentTransitionPayload{
		To:     "research_agent",
		Kind:   "activity",
		Status: "working",
		Reason: "working",
	}))
	m = updated.(Model)

	if cmd == nil {
		t.Fatal("expected wait command for agent transition")
	}
	if len(m.messages) != 0 {
		t.Fatalf("expected no inline transcript message for agent transition, got %d", len(m.messages))
	}
}

func TestInlinePrintMessagesCmdUsesTerminalPrintln(t *testing.T) {
	m := newTestModel()
	m.inline = true
	m.inlineFlow = true

	cmd := m.inlinePrintMessagesCmd([]ChatMessage{{Role: "user", Content: "hello"}})
	if cmd == nil {
		t.Fatal("expected inline print command")
	}
	if got := fmt.Sprintf("%T", cmd()); got != "tea.printLineMessage" {
		t.Fatalf("expected terminal print command, got %s", got)
	}
}

func TestInlineFlowReadyDoesNotDetachHeaderBeforeFirstPrintedMessage(t *testing.T) {
	m := newTestModel()
	m.inline = true
	m.inlineFlow = true
	m.ready = false
	m.showIntro = false
	m.sessionID = "session_123"
	m.provider = "bedrock"
	m.totalRAMGB = 32
	m.computeTarget = "MPS"
	m.freeStorageGB = 41

	updated, cmd := m.handleProtocol(testProtocolMsg(t, protocol.TypeReady, struct{}{}))
	m = updated.(Model)

	if m.inlineBannerPrinted {
		t.Fatal("expected inline banner to remain unprinted immediately after ready")
	}
	if cmd == nil {
		t.Fatal("expected ready handler to return wait command")
	}

	view := m.View()
	if !strings.Contains(view, "● lobster") {
		t.Fatalf("expected inline frame header to remain visible before first printed message, got:\n%s", view)
	}
	if !strings.Contains(view, "└─ Compute:") {
		t.Fatalf("expected inline runtime summary to remain visible before first printed message, got:\n%s", view)
	}
}

func TestInlineFlowFirstPrintedMessageDetachesHeader(t *testing.T) {
	m := newTestModel()
	m.inline = true
	m.inlineFlow = true
	m.ready = true
	m.showIntro = false
	m.sessionID = "session_123"
	m.provider = "bedrock"
	m.totalRAMGB = 32
	m.computeTarget = "MPS"
	m.freeStorageGB = 41

	cmd := m.appendMessage(ChatMessage{Role: "user", Content: "hello"}, false)
	if cmd == nil {
		t.Fatal("expected first printed message command")
	}
	if got := fmt.Sprintf("%T", cmd()); got != "tea.printLineMessage" {
		t.Fatalf("expected print command for first message, got %s", got)
	}
	if !m.inlineBannerPrinted {
		t.Fatal("expected inline banner to be marked as printed after first printed message")
	}

	view := m.View()
	if strings.Contains(view, "● lobster") {
		t.Fatalf("expected inline frame header to be hidden after first printed message, got:\n%s", view)
	}
	if strings.Contains(view, "└─ Compute:") {
		t.Fatalf("expected inline runtime summary to be hidden after first printed message, got:\n%s", view)
	}
}

func TestTaskHandoffQueuesUntilSupervisorMessageFlushes(t *testing.T) {
	m := newTestModel()
	m.inline = false
	m.inlineFlow = false
	m.streamBuf.WriteString("I will search GEO for matching datasets.")
	m.isStreaming = true

	updated, cmd := m.handleProtocol(testProtocolMsg(t, protocol.TypeAgentTransition, protocol.AgentTransitionPayload{
		To:     "research_agent",
		Kind:   "task",
		Reason: "Search GEO for human lung adenocarcinoma scRNA-seq datasets.",
	}))
	m = updated.(Model)

	if cmd == nil {
		t.Fatal("expected wait command for task transition")
	}
	if len(m.messages) != 0 {
		t.Fatalf("expected handoff to remain buffered until stream flush, got %d transcript messages", len(m.messages))
	}
	if len(m.pendingHandoffs) != 1 {
		t.Fatalf("expected one pending handoff, got %d", len(m.pendingHandoffs))
	}

	updated, _ = m.handleProtocol(testProtocolMsg(t, protocol.TypeDone, protocol.DonePayload{}))
	m = updated.(Model)

	if len(m.messages) != 2 {
		t.Fatalf("expected assistant message plus handoff line, got %d messages", len(m.messages))
	}
	if m.messages[0].Role != "assistant" {
		t.Fatalf("expected first message to be assistant, got %q", m.messages[0].Role)
	}
	if m.messages[1].Role != "handoff" {
		t.Fatalf("expected second message to be handoff, got %q", m.messages[1].Role)
	}
	rendered := renderMessage(m.messages[1], m.styles, m.width, nil, false)
	if !strings.Contains(rendered, "└─ Search GEO for human lung adenocarcinoma scRNA-seq datasets.") {
		t.Fatalf("expected rendered handoff line, got:\n%s", rendered)
	}
}

func TestActivityTransitionsUpdateFooterStateOnly(t *testing.T) {
	m := newTestModel()
	m.inline = false
	m.inlineFlow = false

	updated, _ := m.handleProtocol(testProtocolMsg(t, protocol.TypeAgentTransition, protocol.AgentTransitionPayload{
		To:     "research_agent",
		Kind:   "activity",
		Status: "working",
	}))
	m = updated.(Model)

	if len(m.messages) != 0 {
		t.Fatalf("expected activity transition to stay out of transcript, got %d messages", len(m.messages))
	}
	if _, ok := m.activeWorkers["research_agent"]; !ok {
		t.Fatal("expected research_agent to be marked active")
	}
	if !strings.Contains(m.currentStatusLine(), "research_agent") {
		t.Fatalf("expected footer to mention active worker, got %q", m.currentStatusLine())
	}
	if strings.Contains(renderHeader(m), "research_agent") {
		t.Fatalf("expected active worker to stay out of header, got %q", renderHeader(m))
	}

	updated, _ = m.handleProtocol(testProtocolMsg(t, protocol.TypeAgentTransition, protocol.AgentTransitionPayload{
		To:     "research_agent",
		Kind:   "activity",
		Status: "complete",
	}))
	m = updated.(Model)

	if _, ok := m.activeWorkers["research_agent"]; ok {
		t.Fatal("expected research_agent to be cleared after completion")
	}
	if strings.Contains(m.currentStatusLine(), "research_agent") {
		t.Fatalf("expected footer to clear completed worker, got %q", m.currentStatusLine())
	}
}

func TestDoneDoesNotAttributeSupervisorTranscriptToSpecialist(t *testing.T) {
	m := newTestModel()
	m.inline = false
	m.inlineFlow = false
	m.streamBuf.WriteString("Supervisor summary.")

	updated, _ := m.handleProtocol(testProtocolMsg(t, protocol.TypeAgentTransition, protocol.AgentTransitionPayload{
		To:     "research_agent",
		Kind:   "activity",
		Status: "working",
	}))
	m = updated.(Model)

	updated, _ = m.handleProtocol(testProtocolMsg(t, protocol.TypeDone, protocol.DonePayload{}))
	m = updated.(Model)

	if len(m.messages) != 1 {
		t.Fatalf("expected one assistant message, got %d", len(m.messages))
	}
	if m.messages[0].Agent != "" {
		t.Fatalf("expected supervisor transcript to have no specialist attribution, got %q", m.messages[0].Agent)
	}
	rendered := renderMessage(m.messages[0], m.styles, m.width, nil, false)
	if strings.Contains(rendered, "research_agent") {
		t.Fatalf("expected rendered assistant message to avoid specialist label, got:\n%s", rendered)
	}
	if !strings.Contains(rendered, "Supervisor") {
		t.Fatalf("expected rendered assistant message to use supervisor label, got:\n%s", rendered)
	}
}

func TestToolFeedReconcilesSameNameToolsByIdentity(t *testing.T) {
	m := newTestModel()
	m.inline = false
	m.inlineFlow = false

	events := []protocol.ToolExecutionPayload{
		{ToolName: "get_dataset_metadata", ToolCallID: "tc-1", Event: protocol.ToolExecutionStart},
		{ToolName: "get_dataset_metadata", ToolCallID: "tc-2", Event: protocol.ToolExecutionStart},
		{ToolName: "get_dataset_metadata", ToolCallID: "tc-1", Event: protocol.ToolExecutionFinish, Summary: "0.3s"},
	}

	for _, event := range events {
		updated, _ := m.handleProtocol(testProtocolMsg(t, protocol.TypeToolExecution, event))
		m = updated.(Model)
	}

	if len(m.toolFeed) != 2 {
		t.Fatalf("expected two tool-feed rows, got %d", len(m.toolFeed))
	}
	if m.toolFeed[0].ID != "tc-1" || m.toolFeed[0].Event != protocol.ToolExecutionFinish {
		t.Fatalf("expected first tool row to be tc-1 finish, got %#v", m.toolFeed[0])
	}
	if m.toolFeed[1].ID != "tc-2" || m.toolFeed[1].Event != protocol.ToolExecutionStart {
		t.Fatalf("expected second tool row to remain tc-2 start, got %#v", m.toolFeed[1])
	}
}

func TestActiveWorkerSummaryAggregatesMultipleWorkersHonestly(t *testing.T) {
	m := newTestModel()
	m.activeWorkers["research_agent"] = struct{}{}
	m.activeWorkers["data_expert_agent"] = struct{}{}
	m.activeWorkers["metadata_assistant"] = struct{}{}

	if got := m.activeWorkerSummary(); got != "data_expert_agent +2" {
		t.Fatalf("expected sorted aggregate worker summary, got %q", got)
	}
	if got := m.activeWorkerIndicator(); !strings.Contains(got, "● data_expert_agent +2") {
		t.Fatalf("expected active worker indicator with distinct dot, got %q", got)
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
	m.inline = true
	m.inlineFlow = true
	m.ready = true
	m.input.SetValue("/wo")
	m.refreshSuggestions()

	updated, _ := m.handleKey(tea.KeyMsg{Type: tea.KeyTab})
	m = updated.(Model)

	if got := m.input.Value(); got != "/workspace" {
		t.Fatalf("expected tab completion to apply /workspace, got %q", got)
	}
}

func TestKeyTabAppliesSubcommandCompletionAndAppendsSpace(t *testing.T) {
	m := newTestModel()
	m.inline = true
	m.inlineFlow = true
	m.ready = true
	m.input.SetValue("/workspace l")
	m.refreshSuggestions()

	updated, _ := m.handleKey(tea.KeyMsg{Type: tea.KeyTab})
	m = updated.(Model)

	if got := m.input.Value(); got != "/workspace list " {
		t.Fatalf("expected tab completion to apply subcommand with trailing space, got %q", got)
	}
}

func TestKeyTabAppliesPathCompletionWithoutTrailingSpace(t *testing.T) {
	m := newTestModel()
	m.inline = true
	m.inlineFlow = true
	m.ready = true
	m.input.SetValue("/read data")
	m.completionSuggestions = []string{"/read data/file.txt"}
	m.syncCompletionMenuState(m.input.Value())

	updated, _ := m.handleKey(tea.KeyMsg{Type: tea.KeyTab})
	m = updated.(Model)

	if got := m.input.Value(); got != "/read data/file.txt" {
		t.Fatalf("expected path completion without trailing space, got %q", got)
	}
}

func TestCtrlNMovesInlineCompletionSelection(t *testing.T) {
	m := newTestModel()
	m.inline = true
	m.inlineFlow = true
	m.ready = true
	m.input.SetValue("/w")
	m.refreshSuggestions()

	updated, _ := m.handleKey(tea.KeyMsg{Type: tea.KeyCtrlN})
	m = updated.(Model)

	updated, _ = m.handleKey(tea.KeyMsg{Type: tea.KeyTab})
	m = updated.(Model)

	if got := m.input.Value(); got != "/workspace-info" {
		t.Fatalf("expected ctrl+n to select next suggestion, got %q", got)
	}
}

func TestInlineViewRendersCompletionMenuAboveComposer(t *testing.T) {
	m := newTestModel()
	m.inline = true
	m.inlineFlow = true
	m.ready = true
	m.input.SetValue("/wo")
	m.refreshSuggestions()

	view := m.View()
	suggestionPos := strings.Index(view, "/workspace")
	inputPos := strings.LastIndex(view, "/wo")

	if suggestionPos < 0 {
		t.Fatalf("expected inline view to render completion suggestions, got:\n%s", view)
	}
	if inputPos < 0 {
		t.Fatalf("expected inline composer to still render current input, got:\n%s", view)
	}
	if suggestionPos > inputPos {
		t.Fatalf("expected completion menu to render above composer, got:\n%s", view)
	}
	if got := m.currentStatusLine(); !strings.Contains(got, "Tab accept") || !strings.Contains(got, "Ctrl+N/Ctrl+P move") {
		t.Fatalf("expected completion controls in status line, got %q", got)
	}
}

func TestEscapeDismissesInlineCompletionMenuUntilInputChanges(t *testing.T) {
	m := newTestModel()
	m.inline = true
	m.inlineFlow = true
	m.ready = true
	m.input.SetValue("/wo")
	m.refreshSuggestions()

	updated, _ := m.handleKey(tea.KeyMsg{Type: tea.KeyEsc})
	m = updated.(Model)

	if m.completionMenuVisible() {
		t.Fatal("expected escape to dismiss inline completion menu")
	}
	if got := m.currentStatusLine(); strings.Contains(got, "Tab accept") {
		t.Fatalf("expected completion help to clear after dismiss, got %q", got)
	}

	m.input.SetValue("/w")
	m.refreshSuggestions()

	if !m.completionMenuVisible() {
		t.Fatal("expected completion menu to reopen after input changes")
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
		pendingHandoffs:         make([]ChatMessage, 0, 4),
		activeWorkers:           make(map[string]struct{}, 4),
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

// ---------------------------------------------------------------------------
// Cancel vs Quit tests (Phase 1: Query Cancellation)
// ---------------------------------------------------------------------------

func TestCtrlCDuringStreamingArmsCancel(t *testing.T) {
	var out bytes.Buffer
	m := newTestModel()
	m.handler = protocol.NewHandler(strings.NewReader(""), &out)
	m.isStreaming = true

	updated, cmd := m.handleKey(tea.KeyMsg{Type: tea.KeyCtrlC})
	got := updated.(Model)

	if got.quitting {
		t.Fatal("expected model NOT to enter quitting state during streaming")
	}
	if !got.isCanceling {
		t.Fatal("expected isCanceling to be armed after first Ctrl+C")
	}
	if got.statusText != "Press Ctrl+C again to cancel" {
		t.Fatalf("expected cancel prompt, got %q", got.statusText)
	}
	// First press should NOT send cancel — it arms the two-phase flow.
	if out.Len() > 0 {
		t.Fatal("expected no protocol message on first Ctrl+C press")
	}
	if cmd == nil {
		t.Fatal("expected a timer command from first Ctrl+C press")
	}
}

func TestCtrlCDuringSpinnerArmsCancel(t *testing.T) {
	var out bytes.Buffer
	m := newTestModel()
	m.handler = protocol.NewHandler(strings.NewReader(""), &out)
	m.spinnerActive = true

	updated, cmd := m.handleKey(tea.KeyMsg{Type: tea.KeyCtrlC})
	got := updated.(Model)

	if got.quitting {
		t.Fatal("expected model NOT to enter quitting state during spinner")
	}
	if !got.isCanceling {
		t.Fatal("expected isCanceling to be armed after first Ctrl+C")
	}
	if got.statusText != "Press Ctrl+C again to cancel" {
		t.Fatalf("expected cancel prompt, got %q", got.statusText)
	}
	if out.Len() > 0 {
		t.Fatal("expected no protocol message on first Ctrl+C press")
	}
	if cmd == nil {
		t.Fatal("expected a timer command from first Ctrl+C press")
	}
}

func TestCtrlCDoublePressSendsCancel(t *testing.T) {
	var out bytes.Buffer
	m := newTestModel()
	m.handler = protocol.NewHandler(strings.NewReader(""), &out)
	m.isStreaming = true

	// First press: arms cancel.
	updated, _ := m.handleKey(tea.KeyMsg{Type: tea.KeyCtrlC})
	m = updated.(Model)

	if !m.isCanceling {
		t.Fatal("expected isCanceling after first press")
	}

	// Second press: fires cancel.
	updated, cmd := m.handleKey(tea.KeyMsg{Type: tea.KeyCtrlC})
	got := updated.(Model)

	if got.isCanceling {
		t.Fatal("expected isCanceling to be cleared after second press")
	}
	if got.statusText != "Cancelling…" {
		t.Fatalf("expected statusText %q, got %q", "Cancelling…", got.statusText)
	}
	if cmd != nil {
		t.Fatal("expected no tea.Quit command during streaming cancel")
	}

	msg := readSingleProtocolMessage(t, &out)
	if msg.Type != protocol.TypeCancel {
		t.Fatalf("expected %q message, got %q", protocol.TypeCancel, msg.Type)
	}
}

func TestCancelTimerExpiredResetsArm(t *testing.T) {
	m := newTestModel()
	m.isStreaming = true
	m.isCanceling = true
	m.statusText = "Press Ctrl+C again to cancel"

	updated, _ := m.Update(cancelTimerExpired{})
	got := updated.(Model)

	if got.isCanceling {
		t.Fatal("expected isCanceling to be cleared after timer expiry")
	}
	if got.statusText != "" {
		t.Fatalf("expected empty statusText after timer expiry, got %q", got.statusText)
	}
}

func TestCtrlCAtIdleSendsQuit(t *testing.T) {
	var out bytes.Buffer
	m := newTestModel()
	m.handler = protocol.NewHandler(strings.NewReader(""), &out)
	m.isStreaming = false
	m.spinnerActive = false

	updated, cmd := m.handleKey(tea.KeyMsg{Type: tea.KeyCtrlC})
	got := updated.(Model)

	if !got.quitting {
		t.Fatal("expected model to enter quitting state at idle")
	}
	if cmd == nil {
		t.Fatal("expected tea.Quit command at idle")
	}

	msg := readSingleProtocolMessage(t, &out)
	if msg.Type != protocol.TypeQuit {
		t.Fatalf("expected %q message at idle, got %q", protocol.TypeQuit, msg.Type)
	}
}

// ---------------------------------------------------------------------------
// Component render tests (Phase 3: HITL Protocol Extension)
// ---------------------------------------------------------------------------

func TestComponentRenderConfirmRoutesToPendingConfirm(t *testing.T) {
	m := newTestModel()
	m.handler = protocol.NewHandler(strings.NewReader(""), &bytes.Buffer{})

	payload := protocol.ComponentRenderPayload{
		Component:      "confirm",
		Data:           map[string]any{"question": "Proceed with batch correction?", "default": true},
		FallbackPrompt: "Proceed? [Y/n]",
	}

	updated, _ := m.handleComponentRender(payload, "intr-001")
	got := updated.(Model)

	if got.pendingConfirm == nil {
		t.Fatal("expected pendingConfirm to be set")
	}
	if got.pendingConfirm.Message != "Proceed with batch correction?" {
		t.Fatalf("expected confirm message, got %q", got.pendingConfirm.Message)
	}
	if got.pendingConfirmID != "intr-001" {
		t.Fatalf("expected confirm ID 'intr-001', got %q", got.pendingConfirmID)
	}
	if !got.pendingConfirm.Default {
		t.Fatal("expected default=true")
	}
}

func TestComponentRenderSelectRoutesToPendingSelect(t *testing.T) {
	m := newTestModel()
	m.handler = protocol.NewHandler(strings.NewReader(""), &bytes.Buffer{})

	payload := protocol.ComponentRenderPayload{
		Component: "select",
		Data: map[string]any{
			"question": "Which method?",
			"options":  []any{"DESeq2", "CPM", "TMM"},
		},
	}

	updated, _ := m.handleComponentRender(payload, "intr-002")
	got := updated.(Model)

	if got.pendingSelect == nil {
		t.Fatal("expected pendingSelect to be set")
	}
	if len(got.pendingSelect.Options) != 3 {
		t.Fatalf("expected 3 options, got %d", len(got.pendingSelect.Options))
	}
	if got.pendingSelectID != "intr-002" {
		t.Fatalf("expected select ID 'intr-002', got %q", got.pendingSelectID)
	}
	if got.selectIndex != 0 {
		t.Fatalf("expected selectIndex 0, got %d", got.selectIndex)
	}
}

func TestComponentRenderTextInputSetsPendingComponentID(t *testing.T) {
	m := newTestModel()
	m.handler = protocol.NewHandler(strings.NewReader(""), &bytes.Buffer{})

	payload := protocol.ComponentRenderPayload{
		Component:      "text_input",
		Data:           map[string]any{"question": "What cell types?"},
		FallbackPrompt: "What cell types?",
	}

	updated, _ := m.handleComponentRender(payload, "intr-003")
	got := updated.(Model)

	if got.pendingComponentID != "intr-003" {
		t.Fatalf("expected pendingComponentID 'intr-003', got %q", got.pendingComponentID)
	}
	// Should have appended a system message with the fallback prompt.
	if len(got.messages) == 0 {
		t.Fatal("expected system message with fallback prompt")
	}
	last := got.messages[len(got.messages)-1]
	if last.Role != "system" {
		t.Fatalf("expected system message, got %q", last.Role)
	}
	if !strings.Contains(last.Content, "What cell types?") {
		t.Fatalf("expected fallback prompt in content, got %q", last.Content)
	}
}

func TestComponentRenderUnknownFallsBackToTextInput(t *testing.T) {
	m := newTestModel()
	m.handler = protocol.NewHandler(strings.NewReader(""), &bytes.Buffer{})

	payload := protocol.ComponentRenderPayload{
		Component:      "threshold_slider",
		Data:           map[string]any{"label": "Adjust p-value"},
		FallbackPrompt: "Adjust p-value (0.0-1.0, default: 0.05)",
	}

	updated, _ := m.handleComponentRender(payload, "intr-004")
	got := updated.(Model)

	if got.pendingComponentID != "intr-004" {
		t.Fatalf("expected pendingComponentID for threshold fallback, got %q", got.pendingComponentID)
	}
}

func TestTextInputComponentResponseSendsComponentResponse(t *testing.T) {
	var out bytes.Buffer
	m := newTestModel()
	m.handler = protocol.NewHandler(strings.NewReader(""), &out)
	m.pendingComponentID = "intr-005"
	m.ready = true
	m.input.SetValue("T cells, B cells")

	updated, _ := m.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	got := updated.(Model)

	if got.pendingComponentID != "" {
		t.Fatal("expected pendingComponentID to be cleared after submit")
	}

	msg := readSingleProtocolMessage(t, &out)
	if msg.Type != protocol.TypeComponentResponse {
		t.Fatalf("expected %q, got %q", protocol.TypeComponentResponse, msg.Type)
	}

	var p protocol.ComponentResponsePayload
	if err := protocol.DecodePayload(msg, &p); err != nil {
		t.Fatalf("decode component response: %v", err)
	}
	if p.Data["answer"] != "T cells, B cells" {
		t.Fatalf("expected answer 'T cells, B cells', got %v", p.Data["answer"])
	}
}
