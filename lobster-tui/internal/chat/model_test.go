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

	"charm.land/bubbles/v2/textarea"
	"charm.land/bubbles/v2/viewport"
	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"

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

	updated, cmd := m.handleConfirmKey(tea.KeyPressMsg{Code: 'y', Text: "y"})

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

	updated, cmd := m.handleConfirmKey(tea.KeyPressMsg{Code: 'n', Text: "n"})

	got := updated.(Model)
	if got.quitting {
		t.Fatal("expected model to remain active")
	}
	if cmd != nil {
		t.Fatal("expected no quit command on cancel")
	}
	if len(got.messages) == 0 || got.messages[len(got.messages)-1].Content() != "Exit cancelled." {
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

func TestHandleProtocolTableCreatesBlockTable(t *testing.T) {
	m := newTestModel()
	m.streamBuf.WriteString("before table text")

	updated, _ := m.handleProtocol(testProtocolMsg(t, protocol.TypeTable, protocol.TablePayload{
		Headers: []string{"Setting", "Value", "Source"},
		Rows: [][]string{
			{"Provider", "ollama", "global user config (~/.config/lobster/)"},
			{"Profile", "production", "global user config"},
		},
	}))
	m = updated.(Model)

	// streamBuf should have been flushed (table handler calls flushStreamBuffer).
	if m.streamBuf.Len() != 0 {
		t.Fatalf("expected streamBuf to be flushed, got: %q", m.streamBuf.String())
	}

	// Last assistant message should contain both BlockText (flushed) and BlockTable.
	if len(m.messages) == 0 {
		t.Fatal("expected at least one message")
	}
	last := m.messages[len(m.messages)-1]
	if len(last.Blocks) < 2 {
		t.Fatalf("expected at least 2 blocks (text + table), got %d", len(last.Blocks))
	}
	if _, ok := last.Blocks[0].(BlockText); !ok {
		t.Errorf("block 0: want BlockText, got %T", last.Blocks[0])
	}
	tb, ok := last.Blocks[1].(BlockTable)
	if !ok {
		t.Fatalf("block 1: want BlockTable, got %T", last.Blocks[1])
	}
	if len(tb.Headers) != 3 || tb.Headers[0] != "Setting" {
		t.Errorf("unexpected table headers: %v", tb.Headers)
	}
	if len(tb.Rows) != 2 || tb.Rows[0][0] != "Provider" {
		t.Errorf("unexpected table rows: %v", tb.Rows)
	}
}

func TestSelectComponentNavigationViaHandleKey(t *testing.T) {
	m := newTestModel()
	m.handler = protocol.NewHandler(strings.NewReader(""), &bytes.Buffer{})

	// Set up an active select component.
	payload := protocol.ComponentRenderPayload{
		Component: "select",
		Data: map[string]any{
			"question": "Choose one",
			"options":  []any{"a", "b", "c"},
		},
	}
	updated, _ := m.handleComponentRender(payload, "sel-nav")
	m = updated.(Model)
	if m.activeComponent == nil {
		t.Fatal("expected activeComponent to be set")
	}

	// Press Down twice (should go to index 2).
	updated, _ = m.handleKey(tea.KeyPressMsg{Code: tea.KeyDown})
	m = updated.(Model)
	updated, _ = m.handleKey(tea.KeyPressMsg{Code: tea.KeyDown})
	m = updated.(Model)
	// Component should still be active (not submitted).
	if m.activeComponent == nil {
		t.Fatal("expected activeComponent to still be active after navigation")
	}
}

func TestSelectComponentEscCancelsViaHandleKey(t *testing.T) {
	var out bytes.Buffer
	m := newTestModel()
	m.handler = protocol.NewHandler(strings.NewReader(""), &out)

	// Set up an active select component.
	payload := protocol.ComponentRenderPayload{
		Component: "select",
		Data: map[string]any{
			"question": "Choose one",
			"options":  []any{"a", "b", "c"},
		},
	}
	updated, _ := m.handleComponentRender(payload, "sel-1")
	m = updated.(Model)

	// Press Esc to cancel the select component.
	updated, _ = m.handleKey(tea.KeyPressMsg{Code: tea.KeyEscape})
	m = updated.(Model)

	if m.activeComponent != nil {
		t.Fatal("expected activeComponent to be cleared after Esc")
	}

	msg := readSingleProtocolMessage(t, &out)
	if msg.Type != protocol.TypeComponentResponse {
		t.Fatalf("expected %q message, got %q", protocol.TypeComponentResponse, msg.Type)
	}

	var p protocol.ComponentResponsePayload
	if err := protocol.DecodePayload(msg, &p); err != nil {
		t.Fatalf("decode component response payload: %v", err)
	}
	if p.ID != "sel-1" {
		t.Fatalf("expected ID 'sel-1', got %q", p.ID)
	}
}

func TestClearParityLocalAndProtocolTargetsProduceEquivalentState(t *testing.T) {
	local := seededModelForClear()
	local.input.SetValue("/clear")
	updated, _ := local.handleKey(tea.KeyPressMsg{Code: tea.KeyEnter})
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
			Role:   "assistant",
			Blocks: textBlocks(fmt.Sprintf("overflow line %d", i)),
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
	m.messages = append(m.messages, ChatMessage{Role: "assistant", Blocks: textBlocks("older output")})
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

	cmd := m.inlinePrintMessagesCmd([]ChatMessage{{Role: "user", Blocks: textBlocks("hello")}})
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

	cmd := m.appendMessage(ChatMessage{Role: "user", Blocks: textBlocks("hello")}, false)
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

	updated, _ := m.handleKey(tea.KeyPressMsg{Code: tea.KeyEnter})
	m = updated.(Model)

	if got := m.input.Value(); got != "" {
		t.Fatalf("expected composer to clear after submit, got %q", got)
	}
	if len(m.messages) == 0 {
		t.Fatal("expected submitted user message to be appended")
	}
	last := m.messages[len(m.messages)-1]
	if last.Role != "user" || last.Content() != "hello world" {
		t.Fatalf("unexpected submitted message: %#v", last)
	}
}

func TestKeyAltEnterInsertsComposerNewline(t *testing.T) {
	m := newTestModel()
	m.ready = true
	m.input.SetValue("hello")

	updated, _ := m.handleKey(tea.KeyPressMsg{Code: tea.KeyEnter, Mod: tea.ModAlt})
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

	updated, _ := m.handleKey(tea.KeyPressMsg{Code: tea.KeyTab})
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

	updated, _ := m.handleKey(tea.KeyPressMsg{Code: tea.KeyTab})
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

	updated, _ := m.handleKey(tea.KeyPressMsg{Code: tea.KeyTab})
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

	updated, _ := m.handleKey(tea.KeyPressMsg{Code: 'n', Mod: tea.ModCtrl})
	m = updated.(Model)

	updated, _ = m.handleKey(tea.KeyPressMsg{Code: tea.KeyTab})
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

	updated, _ := m.handleKey(tea.KeyPressMsg{Code: tea.KeyEscape})
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

	updated, _ := m.handleKey(tea.KeyPressMsg{Code: tea.KeyUp})
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

	updated, _ := m.handleKey(tea.KeyPressMsg{Code: tea.KeyPgUp})
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

	updated, _ := m.handleKey(tea.KeyPressMsg{Code: tea.KeyUp})
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

	updated, cmd := m.handleKey(tea.KeyPressMsg{Code: 'g', Mod: tea.ModCtrl})
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

	updated, cmd := m.handleKey(tea.KeyPressMsg{Code: 'g', Mod: tea.ModCtrl})
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
	m.messages = append(m.messages, ChatMessage{Role: "assistant", Blocks: textBlocks("one")})
	m.messages = append(m.messages, ChatMessage{Role: "assistant", Blocks: textBlocks("two")})
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
			Role:   "assistant",
			Blocks: textBlocks(fmt.Sprintf("line %d", i)),
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
// Cancel: clean rollback on done(summary=cancelled)
// ---------------------------------------------------------------------------

func TestDoneCancelledDiscardsStreamBufAndToolFeed(t *testing.T) {
	m := newTestModel()
	m.isStreaming = true
	m.isCanceling = true
	m.streamBuf.WriteString("partial AI response that should vanish")
	m.pendingHandoffs = append(m.pendingHandoffs, ChatMessage{
		Role: "assistant", Blocks: []ContentBlock{BlockText{Text: "handoff note"}},
	})
	m.toolFeed = append(m.toolFeed, ToolFeedEntry{
		Name: "analyze_data", Event: "start",
	})
	// Pre-existing chat message that should survive.
	m.messages = append(m.messages, ChatMessage{
		Role: "user", Blocks: []ContentBlock{BlockText{Text: "prior turn"}},
	})

	updated, _ := m.handleProtocol(testProtocolMsg(t, protocol.TypeDone, protocol.DonePayload{
		Summary: "cancelled",
	}))
	got := updated.(Model)

	if got.streamBuf.Len() != 0 {
		t.Fatalf("expected streamBuf to be reset, got %q", got.streamBuf.String())
	}
	if len(got.pendingHandoffs) != 0 {
		t.Fatalf("expected pendingHandoffs cleared, got %d", len(got.pendingHandoffs))
	}
	if len(got.toolFeed) != 0 {
		t.Fatalf("expected toolFeed cleared, got %d", len(got.toolFeed))
	}
	if got.isStreaming {
		t.Fatal("expected isStreaming=false")
	}
	if got.isCanceling {
		t.Fatal("expected isCanceling=false")
	}
	// Prior messages must survive — only streaming buffer is discarded.
	if len(got.messages) != 1 || got.messages[0].Content() != "prior turn" {
		t.Fatalf("expected prior message to survive, got %v", got.messages)
	}
}

func TestDoneNormalStillFlushesStreamBuf(t *testing.T) {
	m := newTestModel()
	m.isStreaming = true
	m.streamBuf.WriteString("final answer")

	updated, _ := m.handleProtocol(testProtocolMsg(t, protocol.TypeDone, protocol.DonePayload{}))
	got := updated.(Model)

	if len(got.messages) != 1 {
		t.Fatalf("expected one message flushed from streamBuf, got %d", len(got.messages))
	}
	if got.messages[0].Content() != "final answer" {
		t.Fatalf("expected flushed content, got %q", got.messages[0].Content())
	}
}

// ---------------------------------------------------------------------------
// Cancel vs Quit tests (Phase 1: Query Cancellation)
// ---------------------------------------------------------------------------

func TestCtrlCDuringStreamingSendsCancelImmediately(t *testing.T) {
	var out bytes.Buffer
	m := newTestModel()
	m.handler = protocol.NewHandler(strings.NewReader(""), &out)
	m.isStreaming = true

	updated, cmd := m.handleKey(tea.KeyPressMsg{Code: 'c', Mod: tea.ModCtrl})
	got := updated.(Model)

	if got.quitting {
		t.Fatal("expected model NOT to quit on first Ctrl+C during streaming")
	}
	if !got.isCanceling {
		t.Fatal("expected isCanceling flag set after cancel sent")
	}
	if got.statusText != "Cancelling…" {
		t.Fatalf("expected statusText %q, got %q", "Cancelling…", got.statusText)
	}
	if cmd != nil {
		t.Fatal("expected no tea.Cmd from cancel (no timer)")
	}

	msg := readSingleProtocolMessage(t, &out)
	if msg.Type != protocol.TypeCancel {
		t.Fatalf("expected %q message, got %q", protocol.TypeCancel, msg.Type)
	}
}

func TestCtrlCDuringSpinnerSendsCancelImmediately(t *testing.T) {
	var out bytes.Buffer
	m := newTestModel()
	m.handler = protocol.NewHandler(strings.NewReader(""), &out)
	m.spinnerActive = true

	updated, cmd := m.handleKey(tea.KeyPressMsg{Code: 'c', Mod: tea.ModCtrl})
	got := updated.(Model)

	if got.quitting {
		t.Fatal("expected model NOT to quit on first Ctrl+C during spinner")
	}
	if !got.isCanceling {
		t.Fatal("expected isCanceling flag set after cancel sent")
	}
	if got.statusText != "Cancelling…" {
		t.Fatalf("expected statusText %q, got %q", "Cancelling…", got.statusText)
	}
	if cmd != nil {
		t.Fatal("expected no tea.Cmd from cancel")
	}

	msg := readSingleProtocolMessage(t, &out)
	if msg.Type != protocol.TypeCancel {
		t.Fatalf("expected %q message, got %q", protocol.TypeCancel, msg.Type)
	}
}

func TestCtrlCDoublePressForcesQuit(t *testing.T) {
	var out bytes.Buffer
	m := newTestModel()
	m.handler = protocol.NewHandler(strings.NewReader(""), &out)
	m.isStreaming = true

	// First press: sends cancel immediately.
	updated, _ := m.handleKey(tea.KeyPressMsg{Code: 'c', Mod: tea.ModCtrl})
	m = updated.(Model)

	if !m.isCanceling {
		t.Fatal("expected isCanceling after first press")
	}

	// Drain the cancel message from the buffer.
	_ = readSingleProtocolMessage(t, &out)

	// Second press: streaming still active → force quit.
	updated, cmd := m.handleKey(tea.KeyPressMsg{Code: 'c', Mod: tea.ModCtrl})
	got := updated.(Model)

	if !got.quitting {
		t.Fatal("expected force quit on second Ctrl+C during active streaming")
	}
	if cmd == nil {
		t.Fatal("expected tea.Quit command on force quit")
	}

	msg := readSingleProtocolMessage(t, &out)
	if msg.Type != protocol.TypeQuit {
		t.Fatalf("expected %q message on force quit, got %q", protocol.TypeQuit, msg.Type)
	}
}

func TestCancellingStatusNotOverwrittenBySpinner(t *testing.T) {
	m := newTestModel()
	m.spinnerActive = true
	m.spinnerLabel = "thinking"
	m.isCanceling = true
	m.statusText = "Cancelling…"

	line := m.currentStatusLine()
	if !strings.Contains(line, "Cancelling…") {
		t.Fatalf("expected Cancelling status to survive spinner, got %q", line)
	}
	if strings.Contains(line, "thinking") {
		t.Fatalf("expected spinner text to be suppressed during cancel, got %q", line)
	}
}

func TestCtrlCAtIdleSendsQuit(t *testing.T) {
	var out bytes.Buffer
	m := newTestModel()
	m.handler = protocol.NewHandler(strings.NewReader(""), &out)
	m.isStreaming = false
	m.spinnerActive = false

	updated, cmd := m.handleKey(tea.KeyPressMsg{Code: 'c', Mod: tea.ModCtrl})
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

func TestComponentRenderConfirmRoutesToActiveComponent(t *testing.T) {
	m := newTestModel()
	m.handler = protocol.NewHandler(strings.NewReader(""), &bytes.Buffer{})

	payload := protocol.ComponentRenderPayload{
		Component:      "confirm",
		Data:           map[string]any{"question": "Proceed with batch correction?", "default": true},
		FallbackPrompt: "Proceed? [Y/n]",
	}

	updated, _ := m.handleComponentRender(payload, "intr-001")
	got := updated.(Model)

	if got.activeComponent == nil {
		t.Fatal("expected activeComponent to be set")
	}
	if got.activeComponent.Component.Name() != "confirm" {
		t.Fatalf("expected component name 'confirm', got %q", got.activeComponent.Component.Name())
	}
	if got.activeComponent.MsgID != "intr-001" {
		t.Fatalf("expected MsgID 'intr-001', got %q", got.activeComponent.MsgID)
	}
}

func TestComponentRenderSelectRoutesToActiveComponent(t *testing.T) {
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

	if got.activeComponent == nil {
		t.Fatal("expected activeComponent to be set")
	}
	if got.activeComponent.Component.Name() != "select" {
		t.Fatalf("expected component name 'select', got %q", got.activeComponent.Component.Name())
	}
	if got.activeComponent.MsgID != "intr-002" {
		t.Fatalf("expected MsgID 'intr-002', got %q", got.activeComponent.MsgID)
	}
}

func TestComponentRenderTextInputSetsActiveComponent(t *testing.T) {
	m := newTestModel()
	m.handler = protocol.NewHandler(strings.NewReader(""), &bytes.Buffer{})

	payload := protocol.ComponentRenderPayload{
		Component:      "text_input",
		Data:           map[string]any{"question": "What cell types?"},
		FallbackPrompt: "What cell types?",
	}

	updated, _ := m.handleComponentRender(payload, "intr-003")
	got := updated.(Model)

	if got.activeComponent == nil {
		t.Fatal("expected activeComponent to be set")
	}
	if got.activeComponent.Component.Name() != "text_input" {
		t.Fatalf("expected component name 'text_input', got %q", got.activeComponent.Component.Name())
	}
	if got.activeComponent.MsgID != "intr-003" {
		t.Fatalf("expected MsgID 'intr-003', got %q", got.activeComponent.MsgID)
	}
}

func TestComponentRenderUnknownFallsBackToTextInput(t *testing.T) {
	m := newTestModel()
	m.handler = protocol.NewHandler(strings.NewReader(""), &bytes.Buffer{})

	// Use a genuinely unknown component name to test the fallback path.
	// Previously this used "threshold_slider" which was accidentally
	// unregistered due to missing blank imports (COMP-01 fix).
	payload := protocol.ComponentRenderPayload{
		Component:      "totally_unknown_widget",
		Data:           map[string]any{"label": "Adjust p-value"},
		FallbackPrompt: "Adjust p-value (0.0-1.0, default: 0.05)",
	}

	updated, _ := m.handleComponentRender(payload, "intr-004")
	got := updated.(Model)

	if got.activeComponent == nil {
		t.Fatal("expected activeComponent to be set for unknown component fallback")
	}
	if got.activeComponent.Component.Name() != "text_input" {
		t.Fatalf("expected text_input fallback, got %q", got.activeComponent.Component.Name())
	}
	if got.activeComponent.MsgID != "intr-004" {
		t.Fatalf("expected MsgID 'intr-004', got %q", got.activeComponent.MsgID)
	}
}

func TestTextInputComponentResponseSendsComponentResponse(t *testing.T) {
	var out bytes.Buffer
	m := newTestModel()
	m.handler = protocol.NewHandler(strings.NewReader(""), &out)
	m.ready = true

	// Set up an active text_input component via handleComponentRender.
	payload := protocol.ComponentRenderPayload{
		Component: "text_input",
		Data:      map[string]any{"question": "What cell types?"},
	}
	updated, _ := m.handleComponentRender(payload, "intr-005")
	m = updated.(Model)

	if m.activeComponent == nil {
		t.Fatal("expected activeComponent to be set")
	}

	// Type text into the component, then press Enter to submit.
	// First type characters into the text_input component.
	for _, r := range "T cells, B cells" {
		m.activeComponent.Component.HandleMsg(tea.KeyPressMsg{Code: r, Text: string(r)})
	}

	// Press Enter to submit.
	updated, _ = m.handleKey(tea.KeyPressMsg{Code: tea.KeyEnter})
	got := updated.(Model)

	if got.activeComponent != nil {
		t.Fatal("expected activeComponent to be cleared after submit")
	}

	msg := readSingleProtocolMessage(t, &out)
	if msg.Type != protocol.TypeComponentResponse {
		t.Fatalf("expected %q, got %q", protocol.TypeComponentResponse, msg.Type)
	}

	var p protocol.ComponentResponsePayload
	if err := protocol.DecodePayload(msg, &p); err != nil {
		t.Fatalf("decode component response: %v", err)
	}
	answer, ok := p.Data["answer"].(string)
	if !ok {
		t.Fatalf("expected string answer, got %T: %v", p.Data["answer"], p.Data["answer"])
	}
	if answer != "T cells, B cells" {
		t.Fatalf("expected answer 'T cells, B cells', got %q", answer)
	}
}
