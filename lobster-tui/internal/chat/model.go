// Package chat implements the interactive chat BubbleTea model for Lobster AI.
//
// The model bridges the protocol handler (JSON Lines IPC with Python) and the
// BubbleTea event loop, rendering streamed text, agent transitions, tool calls,
// and user input into a terminal-based chat interface.
package chat

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"

	"charm.land/bubbles/v2/textarea"
	"charm.land/bubbles/v2/viewport"
	tea "charm.land/bubbletea/v2"
	"github.com/charmbracelet/glamour"
	"charm.land/lipgloss/v2"
	"github.com/the-omics-os/lobster-tui/internal/biocomp"
	_ "github.com/the-omics-os/lobster-tui/internal/biocomp/bioselect"
	_ "github.com/the-omics-os/lobster-tui/internal/biocomp/celltype"
	_ "github.com/the-omics-os/lobster-tui/internal/biocomp/confirm"
	_ "github.com/the-omics-os/lobster-tui/internal/biocomp/qcdash"
	_ "github.com/the-omics-os/lobster-tui/internal/biocomp/textinput"
	_ "github.com/the-omics-os/lobster-tui/internal/biocomp/threshold"
	"github.com/the-omics-os/lobster-tui/internal/protocol"
	"github.com/the-omics-os/lobster-tui/internal/theme"
)

// maxToolFeed is the maximum number of tool execution lines kept in the ring buffer.
const maxToolFeed = 5

// maxInputHistory is the number of entered lines kept for Up/Down recall.
const maxInputHistory = 200

const composerMinHeight = 1
const composerMaxHeight = 8
const maxVisibleCompletionSuggestions = 5

// spinnerInterval controls the animation frame rate (~12 fps).
const spinnerInterval = 80 * time.Millisecond

// spinnerFrames are braille-based animation frames for the active spinner.
var spinnerFrames = []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}

// usdCostRe extracts the first USD amount from status text, e.g. "$0.0123".
var usdCostRe = regexp.MustCompile(`\$(\d+(?:\.\d+)?)`)

// tipInterval controls how often loading tips rotate.
const tipInterval = 5 * time.Second

// welcome animation timing aligned with .planning/charm-tui/animated_logo.md.
const welcomeAnimInterval = 50 * time.Millisecond
const welcomeInitialScrambleFrames = 24
const welcomeFadeStepDuration = 200 * time.Millisecond
const welcomeFadeSteps = 3
const welcomeSporadicDelay = 2200 * time.Millisecond
const welcomeSporadicMinInterval = 2000 * time.Millisecond
const welcomeSporadicJitterRange = 2000 * time.Millisecond
const welcomeSporadicChance = 0.7
const welcomeSporadicFrames = 8
const welcomePersistentSparkFrameInterval = 100 * time.Millisecond
const welcomePersistentSparkFrames = 4
const welcomePersistentSparkMinInterval = 200 * time.Millisecond
const welcomePersistentSparkJitterRange = 200 * time.Millisecond

// completionRequestTimeout prevents stale in-flight request tracking forever.
const completionRequestTimeout = 700 * time.Millisecond

// localExitConfirmID identifies a Go-native confirm flow for /exit.
const localExitConfirmID = "__local_exit__"

// loadingTips are shown during init to keep users engaged.
var loadingTips = []string{
	"Lobster AI has 22 specialist agents across 10 domains",
	"Try /help for available commands",
	"Use /data to inspect loaded modalities",
	"Sessions auto-save — use --session-id latest to resume",
	"Supports scRNA-seq, bulk RNA, proteomics, genomics, metabolomics",
	"Agent transitions show which specialist is working",
	"Press Ctrl+C to exit cleanly at any time",
	"Export reproducible Jupyter notebooks with /pipeline export",
}

// spinnerTick fires periodically to advance the spinner animation frame.
type spinnerTick struct{}

// changeEventTick fires after the debounce interval to flush a pending
// ChangeEvent from an active BioComp component.
type changeEventTick struct{}

// heartbeatCheck fires periodically to verify the backend is still alive.
type heartbeatCheck struct{}

// tipRotate fires periodically to advance to the next loading tip.
type tipRotate struct{}

// welcomeTick advances the inline startup logo animation.
type welcomeTick struct{}

// protocolErr wraps an error read from the protocol handler's Errs() channel.
type protocolErr struct{ err error }

// completionTimeout clears in-flight completion state if Python does not reply.
type completionTimeout struct{ id string }

// ChatMessage represents a single message in the chat history.
type ChatMessage struct {
	// Role is one of "user", "assistant", "system", or "handoff".
	Role string
	// Blocks holds typed content blocks (text, code, table, alert, handoff).
	// Use Content() method to get a backward-compatible flat string.
	Blocks []ContentBlock
	// Agent is optional metadata for agent-specific UI treatment.
	Agent string
	// IsStreaming is true when the message is actively being streamed.
	// Streaming messages are never cached.
	IsStreaming bool
	// cache stores width-keyed rendered output for finalized messages.
	cache renderCache
}

// ToolFeedEntry represents a single tool execution with its lifecycle state.
type ToolFeedEntry struct {
	ID      string
	Name    string
	Event   protocol.ToolExecutionEvent // "start", "finish", "error"
	Summary string
	Agent   string
}

// ModalityInfo stores cached modality metadata for instant /data rendering.
type ModalityInfo struct {
	Name  string
	Shape string
}

// Model is the top-level BubbleTea model for the chat interface.
type Model struct {
	handler *protocol.Handler

	viewport viewport.Model
	input    textarea.Model

	messages        []ChatMessage
	pendingHandoffs []ChatMessage
	activeWorkers   map[string]struct{}
	toolFeed        []ToolFeedEntry
	modalities      []ModalityInfo

	statusText  string
	isStreaming bool
	streamBuf   *strings.Builder

	// Spinner animation state.
	spinnerActive bool
	spinnerFrame  int
	spinnerLabel  string

	// Loading tips rotation state.
	tipIndex int

	// Inline welcome animation state.
	showIntro           bool
	quietStartup        bool
	welcomeActive       bool
	welcomeStart        time.Time
	welcomeFrame        int
	welcomeDNA          string
	welcomeSporadicCell int
	welcomeSporadicTick int
	welcomeNextSporadic time.Time
	welcomeRNG          *rand.Rand

	// Session and metadata.
	sessionID string
	version   string
	provider  string

	// Glamour markdown renderer (lazily initialized, recreated on width change).
	mdRenderer      *glamour.TermRenderer
	mdRendererWidth int

	// Input gating: input is ignored until the backend sends "ready".
	ready bool

	// Local input history ring and navigation state.
	inputHistory      []string
	inputHistoryIndex int
	inputHistoryDraft string

	// Dynamic completion request/response tracking.
	completionSeq                int
	completionPendingID          string
	completionLastRequestedInput string
	completionRequestInputs      map[string]string
	completionCache              map[string][]string
	completionSuggestions        []string
	completionMenuIndex          int
	completionMenuInput          string
	completionDismissedInput     string

	// Heartbeat monitoring: tracks the last heartbeat from the backend.
	lastHeartbeat time.Time

	// Progress bar state.
	progressLabel   string
	progressCurrent int
	progressTotal   int
	progressActive  bool

	// Pending confirm dialog state (local exit confirm only).
	pendingConfirm   *protocol.ConfirmPayload
	pendingConfirmID string

	// Active BioComp component (replaces pendingSelect, pendingComponentID).
	activeComponent *ActiveComponent

	// Active inline form (replaces huh tea.Exec suspension).
	activeForm *inlineFormModel

	// ChangeEvent debounce state for active BioComp components.
	pendingChangeEvent   map[string]any
	changeEventMsgID     string
	changeDebounceActive bool

	width               int
	height              int
	inline              bool
	inlineFlow          bool
	inlineBannerPrinted bool
	mouseCapture        bool
	quitting            bool
	isCanceling         bool // Two-phase cancel: first Ctrl+C arms, second fires.
	styles              theme.Styles

	// Inline runtime banner details.
	totalRAMGB    int
	computeTarget string
	freeStorageGB int
	promptCostUSD float64
}

// protocolMsg wraps a protocol.Message as a tea.Msg.
type protocolMsg struct {
	protocol.Message
}

// protocolEOF signals that the protocol channel was closed (Python exited).
type protocolEOF struct{}

// ActiveComponent tracks the currently displayed BioComp component.
type ActiveComponent struct {
	Component biocomp.BioComponent
	MsgID     string    // correlation ID for protocol response
	CreatedAt time.Time
}

// NewModel creates a new chat Model wired to the given handler and styles.
func NewModel(handler *protocol.Handler, styles theme.Styles, width, height int, inline bool, mouseCapture bool, versionFallback string) Model {
	vp := viewport.New(viewport.WithWidth(width), viewport.WithHeight(1))
	vp.SetContent("")

	ti := textarea.New()
	ti.Prompt = ""
	ti.ShowLineNumbers = false
	if inline {
		ti.Placeholder = ""
	} else {
		ti.Placeholder = "Initializing..."
	}
	ti.Focus()
	ti.CharLimit = 4096
	ti.SetHeight(composerMinHeight)
	ti.SetWidth(width)
	applyComposerStyles(&ti, styles)

	version := resolveLobsterVersion(versionFallback)
	provider := normalizeProviderName(os.Getenv("LOBSTER_TUI_PROVIDER"))
	workspacePath := os.Getenv("LOBSTER_TUI_WORKSPACE")
	if workspacePath == "" {
		wd, err := os.Getwd()
		if err == nil {
			workspacePath = wd
		}
	}
	ramGB := detectTotalRAMGB()
	compute := detectComputeTarget()
	freeGB := detectFreeStorageGB(workspacePath)
	showIntro := shouldShowInlineIntro(inline)
	quietStartup := inline && !showIntro
	welcomeActive := showIntro
	welcomeStart := time.Now()
	welcomeDNA := makeDNASequence(width)

	model := Model{
		handler:                 handler,
		viewport:                vp,
		input:                   ti,
		messages:                make([]ChatMessage, 0, 64),
		pendingHandoffs:         make([]ChatMessage, 0, 8),
		activeWorkers:           make(map[string]struct{}, 4),
		toolFeed:                make([]ToolFeedEntry, 0, maxToolFeed),
		modalities:              make([]ModalityInfo, 0, 8),
		inputHistory:            make([]string, 0, maxInputHistory),
		inputHistoryIndex:       -1,
		streamBuf:               &strings.Builder{},
		completionRequestInputs: make(map[string]string, 8),
		completionCache:         make(map[string][]string, 16),
		ready:                   false,
		lastHeartbeat:           time.Now(),
		styles:                  styles,
		width:                   width,
		height:                  height,
		inline:                  inline,
		inlineFlow:              inline,
		mouseCapture:            mouseCapture,
		version:                 version,
		provider:                provider,
		totalRAMGB:              ramGB,
		computeTarget:           compute,
		freeStorageGB:           freeGB,
		showIntro:               showIntro,
		quietStartup:            quietStartup,
		welcomeActive:           welcomeActive,
		welcomeStart:            welcomeStart,
		welcomeDNA:              welcomeDNA,
		welcomeSporadicCell:     -1,
		welcomeRNG:              rand.New(rand.NewSource(time.Now().UnixNano())),
	}
	model.recalculateViewportHeight()
	return model
}

// Init starts the protocol read loop and returns the initial commands.
func (m Model) Init() tea.Cmd {
	m.handler.StartReadLoop()
	cmds := []tea.Cmd{
		waitForProtocolMsg(m.handler),
		waitForProtocolErr(m.handler),
		func() tea.Msg { return textarea.Blink() },
		tea.Tick(5*time.Second, func(time.Time) tea.Msg { return heartbeatCheck{} }),
	}
	if m.inline && m.showIntro {
		cmds = append(cmds, tea.Tick(welcomeAnimInterval, func(time.Time) tea.Msg { return welcomeTick{} }))
	}
	return tea.Batch(cmds...)
}

// Update handles all incoming messages (protocol events, key presses, window resize).
func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmds []tea.Cmd

	switch msg := msg.(type) {

	case protocolMsg:
		return m.handleProtocol(msg)

	case protocolEOF:
		m.quitting = true
		return m, tea.Quit

	case protocolErr:
		// Log but don't crash — keep draining the error channel.
		return m, waitForProtocolErr(m.handler)

	case changeEventTick:
		m.changeDebounceActive = false
		if m.pendingChangeEvent != nil && m.activeComponent != nil {
			if m.handler != nil {
				_ = m.handler.SendTyped("change_event", map[string]any{
					"id":   m.changeEventMsgID,
					"data": m.pendingChangeEvent,
				}, m.changeEventMsgID)
			}
			m.pendingChangeEvent = nil
		}
		return m, nil

	case spinnerTick:
		if !m.spinnerActive || (m.quietStartup && !m.ready) {
			return m, nil
		}
		m.spinnerFrame = (m.spinnerFrame + 1) % len(spinnerFrames)
		return m, tea.Tick(spinnerInterval, func(time.Time) tea.Msg { return spinnerTick{} })

	case tipRotate:
		if !m.spinnerActive || m.ready {
			return m, nil // Stop rotating once ready or spinner stopped.
		}
		m.tipIndex = (m.tipIndex + 1) % len(loadingTips)
		return m, tea.Tick(tipInterval, func(time.Time) tea.Msg { return tipRotate{} })

	case welcomeTick:
		if !m.inline || !m.showIntro {
			return m, nil
		}
		if m.welcomeRNG == nil {
			m.welcomeRNG = rand.New(rand.NewSource(time.Now().UnixNano()))
		}
		now := time.Now()
		if m.welcomeActive {
			m.welcomeFrame++
			if now.Sub(m.welcomeStart) >= welcomeSporadicDelay {
				if m.welcomeNextSporadic.IsZero() {
					m.welcomeNextSporadic = now.Add(nextWelcomeSporadicDelay(m.welcomeRNG))
				}
				if m.welcomeSporadicCell >= 0 {
					m.welcomeSporadicTick++
					if m.welcomeSporadicTick >= welcomeSporadicFrames {
						m.welcomeSporadicCell = -1
						m.welcomeSporadicTick = 0
					}
				} else if !m.welcomeNextSporadic.After(now) {
					if m.welcomeRNG.Float64() < welcomeSporadicChance {
						cellCount := welcomeTitleCellCount(m.width)
						if cellCount > 0 {
							m.welcomeSporadicCell = m.welcomeRNG.Intn(cellCount)
							m.welcomeSporadicTick = 0
						}
					}
					m.welcomeNextSporadic = now.Add(nextWelcomeSporadicDelay(m.welcomeRNG))
				}
			}
			if m.ready {
				m.welcomeActive = false
				m.welcomeSporadicCell = -1
				m.welcomeSporadicTick = 0
				m.welcomeNextSporadic = now.Add(nextWelcomePersistentSparkDelay(m.welcomeRNG))
			}
			return m, tea.Tick(welcomeAnimInterval, func(time.Time) tea.Msg { return welcomeTick{} })
		}

		// Persistent low-frequency spark mode while user is in chat.
		if m.welcomeSporadicCell >= 0 {
			m.welcomeFrame++
			m.welcomeSporadicTick++
			if m.welcomeSporadicTick >= welcomePersistentSparkFrames {
				m.welcomeSporadicCell = -1
				m.welcomeSporadicTick = 0
				m.welcomeNextSporadic = now.Add(nextWelcomePersistentSparkDelay(m.welcomeRNG))
				delay := time.Until(m.welcomeNextSporadic)
				if delay < 50*time.Millisecond {
					delay = 50 * time.Millisecond
				}
				return m, tea.Tick(delay, func(time.Time) tea.Msg { return welcomeTick{} })
			}
			return m, tea.Tick(welcomePersistentSparkFrameInterval, func(time.Time) tea.Msg { return welcomeTick{} })
		}

		if m.welcomeNextSporadic.IsZero() {
			m.welcomeNextSporadic = now.Add(nextWelcomePersistentSparkDelay(m.welcomeRNG))
		}
		if !m.welcomeNextSporadic.After(now) {
			if m.welcomeRNG.Float64() < welcomeSporadicChance {
				cellCount := welcomeTitleCellCount(m.width)
				if cellCount > 0 {
					m.welcomeSporadicCell = m.welcomeRNG.Intn(cellCount)
					m.welcomeSporadicTick = 0
					m.welcomeFrame++
					return m, tea.Tick(welcomePersistentSparkFrameInterval, func(time.Time) tea.Msg { return welcomeTick{} })
				}
			}
			m.welcomeNextSporadic = now.Add(nextWelcomePersistentSparkDelay(m.welcomeRNG))
		}

		delay := time.Until(m.welcomeNextSporadic)
		if delay < 50*time.Millisecond {
			delay = 50 * time.Millisecond
		}
		return m, tea.Tick(delay, func(time.Time) tea.Msg { return welcomeTick{} })

	case completionTimeout:
		delete(m.completionRequestInputs, msg.id)
		if m.completionPendingID == msg.id {
			m.completionPendingID = ""
		}
		return m, nil

	case heartbeatCheck:
		if m.ready {
			return m, nil // Stop checking once backend is ready.
		}
		elapsed := time.Since(m.lastHeartbeat)
		if elapsed > 15*time.Second {
			m.statusText = "Backend not responding..."
		}
		return m, tea.Tick(5*time.Second, func(time.Time) tea.Msg { return heartbeatCheck{} })

	case tea.ResumeMsg:
		// After tea.Suspend returns, BubbleTea fires ResumeMsg.
		// Re-subscribe to the protocol read loop which was broken by Suspend.
		return m, waitForProtocolMsg(m.handler)

	case tea.KeyPressMsg:
		// If an inline form is active, delegate all key input to it.
		if m.activeForm != nil {
			return m.updateActiveForm(msg)
		}
		return m.handleKey(msg)

	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		m.viewport.SetWidth(m.width)
		m.recalculateViewportHeight()

		// Invalidate glamour renderer on width change (recreated lazily).
		m.mdRenderer = nil

		// Re-render messages with new width.
		m.rebuildViewport()

		// Inform the Python side of the new dimensions.
		_ = m.handler.SendTyped(protocol.TypeResize, protocol.ResizePayload{
			Width:  m.width,
			Height: m.height,
		}, "")

		return m, nil
	}

	// Forward unhandled messages to sub-models.
	var cmd tea.Cmd
	m.viewport, cmd = m.viewport.Update(msg)
	if cmd != nil {
		cmds = append(cmds, cmd)
	}
	m.input, cmd = m.input.Update(msg)
	if cmd != nil {
		cmds = append(cmds, cmd)
	}

	return m, tea.Batch(cmds...)
}

// View renders the full TUI layout.
func (m Model) View() tea.View {
	if m.quitting {
		return tea.NewView("")
	}

	var b strings.Builder
	completionView := m.renderCompletionMenu()

	if m.inline {
		intro := renderInlineIntro(m)
		if intro != "" {
			b.WriteString(intro)
			b.WriteByte('\n')
		}
	}

	// Header/runtime chrome.
	if m.shouldRenderHeaderInFrame() {
		b.WriteString(renderHeader(m))
		b.WriteByte('\n')
		if m.inline {
			runtimeSummary := renderRuntimeSummary(m)
			if runtimeSummary != "" {
				b.WriteString(runtimeSummary)
				b.WriteString("\n\n")
			}
		}
	}

	// Viewport (scrollable message history).
	// In inline mode, trim Bubble viewport padding so the input appears
	// directly below content instead of at the bottom of the terminal.
	vpView := m.viewport.View()
	if m.inline {
		vpView = strings.TrimRight(vpView, " \n\r\t")
	}
	if !m.inlineFlowMode() {
		vpView = renderViewportWithScrollbar(vpView, m.viewport, m.styles)
	}
	if vpView != "" {
		b.WriteString(vpView)
		b.WriteByte('\n')
	} else if !m.inline {
		b.WriteByte('\n')
	}

	// Tool feed (0-N lines, dim).
	tf := renderToolFeed(m.toolFeed, m.styles, m.width, m.inline)
	if tf != "" {
		b.WriteString(tf)
		b.WriteByte('\n')
	}

	// Inline BioComp component (rendered as a block above the composer, non-interactive).
	if m.activeComponent != nil && m.activeComponent.Component.Mode() == "inline" {
		comp := m.activeComponent.Component
		inlineView, panicked := safeView(comp, m.width, m.height)
		if panicked {
			m.sendComponentResponse(m.activeComponent.MsgID, "error", map[string]any{"error": "view_panic"})
			m.activeComponent = nil
		} else if inlineView != "" {
			b.WriteString(inlineView)
			b.WriteByte('\n')
		}
	}

	// Progress bar (0-1 line).
	if m.progressActive {
		b.WriteString(renderProgressBar(m.progressLabel, m.progressCurrent, m.progressTotal, m.width, m.styles))
		b.WriteByte('\n')
	}

	// Inline form (protocol form rendered in-place, replaces huh tea.Exec).
	if m.activeForm != nil {
		b.WriteString(m.activeForm.View())
		b.WriteByte('\n')
	}

	// Input field (1 line) — replaced by overlay component or local confirm when active.
	overlayRendered := false
	if m.activeComponent != nil && m.activeComponent.Component.Mode() == "overlay" {
		comp := m.activeComponent.Component
		overlayW, overlayH := biocomp.OverlaySize(m.width, m.height, "small")
		switch comp.Name() {
		case "cell_type_selector", "ontology_browser":
			overlayW, overlayH = biocomp.OverlaySize(m.width, m.height, "large")
		case "threshold_slider":
			overlayW, overlayH = biocomp.OverlaySize(m.width, m.height, "medium")
		}
		contentW := overlayW - 4
		contentH := overlayH - 6
		if contentW < 1 {
			contentW = 1
		}
		if contentH < 1 {
			contentH = 1
		}
		helpBar := biocomp.RenderHelpBar(comp.KeyBindings(), contentW)
		content, viewPanicked := safeView(comp, contentW, contentH)
		if viewPanicked {
			m.sendComponentResponse(m.activeComponent.MsgID, "error", map[string]any{"error": "view_panic"})
			m.activeComponent = nil
		} else {
			frame := biocomp.RenderFrame(comp.Name(), content, helpBar, overlayW, overlayH)
			centered := lipgloss.Place(m.width, m.height, lipgloss.Center, lipgloss.Center, frame)
			b.WriteString(centered)
			b.WriteByte('\n')
			overlayRendered = true
		}
	}
	if overlayRendered {
		// Overlay was rendered; skip composer/confirm.
	} else if m.pendingConfirm != nil {
		// Local exit confirm only (not protocol components).
		b.WriteString(renderConfirmPrompt(m.pendingConfirm, m.styles, m.width))
		b.WriteByte('\n')
	} else {
		// Add breathing room before the prompt whenever there is visible
		// output above it in inline mode.
		if m.inline {
			hasOutputAbove := strings.TrimSpace(vpView) != "" || tf != "" || m.progressActive
			if hasOutputAbove {
				b.WriteByte('\n')
			}
		}
		if completionView != "" {
			b.WriteString(completionView)
			b.WriteByte('\n')
		}
		b.WriteString(m.renderComposer())
		b.WriteByte('\n')
	}

	// Status bar (1 line). Override status text with animated spinner when active.
	statusText := m.currentStatusLine()
	if m.inline {
		if strings.TrimSpace(statusText) != "" {
			b.WriteString(m.styles.Dimmed.Render(statusText))
		}
	} else {
		b.WriteString(renderStatusBar(statusText, m.styles, m.width))
	}

	v := tea.NewView(b.String())
	if !m.inline {
		v.AltScreen = true
	}
	if m.mouseCapture {
		v.MouseMode = tea.MouseModeCellMotion
	}
	return v
}

// --------------------------------------------------------------------------
// Protocol message dispatch
// --------------------------------------------------------------------------

func (m Model) handleProtocol(msg protocolMsg) (tea.Model, tea.Cmd) {
	switch msg.Type {

	case protocol.TypeText:
		var p protocol.TextPayload
		if err := protocol.DecodePayload(msg.Message, &p); err == nil {
			m.isStreaming = true
			m.streamBuf.WriteString(p.Content)
			if m.shouldRenderStreamingTranscript() {
				m.rebuildViewport()
			}
		}
		return m, waitForProtocolMsg(m.handler)

	case protocol.TypeMarkdown:
		// Forward-compatible: treat markdown blocks same as text (glamour renders in View).
		var p protocol.MarkdownPayload
		if err := protocol.DecodePayload(msg.Message, &p); err == nil {
			m.isStreaming = true
			m.streamBuf.WriteString(p.Content)
			if m.shouldRenderStreamingTranscript() {
				m.rebuildViewport()
			}
		}
		return m, waitForProtocolMsg(m.handler)

	case protocol.TypeCode:
		// Create a typed BlockCode instead of writing markdown fences to streamBuf.
		var p protocol.CodePayload
		if err := protocol.DecodePayload(msg.Message, &p); err == nil {
			m.isStreaming = true
			m.flushStreamBuffer()
			m.appendBlock(BlockCode{Language: p.Language, Content: p.Content})
			if m.shouldRenderStreamingTranscript() {
				m.rebuildViewport()
			}
		}
		return m, waitForProtocolMsg(m.handler)

	case protocol.TypeAgentTransition:
		var p protocol.AgentTransitionPayload
		if err := protocol.DecodePayload(msg.Message, &p); err == nil {
			if m.isActivityTransition(p) {
				m.applyWorkerActivity(p)
				return m, waitForProtocolMsg(m.handler)
			}

			task := strings.TrimSpace(p.Reason)
			if task == "" {
				task = fmt.Sprintf("handoff to %s", p.To)
			}
			printCmd := m.recordHandoff(task, p.To)
			return m, tea.Batch(waitForProtocolMsg(m.handler), printCmd)
		}
		return m, waitForProtocolMsg(m.handler)

	case protocol.TypeToolExecution:
		var p protocol.ToolExecutionPayload
		if err := protocol.DecodePayload(msg.Message, &p); err == nil {
			entry := ToolFeedEntry{
				ID:      strings.TrimSpace(p.ToolCallID),
				Name:    p.ToolName,
				Event:   p.Event,
				Summary: p.Summary,
				Agent:   p.Agent,
			}
			m.applyToolFeedEntry(entry)
			m.recalculateViewportHeight()
			m.rebuildViewport()
		}
		return m, waitForProtocolMsg(m.handler)

	case protocol.TypeDone:
		var dp protocol.DonePayload
		_ = protocol.DecodePayload(msg.Message, &dp)

		if dp.Summary == "cancelled" {
			// Cancel: discard partial stream, don't append to chat history.
			m.streamBuf.Reset()
			m.pendingHandoffs = m.pendingHandoffs[:0]
			m.isStreaming = false
			m.isCanceling = false
			m.toolFeed = m.toolFeed[:0]
			m.recalculateViewportHeight()
			m.rebuildViewportWithMode(true)
			return m, waitForProtocolMsg(m.handler)
		}

		// Flush any remaining streamed text into typed blocks, then append
		// buffered handoff lines that belong directly beneath the response.
		m.flushStreamBuffer()

		toAppend := make([]ChatMessage, 0, len(m.pendingHandoffs))
		if len(m.pendingHandoffs) > 0 {
			toAppend = append(toAppend, m.pendingHandoffs...)
			m.pendingHandoffs = m.pendingHandoffs[:0]
		}
		m.isStreaming = false
		m.isCanceling = false
		printCmd := m.appendMessages(toAppend, true)
		if len(toAppend) == 0 {
			m.rebuildViewportWithMode(true)
		}
		return m, tea.Batch(waitForProtocolMsg(m.handler), printCmd)

	case protocol.TypeStatus:
		var p protocol.StatusPayload
		if err := protocol.DecodePayload(msg.Message, &p); err == nil {
			text := strings.TrimSpace(p.Text)
			// Extract session ID if present.
			if strings.HasPrefix(text, "Session: ") {
				m.sessionID = strings.TrimSpace(strings.TrimPrefix(text, "Session: "))
				return m, waitForProtocolMsg(m.handler)
			}
			// Extract provider if present.
			if strings.HasPrefix(text, "Provider: ") {
				m.provider = normalizeProviderName(strings.TrimSpace(strings.TrimPrefix(text, "Provider: ")))
				return m, waitForProtocolMsg(m.handler)
			}

			m.statusText = text
			if cost, ok := extractUSDCost(text); ok {
				m.promptCostUSD = cost
			}
			m.recalculateViewportHeight()
		}
		return m, waitForProtocolMsg(m.handler)

	case protocol.TypeAlert:
		var p protocol.AlertPayload
		if err := protocol.DecodePayload(msg.Message, &p); err == nil {
			printCmd := m.appendMessage(ChatMessage{
				Role:   "alert_" + string(p.Level),
				Blocks: []ContentBlock{BlockAlert{Level: string(p.Level), Message: p.Message}},
			}, false)
			return m, tea.Batch(waitForProtocolMsg(m.handler), printCmd)
		}
		return m, waitForProtocolMsg(m.handler)

	case protocol.TypeSpinner:
		var p protocol.SpinnerPayload
		if err := protocol.DecodePayload(msg.Message, &p); err == nil {
			if p.Active {
				m.spinnerActive = true
				m.spinnerLabel = p.Label
				if m.spinnerLabel == "" {
					m.spinnerLabel = "thinking"
				}
				m.spinnerFrame = 0
				m.recalculateViewportHeight()
				// Quiet startup keeps a static status line for automation-friendly PTY capture.
				cmds := []tea.Cmd{waitForProtocolMsg(m.handler)}
				if !m.quietStartup || m.ready {
					cmds = append(cmds, tea.Tick(spinnerInterval, func(time.Time) tea.Msg { return spinnerTick{} }))
				}
				if !m.ready && !m.quietStartup {
					cmds = append(cmds, tea.Tick(tipInterval, func(time.Time) tea.Msg { return tipRotate{} }))
				}
				return m, tea.Batch(cmds...)
			} else {
				m.spinnerActive = false
				m.statusText = ""
				m.recalculateViewportHeight()
			}
		}
		return m, waitForProtocolMsg(m.handler)

	case protocol.TypeReady:
		m.ready = true
		m.welcomeActive = false
		m.welcomeSporadicCell = -1
		m.welcomeSporadicTick = 0
		if m.showIntro {
			m.welcomeNextSporadic = time.Now().Add(nextWelcomePersistentSparkDelay(m.welcomeRNG))
		} else {
			m.welcomeNextSporadic = time.Time{}
		}
		if m.inline {
			m.input.Placeholder = ""
		} else {
			m.input.Placeholder = "Ask Lobster anything..."
		}
		m.input.Focus()
		m.recalculateViewportHeight()
		return m, waitForProtocolMsg(m.handler)

	case protocol.TypeHeartbeat:
		m.lastHeartbeat = time.Now()
		return m, waitForProtocolMsg(m.handler)

	case protocol.TypeCompletionResponse:
		var p protocol.CompletionResponsePayload
		if err := protocol.DecodePayload(msg.Message, &p); err == nil {
			requestInput := ""
			if msg.ID != "" {
				if in, ok := m.completionRequestInputs[msg.ID]; ok {
					requestInput = in
					delete(m.completionRequestInputs, msg.ID)
				} else {
					return m, waitForProtocolMsg(m.handler)
				}
				if m.completionPendingID == msg.ID {
					m.completionPendingID = ""
				}
			}

			if requestInput != "" {
				if p.Error == "" {
					suggestions := sanitizeSuggestions(p.Suggestions)
					if len(suggestions) > 0 {
						if len(m.completionCache) >= 256 {
							m.completionCache = make(map[string][]string, 16)
						}
						m.completionCache[requestInput] = suggestions
					} else {
						delete(m.completionCache, requestInput)
					}
				}
				if requestInput == m.input.Value() {
					if cmd := m.refreshSuggestions(); cmd != nil {
						return m, tea.Batch(waitForProtocolMsg(m.handler), cmd)
					}
				}
			}
		}
		return m, waitForProtocolMsg(m.handler)

	case protocol.TypeSuspend:
		return m, tea.Suspend

	case protocol.TypeResume:
		// BubbleTea auto-redraws on resume; nothing extra needed.
		return m, waitForProtocolMsg(m.handler)

	case protocol.TypeClear:
		var p protocol.ClearPayload
		if err := protocol.DecodePayload(msg.Message, &p); err == nil {
			m.applyClearTarget(p.Target)
		}
		return m, waitForProtocolMsg(m.handler)

	case protocol.TypeModalityLoaded:
		var p protocol.ModalityLoadedPayload
		if err := protocol.DecodePayload(msg.Message, &p); err == nil {
			info := fmt.Sprintf("📊 Loaded: %s", p.Name)
			if p.Shape != "" {
				info += fmt.Sprintf(" (%s)", p.Shape)
			}
			printCmd := m.appendMessage(ChatMessage{
				Role:   "system",
				Blocks: textBlocks(m.styles.ModalityLoaded.Render(info)),
			}, false)
			m.modalities = append(m.modalities, ModalityInfo{Name: p.Name, Shape: p.Shape})
			return m, tea.Batch(waitForProtocolMsg(m.handler), printCmd)
		}
		return m, waitForProtocolMsg(m.handler)

	case protocol.TypeProgress:
		var p protocol.ProgressPayload
		if err := protocol.DecodePayload(msg.Message, &p); err == nil {
			if p.Done {
				m.progressActive = false
				m.progressLabel = ""
				m.progressCurrent = 0
				m.progressTotal = 0
			} else {
				m.progressActive = true
				m.progressLabel = p.Label
				m.progressCurrent = p.Current
				m.progressTotal = p.Total
			}
			m.recalculateViewportHeight()
		}
		return m, waitForProtocolMsg(m.handler)

	case protocol.TypeTable:
		var p protocol.TablePayload
		if err := protocol.DecodePayload(msg.Message, &p); err == nil && len(p.Headers) > 0 {
			m.isStreaming = true
			m.flushStreamBuffer()
			m.appendBlock(BlockTable{Headers: p.Headers, Rows: p.Rows})
			if m.shouldRenderStreamingTranscript() {
				m.rebuildViewport()
			}
		}
		return m, waitForProtocolMsg(m.handler)

	case protocol.TypeForm:
		var p protocol.FormPayload
		if err := protocol.DecodePayload(msg.Message, &p); err == nil {
			m, cmd := m.handleFormMessage(p, msg.ID)
			return m, tea.Batch(waitForProtocolMsg(m.handler), cmd)
		}
		return m, waitForProtocolMsg(m.handler)

	case protocol.TypeConfirm:
		var p protocol.ConfirmPayload
		if err := protocol.DecodePayload(msg.Message, &p); err == nil {
			crp := protocol.ComponentRenderPayload{
				Component: "confirm",
				Data: map[string]any{
					"question": p.Message,
					"default":  p.Default,
				},
			}
			return m.handleComponentRender(crp, msg.ID)
		}
		return m, waitForProtocolMsg(m.handler)

	case protocol.TypeSelect:
		var p protocol.SelectPayload
		if err := protocol.DecodePayload(msg.Message, &p); err == nil {
			crp := protocol.ComponentRenderPayload{
				Component: "select",
				Data: map[string]any{
					"question": p.Message,
					"options":  p.Options,
				},
			}
			return m.handleComponentRender(crp, msg.ID)
		}
		return m, waitForProtocolMsg(m.handler)

	case protocol.TypeComponentRender:
		var p protocol.ComponentRenderPayload
		if err := protocol.DecodePayload(msg.Message, &p); err == nil {
			return m.handleComponentRender(p, msg.ID)
		}
		return m, waitForProtocolMsg(m.handler)

	case protocol.TypeComponentSetData:
		var p protocol.ComponentSetDataPayload
		if err := protocol.DecodePayload(msg.Message, &p); err == nil {
			// COMP-08: Stale guard — only route if MsgID matches.
			if m.activeComponent != nil && m.activeComponent.MsgID == p.ID {
				_ = m.activeComponent.Component.SetData(p.Data)
			}
			// Silently discard if no active component or MsgID mismatch.
		}
		return m, waitForProtocolMsg(m.handler)

	default:
		// Unknown message types are silently ignored for forward compatibility.
		return m, waitForProtocolMsg(m.handler)
	}
}

// --------------------------------------------------------------------------
// Key dispatch
// --------------------------------------------------------------------------

func (m Model) handleKey(msg tea.KeyPressMsg) (tea.Model, tea.Cmd) {
	if msg.String() == "ctrl+g" {
		m.mouseCapture = !m.mouseCapture
		return m, mouseCaptureCmd(m.mouseCapture)
	}

	// Intercept keys for active BioComp overlay component.
	if m.activeComponent != nil && m.activeComponent.Component.Mode() == "overlay" {
		result, panicked := safeHandleMsg(m.activeComponent.Component, msg)
		if panicked {
			m.sendComponentResponse(m.activeComponent.MsgID, "error", map[string]any{"error": "handleMsg_panic"})
			m.activeComponent = nil
			m.recalculateViewportHeight()
			return m, nil
		}
		if result != nil {
			m.sendComponentResponse(m.activeComponent.MsgID, result.Action, result.Data)
			m.activeComponent = nil
			m.pendingChangeEvent = nil
			m.changeDebounceActive = false
			m.recalculateViewportHeight()
			return m, nil
		}
		// COMP-05: Poll ChangeEvent after HandleMsg when result is nil
		// (user still interacting). Debounce at 50ms to avoid flooding.
		if ce := m.activeComponent.Component.ChangeEvent(); ce != nil {
			m.pendingChangeEvent = ce
			m.changeEventMsgID = m.activeComponent.MsgID
			if !m.changeDebounceActive {
				m.changeDebounceActive = true
				return m, tea.Tick(50*time.Millisecond, func(time.Time) tea.Msg {
					return changeEventTick{}
				})
			}
		}
		return m, nil
	}

	// Intercept keys for pending LOCAL confirm dialog (exit/quit only).
	if m.pendingConfirm != nil {
		return m.handleConfirmKey(msg)
	}

	if m.completionMenuVisible() {
		switch msg.String() {
		case "esc":
			m.dismissCompletionMenu()
			return m, nil
		case "tab":
			if m.applySelectedCompletionSuggestion() {
				m.recalculateViewportHeight()
				return m, m.refreshSuggestions()
			}
			return m, nil
		case "ctrl+n":
			if m.moveCompletionSelection(1) {
				return m, nil
			}
		case "ctrl+p":
			if m.moveCompletionSelection(-1) {
				return m, nil
			}
		}
	}

	switch msg.String() {

	case "ctrl+c":
		if m.isStreaming || m.spinnerActive {
			if m.isCanceling {
				// Cancel already sent but streaming persists — force quit.
				if m.handler != nil {
					_ = m.handler.SendTyped(protocol.TypeQuit, protocol.QuitPayload{}, "")
				}
				m.quitting = true
				return m, tea.Quit
			}
			// First press: send cancel immediately.
			m.isCanceling = true
			m.statusText = "Cancelling…"
			if m.handler != nil {
				_ = m.handler.SendTyped(protocol.TypeCancel, protocol.CancelPayload{}, "")
			}
			return m, nil
		}
		// Idle prompt — quit.
		if m.handler != nil {
			_ = m.handler.SendTyped(protocol.TypeQuit, protocol.QuitPayload{}, "")
		}
		m.quitting = true
		return m, tea.Quit

	case "enter", "alt+enter", "shift+enter":
		if isComposerInsertNewlineKey(msg) {
			prevVal := m.input.Value()
			m.input.InsertRune('\n')
			if m.inputHistoryIndex != -1 && m.input.Value() != prevVal {
				m.resetHistoryNavigation()
			}
			m.recalculateViewportHeight()
			return m, m.refreshSuggestions()
		}

		if !m.ready {
			return m, nil // Silently ignore input until backend is ready.
		}
		val := strings.TrimSpace(m.input.Value())
		if val == "" {
			return m, nil
		}
		m.pushInputHistory(val)

		if strings.HasPrefix(val, "/") {
			parts := strings.SplitN(val[1:], " ", 2)
			cmd := strings.ToLower(parts[0])
			args := ""
			if len(parts) > 1 {
				args = parts[1]
			}
			_ = args // suppress unused warning for Go-native cases

			switch cmd {
			case "clear":
				m.applyClearTarget("output")
				m.input.SetValue("")
				m.recalculateViewportHeight()
				return m, m.refreshSuggestions()
			case "exit", "quit":
				m.pendingConfirm = &protocol.ConfirmPayload{
					Title:   "Confirm Exit",
					Message: "Exit Lobster session?",
					Default: false,
				}
				m.pendingConfirmID = localExitConfirmID
				return m, nil
			case "dashboard":
				printCmd := m.appendMessage(ChatMessage{
					Role:   "system",
					Blocks: textBlocks("Dashboard is not available in Go TUI mode. Use `lobster chat --ui classic` for the Textual dashboard."),
				}, false)
				m.input.SetValue("")
				m.recalculateViewportHeight()
				cmd := m.refreshSuggestions()
				return m, tea.Batch(cmd, printCmd)
			default:
				// Forward to Python for handling.
				if m.handler != nil {
					_ = m.handler.SendTyped(protocol.TypeSlashCommand, protocol.SlashCommandPayload{
						Command: cmd,
						Args:    args,
					}, "")
				}
			}
		} else {
			if m.handler != nil {
				_ = m.handler.SendTyped(protocol.TypeInput, protocol.InputPayload{
					Content: val,
				}, "")
			}
		}

		printCmd := m.appendMessage(ChatMessage{
			Role:   "user",
			Blocks: textBlocks(val),
		}, false)
		m.isStreaming = true
		m.input.SetValue("")
		m.recalculateViewportHeight()
		cmd := m.refreshSuggestions()

		return m, tea.Batch(cmd, printCmd)

	case "tab":
		if m.applySelectedCompletionSuggestion() {
			m.recalculateViewportHeight()
			return m, m.refreshSuggestions()
		}
		return m, nil

	case "pgup", "pgdown":
		// Transcript scrollback controls are dedicated to page keys.
		var cmd tea.Cmd
		m.viewport, cmd = m.viewport.Update(msg)
		return m, cmd

	case "up":
		// Up/Down are reserved for input history navigation.
		if m.recallHistoryUp() {
			m.recalculateViewportHeight()
			return m, m.refreshSuggestions()
		}
		return m, nil

	case "down":
		// Up/Down are reserved for input history navigation.
		if m.recallHistoryDown() {
			m.recalculateViewportHeight()
			return m, m.refreshSuggestions()
		}
		return m, nil

	default:
		// Forward keys to text input first, then conditionally to viewport.
		var cmds []tea.Cmd
		var cmd tea.Cmd
		prevVal := m.input.Value()

		m.input, cmd = m.input.Update(msg)
		if cmd != nil {
			cmds = append(cmds, cmd)
		}
		if m.inputHistoryIndex != -1 && m.input.Value() != prevVal {
			m.resetHistoryNavigation()
		}
		m.recalculateViewportHeight()

		// Refresh autocomplete suggestions after every keystroke.
		cmd = m.refreshSuggestions()
		if cmd != nil {
			cmds = append(cmds, cmd)
		}

		m.viewport, cmd = m.viewport.Update(msg)
		if cmd != nil {
			cmds = append(cmds, cmd)
		}

		return m, tea.Batch(cmds...)
	}
}

func isComposerInsertNewlineKey(msg tea.KeyPressMsg) bool {
	key := strings.ToLower(strings.TrimSpace(msg.String()))
	return key == "shift+enter" || key == "alt+enter"
}

func (m *Model) applySelectedCompletionSuggestion() bool {
	if !m.completionMenuVisible() {
		return false
	}

	limit := m.visibleCompletionSuggestionCount()
	if limit == 0 {
		return false
	}
	selected := clampIndex(m.completionMenuIndex, limit)
	input := m.input.Value()
	value := m.completionSuggestions[selected]
	if shouldAppendCompletionSpace(input, value) {
		value += " "
	}

	m.input.SetValue(value)
	m.input.CursorEnd()
	return true
}

func containsSuggestion(items []string, target string) bool {
	return suggestionIndex(items, target) >= 0
}

func suggestionIndex(items []string, target string) int {
	for i, item := range items {
		if item == target {
			return i
		}
	}
	return -1
}

func shouldAppendCompletionSpace(input string, selected string) bool {
	if strings.HasSuffix(selected, " ") || strings.HasSuffix(input, " ") {
		return false
	}
	if _, ok := parsePathCompletionContext(input); ok {
		return false
	}
	return len(strings.Fields(strings.TrimSpace(selected))) > 1
}

func (m *Model) syncCompletionMenuState(input string) {
	limit := m.visibleCompletionSuggestionCount()
	if limit == 0 {
		m.completionMenuIndex = 0
		m.completionMenuInput = ""
		return
	}

	if m.completionMenuInput != input {
		if idx := suggestionIndex(m.completionSuggestions[:limit], input); idx >= 0 {
			m.completionMenuIndex = idx
		} else {
			m.completionMenuIndex = 0
		}
		m.completionMenuInput = input
		return
	}

	m.completionMenuIndex = clampIndex(m.completionMenuIndex, limit)
}

func (m Model) visibleCompletionSuggestionCount() int {
	count := len(m.completionSuggestions)
	if count > maxVisibleCompletionSuggestions {
		count = maxVisibleCompletionSuggestions
	}
	return count
}

func (m Model) completionMenuVisible() bool {
	if !m.inline || m.pendingConfirm != nil || m.activeComponent != nil {
		return false
	}
	if len(m.completionSuggestions) == 0 {
		return false
	}

	input := m.input.Value()
	if input == "" || strings.ContainsAny(input, "\r\n") {
		return false
	}
	if m.completionDismissedInput != "" && input == m.completionDismissedInput {
		return false
	}
	if !strings.HasPrefix(strings.TrimLeft(input, " \t"), "/") {
		return false
	}

	lineInfo := m.input.LineInfo()
	return lineInfo.StartColumn+lineInfo.ColumnOffset == len(input)
}

func (m *Model) dismissCompletionMenu() {
	if !m.completionMenuVisible() {
		return
	}
	m.completionDismissedInput = m.input.Value()
}

func (m *Model) moveCompletionSelection(delta int) bool {
	limit := m.visibleCompletionSuggestionCount()
	if !m.completionMenuVisible() || limit == 0 {
		return false
	}

	next := (m.completionMenuIndex + delta) % limit
	if next < 0 {
		next += limit
	}
	m.completionMenuIndex = next
	return true
}

func completionMenuAnchorIndex(input string) int {
	if input == "" {
		return 0
	}
	if idx := strings.LastIndexAny(input, " \t"); idx >= 0 {
		return idx + 1
	}
	return 0
}

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------

func renderViewportWithScrollbar(view string, vp viewport.Model, styles theme.Styles) string {
	if view == "" {
		return ""
	}
	total := vp.TotalLineCount()
	visible := vp.VisibleLineCount()
	if visible <= 0 || total <= visible {
		return view
	}

	lines := strings.Split(view, "\n")
	height := len(lines)
	if height <= 0 {
		return view
	}

	thumbSize := int(math.Round(float64(visible) / float64(total) * float64(height)))
	if thumbSize < 1 {
		thumbSize = 1
	}
	if thumbSize > height {
		thumbSize = height
	}

	maxOffset := total - visible
	if maxOffset < 0 {
		maxOffset = 0
	}
	trackSpan := height - thumbSize
	thumbTop := 0
	if trackSpan > 0 && maxOffset > 0 {
		thumbTop = int(math.Round(float64(vp.YOffset()) / float64(maxOffset) * float64(trackSpan)))
	}
	if thumbTop < 0 {
		thumbTop = 0
	}
	if thumbTop > trackSpan {
		thumbTop = trackSpan
	}

	for i, line := range lines {
		marker := styles.Dimmed.Render("│")
		if i >= thumbTop && i < thumbTop+thumbSize {
			marker = styles.Muted.Render("█")
		}
		lines[i] = line + marker
	}
	return strings.Join(lines, "\n")
}

// getMarkdownRenderer returns a cached glamour renderer, creating one if needed.
func (m *Model) getMarkdownRenderer() *glamour.TermRenderer {
	// Content width: leave room for message border + padding (6 chars).
	renderWidth := m.width - 6
	if renderWidth < 40 {
		renderWidth = 40
	}
	if m.mdRenderer != nil && m.mdRendererWidth == renderWidth {
		return m.mdRenderer
	}
	r, err := glamour.NewTermRenderer(
		glamour.WithAutoStyle(),
		glamour.WithWordWrap(renderWidth),
	)
	if err != nil {
		return nil
	}
	m.mdRenderer = r
	m.mdRendererWidth = renderWidth
	return r
}

// rebuildViewport reconstructs the viewport content from messages + active stream.
func (m *Model) rebuildViewport() {
	m.rebuildViewportWithMode(false)
}

func (m *Model) appendMessage(msg ChatMessage, forceBottom bool) tea.Cmd {
	return m.appendMessages([]ChatMessage{msg}, forceBottom)
}

func (m *Model) appendMessages(msgs []ChatMessage, forceBottom bool) tea.Cmd {
	if len(msgs) == 0 {
		if forceBottom {
			m.rebuildViewportWithMode(true)
		}
		return nil
	}

	m.messages = append(m.messages, msgs...)
	m.rebuildViewportWithMode(forceBottom)
	return m.inlinePrintMessagesCmd(msgs)
}

// rebuildViewportWithMode reconstructs the viewport while preserving user
// scroll position unless forceBottom is requested.
func (m *Model) rebuildViewportWithMode(forceBottom bool) {
	m.recalculateViewportHeight()
	wasAtBottom := m.viewport.AtBottom()
	renderer := m.getMarkdownRenderer()
	var b strings.Builder
	if !m.inlineFlowMode() {
		for _, msg := range m.messages {
			b.WriteString(renderMessage(msg, m.styles, m.width, renderer, m.inline))
			b.WriteByte('\n')
			// Add breathing room between a user turn and assistant response.
			if m.inline && msg.Role == "user" {
				b.WriteByte('\n')
			}
		}
	}
	// Append in-progress streaming text only in framed transcript mode.
	// Inline flow mode prints finalized messages to terminal scrollback;
	// rendering partial stream text here causes visual duplication/noise.
	if !m.inlineFlowMode() && m.streamBuf.Len() > 0 {
		partial := ChatMessage{
			Role:        "assistant",
			Blocks:      textBlocks(m.streamBuf.String()),
			IsStreaming: true,
		}
		b.WriteString(renderMessage(partial, m.styles, m.width, renderer, m.inline))
		b.WriteByte('\n')
	}
	if !m.inlineFlowMode() && len(m.pendingHandoffs) > 0 {
		for _, msg := range m.pendingHandoffs {
			b.WriteString(renderMessage(msg, m.styles, m.width, renderer, m.inline))
			b.WriteByte('\n')
		}
	}
	m.viewport.SetContent(b.String())
	if m.inlineFlowMode() {
		m.recalculateViewportHeight()
	}
	if forceBottom || wasAtBottom {
		m.viewport.GotoBottom()
	}
}

func (m *Model) inlinePrintMessagesCmd(msgs []ChatMessage) tea.Cmd {
	if !m.inlineFlowMode() {
		return nil
	}

	renderedParts := make([]string, 0, len(msgs))
	renderer := m.getMarkdownRenderer()
	for _, msg := range msgs {
		rendered := strings.TrimRight(renderMessage(msg, m.styles, m.width, renderer, true), "\n")
		if strings.TrimSpace(rendered) == "" {
			continue
		}
		renderedParts = append(renderedParts, rendered)
	}
	if len(renderedParts) == 0 {
		return nil
	}

	if !m.inlineBannerPrinted {
		m.inlineBannerPrinted = true
		header := strings.TrimSpace(renderHeader(*m))
		runtimeSummary := strings.TrimSpace(renderRuntimeSummary(*m))
		parts := make([]string, 0, 3)
		if header != "" {
			parts = append(parts, header)
		}
		if runtimeSummary != "" {
			parts = append(parts, runtimeSummary)
		}
		parts = append(parts, renderedParts...)
		return tea.Println(strings.Join(parts, "\n"))
	}

	return tea.Println(strings.Join(renderedParts, "\n"))
}

func (m *Model) applyClearTarget(target string) {
	switch target {
	case "output", "all", "":
		m.messages = m.messages[:0]
		m.pendingHandoffs = m.pendingHandoffs[:0]
		m.streamBuf.Reset()
		m.isStreaming = false
		m.toolFeed = m.toolFeed[:0]
		m.modalities = m.modalities[:0]
		m.progressActive = false
		m.progressLabel = ""
		m.progressCurrent = 0
		m.progressTotal = 0

		m.recalculateViewportHeight()
		m.rebuildViewportWithMode(true)
	case "status":
		m.statusText = ""
		m.recalculateViewportHeight()
	}
}

// renderHelp appends a help message listing available commands.
func (m *Model) renderHelp() {
	help := `## Commands
| Command | Description |
| --- | --- |
| /data | Current data summary |
| /workspace | Workspace info |
| /plots | List generated plots |
| /tokens | Token usage & costs |
| /read <file> | Load data file |
| /pipeline export | Export as Jupyter notebook |
| /clear | Clear screen |
| /exit | Exit |

## Controls
- Inline mode uses natural terminal scrollback for transcript reading
- Fullscreen mode supports PgUp/PgDn transcript scrolling
- Ctrl+G optionally toggles mouse capture

*Ask anything in natural language — Lobster routes to the right specialist agent.*`

	m.appendMessage(ChatMessage{
		Role:   "assistant",
		Blocks: textBlocks(help),
	}, false)
}

// renderDataSummary appends a summary of loaded modalities to the chat.
func (m *Model) renderDataSummary() {
	if len(m.modalities) == 0 {
		m.appendMessage(ChatMessage{
			Role:   "system",
			Blocks: textBlocks("No data loaded. Use /read <file> or ask Lobster to load data."),
		}, false)
		return
	}
	var tb strings.Builder
	tb.WriteString("## Loaded Data\n\n")
	tb.WriteString("| Modality | Shape |\n| --- | --- |\n")
	for _, mod := range m.modalities {
		tb.WriteString(fmt.Sprintf("| %s | %s |\n", mod.Name, mod.Shape))
	}
	m.appendMessage(ChatMessage{
		Role:   "assistant",
		Blocks: textBlocks(tb.String()),
	}, false)
}

// pushInputHistory adds a line to local input history with ring semantics.
func (m *Model) pushInputHistory(line string) {
	line = strings.TrimSpace(line)
	if line == "" {
		return
	}
	if n := len(m.inputHistory); n > 0 && m.inputHistory[n-1] == line {
		m.resetHistoryNavigation()
		return
	}
	if len(m.inputHistory) >= maxInputHistory {
		m.inputHistory = m.inputHistory[1:]
	}
	m.inputHistory = append(m.inputHistory, line)
	m.resetHistoryNavigation()
}

func (m *Model) resetHistoryNavigation() {
	m.inputHistoryIndex = -1
	m.inputHistoryDraft = ""
}

func (m *Model) recallHistoryUp() bool {
	if len(m.inputHistory) == 0 {
		return false
	}
	if m.inputHistoryIndex == -1 {
		m.inputHistoryDraft = m.input.Value()
		m.inputHistoryIndex = len(m.inputHistory) - 1
	} else if m.inputHistoryIndex > 0 {
		m.inputHistoryIndex--
	}
	m.input.SetValue(m.inputHistory[m.inputHistoryIndex])
	return true
}

func (m *Model) recallHistoryDown() bool {
	if m.inputHistoryIndex == -1 || len(m.inputHistory) == 0 {
		return false
	}
	if m.inputHistoryIndex < len(m.inputHistory)-1 {
		m.inputHistoryIndex++
		m.input.SetValue(m.inputHistory[m.inputHistoryIndex])
		return true
	}
	m.inputHistoryIndex = -1
	m.input.SetValue(m.inputHistoryDraft)
	m.inputHistoryDraft = ""
	return true
}

// pushToolFeed adds an entry to the tool feed ring buffer, evicting the oldest
// entry if the buffer is at capacity.
func (m *Model) pushToolFeed(entry ToolFeedEntry) {
	if len(m.toolFeed) >= maxToolFeed {
		m.toolFeed = m.toolFeed[1:]
	}
	m.toolFeed = append(m.toolFeed, entry)
}

func (m *Model) applyToolFeedEntry(entry ToolFeedEntry) {
	if idx := m.toolFeedIndex(entry); idx >= 0 {
		m.toolFeed[idx] = entry
		return
	}
	m.pushToolFeed(entry)
}

func (m Model) toolFeedIndex(entry ToolFeedEntry) int {
	if entry.ID != "" {
		for i := len(m.toolFeed) - 1; i >= 0; i-- {
			if m.toolFeed[i].ID == entry.ID {
				return i
			}
		}
	}

	if entry.Event == protocol.ToolExecutionFinish || entry.Event == protocol.ToolExecutionError {
		for i := len(m.toolFeed) - 1; i >= 0; i-- {
			if m.toolFeed[i].Name == entry.Name && m.toolFeed[i].Event == protocol.ToolExecutionStart {
				return i
			}
		}
	}

	return -1
}

func lineCount(text string) int {
	if text == "" {
		return 0
	}
	return strings.Count(text, "\n") + 1
}

func (m Model) shouldRenderStreamingTranscript() bool {
	return !m.inlineFlowMode()
}

func (m Model) currentStatusLine() string {
	statusText := m.statusText
	if m.spinnerActive && !m.isCanceling {
		spinnerText := spinnerFrames[m.spinnerFrame] + " " + m.spinnerLabel + "..."
		if m.quietStartup && !m.ready {
			spinnerText = "· " + m.spinnerLabel + "..."
		}
		if !m.ready && len(loadingTips) > 0 && !m.quietStartup {
			spinnerText += "  ·  " + m.styles.Muted.Render("💡 "+loadingTips[m.tipIndex])
		}
		statusText = spinnerText
	}
	workerStatus := m.activeWorkerIndicator()
	if !m.ready {
		return joinStatusParts(statusText, workerStatus)
	}
	if m.completionMenuVisible() {
		return joinStatusParts(statusText, workerStatus, "Tab accept", "Ctrl+N/Ctrl+P move", "Esc dismiss")
	}

	if m.inlineFlowMode() {
		if strings.TrimSpace(statusText) == "" {
			return joinStatusParts(
				workerStatus,
				"inline flow: use terminal scrollback",
				"Ctrl+G optional mouse capture",
			)
		}
		return joinStatusParts(statusText, workerStatus, "inline flow via terminal scrollback")
	}

	mouseLabel := m.mouseModeLabel()
	if strings.TrimSpace(statusText) == "" {
		return joinStatusParts(workerStatus, mouseLabel, "Ctrl+G toggles")
	}
	return joinStatusParts(statusText, workerStatus, mouseLabel)
}

func joinStatusParts(parts ...string) string {
	filtered := make([]string, 0, len(parts))
	for _, part := range parts {
		if strings.TrimSpace(part) == "" {
			continue
		}
		filtered = append(filtered, part)
	}
	return strings.Join(filtered, "  ·  ")
}

func (m Model) activeWorkerIndicator() string {
	summary := m.activeWorkerSummary()
	if summary == "" {
		return ""
	}
	return m.styles.AgentTransition.Render("● " + summary)
}

func (m Model) activeWorkerSummary() string {
	if len(m.activeWorkers) == 0 {
		return ""
	}

	names := make([]string, 0, len(m.activeWorkers))
	for name := range m.activeWorkers {
		names = append(names, name)
	}
	sort.Strings(names)

	if len(names) == 1 {
		return names[0]
	}
	if len(names) > 3 {
		return fmt.Sprintf("%d agents active", len(names))
	}
	return fmt.Sprintf("%s +%d", names[0], len(names)-1)
}

func (m Model) isActivityTransition(p protocol.AgentTransitionPayload) bool {
	kind := strings.TrimSpace(p.Kind)
	if kind == "activity" {
		return true
	}
	if kind == "task" {
		return false
	}

	status := strings.TrimSpace(p.Status)
	if status != "" {
		return true
	}

	switch strings.TrimSpace(p.Reason) {
	case "working", "complete", "chain_start", "return":
		return true
	default:
		return false
	}
}

func (m *Model) applyWorkerActivity(p protocol.AgentTransitionPayload) {
	worker := strings.TrimSpace(p.To)
	if worker == "" {
		worker = strings.TrimSpace(p.From)
	}
	if worker == "" {
		return
	}
	if m.activeWorkers == nil {
		m.activeWorkers = make(map[string]struct{}, 4)
	}

	status := strings.TrimSpace(p.Status)
	if status == "" {
		status = strings.TrimSpace(p.Reason)
	}

	switch status {
	case "working":
		m.activeWorkers[worker] = struct{}{}
	case "complete", "return":
		delete(m.activeWorkers, worker)
	}
}

func (m *Model) recordHandoff(task string, agent string) tea.Cmd {
	msg := ChatMessage{
		Role:   "handoff",
		Blocks: []ContentBlock{BlockHandoff{From: "", To: agent, Reason: task}},
		Agent:  agent,
	}

	if m.streamBuf.Len() > 0 {
		m.pendingHandoffs = append(m.pendingHandoffs, msg)
		if m.shouldRenderStreamingTranscript() {
			m.rebuildViewport()
		}
		return nil
	}

	return m.appendMessage(msg, false)
}

func (m Model) mouseModeLabel() string {
	if m.mouseCapture {
		return "mouse: scroll"
	}
	return "mouse: select"
}

func mouseCaptureCmd(_ bool) tea.Cmd {
	// In v2, mouse mode is controlled via View() fields (View.MouseMode).
	// Toggling m.mouseCapture is sufficient; the next View() render picks it up.
	return nil
}

// layoutReservedRows estimates non-viewport rows currently occupied by UI chrome.
func (m Model) layoutReservedRows() int {
	rows := 0

	if m.inline {
		if intro := renderInlineIntro(m); intro != "" {
			rows += lineCount(intro)
		}
	}
	if m.shouldRenderHeaderInFrame() {
		rows += lineCount(renderHeader(m))
		if m.inline {
			if runtimeSummary := renderRuntimeSummary(m); runtimeSummary != "" {
				rows += lineCount(runtimeSummary) + 2 // explicit "\n\n" spacer in View()
			}
		}
	}

	if tf := renderToolFeed(m.toolFeed, m.styles, m.width, m.inline); tf != "" {
		rows += lineCount(tf)
	}
	if m.activeComponent != nil && m.activeComponent.Component.Mode() == "inline" {
		inlineView := m.activeComponent.Component.View(m.width, m.height)
		if inlineView != "" {
			rows += lineCount(inlineView)
		}
	}
	if m.progressActive {
		rows += lineCount(renderProgressBar(m.progressLabel, m.progressCurrent, m.progressTotal, m.width, m.styles))
	}

	if m.activeComponent != nil && m.activeComponent.Component.Mode() == "overlay" {
		_, oh := biocomp.OverlaySize(m.width, m.height, "small")
		rows += oh
	} else if m.pendingConfirm != nil {
		rows += lineCount(renderConfirmPrompt(m.pendingConfirm, m.styles, m.width))
	} else {
		if m.inline {
			rows += 1 // reserve optional prompt spacer in inline mode
		}
		if completionView := m.renderCompletionMenu(); completionView != "" {
			rows += lineCount(completionView)
		}
		rows += lineCount(m.renderComposer())
	}

	statusText := m.currentStatusLine()
	if m.inline {
		if strings.TrimSpace(statusText) != "" {
			rows += 1
		}
	} else {
		rows += lineCount(renderStatusBar(statusText, m.styles, m.width))
	}

	return rows
}

func (m *Model) recalculateViewportHeight() {
	m.recalculateComposerDimensions()
	if m.inlineFlowMode() {
		budget := m.height - m.layoutReservedRows()
		if budget < 1 {
			budget = 1
		}
		h := m.viewport.TotalLineCount()
		if h < 1 {
			h = 1
		}
		if h > budget {
			h = budget
		}
		m.viewport.SetHeight(h)
		return
	}

	h := m.height - m.layoutReservedRows()
	if h < 1 {
		h = 1
	}
	m.viewport.SetHeight(h)
}

func (m Model) inlineFlowMode() bool {
	return m.inline && m.inlineFlow
}

func (m Model) shouldRenderHeaderInFrame() bool {
	if !m.inline {
		return true
	}
	if m.inlineFlowMode() && m.inlineBannerPrinted {
		return false
	}
	return true
}

func (m Model) composerInitialized() bool {
	return !(m.input.MaxHeight == 0 && m.input.Width() == 0 && m.input.Height() == 0)
}

func applyComposerStyles(input *textarea.Model, styles theme.Styles) {
	s := textarea.DefaultDarkStyles()

	textColor := styles.InputField.GetForeground()
	promptColor := styles.InputPrompt.GetForeground()
	mutedColor := styles.Muted.GetForeground()
	dimColor := styles.Dimmed.GetForeground()

	base := lipgloss.NewStyle().UnsetBackground()

	s.Focused.Base = base
	s.Blurred.Base = base

	s.Focused.Text = lipgloss.NewStyle().Foreground(textColor)
	s.Blurred.Text = lipgloss.NewStyle().Foreground(textColor)

	s.Focused.CursorLine = lipgloss.NewStyle().Foreground(textColor)
	s.Blurred.CursorLine = lipgloss.NewStyle().Foreground(textColor)

	s.Focused.Placeholder = lipgloss.NewStyle().Foreground(mutedColor)
	s.Blurred.Placeholder = lipgloss.NewStyle().Foreground(mutedColor)

	s.Focused.Prompt = lipgloss.NewStyle().Foreground(promptColor)
	s.Blurred.Prompt = lipgloss.NewStyle().Foreground(promptColor)

	s.Focused.LineNumber = lipgloss.NewStyle().Foreground(dimColor)
	s.Blurred.LineNumber = lipgloss.NewStyle().Foreground(dimColor)
	s.Focused.CursorLineNumber = lipgloss.NewStyle().Foreground(mutedColor)
	s.Blurred.CursorLineNumber = lipgloss.NewStyle().Foreground(mutedColor)

	s.Focused.EndOfBuffer = lipgloss.NewStyle().Foreground(lipgloss.NoColor{})
	s.Blurred.EndOfBuffer = lipgloss.NewStyle().Foreground(lipgloss.NoColor{})

	input.SetStyles(s)
	input.Focus()
}

func (m *Model) recalculateComposerDimensions() {
	if !m.composerInitialized() {
		return
	}
	promptWidth := lipgloss.Width(m.styles.InputPrompt.Render(m.inputPromptText()))
	inputWidth := m.width - promptWidth
	if inputWidth < 1 {
		inputWidth = 1
	}
	if m.input.Width() != inputWidth {
		m.input.SetWidth(inputWidth)
	}
	m.recalculateComposerHeight()
}

func (m *Model) recalculateComposerHeight() {
	height := m.composerHeightForValue(m.input.Value())
	if m.input.Height() == height {
		return
	}
	m.input.SetHeight(height)
}

func (m Model) composerHeightForValue(value string) int {
	width := m.input.Width()
	if width < 1 {
		width = 1
	}

	lines := strings.Split(strings.ReplaceAll(value, "\r\n", "\n"), "\n")
	if len(lines) == 0 {
		lines = []string{""}
	}

	visualLines := 0
	for _, line := range lines {
		if line == "" {
			visualLines++
			continue
		}
		visualLines += hardWrapLineCount(line, width)
	}
	if visualLines < composerMinHeight {
		visualLines = composerMinHeight
	}
	if visualLines > composerMaxHeight {
		visualLines = composerMaxHeight
	}
	return visualLines
}

// hardWrapLineCount returns the number of visual lines a string occupies
// when hard-wrapped to the given width.
func hardWrapLineCount(value string, width int) int {
	w := lipgloss.Width(value)
	if w <= width {
		return 1
	}
	count := 0
	currentWidth := 0
	for _, r := range value {
		rw := lipgloss.Width(string(r))
		if currentWidth+rw > width && currentWidth > 0 {
			count++
			currentWidth = 0
		}
		currentWidth += rw
	}
	if currentWidth > 0 {
		count++
	}
	if count == 0 {
		return 1
	}
	return count
}

func (m Model) renderComposer() string {
	prompt := m.styles.InputPrompt.Render(m.inputPromptText())
	if !m.composerInitialized() {
		return prompt
	}
	view := m.input.View()
	lines := strings.Split(view, "\n")
	if len(lines) == 0 {
		return prompt
	}

	prefix := strings.Repeat(" ", lipgloss.Width(prompt))
	var b strings.Builder
	for i, line := range lines {
		if i > 0 {
			b.WriteByte('\n')
		}
		if i == 0 {
			b.WriteString(prompt)
		} else {
			b.WriteString(prefix)
		}
		b.WriteString(line)
	}
	return b.String()
}

func (m Model) renderCompletionMenu() string {
	if !m.completionMenuVisible() {
		return ""
	}

	limit := m.visibleCompletionSuggestionCount()
	selected := clampIndex(m.completionMenuIndex, limit)
	promptWidth := lipgloss.Width(m.styles.InputPrompt.Render(m.inputPromptText()))
	lineInfo := m.input.LineInfo()
	anchorColumn := completionMenuAnchorIndex(m.input.Value()) - lineInfo.StartColumn
	if anchorColumn < 0 {
		anchorColumn = 0
	}

	indentWidth := promptWidth + anchorColumn
	menuWidth := m.width - indentWidth
	if menuWidth < 18 {
		indentWidth -= 18 - menuWidth
		if indentWidth < promptWidth {
			indentWidth = promptWidth
		}
		menuWidth = m.width - indentWidth
	}
	if menuWidth < 12 {
		menuWidth = 12
	}

	indent := strings.Repeat(" ", indentWidth)
	selectedStyle := lipgloss.NewStyle().
		Foreground(m.styles.InputPrompt.GetForeground()).
		Bold(true)

	lines := make([]string, 0, limit+1)
	if extra := len(m.completionSuggestions) - limit; extra > 0 {
		lines = append(lines, indent+m.styles.Dimmed.Render(fmt.Sprintf("+%d more", extra)))
	}
	for i := limit - 1; i >= 0; i-- {
		label := truncateMiddle(m.completionSuggestions[i], menuWidth-2)
		line := "  " + label
		style := m.styles.Muted
		if i == selected {
			line = "› " + label
			style = selectedStyle
		}
		lines = append(lines, indent+style.Render(line))
	}

	return strings.Join(lines, "\n")
}

func (m Model) inputPromptText() string {
	if m.inline {
		return fmt.Sprintf("%s · $%.4f ❯ ", providerIcon(m.provider), m.promptCostUSD)
	}
	return "> "
}

func protocolTableRenderWidth(viewWidth int) int {
	width := viewWidth - 12
	if width < 32 {
		width = 32
	}
	return width
}

func appendProtocolCodeBlock(buf *strings.Builder, block string) {
	if strings.TrimSpace(block) == "" {
		return
	}

	if buf.Len() > 0 {
		existing := buf.String()
		switch {
		case strings.HasSuffix(existing, "\n\n"):
		case strings.HasSuffix(existing, "\n"):
			buf.WriteByte('\n')
		default:
			buf.WriteString("\n\n")
		}
	}

	buf.WriteString("```text\n")
	buf.WriteString(block)
	buf.WriteString("\n```\n")
}

// sendComponentResponse sends a component_response protocol message to Python.
// The action parameter is always included in the Data map so Python can
// distinguish submit, cancel, and error responses.
func (m *Model) sendComponentResponse(msgID, action string, data map[string]any) {
	if m.handler == nil {
		return
	}
	if data == nil {
		data = map[string]any{}
	}
	data["action"] = action
	_ = m.handler.SendTyped(protocol.TypeComponentResponse, protocol.ComponentResponsePayload{
		ID:   msgID,
		Data: data,
	}, msgID)
}

// safeHandleMsg wraps comp.HandleMsg in a recover boundary so a panicking
// component does not crash the entire TUI.
func safeHandleMsg(comp biocomp.BioComponent, msg tea.Msg) (result *biocomp.ComponentResult, panicked bool) {
	defer func() {
		if r := recover(); r != nil {
			panicked = true
		}
	}()
	return comp.HandleMsg(msg), false
}

// safeView wraps comp.View in a recover boundary. Returns empty string on panic.
func safeView(comp biocomp.BioComponent, w, h int) (content string, panicked bool) {
	defer func() {
		if r := recover(); r != nil {
			panicked = true
			content = ""
		}
	}()
	return comp.View(w, h), false
}

// handleConfirmKey handles key presses for the local exit confirm dialog.
func (m Model) handleConfirmKey(msg tea.KeyPressMsg) (tea.Model, tea.Cmd) {
	switch msg.String() {
	case "ctrl+c":
		// Dismiss confirm and send decline.
		return m.resolveConfirm(false)
	case "enter":
		// Enter accepts the default.
		confirm := m.pendingConfirm.Default
		return m.resolveConfirm(confirm)
	case "y", "Y":
		return m.resolveConfirm(true)
	case "n", "N":
		return m.resolveConfirm(false)
	}
	return m, nil
}

// resolveConfirm finalizes the local exit confirm dialog.
// Protocol-originated confirms now go through the BioComp activeComponent path.
func (m Model) resolveConfirm(confirm bool) (tea.Model, tea.Cmd) {
	m.pendingConfirm = nil
	m.pendingConfirmID = ""
	m.recalculateViewportHeight()
	if confirm {
		if m.handler != nil {
			_ = m.handler.SendTyped(protocol.TypeQuit, protocol.QuitPayload{}, "")
		}
		m.quitting = true
		return m, tea.Quit
	}
	printCmd := m.appendMessage(ChatMessage{
		Role:   "system",
		Blocks: textBlocks("Exit cancelled."),
	}, false)
	return m, printCmd
}

// handleComponentRender routes a generic HITL component_render to a
// BioComp component from the registry. Unknown components fall back to
// text_input with the fallback prompt.
func (m Model) handleComponentRender(p protocol.ComponentRenderPayload, msgID string) (tea.Model, tea.Cmd) {
	// Close existing component if any.
	if m.activeComponent != nil {
		m.sendComponentResponse(m.activeComponent.MsgID, "cancel", map[string]any{"reason": "replaced"})
		m.activeComponent = nil
	}

	factory := biocomp.Get(p.Component)
	if factory == nil {
		// Unknown component — use text_input as fallback if available.
		factory = biocomp.Get("text_input")
		if factory == nil {
			m.sendComponentResponse(msgID, "error", map[string]any{"error": "unknown_component", "name": p.Component})
			return m, waitForProtocolMsg(m.handler)
		}
		// Rewrite data to use fallback prompt as question.
		prompt := p.FallbackPrompt
		if prompt == "" {
			if q, ok := p.Data["question"].(string); ok && q != "" {
				prompt = q
			} else {
				prompt = "Please provide input"
			}
		}
		p.Data = map[string]any{"question": prompt}
	}

	comp := factory()
	dataBytes, err := json.Marshal(p.Data)
	if err != nil {
		m.sendComponentResponse(msgID, "error", map[string]any{"error": "marshal_failed", "detail": err.Error()})
		return m, waitForProtocolMsg(m.handler)
	}
	if err := comp.Init(dataBytes); err != nil {
		m.sendComponentResponse(msgID, "error", map[string]any{"error": "init_failed", "detail": err.Error()})
		return m, waitForProtocolMsg(m.handler)
	}

	m.activeComponent = &ActiveComponent{
		Component: comp,
		MsgID:     msgID,
		CreatedAt: time.Now(),
	}
	m.recalculateViewportHeight()
	return m, waitForProtocolMsg(m.handler)
}

func clampIndex(value, count int) int {
	if count <= 0 {
		return 0
	}
	if value < 0 {
		return 0
	}
	if value >= count {
		return count - 1
	}
	return value
}

// waitForProtocolMsg returns a tea.Cmd that reads the next message from the
// protocol handler channel. If the channel is closed, it returns protocolEOF.
func waitForProtocolMsg(h *protocol.Handler) tea.Cmd {
	return func() tea.Msg {
		msg, ok := <-h.Msgs()
		if !ok {
			return protocolEOF{}
		}
		return protocolMsg{msg}
	}
}

// waitForProtocolErr returns a tea.Cmd that reads the next error from the
// protocol handler's error channel. Errors are non-fatal parse/read issues.
func waitForProtocolErr(h *protocol.Handler) tea.Cmd {
	return func() tea.Msg {
		err, ok := <-h.Errs()
		if !ok {
			return nil
		}
		return protocolErr{err: err}
	}
}

func resolveLobsterVersion(fallback string) string {
	if envVer := strings.TrimSpace(os.Getenv("LOBSTER_TUI_APP_VERSION")); envVer != "" {
		return trimVersionPrefix(envVer)
	}

	paths := make([]string, 0, 2)
	if wd, err := os.Getwd(); err == nil {
		paths = append(paths, wd)
	}
	if exe, err := os.Executable(); err == nil {
		paths = append(paths, filepath.Dir(exe))
	}

	re := regexp.MustCompile(`(?m)^version\s*=\s*"([^"]+)"\s*$`)
	for _, start := range paths {
		dir := start
		for i := 0; i < 8; i++ {
			pp := filepath.Join(dir, "pyproject.toml")
			data, err := os.ReadFile(pp)
			if err == nil {
				if m := re.FindStringSubmatch(string(data)); len(m) == 2 {
					return trimVersionPrefix(m[1])
				}
			}
			parent := filepath.Dir(dir)
			if parent == dir {
				break
			}
			dir = parent
		}
	}

	return trimVersionPrefix(fallback)
}

func trimVersionPrefix(v string) string {
	v = strings.TrimSpace(v)
	if strings.HasPrefix(strings.ToLower(v), "v") && len(v) > 1 {
		return v[1:]
	}
	return v
}

func normalizeProviderName(p string) string {
	p = strings.TrimSpace(strings.ToLower(p))
	if p == "" {
		return "auto"
	}
	return p
}

func providerIcon(provider string) string {
	switch normalizeProviderName(provider) {
	case "bedrock":
		return "🟠"
	case "openai":
		return "🟢"
	case "anthropic":
		return "🟣"
	case "ollama":
		return "🟡"
	case "openrouter":
		return "🔵"
	case "azure":
		return "🔷"
	case "gemini":
		return "🔶"
	default:
		return "⚪"
	}
}

func extractUSDCost(text string) (float64, bool) {
	m := usdCostRe.FindStringSubmatch(text)
	if len(m) != 2 {
		return 0, false
	}
	v, err := strconv.ParseFloat(m[1], 64)
	if err != nil {
		return 0, false
	}
	return v, true
}

func detectComputeTarget() string {
	if runtime.GOOS == "darwin" {
		return "MPS"
	}
	if _, err := exec.LookPath("nvidia-smi"); err == nil {
		return "CUDA"
	}
	return "CPU"
}

func detectTotalRAMGB() int {
	if runtime.GOOS == "darwin" {
		out, err := exec.Command("sysctl", "-n", "hw.memsize").Output()
		if err == nil {
			if bytes, parseErr := strconv.ParseInt(strings.TrimSpace(string(out)), 10, 64); parseErr == nil && bytes > 0 {
				return int(bytes / (1024 * 1024 * 1024))
			}
		}
		return 0
	}

	if runtime.GOOS == "linux" {
		f, err := os.Open("/proc/meminfo")
		if err != nil {
			return 0
		}
		defer f.Close()

		sc := bufio.NewScanner(f)
		for sc.Scan() {
			line := sc.Text()
			if strings.HasPrefix(line, "MemTotal:") {
				fields := strings.Fields(line)
				if len(fields) >= 2 {
					if kb, parseErr := strconv.ParseInt(fields[1], 10, 64); parseErr == nil && kb > 0 {
						return int((kb * 1024) / (1024 * 1024 * 1024))
					}
				}
				break
			}
		}
	}

	return 0
}

func detectFreeStorageGB(path string) int {
	if strings.TrimSpace(path) == "" {
		return -1
	}
	var stat syscall.Statfs_t
	if err := syscall.Statfs(path, &stat); err != nil {
		return -1
	}
	free := stat.Bavail * uint64(stat.Bsize)
	return int(free / (1024 * 1024 * 1024))
}

func makeDNASequence(width int) string {
	if width < 20 {
		width = 20
	}
	seq := make([]byte, width)
	bases := []byte{'A', 'T', 'G', 'C'}
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	for i := range seq {
		seq[i] = bases[r.Intn(len(bases))]
	}
	return string(seq)
}

func nextWelcomeSporadicDelay(r *rand.Rand) time.Duration {
	if r == nil {
		return welcomeSporadicMinInterval
	}
	return welcomeSporadicMinInterval + time.Duration(r.Int63n(int64(welcomeSporadicJitterRange)))
}

func nextWelcomePersistentSparkDelay(r *rand.Rand) time.Duration {
	if r == nil {
		return welcomePersistentSparkMinInterval
	}
	return welcomePersistentSparkMinInterval + time.Duration(r.Int63n(int64(welcomePersistentSparkJitterRange)))
}
