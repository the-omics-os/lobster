// Package chat implements the interactive chat BubbleTea model for Lobster AI.
//
// The model bridges the protocol handler (JSON Lines IPC with Python) and the
// BubbleTea event loop, rendering streamed text, agent transitions, tool calls,
// and user input into a terminal-based chat interface.
package chat

import (
	"bufio"
	"fmt"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/glamour"
	"github.com/charmbracelet/lipgloss"
	"github.com/muesli/reflow/wordwrap"

	"github.com/the-omics-os/lobster-tui/internal/protocol"
	"github.com/the-omics-os/lobster-tui/internal/theme"
)

// maxToolFeed is the maximum number of tool execution lines kept in the ring buffer.
const maxToolFeed = 5

// maxInputHistory is the number of entered lines kept for Up/Down recall.
const maxInputHistory = 200

const composerMinHeight = 1
const composerMaxHeight = 8

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
	// Role is one of "user", "assistant", or "system".
	Role string
	// Content is the rendered text content of the message.
	Content string
	// Agent is the name of the specialist agent (empty for user/system).
	Agent string
}

// ToolFeedEntry represents a single tool execution with its lifecycle state.
type ToolFeedEntry struct {
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

	messages    []ChatMessage
	activeAgent string
	toolFeed    []ToolFeedEntry
	modalities  []ModalityInfo

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
	completionCycleBase          string
	completionCycleIndex         int
	completionCycleSuggestions   []string

	// Heartbeat monitoring: tracks the last heartbeat from the backend.
	lastHeartbeat time.Time

	// Progress bar state.
	progressLabel   string
	progressCurrent int
	progressTotal   int
	progressActive  bool

	// Pending confirm dialog state.
	pendingConfirm   *protocol.ConfirmPayload
	pendingConfirmID string

	// Pending select dialog state.
	pendingSelect   *protocol.SelectPayload
	pendingSelectID string
	selectIndex     int

	width        int
	height       int
	inline       bool
	inlineFlow   bool
	mouseCapture bool
	quitting     bool
	styles       theme.Styles

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

// NewModel creates a new chat Model wired to the given handler and styles.
func NewModel(handler *protocol.Handler, styles theme.Styles, width, height int, inline bool, mouseCapture bool, versionFallback string) Model {
	vp := viewport.New(width, 1)
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

	case formResult:
		// Form completed (from tea.Exec). Send the response back to Python.
		if msg.err != nil {
			// User cancelled or huh error — send empty values.
			_ = m.handler.SendTyped(protocol.TypeFormResponse, protocol.FormResponsePayload{
				ID:     msg.id,
				Values: map[string]string{},
			}, msg.id)
		} else {
			_ = m.handler.SendTyped(protocol.TypeFormResponse, protocol.FormResponsePayload{
				ID:     msg.id,
				Values: msg.values,
			}, msg.id)
		}
		return m, waitForProtocolMsg(m.handler)

	case tea.KeyMsg:
		return m.handleKey(msg)

	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		m.viewport.Width = m.width
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
func (m Model) View() string {
	if m.quitting {
		return ""
	}

	var b strings.Builder

	if m.inline {
		intro := renderInlineIntro(m)
		if intro != "" {
			b.WriteString(intro)
			b.WriteByte('\n')
		}
	}

	// Header (1 line).
	b.WriteString(renderHeader(m))
	b.WriteByte('\n')
	if m.inline {
		runtimeSummary := renderRuntimeSummary(m)
		if runtimeSummary != "" {
			b.WriteString(runtimeSummary)
			b.WriteString("\n\n")
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

	// Progress bar (0-1 line).
	if m.progressActive {
		b.WriteString(renderProgressBar(m.progressLabel, m.progressCurrent, m.progressTotal, m.width, m.styles))
		b.WriteByte('\n')
	}

	// Input field (1 line) — replaced by confirm/select when pending.
	if m.pendingConfirm != nil {
		b.WriteString(renderConfirmPrompt(m.pendingConfirm, m.styles, m.width))
		b.WriteByte('\n')
	} else if m.pendingSelect != nil {
		b.WriteString(renderSelectPrompt(m.pendingSelect, m.selectIndex, m.styles, m.width))
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

	return b.String()
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
			m.rebuildViewport()
		}
		return m, waitForProtocolMsg(m.handler)

	case protocol.TypeMarkdown:
		// Forward-compatible: treat markdown blocks same as text (glamour renders in View).
		var p protocol.MarkdownPayload
		if err := protocol.DecodePayload(msg.Message, &p); err == nil {
			m.isStreaming = true
			m.streamBuf.WriteString(p.Content)
			m.rebuildViewport()
		}
		return m, waitForProtocolMsg(m.handler)

	case protocol.TypeCode:
		// Wrap code content in fenced code block so glamour renders it.
		var p protocol.CodePayload
		if err := protocol.DecodePayload(msg.Message, &p); err == nil {
			m.isStreaming = true
			m.streamBuf.WriteString("\n```")
			m.streamBuf.WriteString(p.Language)
			m.streamBuf.WriteByte('\n')
			m.streamBuf.WriteString(p.Content)
			m.streamBuf.WriteString("\n```\n")
			m.rebuildViewport()
		}
		return m, waitForProtocolMsg(m.handler)

	case protocol.TypeAgentTransition:
		var p protocol.AgentTransitionPayload
		if err := protocol.DecodePayload(msg.Message, &p); err == nil {
			m.activeAgent = p.To
			reason := p.Reason
			if reason == "" {
				reason = fmt.Sprintf("handing off to %s", p.To)
			}
			m.messages = append(m.messages, ChatMessage{
				Role:    "system",
				Content: reason,
				Agent:   p.To,
			})
			m.rebuildViewport()
		}
		return m, waitForProtocolMsg(m.handler)

	case protocol.TypeToolExecution:
		var p protocol.ToolExecutionPayload
		if err := protocol.DecodePayload(msg.Message, &p); err == nil {
			entry := ToolFeedEntry{
				Name:    p.ToolName,
				Event:   p.Event,
				Summary: p.Summary,
				Agent:   m.activeAgent,
			}
			// In-place update: find the most recent entry with same name in "start" state.
			updated := false
			if p.Event == protocol.ToolExecutionFinish || p.Event == protocol.ToolExecutionError {
				for i := len(m.toolFeed) - 1; i >= 0; i-- {
					if m.toolFeed[i].Name == p.ToolName && m.toolFeed[i].Event == protocol.ToolExecutionStart {
						m.toolFeed[i] = entry
						updated = true
						break
					}
				}
			}
			if !updated {
				m.pushToolFeed(entry)
			}
			m.recalculateViewportHeight()
			m.rebuildViewport()
		}
		return m, waitForProtocolMsg(m.handler)

	case protocol.TypeDone:
		// Flush the stream buffer as an assistant message.
		if m.streamBuf.Len() > 0 {
			m.messages = append(m.messages, ChatMessage{
				Role:    "assistant",
				Content: m.streamBuf.String(),
				Agent:   m.activeAgent,
			})
			m.streamBuf.Reset()
		}
		m.isStreaming = false
		m.rebuildViewportWithMode(true)
		return m, waitForProtocolMsg(m.handler)

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
			m.messages = append(m.messages, ChatMessage{
				Role:    "alert_" + string(p.Level),
				Content: p.Message,
			})
			m.rebuildViewport()
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
			m.messages = append(m.messages, ChatMessage{
				Role:    "system",
				Content: m.styles.ModalityLoaded.Render(info),
			})
			m.modalities = append(m.modalities, ModalityInfo{Name: p.Name, Shape: p.Shape})
			m.rebuildViewport()
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
			tb := renderProtocolTable(p.Headers, p.Rows, protocolTableRenderWidth(m.width))
			m.isStreaming = true
			appendProtocolCodeBlock(m.streamBuf, tb)
			m.rebuildViewport()
		}
		return m, waitForProtocolMsg(m.handler)

	case protocol.TypeForm:
		var p protocol.FormPayload
		if err := protocol.DecodePayload(msg.Message, &p); err == nil {
			formCmd := m.runFormSuspended(p, msg.ID)
			return m, tea.Batch(waitForProtocolMsg(m.handler), formCmd)
		}
		return m, waitForProtocolMsg(m.handler)

	case protocol.TypeConfirm:
		var p protocol.ConfirmPayload
		if err := protocol.DecodePayload(msg.Message, &p); err == nil {
			m.pendingConfirm = &p
			m.pendingConfirmID = msg.ID
			m.recalculateViewportHeight()
		}
		return m, waitForProtocolMsg(m.handler)

	case protocol.TypeSelect:
		var p protocol.SelectPayload
		if err := protocol.DecodePayload(msg.Message, &p); err == nil {
			m.pendingSelect = &p
			m.pendingSelectID = msg.ID
			m.selectIndex = 0
			m.recalculateViewportHeight()
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

func (m Model) handleKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	if msg.Type == tea.KeyCtrlG {
		m.mouseCapture = !m.mouseCapture
		return m, mouseCaptureCmd(m.mouseCapture)
	}

	// Intercept keys for pending confirm dialog.
	if m.pendingConfirm != nil {
		return m.handleConfirmKey(msg)
	}
	// Intercept keys for pending select dialog.
	if m.pendingSelect != nil {
		return m.handleSelectKey(msg)
	}

	switch msg.Type {

	case tea.KeyCtrlC:
		if m.handler != nil {
			_ = m.handler.SendTyped(protocol.TypeQuit, protocol.QuitPayload{}, "")
		}
		m.quitting = true
		return m, tea.Quit

	case tea.KeyEnter:
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
				m.messages = append(m.messages, ChatMessage{
					Role:    "system",
					Content: "Dashboard is not available in Go TUI mode. Use `lobster chat --ui classic` for the Textual dashboard.",
				})
				m.input.SetValue("")
				m.recalculateViewportHeight()
				cmd := m.refreshSuggestions()
				m.rebuildViewport()
				return m, cmd
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

		m.messages = append(m.messages, ChatMessage{
			Role:    "user",
			Content: val,
		})
		m.isStreaming = true
		m.input.SetValue("")
		m.recalculateViewportHeight()
		cmd := m.refreshSuggestions()
		m.rebuildViewport()

		return m, cmd

	case tea.KeyTab:
		if m.applyNextCompletionSuggestion() {
			m.recalculateViewportHeight()
			return m, m.refreshSuggestions()
		}
		return m, nil

	case tea.KeyPgUp, tea.KeyPgDown:
		// Transcript scrollback controls are dedicated to page keys.
		var cmd tea.Cmd
		m.viewport, cmd = m.viewport.Update(msg)
		return m, cmd

	case tea.KeyUp:
		// Up/Down are reserved for input history navigation.
		if m.recallHistoryUp() {
			m.recalculateViewportHeight()
			return m, m.refreshSuggestions()
		}
		return m, nil

	case tea.KeyDown:
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

func isComposerInsertNewlineKey(msg tea.KeyMsg) bool {
	key := strings.ToLower(strings.TrimSpace(msg.String()))
	return key == "shift+enter" || key == "alt+enter"
}

func (m *Model) applyNextCompletionSuggestion() bool {
	suggestions := m.completionSuggestions
	if len(suggestions) == 0 {
		return false
	}

	input := m.input.Value()
	needsReset := !suggestionSlicesEqual(m.completionCycleSuggestions, suggestions)
	if needsReset || (input != m.completionCycleBase && !containsSuggestion(suggestions, input)) {
		m.completionCycleBase = input
		m.completionCycleIndex = 0
		m.completionCycleSuggestions = append([]string(nil), suggestions...)
	} else {
		if idx := suggestionIndex(m.completionCycleSuggestions, input); idx >= 0 {
			m.completionCycleIndex = (idx + 1) % len(m.completionCycleSuggestions)
		} else {
			m.completionCycleIndex = 0
		}
	}

	if len(m.completionCycleSuggestions) == 0 {
		return false
	}

	m.input.SetValue(m.completionCycleSuggestions[m.completionCycleIndex])
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

func suggestionSlicesEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
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
		thumbTop = int(math.Round(float64(vp.YOffset) / float64(maxOffset) * float64(trackSpan)))
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

// rebuildViewportWithMode reconstructs the viewport while preserving user
// scroll position unless forceBottom is requested.
func (m *Model) rebuildViewportWithMode(forceBottom bool) {
	m.recalculateViewportHeight()
	wasAtBottom := m.viewport.AtBottom()
	renderer := m.getMarkdownRenderer()
	var b strings.Builder
	for _, msg := range m.messages {
		b.WriteString(renderMessage(msg, m.styles, m.width, renderer, m.inline))
		b.WriteByte('\n')
		// Add breathing room between a user turn and assistant response.
		if m.inline && msg.Role == "user" {
			b.WriteByte('\n')
		}
	}
	// Append in-progress streaming text.
	if m.streamBuf.Len() > 0 {
		partial := ChatMessage{
			Role:    "assistant",
			Content: m.streamBuf.String(),
			Agent:   m.activeAgent,
		}
		b.WriteString(renderMessage(partial, m.styles, m.width, renderer, m.inline))
		b.WriteByte('\n')
	}
	m.viewport.SetContent(b.String())
	if m.inlineFlowMode() {
		m.recalculateViewportHeight()
	}
	if forceBottom || wasAtBottom {
		m.viewport.GotoBottom()
	}
}

func (m *Model) applyClearTarget(target string) {
	switch target {
	case "output", "all", "":
		m.messages = m.messages[:0]
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

	m.messages = append(m.messages, ChatMessage{
		Role:    "assistant",
		Content: help,
	})
}

// renderDataSummary appends a summary of loaded modalities to the chat.
func (m *Model) renderDataSummary() {
	if len(m.modalities) == 0 {
		m.messages = append(m.messages, ChatMessage{
			Role:    "system",
			Content: "No data loaded. Use /read <file> or ask Lobster to load data.",
		})
		return
	}
	var tb strings.Builder
	tb.WriteString("## Loaded Data\n\n")
	tb.WriteString("| Modality | Shape |\n| --- | --- |\n")
	for _, mod := range m.modalities {
		tb.WriteString(fmt.Sprintf("| %s | %s |\n", mod.Name, mod.Shape))
	}
	m.messages = append(m.messages, ChatMessage{
		Role:    "assistant",
		Content: tb.String(),
	})
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

func lineCount(text string) int {
	if text == "" {
		return 0
	}
	return strings.Count(text, "\n") + 1
}

func (m Model) currentStatusLine() string {
	statusText := m.statusText
	if m.spinnerActive {
		spinnerText := spinnerFrames[m.spinnerFrame] + " " + m.spinnerLabel + "..."
		if m.quietStartup && !m.ready {
			spinnerText = "· " + m.spinnerLabel + "..."
		}
		if !m.ready && len(loadingTips) > 0 && !m.quietStartup {
			spinnerText += "  ·  " + m.styles.Muted.Render("💡 "+loadingTips[m.tipIndex])
		}
		statusText = spinnerText
	}
	if !m.ready {
		return statusText
	}

	if m.inlineFlowMode() {
		if strings.TrimSpace(statusText) == "" {
			return "inline flow: use terminal scrollback  ·  Ctrl+G optional mouse capture"
		}
		return statusText + "  ·  inline flow via terminal scrollback"
	}

	mouseLabel := m.mouseModeLabel()
	if strings.TrimSpace(statusText) == "" {
		return mouseLabel + "  ·  Ctrl+G toggles"
	}
	return statusText + "  ·  " + mouseLabel
}

func (m Model) mouseModeLabel() string {
	if m.mouseCapture {
		return "mouse: scroll"
	}
	return "mouse: select"
}

func mouseCaptureCmd(enabled bool) tea.Cmd {
	return func() tea.Msg {
		if enabled {
			return tea.EnableMouseCellMotion()
		}
		return tea.DisableMouse()
	}
}

// layoutReservedRows estimates non-viewport rows currently occupied by UI chrome.
func (m Model) layoutReservedRows() int {
	rows := 0

	if m.inline {
		if intro := renderInlineIntro(m); intro != "" {
			rows += lineCount(intro)
		}
	}
	rows += lineCount(renderHeader(m))
	if m.inline {
		if runtimeSummary := renderRuntimeSummary(m); runtimeSummary != "" {
			rows += lineCount(runtimeSummary) + 2 // explicit "\n\n" spacer in View()
		}
	}

	if tf := renderToolFeed(m.toolFeed, m.styles, m.width, m.inline); tf != "" {
		rows += lineCount(tf)
	}
	if m.progressActive {
		rows += lineCount(renderProgressBar(m.progressLabel, m.progressCurrent, m.progressTotal, m.width, m.styles))
	}

	if m.pendingConfirm != nil {
		rows += lineCount(renderConfirmPrompt(m.pendingConfirm, m.styles, m.width))
	} else if m.pendingSelect != nil {
		rows += lineCount(renderSelectPrompt(m.pendingSelect, m.selectIndex, m.styles, m.width))
	} else {
		if m.inline {
			rows += 1 // reserve optional prompt spacer in inline mode
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
		m.viewport.Height = h
		return
	}

	h := m.height - m.layoutReservedRows()
	if h < 1 {
		h = 1
	}
	m.viewport.Height = h
}

func (m Model) inlineFlowMode() bool {
	return m.inline && m.inlineFlow
}

func (m Model) composerInitialized() bool {
	return !(m.input.MaxHeight == 0 && m.input.Width() == 0 && m.input.Height() == 0)
}

func applyComposerStyles(input *textarea.Model, styles theme.Styles) {
	focused, blurred := textarea.DefaultStyles()

	textColor := styles.InputField.GetForeground()
	promptColor := styles.InputPrompt.GetForeground()
	mutedColor := styles.Muted.GetForeground()
	dimColor := styles.Dimmed.GetForeground()

	base := lipgloss.NewStyle().UnsetBackground()

	focused.Base = base
	blurred.Base = base

	focused.Text = lipgloss.NewStyle().Foreground(textColor)
	blurred.Text = lipgloss.NewStyle().Foreground(textColor)

	focused.CursorLine = lipgloss.NewStyle().Foreground(textColor)
	blurred.CursorLine = lipgloss.NewStyle().Foreground(textColor)

	focused.Placeholder = lipgloss.NewStyle().Foreground(mutedColor)
	blurred.Placeholder = lipgloss.NewStyle().Foreground(mutedColor)

	focused.Prompt = lipgloss.NewStyle().Foreground(promptColor)
	blurred.Prompt = lipgloss.NewStyle().Foreground(promptColor)

	focused.LineNumber = lipgloss.NewStyle().Foreground(dimColor)
	blurred.LineNumber = lipgloss.NewStyle().Foreground(dimColor)
	focused.CursorLineNumber = lipgloss.NewStyle().Foreground(mutedColor)
	blurred.CursorLineNumber = lipgloss.NewStyle().Foreground(mutedColor)

	focused.EndOfBuffer = lipgloss.NewStyle().Foreground(lipgloss.NoColor{})
	blurred.EndOfBuffer = lipgloss.NewStyle().Foreground(lipgloss.NoColor{})

	input.FocusedStyle = focused
	input.BlurredStyle = blurred
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
		visualLines += len(hardWrapProtocolTableCell(line, width))
	}
	if visualLines < composerMinHeight {
		visualLines = composerMinHeight
	}
	if visualLines > composerMaxHeight {
		visualLines = composerMaxHeight
	}
	return visualLines
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

func renderProtocolTable(headers []string, rows [][]string, totalWidth int) string {
	colCount := len(headers)
	if colCount == 0 {
		return ""
	}

	separatorWidth := lipgloss.Width(" │ ") * (colCount - 1)
	available := totalWidth - separatorWidth
	if available < colCount*6 {
		available = colCount * 6
	}

	widths := make([]int, colCount)
	minWidths := make([]int, colCount)
	for i, header := range headers {
		desired := lipgloss.Width(strings.TrimSpace(header))
		for _, row := range rows {
			if i >= len(row) {
				continue
			}
			for _, line := range strings.Split(strings.ReplaceAll(row[i], "\r\n", "\n"), "\n") {
				if w := lipgloss.Width(line); w > desired {
					desired = w
				}
			}
		}

		if desired < 8 {
			desired = 8
		}
		if desired > 42 {
			desired = 42
		}
		widths[i] = desired
		minWidths[i] = 6
	}

	widthBudget := 0
	for _, width := range widths {
		widthBudget += width
	}
	for widthBudget > available {
		shrinkIdx := -1
		for i := range widths {
			if widths[i] <= minWidths[i] {
				continue
			}
			if shrinkIdx == -1 || widths[i] > widths[shrinkIdx] {
				shrinkIdx = i
			}
		}
		if shrinkIdx == -1 {
			break
		}
		widths[shrinkIdx]--
		widthBudget--
	}
	if widthBudget < available {
		widths[len(widths)-1] += available - widthBudget
	}

	headerLines := wrapProtocolTableRow(headers, widths)
	var out strings.Builder
	for _, line := range headerLines {
		out.WriteString(line)
		out.WriteByte('\n')
	}
	out.WriteString(protocolTableDivider(widths))
	for _, row := range rows {
		out.WriteByte('\n')
		for _, line := range wrapProtocolTableRow(row, widths) {
			out.WriteString(line)
			out.WriteByte('\n')
		}
	}

	return strings.TrimRight(out.String(), "\n")
}

func wrapProtocolTableRow(row []string, widths []int) []string {
	cells := make([][]string, len(widths))
	rowHeight := 1
	for i, width := range widths {
		value := ""
		if i < len(row) {
			value = row[i]
		}
		cells[i] = wrapProtocolTableCell(value, width)
		if len(cells[i]) > rowHeight {
			rowHeight = len(cells[i])
		}
	}

	lines := make([]string, 0, rowHeight)
	for lineIdx := 0; lineIdx < rowHeight; lineIdx++ {
		parts := make([]string, len(widths))
		for colIdx, width := range widths {
			cellLine := ""
			if lineIdx < len(cells[colIdx]) {
				cellLine = cells[colIdx][lineIdx]
			}
			parts[colIdx] = padProtocolTableCell(cellLine, width)
		}
		lines = append(lines, strings.Join(parts, " │ "))
	}

	return lines
}

func wrapProtocolTableCell(value string, width int) []string {
	if width < 1 {
		return []string{""}
	}

	normalized := strings.ReplaceAll(value, "\r\n", "\n")
	rawLines := strings.Split(normalized, "\n")
	lines := make([]string, 0, len(rawLines))
	for _, raw := range rawLines {
		if raw == "" {
			lines = append(lines, "")
			continue
		}
		wrapped := wordwrap.String(raw, width)
		for _, segment := range strings.Split(wrapped, "\n") {
			lines = append(lines, hardWrapProtocolTableCell(segment, width)...)
		}
	}
	if len(lines) == 0 {
		return []string{""}
	}
	return lines
}

func hardWrapProtocolTableCell(value string, width int) []string {
	if lipgloss.Width(value) <= width {
		return []string{value}
	}

	lines := make([]string, 0, 2)
	var current strings.Builder
	currentWidth := 0
	for _, r := range value {
		rw := lipgloss.Width(string(r))
		if currentWidth+rw > width && current.Len() > 0 {
			lines = append(lines, current.String())
			current.Reset()
			currentWidth = 0
		}
		current.WriteRune(r)
		currentWidth += rw
	}
	if current.Len() > 0 {
		lines = append(lines, current.String())
	}
	if len(lines) == 0 {
		return []string{""}
	}
	return lines
}

func padProtocolTableCell(value string, width int) string {
	padding := width - lipgloss.Width(value)
	if padding <= 0 {
		return value
	}
	return value + strings.Repeat(" ", padding)
}

func protocolTableDivider(widths []int) string {
	parts := make([]string, len(widths))
	for i, width := range widths {
		parts[i] = strings.Repeat("─", width)
	}
	return strings.Join(parts, "─┼─")
}

// handleConfirmKey handles key presses when a confirm dialog is pending.
func (m Model) handleConfirmKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.Type {
	case tea.KeyCtrlC:
		// Dismiss confirm and send decline.
		return m.resolveConfirm(false)
	case tea.KeyEnter:
		// Enter accepts the default.
		confirm := m.pendingConfirm.Default
		return m.resolveConfirm(confirm)
	case tea.KeyRunes:
		s := msg.String()
		if s == "y" || s == "Y" {
			return m.resolveConfirm(true)
		}
		if s == "n" || s == "N" {
			return m.resolveConfirm(false)
		}
	}
	return m, nil
}

// resolveConfirm finalizes either a Python-originated or local confirm dialog.
func (m Model) resolveConfirm(confirm bool) (tea.Model, tea.Cmd) {
	if m.pendingConfirmID == localExitConfirmID {
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
		m.messages = append(m.messages, ChatMessage{
			Role:    "system",
			Content: "Exit cancelled.",
		})
		m.rebuildViewport()
		return m, nil
	}

	if m.handler != nil {
		_ = m.handler.SendTyped(protocol.TypeConfirmResponse, protocol.ConfirmResponsePayload{
			ID:      m.pendingConfirmID,
			Confirm: confirm,
		}, m.pendingConfirmID)
	}
	m.pendingConfirm = nil
	m.pendingConfirmID = ""
	m.recalculateViewportHeight()
	return m, nil
}

// handleSelectKey handles key presses when a select dialog is pending.
func (m Model) handleSelectKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	if m.pendingSelect == nil || len(m.pendingSelect.Options) == 0 {
		return m, nil
	}
	optCount := len(m.pendingSelect.Options)

	switch msg.Type {
	case tea.KeyCtrlC:
		selected := clampIndex(m.selectIndex, optCount)
		_ = m.handler.SendTyped(protocol.TypeSelectResponse, protocol.SelectResponsePayload{
			ID:    m.pendingSelectID,
			Value: m.pendingSelect.Options[selected],
			Index: selected,
		}, m.pendingSelectID)
		m.pendingSelect = nil
		m.pendingSelectID = ""
		m.recalculateViewportHeight()
		return m, nil
	case tea.KeyUp, tea.KeyShiftTab:
		m.selectIndex--
		if m.selectIndex < 0 {
			m.selectIndex = optCount - 1
		}
		return m, nil
	case tea.KeyDown, tea.KeyTab:
		m.selectIndex++
		if m.selectIndex >= optCount {
			m.selectIndex = 0
		}
		return m, nil
	case tea.KeyEnter:
		selected := clampIndex(m.selectIndex, optCount)
		_ = m.handler.SendTyped(protocol.TypeSelectResponse, protocol.SelectResponsePayload{
			ID:    m.pendingSelectID,
			Value: m.pendingSelect.Options[selected],
			Index: selected,
		}, m.pendingSelectID)
		m.pendingSelect = nil
		m.pendingSelectID = ""
		m.recalculateViewportHeight()
		return m, nil
	}
	return m, nil
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
