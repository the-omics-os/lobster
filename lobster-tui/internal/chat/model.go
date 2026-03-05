// Package chat implements the interactive chat BubbleTea model for Lobster AI.
//
// The model bridges the protocol handler (JSON Lines IPC with Python) and the
// BubbleTea event loop, rendering streamed text, agent transitions, tool calls,
// and user input into a terminal-based chat interface.
package chat

import (
	"fmt"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/textinput"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/glamour"

	"github.com/the-omics-os/lobster-tui/internal/protocol"
	"github.com/the-omics-os/lobster-tui/internal/theme"
)

// maxToolFeed is the maximum number of tool execution lines kept in the ring buffer.
const maxToolFeed = 5

// maxInputHistory is the number of entered lines kept for Up/Down recall.
const maxInputHistory = 200

// spinnerInterval controls the animation frame rate (~12 fps).
const spinnerInterval = 80 * time.Millisecond

// spinnerFrames are braille-based animation frames for the active spinner.
var spinnerFrames = []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}

// tipInterval controls how often loading tips rotate.
const tipInterval = 5 * time.Second

// completionRequestTimeout prevents stale in-flight request tracking forever.
const completionRequestTimeout = 700 * time.Millisecond

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
	input    textinput.Model

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

	// Session and metadata.
	sessionID string

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

	width    int
	height   int
	quitting bool
	styles   theme.Styles
}

// protocolMsg wraps a protocol.Message as a tea.Msg.
type protocolMsg struct {
	protocol.Message
}

// protocolEOF signals that the protocol channel was closed (Python exited).
type protocolEOF struct{}

// NewModel creates a new chat Model wired to the given handler and styles.
func NewModel(handler *protocol.Handler, styles theme.Styles, width, height int) Model {
	vp := viewport.New(width, viewportHeight(height, 0, 0))
	vp.SetContent("")

	ti := textinput.New()
	ti.Prompt = ""
	ti.Placeholder = "Initializing..."
	ti.Focus()
	ti.CharLimit = 4096
	ti.Width = width - 4 // leave room for prompt and padding
	ti.ShowSuggestions = true
	ti.CompletionStyle = styles.Muted

	return Model{
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
	}
}

// Init starts the protocol read loop and returns the initial commands.
func (m Model) Init() tea.Cmd {
	m.handler.StartReadLoop()
	return tea.Batch(
		waitForProtocolMsg(m.handler),
		waitForProtocolErr(m.handler),
		textinput.Blink,
		tea.Tick(5*time.Second, func(time.Time) tea.Msg { return heartbeatCheck{} }),
	)
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
		if !m.spinnerActive {
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
		tfh := toolFeedHeight(m.toolFeed)
		ph := progressHeight(m.progressActive)
		m.viewport.Width = m.width
		m.viewport.Height = viewportHeight(m.height, tfh, ph)
		m.input.Width = m.width - 4

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

	// Header (1 line).
	b.WriteString(renderHeader(m))
	b.WriteByte('\n')

	// Viewport (scrollable message history).
	b.WriteString(m.viewport.View())
	b.WriteByte('\n')

	// Tool feed (0-N lines, dim).
	tf := renderToolFeed(m.toolFeed, m.styles, m.width)
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
		b.WriteString(m.styles.InputPrompt.Render("> "))
		b.WriteString(m.input.View())
		b.WriteByte('\n')
	}

	// Status bar (1 line). Override status text with animated spinner when active.
	statusText := m.statusText
	if m.spinnerActive {
		spinnerText := spinnerFrames[m.spinnerFrame] + " " + m.spinnerLabel + "..."
		// Show rotating tips during init (not yet ready).
		if !m.ready && len(loadingTips) > 0 {
			spinnerText += "  ·  " + m.styles.Muted.Render("💡 "+loadingTips[m.tipIndex])
		}
		statusText = spinnerText
	}
	b.WriteString(renderStatusBar(statusText, m.styles, m.width))

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
			// Recalculate viewport height when tool feed changes.
			tfh := toolFeedHeight(m.toolFeed)
			ph := progressHeight(m.progressActive)
			m.viewport.Height = viewportHeight(m.height, tfh, ph)
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
		m.rebuildViewport()
		return m, waitForProtocolMsg(m.handler)

	case protocol.TypeStatus:
		var p protocol.StatusPayload
		if err := protocol.DecodePayload(msg.Message, &p); err == nil {
			m.statusText = p.Text
			// Extract session ID if present.
			if strings.HasPrefix(p.Text, "Session: ") {
				m.sessionID = strings.TrimPrefix(p.Text, "Session: ")
			}
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
				// Start spinner tick + tip rotation (tips only matter during init).
				cmds := []tea.Cmd{
					waitForProtocolMsg(m.handler),
					tea.Tick(spinnerInterval, func(time.Time) tea.Msg { return spinnerTick{} }),
				}
				if !m.ready {
					cmds = append(cmds, tea.Tick(tipInterval, func(time.Time) tea.Msg { return tipRotate{} }))
				}
				return m, tea.Batch(cmds...)
			} else {
				m.spinnerActive = false
				m.statusText = ""
			}
		}
		return m, waitForProtocolMsg(m.handler)

	case protocol.TypeReady:
		m.ready = true
		m.input.Placeholder = "Ask Lobster anything..."
		m.input.Focus()
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
			switch p.Target {
			case "output", "all", "":
				m.messages = m.messages[:0]
				m.streamBuf.Reset()
				m.modalities = m.modalities[:0]
				m.rebuildViewport()
			case "status":
				m.statusText = ""
			}
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
			// Recalculate viewport height when progress bar changes.
			tfh := toolFeedHeight(m.toolFeed)
			ph := progressHeight(m.progressActive)
			m.viewport.Height = viewportHeight(m.height, tfh, ph)
		}
		return m, waitForProtocolMsg(m.handler)

	case protocol.TypeTable:
		var p protocol.TablePayload
		if err := protocol.DecodePayload(msg.Message, &p); err == nil && len(p.Headers) > 0 {
			var tb strings.Builder
			tb.WriteString("\n| ")
			tb.WriteString(strings.Join(escapePipes(p.Headers), " | "))
			tb.WriteString(" |\n| ")
			for i := range p.Headers {
				if i > 0 {
					tb.WriteString(" | ")
				}
				tb.WriteString("---")
			}
			tb.WriteString(" |\n")
			for _, row := range p.Rows {
				tb.WriteString("| ")
				tb.WriteString(strings.Join(escapePipes(row), " | "))
				tb.WriteString(" |\n")
			}
			m.isStreaming = true
			m.streamBuf.WriteString(tb.String())
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
		}
		return m, waitForProtocolMsg(m.handler)

	case protocol.TypeSelect:
		var p protocol.SelectPayload
		if err := protocol.DecodePayload(msg.Message, &p); err == nil {
			m.pendingSelect = &p
			m.pendingSelectID = msg.ID
			m.selectIndex = 0
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
		_ = m.handler.SendTyped(protocol.TypeQuit, protocol.QuitPayload{}, "")
		m.quitting = true
		return m, tea.Quit

	case tea.KeyEnter:
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
				m.messages = m.messages[:0]
				m.streamBuf.Reset()
				m.toolFeed = m.toolFeed[:0]
				m.rebuildViewport()
				m.input.SetValue("")
				return m, m.refreshSuggestions()
			case "exit", "quit":
				_ = m.handler.SendTyped(protocol.TypeQuit, protocol.QuitPayload{}, "")
				m.quitting = true
				return m, tea.Quit
			case "dashboard":
				m.messages = append(m.messages, ChatMessage{
					Role:    "system",
					Content: "Dashboard is not available in Go TUI mode. Use `lobster chat --ui classic` for the Textual dashboard.",
				})
				m.input.SetValue("")
				cmd := m.refreshSuggestions()
				m.rebuildViewport()
				return m, cmd
			default:
				// Forward to Python for handling.
				_ = m.handler.SendTyped(protocol.TypeSlashCommand, protocol.SlashCommandPayload{
					Command: cmd,
					Args:    args,
				}, "")
			}
		} else {
			_ = m.handler.SendTyped(protocol.TypeInput, protocol.InputPayload{
				Content: val,
			}, "")
		}

		m.messages = append(m.messages, ChatMessage{
			Role:    "user",
			Content: val,
		})
		m.isStreaming = true
		m.input.SetValue("")
		cmd := m.refreshSuggestions()
		m.rebuildViewport()

		return m, cmd

	default:
		trimmedInput := strings.TrimLeft(m.input.Value(), " \t")
		hasSlashSuggestions := strings.HasPrefix(trimmedInput, "/") && len(m.input.MatchedSuggestions()) > 0

		if msg.Type == tea.KeyUp && !hasSlashSuggestions {
			if m.recallHistoryUp() {
				return m, m.refreshSuggestions()
			}
		}
		if msg.Type == tea.KeyDown && !hasSlashSuggestions {
			if m.recallHistoryDown() {
				return m, m.refreshSuggestions()
			}
		}

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

		// Refresh autocomplete suggestions after every keystroke.
		cmd = m.refreshSuggestions()
		if cmd != nil {
			cmds = append(cmds, cmd)
		}

		// Gate viewport scrolling: when autocomplete has matches, Tab should
		// navigate/accept suggestions rather than scrolling the viewport.
		hasSuggestions := len(m.input.MatchedSuggestions()) > 0
		skipViewport := hasSuggestions && msg.Type == tea.KeyTab

		if !skipViewport {
			m.viewport, cmd = m.viewport.Update(msg)
			if cmd != nil {
				cmds = append(cmds, cmd)
			}
		}

		return m, tea.Batch(cmds...)
	}
}

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------

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
	renderer := m.getMarkdownRenderer()
	var b strings.Builder
	for _, msg := range m.messages {
		b.WriteString(renderMessage(msg, m.styles, m.width, renderer))
		b.WriteByte('\n')
	}
	// Append in-progress streaming text.
	if m.streamBuf.Len() > 0 {
		partial := ChatMessage{
			Role:    "assistant",
			Content: m.streamBuf.String(),
			Agent:   m.activeAgent,
		}
		b.WriteString(renderMessage(partial, m.styles, m.width, renderer))
		b.WriteByte('\n')
	}
	m.viewport.SetContent(b.String())
	m.viewport.GotoBottom()
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

// toolFeedHeight returns the number of lines occupied by the tool feed.
func toolFeedHeight(feed []ToolFeedEntry) int {
	if len(feed) == 0 {
		return 0
	}
	// One line per entry, plus one for the trailing newline separator.
	return len(feed) + 1
}

// progressHeight returns 1 if a progress bar is active, 0 otherwise.
func progressHeight(active bool) int {
	if active {
		return 1
	}
	return 0
}

// viewportHeight calculates the available height for the viewport.
// Layout: header(1) + viewport + toolFeed(tfh) + progress(ph) + input(2) + status(1) = height.
func viewportHeight(totalHeight, toolFeedH, progressH int) int {
	h := totalHeight - 4 - toolFeedH - progressH // 1 header + 2 input + 1 status
	if h < 1 {
		h = 1
	}
	return h
}

// escapePipes replaces pipe characters in strings to prevent breaking markdown tables.
func escapePipes(items []string) []string {
	out := make([]string, len(items))
	for i, s := range items {
		out[i] = strings.ReplaceAll(s, "|", "\\|")
	}
	return out
}

// handleConfirmKey handles key presses when a confirm dialog is pending.
func (m Model) handleConfirmKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.Type {
	case tea.KeyCtrlC:
		// Dismiss confirm and send decline.
		_ = m.handler.SendTyped(protocol.TypeConfirmResponse, protocol.ConfirmResponsePayload{
			ID:      m.pendingConfirmID,
			Confirm: false,
		}, m.pendingConfirmID)
		m.pendingConfirm = nil
		m.pendingConfirmID = ""
		return m, nil
	case tea.KeyEnter:
		// Enter accepts the default.
		confirm := m.pendingConfirm.Default
		_ = m.handler.SendTyped(protocol.TypeConfirmResponse, protocol.ConfirmResponsePayload{
			ID:      m.pendingConfirmID,
			Confirm: confirm,
		}, m.pendingConfirmID)
		m.pendingConfirm = nil
		m.pendingConfirmID = ""
		return m, nil
	case tea.KeyRunes:
		s := msg.String()
		if s == "y" || s == "Y" {
			_ = m.handler.SendTyped(protocol.TypeConfirmResponse, protocol.ConfirmResponsePayload{
				ID:      m.pendingConfirmID,
				Confirm: true,
			}, m.pendingConfirmID)
			m.pendingConfirm = nil
			m.pendingConfirmID = ""
			return m, nil
		}
		if s == "n" || s == "N" {
			_ = m.handler.SendTyped(protocol.TypeConfirmResponse, protocol.ConfirmResponsePayload{
				ID:      m.pendingConfirmID,
				Confirm: false,
			}, m.pendingConfirmID)
			m.pendingConfirm = nil
			m.pendingConfirmID = ""
			return m, nil
		}
	}
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
		// Dismiss select — send first option as default.
		_ = m.handler.SendTyped(protocol.TypeSelectResponse, protocol.SelectResponsePayload{
			ID:    m.pendingSelectID,
			Value: m.pendingSelect.Options[0],
			Index: 0,
		}, m.pendingSelectID)
		m.pendingSelect = nil
		m.pendingSelectID = ""
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
		_ = m.handler.SendTyped(protocol.TypeSelectResponse, protocol.SelectResponsePayload{
			ID:    m.pendingSelectID,
			Value: m.pendingSelect.Options[m.selectIndex],
			Index: m.selectIndex,
		}, m.pendingSelectID)
		m.pendingSelect = nil
		m.pendingSelectID = ""
		return m, nil
	}
	return m, nil
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
