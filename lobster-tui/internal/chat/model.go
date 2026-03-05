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

	"github.com/the-omics-os/lobster-tui/internal/protocol"
	"github.com/the-omics-os/lobster-tui/internal/theme"
)

// maxToolFeed is the maximum number of tool execution lines kept in the ring buffer.
const maxToolFeed = 5

// spinnerInterval controls the animation frame rate (~12 fps).
const spinnerInterval = 80 * time.Millisecond

// spinnerFrames are braille-based animation frames for the active spinner.
var spinnerFrames = []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}

// spinnerTick fires periodically to advance the spinner animation frame.
type spinnerTick struct{}

// heartbeatCheck fires periodically to verify the backend is still alive.
type heartbeatCheck struct{}

// protocolErr wraps an error read from the protocol handler's Errs() channel.
type protocolErr struct{ err error }

// ChatMessage represents a single message in the chat history.
type ChatMessage struct {
	// Role is one of "user", "assistant", or "system".
	Role string
	// Content is the rendered text content of the message.
	Content string
	// Agent is the name of the specialist agent (empty for user/system).
	Agent string
}

// Model is the top-level BubbleTea model for the chat interface.
type Model struct {
	handler *protocol.Handler

	viewport viewport.Model
	input    textinput.Model

	messages    []ChatMessage
	activeAgent string
	toolFeed    []string

	statusText  string
	isStreaming bool
	streamBuf   strings.Builder

	// Spinner animation state.
	spinnerActive bool
	spinnerFrame  int
	spinnerLabel  string

	// Input gating: input is ignored until the backend sends "ready".
	ready bool

	// Heartbeat monitoring: tracks the last heartbeat from the backend.
	lastHeartbeat time.Time

	width   int
	height  int
	quitting bool
	styles  theme.Styles
}

// protocolMsg wraps a protocol.Message as a tea.Msg.
type protocolMsg struct {
	protocol.Message
}

// protocolEOF signals that the protocol channel was closed (Python exited).
type protocolEOF struct{}

// NewModel creates a new chat Model wired to the given handler and styles.
func NewModel(handler *protocol.Handler, styles theme.Styles, width, height int) Model {
	vp := viewport.New(width, viewportHeight(height, 0))
	vp.SetContent("")

	ti := textinput.New()
	ti.Placeholder = "Initializing..."
	ti.Focus()
	ti.CharLimit = 4096
	ti.Width = width - 4 // leave room for prompt and padding

	return Model{
		handler:       handler,
		viewport:      vp,
		input:         ti,
		messages:      make([]ChatMessage, 0, 64),
		toolFeed:      make([]string, 0, maxToolFeed),
		ready:         false,
		lastHeartbeat: time.Now(),
		styles:        styles,
		width:         width,
		height:        height,
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

	case heartbeatCheck:
		if m.ready {
			return m, nil // Stop checking once backend is ready.
		}
		elapsed := time.Since(m.lastHeartbeat)
		if elapsed > 15*time.Second {
			m.statusText = "Backend not responding..."
		}
		return m, tea.Tick(5*time.Second, func(time.Time) tea.Msg { return heartbeatCheck{} })

	case tea.KeyMsg:
		return m.handleKey(msg)

	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		tfh := toolFeedHeight(m.toolFeed)
		m.viewport.Width = m.width
		m.viewport.Height = viewportHeight(m.height, tfh)
		m.input.Width = m.width - 4

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

	// Input field (1 line).
	b.WriteString(m.styles.InputPrompt.Render("> "))
	b.WriteString(m.input.View())
	b.WriteByte('\n')

	// Status bar (1 line). Override status text with animated spinner when active.
	statusText := m.statusText
	if m.spinnerActive {
		statusText = spinnerFrames[m.spinnerFrame] + " " + m.spinnerLabel + "..."
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
			line := fmt.Sprintf("%s %s", p.ToolName, string(p.Event))
			if p.Summary != "" {
				line = fmt.Sprintf("%s: %s", p.ToolName, p.Summary)
			}
			m.pushToolFeed(line)
			// Recalculate viewport height when tool feed changes.
			tfh := toolFeedHeight(m.toolFeed)
			m.viewport.Height = viewportHeight(m.height, tfh)
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
		}
		return m, waitForProtocolMsg(m.handler)

	case protocol.TypeAlert:
		var p protocol.AlertPayload
		if err := protocol.DecodePayload(msg.Message, &p); err == nil {
			alertText := renderAlert(string(p.Level), p.Message, m.styles, m.width)
			m.messages = append(m.messages, ChatMessage{
				Role:    "system",
				Content: alertText,
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
				// Start the spinner tick loop.
				return m, tea.Batch(
					waitForProtocolMsg(m.handler),
					tea.Tick(spinnerInterval, func(time.Time) tea.Msg { return spinnerTick{} }),
				)
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

	case protocol.TypeSuspend:
		return m, tea.Suspend

	case protocol.TypeResume:
		// BubbleTea auto-redraws on resume; nothing extra needed.
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

		if strings.HasPrefix(val, "/") {
			// Slash command: split into command + args.
			parts := strings.SplitN(val[1:], " ", 2)
			cmd := parts[0]
			args := ""
			if len(parts) > 1 {
				args = parts[1]
			}
			_ = m.handler.SendTyped(protocol.TypeSlashCommand, protocol.SlashCommandPayload{
				Command: cmd,
				Args:    args,
			}, "")
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
		m.rebuildViewport()

		return m, nil

	default:
		// Forward all other keys to the text input and viewport.
		var cmds []tea.Cmd
		var cmd tea.Cmd

		m.input, cmd = m.input.Update(msg)
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

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------

// rebuildViewport reconstructs the viewport content from messages + active stream.
func (m *Model) rebuildViewport() {
	var b strings.Builder
	for _, msg := range m.messages {
		b.WriteString(renderMessage(msg, m.styles, m.width))
		b.WriteByte('\n')
	}
	// Append in-progress streaming text.
	if m.streamBuf.Len() > 0 {
		partial := ChatMessage{
			Role:    "assistant",
			Content: m.streamBuf.String(),
			Agent:   m.activeAgent,
		}
		b.WriteString(renderMessage(partial, m.styles, m.width))
		b.WriteByte('\n')
	}
	m.viewport.SetContent(b.String())
	m.viewport.GotoBottom()
}

// pushToolFeed adds a line to the tool feed ring buffer, evicting the oldest
// entry if the buffer is at capacity.
func (m *Model) pushToolFeed(line string) {
	if len(m.toolFeed) >= maxToolFeed {
		m.toolFeed = m.toolFeed[1:]
	}
	m.toolFeed = append(m.toolFeed, line)
}

// toolFeedHeight returns the number of lines occupied by the tool feed.
func toolFeedHeight(feed []string) int {
	if len(feed) == 0 {
		return 0
	}
	// One line per entry, plus one for the trailing newline separator.
	return len(feed) + 1
}

// viewportHeight calculates the available height for the viewport.
// Layout: header(1) + viewport + toolFeed(tfh) + input(2) + status(1) = height.
func viewportHeight(totalHeight, toolFeedH int) int {
	h := totalHeight - 4 - toolFeedH // 1 header + 2 input + 1 status
	if h < 1 {
		h = 1
	}
	return h
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
