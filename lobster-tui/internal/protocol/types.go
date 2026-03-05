// Package protocol defines the JSON message envelope and all message types
// exchanged between the Lobster AI Python process and this TUI over stdin/stdout.
//
// Direction conventions:
//   - Python → Go  (server → TUI): display/render messages
//   - Go → Python  (TUI → server): input/control messages
//
// Wire format (newline-delimited JSON):
//
//	{"version":1,"type":"text","payload":{...}}\n
package protocol

import "encoding/json"

// Version is the protocol version. Both sides must agree on this value.
const Version = 1

// ---- Envelope -------------------------------------------------------------

// Message is the top-level wire envelope. Every message on the stream uses
// this structure. Payload is decoded separately based on Type.
type Message struct {
	// Version allows future backwards-incompatible changes. Both sides should
	// reject messages with an unknown version.
	Version int `json:"version"`

	// Type identifies the payload schema (see constants below).
	Type string `json:"type"`

	// Payload contains the message-specific data as raw JSON.
	Payload json.RawMessage `json:"payload"`

	// ID is an optional correlation token used for request/response pairing
	// (e.g. form submissions).
	ID string `json:"id,omitempty"`
}

// ---- Python → Go message type constants -----------------------------------

const (
	// TypeText is a plain text chunk streamed from the assistant.
	TypeText = "text"

	// TypeMarkdown is a fully-rendered markdown block.
	TypeMarkdown = "markdown"

	// TypeCode is a code block with an optional language hint.
	TypeCode = "code"

	// TypeTable is tabular data (headers + rows).
	TypeTable = "table"

	// TypeForm requests the user to fill in a structured form.
	TypeForm = "form"

	// TypeConfirm requests a yes/no confirmation.
	TypeConfirm = "confirm"

	// TypeSelect presents a single-choice selector.
	TypeSelect = "select"

	// TypeProgress updates a progress bar.
	TypeProgress = "progress"

	// TypeAlert displays a coloured alert box (success/warning/error/info).
	TypeAlert = "alert"

	// TypeSpinner shows or hides a spinner.
	TypeSpinner = "spinner"

	// TypeStatus updates the status-bar text.
	TypeStatus = "status"

	// TypeClear clears the current output.
	TypeClear = "clear"

	// TypeDone signals that the current assistant turn is complete.
	TypeDone = "done"

	// TypeAgentTransition indicates the supervisor has handed off to a specialist.
	TypeAgentTransition = "agent_transition"

	// TypeModalityLoaded signals that a data modality has been loaded into the
	// DataManagerV2 workspace.
	TypeModalityLoaded = "modality_loaded"

	// TypeToolExecution reports the start or completion of a tool call.
	TypeToolExecution = "tool_execution"

	// TypeSuspend requests the TUI to suspend (return to shell).
	TypeSuspend = "suspend"

	// TypeResume signals the TUI to resume after a suspend.
	TypeResume = "resume"
)

// ---- Go → Python message type constants -----------------------------------

const (
	// TypeInput carries a free-text user message.
	TypeInput = "input"

	// TypeFormResponse carries filled form values.
	TypeFormResponse = "form_response"

	// TypeConfirmResponse carries a yes/no answer.
	TypeConfirmResponse = "confirm_response"

	// TypeSelectResponse carries the selected option index or value.
	TypeSelectResponse = "select_response"

	// TypeCancel requests cancellation of the current operation.
	TypeCancel = "cancel"

	// TypeQuit requests a clean shutdown of the Python process.
	TypeQuit = "quit"

	// TypeResize informs the Python process of terminal dimensions.
	TypeResize = "resize"

	// TypeSlashCommand carries a parsed slash-command invocation.
	TypeSlashCommand = "slash_command"

	// TypeHandshake is the first message sent by the TUI on connection to
	// negotiate version and capabilities.
	TypeHandshake = "handshake"
)

// ---- Python → Go payloads -------------------------------------------------

// TextPayload carries a streamed text chunk.
type TextPayload struct {
	Content string `json:"content"`
}

// MarkdownPayload carries a complete markdown block.
type MarkdownPayload struct {
	Content string `json:"content"`
}

// CodePayload carries a code block.
type CodePayload struct {
	Language string `json:"language,omitempty"`
	Content  string `json:"content"`
}

// TablePayload carries tabular data.
type TablePayload struct {
	Headers []string   `json:"headers"`
	Rows    [][]string `json:"rows"`
}

// FormField describes a single field in a form request.
type FormField struct {
	// Key is the machine-readable field identifier returned in FormResponse.
	Key string `json:"key"`
	// Label is the human-readable field label.
	Label string `json:"label"`
	// Type is one of "text", "password", "select", "multiselect", "confirm".
	Type string `json:"type"`
	// Options is populated for "select" and "multiselect" types.
	Options []string `json:"options,omitempty"`
	// Default is the pre-filled value shown to the user.
	Default string `json:"default,omitempty"`
	// Required marks the field as mandatory.
	Required bool `json:"required,omitempty"`
	// Description is an optional helper text shown below the field.
	Description string `json:"description,omitempty"`
}

// FormPayload requests the user to fill in a structured form.
type FormPayload struct {
	Title  string      `json:"title,omitempty"`
	Fields []FormField `json:"fields"`
}

// ConfirmPayload requests a yes/no confirmation.
type ConfirmPayload struct {
	Title   string `json:"title,omitempty"`
	Message string `json:"message"`
	Default bool   `json:"default,omitempty"`
}

// SelectPayload presents a single-choice list.
type SelectPayload struct {
	Title   string   `json:"title,omitempty"`
	Message string   `json:"message"`
	Options []string `json:"options"`
}

// ProgressPayload updates a named progress bar.
type ProgressPayload struct {
	// Label is the display name of this progress indicator.
	Label string `json:"label"`
	// Current is the number of completed units.
	Current int `json:"current"`
	// Total is the total number of units (-1 = indeterminate).
	Total int `json:"total"`
	// Done marks the progress as completed when true.
	Done bool `json:"done,omitempty"`
}

// AlertLevel classifies the severity of an alert.
type AlertLevel string

const (
	AlertSuccess AlertLevel = "success"
	AlertWarning AlertLevel = "warning"
	AlertError   AlertLevel = "error"
	AlertInfo    AlertLevel = "info"
)

// AlertPayload displays a coloured notification box.
type AlertPayload struct {
	Level   AlertLevel `json:"level"`
	Title   string     `json:"title,omitempty"`
	Message string     `json:"message"`
}

// SpinnerPayload shows or hides a named spinner.
type SpinnerPayload struct {
	Label  string `json:"label,omitempty"`
	Active bool   `json:"active"`
}

// StatusPayload updates the status-bar line.
type StatusPayload struct {
	Text string `json:"text"`
}

// ClearPayload optionally targets a specific region.
type ClearPayload struct {
	// Target is "all" (default), "output", or "status".
	Target string `json:"target,omitempty"`
}

// DonePayload signals the end of an assistant turn.
type DonePayload struct {
	// Summary may contain a brief end-of-turn note.
	Summary string `json:"summary,omitempty"`
}

// AgentTransitionPayload reports a supervisor → specialist handoff.
type AgentTransitionPayload struct {
	// From is the name of the sending agent (empty if supervisor).
	From string `json:"from,omitempty"`
	// To is the name of the receiving agent.
	To string `json:"to"`
	// Reason is an optional human-readable description of the handoff.
	Reason string `json:"reason,omitempty"`
}

// ModalityLoadedPayload reports that a data modality entered the workspace.
type ModalityLoadedPayload struct {
	// Name is the user-facing modality label (e.g. "scRNA-seq").
	Name string `json:"name"`
	// Shape is an optional dimension hint (e.g. "3245 cells × 22000 genes").
	Shape string `json:"shape,omitempty"`
	// Workspace is the DataManagerV2 workspace name.
	Workspace string `json:"workspace,omitempty"`
}

// ToolExecutionEvent describes one stage of a tool invocation.
type ToolExecutionEvent string

const (
	ToolExecutionStart  ToolExecutionEvent = "start"
	ToolExecutionFinish ToolExecutionEvent = "finish"
	ToolExecutionError  ToolExecutionEvent = "error"
)

// ToolExecutionPayload reports a tool-call lifecycle event.
type ToolExecutionPayload struct {
	ToolName string             `json:"tool_name"`
	Event    ToolExecutionEvent `json:"event"`
	// Summary is an optional one-line description shown to the user.
	Summary string `json:"summary,omitempty"`
}

// ---- Go → Python payloads -------------------------------------------------

// InputPayload carries user free text.
type InputPayload struct {
	Content string `json:"content"`
}

// FormResponsePayload carries filled form values keyed by FormField.Key.
type FormResponsePayload struct {
	// ID matches the Message.ID of the originating FormPayload.
	ID     string            `json:"id"`
	Values map[string]string `json:"values"`
}

// ConfirmResponsePayload carries a yes/no answer.
type ConfirmResponsePayload struct {
	// ID matches the Message.ID of the originating ConfirmPayload.
	ID      string `json:"id"`
	Confirm bool   `json:"confirm"`
}

// SelectResponsePayload carries the chosen option.
type SelectResponsePayload struct {
	// ID matches the Message.ID of the originating SelectPayload.
	ID    string `json:"id"`
	Value string `json:"value"`
	// Index is the 0-based position of the chosen option.
	Index int `json:"index"`
}

// CancelPayload requests cancellation of the in-progress operation.
type CancelPayload struct{}

// QuitPayload requests a clean shutdown.
type QuitPayload struct{}

// ResizePayload notifies the Python side of terminal dimensions.
type ResizePayload struct {
	Width  int `json:"width"`
	Height int `json:"height"`
}

// SlashCommandPayload carries a parsed slash command.
type SlashCommandPayload struct {
	// Command is the command name without the leading '/'.
	Command string `json:"command"`
	// Args is the remainder of the line after the command name.
	Args string `json:"args,omitempty"`
}

// HandshakePayload is the first message sent by the TUI on connection.
type HandshakePayload struct {
	// ClientVersion is the lobster-tui binary version.
	ClientVersion string `json:"client_version"`
	// ProtocolVersion is the protocol version this client implements.
	ProtocolVersion int `json:"protocol_version"`
	// TerminalWidth and TerminalHeight are the initial terminal dimensions.
	TerminalWidth  int `json:"terminal_width"`
	TerminalHeight int `json:"terminal_height"`
}
