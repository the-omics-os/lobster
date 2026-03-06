// Package biocomp provides the BioCharm component library for Lobster TUI.
//
// Each BioComponent is a self-contained interactive widget that can be
// rendered as an overlay, inline, or fullscreen element. Components register
// themselves via init() and are instantiated by the chat model when Python
// sends a component_show protocol message.
package biocomp

import (
	"encoding/json"

	"github.com/charmbracelet/bubbles/key"
	tea "github.com/charmbracelet/bubbletea"
)

// ComponentResult is returned from HandleMsg when the user completes interaction.
// Nil return means the user is still interacting.
type ComponentResult struct {
	Action string         // "submit" | "cancel"
	Data   map[string]any // Component-specific response payload
}

// BioComponent is the interface all bioinformatics TUI components implement.
type BioComponent interface {
	// Init validates the JSON payload and initializes component state.
	Init(data json.RawMessage) error

	// HandleMsg processes a BubbleTea message. Returns non-nil ComponentResult
	// when the user submits or cancels. Returns nil while still interacting.
	HandleMsg(msg tea.Msg) *ComponentResult

	// View renders the component content (inside the overlay frame).
	View(width, height int) string

	// SetData receives updated data from Python (streaming updates).
	SetData(data json.RawMessage) error

	// Name returns the component registry key (e.g. "confirm", "select").
	Name() string

	// Mode returns the rendering mode: "inline", "overlay", or "fullscreen".
	Mode() string

	// KeyBindings returns the help bar bindings for this component.
	KeyBindings() []key.Binding

	// ChangeEvent returns a pending intermediate value to send to Python,
	// or nil if no change event is pending. Called after HandleMsg.
	ChangeEvent() map[string]any
}
