// Package biocomp provides the BioCharm component library for Lobster TUI.
//
// Each BioComponent is a self-contained interactive widget that can be
// rendered as an overlay, inline, or fullscreen element. Components register
// themselves via init() and are instantiated by the chat model when Python
// sends a component_show protocol message.
package biocomp

import (
	"encoding/json"

	"charm.land/bubbles/v2/key"
	tea "charm.land/bubbletea/v2"
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

	// InitCmd returns a one-shot tea.Cmd that should be executed after Init
	// completes (e.g. cursor blink starter). Returns nil if none needed.
	// The returned cmd is cleared after the first call.
	InitCmd() tea.Cmd

	// HandleMsg processes a BubbleTea message. Returns non-nil ComponentResult
	// when the user submits or cancels. Returns (nil, cmd) while still interacting.
	// The tea.Cmd enables Bubbles widgets (cursor blink, timers, animations).
	HandleMsg(msg tea.Msg) (*ComponentResult, tea.Cmd)

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
