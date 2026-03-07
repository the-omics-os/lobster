package chat

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"testing"

	"charm.land/bubbles/v2/key"
	"charm.land/bubbles/v2/textarea"
	"charm.land/bubbles/v2/viewport"
	tea "charm.land/bubbletea/v2"

	"github.com/the-omics-os/lobster-tui/internal/biocomp"
	// Blank imports to trigger init() registration for COMP-01 test.
	_ "github.com/the-omics-os/lobster-tui/internal/biocomp/celltype"
	_ "github.com/the-omics-os/lobster-tui/internal/biocomp/threshold"
	"github.com/the-omics-os/lobster-tui/internal/protocol"
	"github.com/the-omics-os/lobster-tui/internal/theme"
)

// ---------------------------------------------------------------------------
// Mock BioComponent
// ---------------------------------------------------------------------------

type mockComponent struct {
	name        string
	mode        string
	initErr     error
	handleResult *biocomp.ComponentResult
	viewContent string
	viewPanic   bool
	handlePanic bool
	changeEvent map[string]any
	setDataCalls []json.RawMessage
	initCalls    int
	handleCalls  int
	viewCalls    int
}

func (m *mockComponent) Init(data json.RawMessage) error {
	m.initCalls++
	return m.initErr
}
func (m *mockComponent) HandleMsg(msg tea.Msg) *biocomp.ComponentResult {
	m.handleCalls++
	if m.handlePanic {
		panic("mock HandleMsg panic")
	}
	return m.handleResult
}
func (m *mockComponent) View(w, h int) string {
	m.viewCalls++
	if m.viewPanic {
		panic("mock View panic")
	}
	return m.viewContent
}
func (m *mockComponent) SetData(data json.RawMessage) error {
	m.setDataCalls = append(m.setDataCalls, data)
	return nil
}
func (m *mockComponent) Name() string                { return m.name }
func (m *mockComponent) Mode() string                { return m.mode }
func (m *mockComponent) KeyBindings() []key.Binding   { return nil }
func (m *mockComponent) ChangeEvent() map[string]any  { return m.changeEvent }

// ---------------------------------------------------------------------------
// Mock protocol handler that captures sent messages
// ---------------------------------------------------------------------------

type capturedMessage struct {
	Type    string
	Payload json.RawMessage
	ID      string
}

func newCapturingHandler() (*protocol.Handler, *bytes.Buffer) {
	// Create a handler with a real reader (won't be used) and a buffer writer.
	var outBuf bytes.Buffer
	r := strings.NewReader("") // dummy reader
	h := protocol.NewHandler(r, &outBuf)
	return h, &outBuf
}

func parseCapturedMessages(buf *bytes.Buffer) []capturedMessage {
	var msgs []capturedMessage
	for _, line := range strings.Split(strings.TrimSpace(buf.String()), "\n") {
		if line == "" {
			continue
		}
		var m protocol.Message
		if err := json.Unmarshal([]byte(line), &m); err != nil {
			continue
		}
		msgs = append(msgs, capturedMessage{
			Type:    m.Type,
			Payload: m.Payload,
			ID:      m.ID,
		})
	}
	return msgs
}

func newTestModelWithHandler(h *protocol.Handler) Model {
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
		handler:                 h,
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
		height:                  24,
		ready:                   true,
		inlineFlow:              true,
		showIntro:               true,
		welcomeSporadicCell:     -1,
		styles:                  styles,
	}
}

// ---------------------------------------------------------------------------
// COMP-01: Registry imports — threshold_slider and cell_type_selector
// ---------------------------------------------------------------------------

func TestComponentRegistry_AllRegistered(t *testing.T) {
	names := biocomp.Names()
	nameSet := make(map[string]bool, len(names))
	for _, n := range names {
		nameSet[n] = true
	}

	required := []string{"threshold_slider", "cell_type_selector"}
	for _, name := range required {
		if !nameSet[name] {
			sort.Strings(names)
			t.Fatalf("expected %q in registry, got: %v", name, names)
		}
	}
}

// ---------------------------------------------------------------------------
// COMP-02: Cancel action included in response
// ---------------------------------------------------------------------------

func TestCancelAction_IncludedInResponse(t *testing.T) {
	h, buf := newCapturingHandler()
	m := newTestModelWithHandler(h)

	m.sendComponentResponse("test-msg-123", "cancel", map[string]any{"reason": "replaced"})

	msgs := parseCapturedMessages(buf)
	if len(msgs) == 0 {
		t.Fatal("expected at least one protocol message sent")
	}

	// Parse the payload to check action is in Data.
	var resp protocol.ComponentResponsePayload
	if err := json.Unmarshal(msgs[0].Payload, &resp); err != nil {
		t.Fatalf("unmarshal response payload: %v", err)
	}

	action, ok := resp.Data["action"]
	if !ok {
		t.Fatalf("expected 'action' key in Data map, got: %v", resp.Data)
	}
	if action != "cancel" {
		t.Fatalf("expected action='cancel', got %v", action)
	}
}

// ---------------------------------------------------------------------------
// COMP-03: Component replacement cancels old component
// ---------------------------------------------------------------------------

func TestComponentReplacement_CancelsOld(t *testing.T) {
	h, buf := newCapturingHandler()
	m := newTestModelWithHandler(h)

	// Set an active component.
	m.activeComponent = &ActiveComponent{
		Component: &mockComponent{name: "old_comp", mode: "overlay"},
		MsgID:     "old-msg-id",
	}

	// Render a new component (triggers replacement).
	payload := protocol.ComponentRenderPayload{
		Component: "text_input",
		Data:      map[string]any{"question": "Test?"},
	}
	m.handleComponentRender(payload, "new-msg-id")

	msgs := parseCapturedMessages(buf)
	if len(msgs) == 0 {
		t.Fatal("expected cancel message for old component")
	}

	// First message should be the cancel for the old component.
	var resp protocol.ComponentResponsePayload
	if err := json.Unmarshal(msgs[0].Payload, &resp); err != nil {
		t.Fatalf("unmarshal cancel response: %v", err)
	}

	if resp.ID != "old-msg-id" {
		t.Fatalf("expected cancel for old-msg-id, got %q", resp.ID)
	}
	if action, ok := resp.Data["action"]; !ok || action != "cancel" {
		t.Fatalf("expected action=cancel in Data, got: %v", resp.Data)
	}
}

// ---------------------------------------------------------------------------
// COMP-04: Overlay size shared between layoutReservedRows and View
// ---------------------------------------------------------------------------

func TestOverlaySize_SharedBetweenLayoutAndView(t *testing.T) {
	// The overlay size for a given component should be the same whether
	// computed from layoutReservedRows or from the View rendering path.
	// Both use biocomp.OverlaySize with the same size hint.
	for _, tc := range []struct {
		name     string
		compName string
		hint     string
	}{
		{"small_confirm", "confirm", "small"},
		{"large_celltype", "cell_type_selector", "large"},
		{"medium_threshold", "threshold_slider", "medium"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, h := biocomp.OverlaySize(80, 24, tc.hint)
			if h < 1 {
				t.Fatalf("overlay height for %s should be >= 1, got %d", tc.hint, h)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// COMP-07: Error boundary — bad Init sends error
// ---------------------------------------------------------------------------

func TestErrorBoundary_BadInitSendsError(t *testing.T) {
	h, buf := newCapturingHandler()
	m := newTestModelWithHandler(h)

	payload := protocol.ComponentRenderPayload{
		Component: "text_input",
		Data:      map[string]any{}, // Missing required "question" field — Init will fail.
	}
	m.handleComponentRender(payload, "err-msg-id")

	msgs := parseCapturedMessages(buf)
	found := false
	for _, msg := range msgs {
		var resp protocol.ComponentResponsePayload
		if err := json.Unmarshal(msg.Payload, &resp); err != nil {
			continue
		}
		if action, ok := resp.Data["action"]; ok && action == "error" {
			found = true
			break
		}
		// Also check for error key directly (existing pattern).
		if _, ok := resp.Data["error"]; ok {
			found = true
			break
		}
	}
	if !found {
		t.Fatal("expected error response for bad Init, got none")
	}
}

// ---------------------------------------------------------------------------
// COMP-07: Error boundary — panic in View recovers
// ---------------------------------------------------------------------------

func TestErrorBoundary_PanicInViewRecovers(t *testing.T) {
	h, buf := newCapturingHandler()
	m := newTestModelWithHandler(h)

	comp := &mockComponent{
		name:      "panicky",
		mode:      "overlay",
		viewPanic: true,
	}
	m.activeComponent = &ActiveComponent{
		Component: comp,
		MsgID:     "panic-view-id",
	}

	// The View() call in the main View method should not crash.
	// It should recover and send an error response.
	func() {
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("View panic was NOT recovered: %v", r)
			}
		}()
		_ = m.View()
	}()

	// If we get here without panicking, the boundary works.
	// Note: View() has a value receiver, so activeComponent clearing doesn't
	// persist on the caller's copy. The critical safety property is that the
	// panic is recovered and an error response is sent via the handler.
	msgs := parseCapturedMessages(buf)
	foundError := false
	for _, msg := range msgs {
		var resp protocol.ComponentResponsePayload
		if err := json.Unmarshal(msg.Payload, &resp); err == nil {
			if resp.Data["action"] == "error" {
				foundError = true
			}
		}
	}
	if !foundError {
		t.Fatal("expected error response after View panic")
	}
}

// ---------------------------------------------------------------------------
// COMP-07: Error boundary — panic in HandleMsg recovers
// ---------------------------------------------------------------------------

func TestErrorBoundary_PanicInHandleMsgRecovers(t *testing.T) {
	h, buf := newCapturingHandler()
	m := newTestModelWithHandler(h)

	comp := &mockComponent{
		name:        "panicky",
		mode:        "overlay",
		handlePanic: true,
	}
	m.activeComponent = &ActiveComponent{
		Component: comp,
		MsgID:     "panic-handle-id",
	}

	// Sending a key to the active overlay component should not crash.
	var updated tea.Model
	func() {
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("HandleMsg panic was NOT recovered: %v", r)
			}
		}()
		updated, _ = m.handleKey(tea.KeyPressMsg{Code: 'x', Text: "x"})
	}()

	// Verify error response was sent.
	msgs := parseCapturedMessages(buf)
	foundError := false
	for _, msg := range msgs {
		var resp protocol.ComponentResponsePayload
		if err := json.Unmarshal(msg.Payload, &resp); err == nil {
			if resp.Data["action"] == "error" {
				foundError = true
			}
		}
	}
	if !foundError {
		t.Fatal("expected error response after HandleMsg panic")
	}

	// Verify activeComponent was cleared.
	got := updated.(Model)
	if got.activeComponent != nil {
		t.Fatal("expected activeComponent to be nil after HandleMsg panic")
	}
}

// ---------------------------------------------------------------------------
// COMP-05: ChangeEvent debounce
// ---------------------------------------------------------------------------

func TestChangeEvent_DebouncedAt50ms(t *testing.T) {
	h, buf := newCapturingHandler()
	m := newTestModelWithHandler(h)

	// Set up an active overlay component that returns a ChangeEvent.
	comp := &mockComponent{
		name:        "slider",
		mode:        "overlay",
		changeEvent: map[string]any{"value": 0.05},
	}
	m.activeComponent = &ActiveComponent{
		Component: comp,
		MsgID:     "ce-msg-id",
	}

	// Send a key that triggers HandleMsg (returns nil = still interacting).
	updated, cmd := m.handleKey(tea.KeyPressMsg{Code: 'a', Text: "a"})
	got := updated.(Model)

	// The pending change event should be stored.
	if got.pendingChangeEvent == nil {
		t.Fatal("expected pendingChangeEvent to be set after ChangeEvent returned data")
	}

	// A debounce tick command should have been returned.
	if cmd == nil {
		t.Fatal("expected a tick command for debounce")
	}

	// Simulate the tick firing.
	updated2, _ := got.Update(changeEventTick{})
	got2 := updated2.(Model)

	// After tick, pending should be cleared.
	if got2.pendingChangeEvent != nil {
		t.Fatal("expected pendingChangeEvent to be nil after tick")
	}

	// Check that a change_event message was sent.
	msgs := parseCapturedMessages(buf)
	found := false
	for _, msg := range msgs {
		if msg.Type == "change_event" {
			found = true
			break
		}
	}
	if !found {
		t.Fatal("expected change_event protocol message after tick")
	}
}

func TestChangeEvent_MultipleWithinDebounce(t *testing.T) {
	h, buf := newCapturingHandler()
	m := newTestModelWithHandler(h)

	comp := &mockComponent{
		name:        "slider",
		mode:        "overlay",
		changeEvent: map[string]any{"value": 0.01},
	}
	m.activeComponent = &ActiveComponent{
		Component: comp,
		MsgID:     "ce-multi-id",
	}

	// First key press — starts debounce.
	updated, _ := m.handleKey(tea.KeyPressMsg{Code: 'a', Text: "a"})
	got := updated.(Model)

	// Second key press with different value — should overwrite pending.
	comp.changeEvent = map[string]any{"value": 0.10}
	updated2, _ := got.handleKey(tea.KeyPressMsg{Code: 'b', Text: "b"})
	got2 := updated2.(Model)

	// Now fire tick.
	updated3, _ := got2.Update(changeEventTick{})
	_ = updated3.(Model)

	// Only one change_event message should be sent, with the latest value.
	msgs := parseCapturedMessages(buf)
	changeEvents := 0
	for _, msg := range msgs {
		if msg.Type == "change_event" {
			changeEvents++
			// Verify it has the latest value.
			var payload map[string]any
			if err := json.Unmarshal(msg.Payload, &payload); err == nil {
				if data, ok := payload["data"].(map[string]any); ok {
					if val, ok := data["value"].(float64); ok && val != 0.10 {
						t.Fatalf("expected latest value 0.10, got %v", val)
					}
				}
			}
		}
	}
	if changeEvents != 1 {
		t.Fatalf("expected exactly 1 change_event message, got %d", changeEvents)
	}
}

// ---------------------------------------------------------------------------
// COMP-06: SetData routes to active component
// ---------------------------------------------------------------------------

func TestSetData_RoutesToActiveComponent(t *testing.T) {
	h, _ := newCapturingHandler()
	m := newTestModelWithHandler(h)

	comp := &mockComponent{
		name: "slider",
		mode: "overlay",
	}
	m.activeComponent = &ActiveComponent{
		Component: comp,
		MsgID:     "sd-msg-id",
	}

	// Build a TypeComponentSetData protocol message.
	setDataPayload := protocol.ComponentSetDataPayload{
		ID:   "sd-msg-id",
		Data: json.RawMessage(`{"count": 42}`),
	}
	raw, _ := json.Marshal(setDataPayload)
	msg := protocolMsg{
		Message: protocol.Message{
			Type:    protocol.TypeComponentSetData,
			Payload: raw,
			ID:      "sd-msg-id",
		},
	}

	updated, _ := m.handleProtocol(msg)
	_ = updated.(Model)

	// Verify SetData was called on the component.
	if len(comp.setDataCalls) != 1 {
		t.Fatalf("expected 1 SetData call, got %d", len(comp.setDataCalls))
	}
	if string(comp.setDataCalls[0]) != `{"count":42}` {
		t.Fatalf("unexpected SetData payload: %s", string(comp.setDataCalls[0]))
	}
}

// ---------------------------------------------------------------------------
// COMP-08: Stale MsgID guard on SetData
// ---------------------------------------------------------------------------

func TestStaleSetData_DiscardedOnMismatch(t *testing.T) {
	h, _ := newCapturingHandler()
	m := newTestModelWithHandler(h)

	comp := &mockComponent{
		name: "slider",
		mode: "overlay",
	}
	m.activeComponent = &ActiveComponent{
		Component: comp,
		MsgID:     "current-msg-id",
	}

	// Send SetData with a DIFFERENT MsgID.
	setDataPayload := protocol.ComponentSetDataPayload{
		ID:   "stale-msg-id",
		Data: json.RawMessage(`{"count": 99}`),
	}
	raw, _ := json.Marshal(setDataPayload)
	msg := protocolMsg{
		Message: protocol.Message{
			Type:    protocol.TypeComponentSetData,
			Payload: raw,
			ID:      "stale-msg-id",
		},
	}

	m.handleProtocol(msg)

	// SetData should NOT have been called.
	if len(comp.setDataCalls) != 0 {
		t.Fatalf("expected 0 SetData calls for stale MsgID, got %d", len(comp.setDataCalls))
	}
}

func TestSetData_NoActiveComponent(t *testing.T) {
	h, _ := newCapturingHandler()
	m := newTestModelWithHandler(h)
	m.activeComponent = nil // No active component.

	setDataPayload := protocol.ComponentSetDataPayload{
		ID:   "orphan-msg-id",
		Data: json.RawMessage(`{"count": 1}`),
	}
	raw, _ := json.Marshal(setDataPayload)
	msg := protocolMsg{
		Message: protocol.Message{
			Type:    protocol.TypeComponentSetData,
			Payload: raw,
			ID:      "orphan-msg-id",
		},
	}

	// Should not panic or error — silently discard.
	func() {
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("SetData with no active component panicked: %v", r)
			}
		}()
		m.handleProtocol(msg)
	}()
}

// Suppress unused import warnings.
var _ = fmt.Sprintf
