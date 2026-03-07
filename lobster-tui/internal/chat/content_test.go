package chat

import (
	"strings"
	"testing"

	"github.com/the-omics-os/lobster-tui/internal/protocol"
)

func TestContentBlock_BlockTypes(t *testing.T) {
	// Each of the 5 block types must satisfy the ContentBlock interface.
	blocks := []ContentBlock{
		BlockText{Text: "hello"},
		BlockCode{Language: "python", Content: "print('hi')"},
		BlockTable{Headers: []string{"A"}, Rows: [][]string{{"1"}}},
		BlockAlert{Level: "error", Message: "oops"},
		BlockHandoff{From: "supervisor", To: "data_expert", Reason: "download task"},
	}
	for i, b := range blocks {
		bt := b.blockType()
		if bt == "" {
			t.Errorf("block %d returned empty blockType()", i)
		}
	}
}

func TestContentHelper_TextOnly(t *testing.T) {
	msg := ChatMessage{
		Role: "assistant",
		Blocks: []ContentBlock{
			BlockText{Text: "hello"},
			BlockText{Text: " world"},
		},
	}
	got := msg.Content()
	if got != "hello world" {
		t.Errorf("Content() = %q, want %q", got, "hello world")
	}
}

func TestContentHelper_Mixed(t *testing.T) {
	msg := ChatMessage{
		Role: "assistant",
		Blocks: []ContentBlock{
			BlockText{Text: "intro\n"},
			BlockCode{Language: "python", Content: "x = 1"},
			BlockTable{Headers: []string{"Name", "Value"}, Rows: [][]string{{"a", "1"}}},
			BlockAlert{Level: "warning", Message: "watch out"},
			BlockHandoff{From: "sup", To: "data", Reason: "handoff reason"},
		},
	}
	got := msg.Content()

	// Text block
	if !strings.Contains(got, "intro\n") {
		t.Error("Content() missing text block")
	}
	// Code block should be fenced
	if !strings.Contains(got, "```python") {
		t.Error("Content() missing code fence start")
	}
	if !strings.Contains(got, "x = 1") {
		t.Error("Content() missing code content")
	}
	// Table block should produce pipe-delimited markdown fallback
	if !strings.Contains(got, "Name") {
		t.Error("Content() missing table header")
	}
	// Alert
	if !strings.Contains(got, "watch out") {
		t.Error("Content() missing alert message")
	}
	// Handoff
	if !strings.Contains(got, "handoff reason") {
		t.Error("Content() missing handoff reason")
	}
}

func TestContentHelper_Empty(t *testing.T) {
	msg := ChatMessage{
		Role:   "assistant",
		Blocks: nil,
	}
	if got := msg.Content(); got != "" {
		t.Errorf("Content() = %q, want empty", got)
	}
}

func TestFlushStreamBuffer(t *testing.T) {
	m := Model{
		messages:  make([]ChatMessage, 0, 8),
		streamBuf: &strings.Builder{},
	}

	// Flush on empty buffer should do nothing.
	m.flushStreamBuffer()
	if len(m.messages) != 0 {
		t.Fatalf("flush on empty buf created %d messages, want 0", len(m.messages))
	}

	// Add an assistant message and write to streamBuf.
	m.messages = append(m.messages, ChatMessage{Role: "assistant", Blocks: []ContentBlock{}})
	m.streamBuf.WriteString("streamed text")
	m.flushStreamBuffer()

	if m.streamBuf.Len() != 0 {
		t.Error("streamBuf not reset after flush")
	}
	last := m.messages[len(m.messages)-1]
	if len(last.Blocks) != 1 {
		t.Fatalf("expected 1 block after flush, got %d", len(last.Blocks))
	}
	bt, ok := last.Blocks[0].(BlockText)
	if !ok {
		t.Fatalf("expected BlockText, got %T", last.Blocks[0])
	}
	if bt.Text != "streamed text" {
		t.Errorf("BlockText.Text = %q, want %q", bt.Text, "streamed text")
	}
}

// --- Integration tests: protocol handlers create typed blocks ---

func TestHandleProtocol_TableCreatesBlockTable(t *testing.T) {
	m := newTestModel()

	updated, _ := m.handleProtocol(testProtocolMsg(t, protocol.TypeTable, protocol.TablePayload{
		Headers: []string{"Gene", "LogFC"},
		Rows:    [][]string{{"TP53", "2.1"}, {"BRCA1", "-1.5"}},
	}))
	m = updated.(Model)

	if len(m.messages) == 0 {
		t.Fatal("expected at least one message")
	}
	last := m.messages[len(m.messages)-1]
	found := false
	for _, b := range last.Blocks {
		if tb, ok := b.(BlockTable); ok {
			if len(tb.Headers) != 2 || tb.Headers[0] != "Gene" {
				t.Errorf("unexpected headers: %v", tb.Headers)
			}
			if len(tb.Rows) != 2 || tb.Rows[0][0] != "TP53" {
				t.Errorf("unexpected rows: %v", tb.Rows)
			}
			found = true
		}
	}
	if !found {
		t.Error("no BlockTable found in message blocks")
	}
}

func TestHandleProtocol_CodeCreatesBlockCode(t *testing.T) {
	m := newTestModel()

	updated, _ := m.handleProtocol(testProtocolMsg(t, protocol.TypeCode, protocol.CodePayload{
		Language: "python",
		Content:  "import scanpy as sc",
	}))
	m = updated.(Model)

	if len(m.messages) == 0 {
		t.Fatal("expected at least one message")
	}
	last := m.messages[len(m.messages)-1]
	found := false
	for _, b := range last.Blocks {
		if cb, ok := b.(BlockCode); ok {
			if cb.Language != "python" {
				t.Errorf("expected language 'python', got %q", cb.Language)
			}
			if cb.Content != "import scanpy as sc" {
				t.Errorf("unexpected code content: %q", cb.Content)
			}
			found = true
		}
	}
	if !found {
		t.Error("no BlockCode found in message blocks")
	}
}

func TestStreamThenTable_BothBlocksPreserved(t *testing.T) {
	m := newTestModel()

	// Simulate text streaming.
	updated, _ := m.handleProtocol(testProtocolMsg(t, protocol.TypeText, protocol.TextPayload{
		Content: "Here are the results:\n",
	}))
	m = updated.(Model)

	// Then a table arrives.
	updated, _ = m.handleProtocol(testProtocolMsg(t, protocol.TypeTable, protocol.TablePayload{
		Headers: []string{"Sample", "Count"},
		Rows:    [][]string{{"A", "100"}},
	}))
	m = updated.(Model)

	if len(m.messages) == 0 {
		t.Fatal("expected at least one message")
	}
	last := m.messages[len(m.messages)-1]
	if len(last.Blocks) < 2 {
		t.Fatalf("expected at least 2 blocks (text + table), got %d", len(last.Blocks))
	}

	// First block should be the flushed text.
	bt, ok := last.Blocks[0].(BlockText)
	if !ok {
		t.Fatalf("block 0: want BlockText, got %T", last.Blocks[0])
	}
	if !strings.Contains(bt.Text, "results") {
		t.Errorf("block 0 text missing expected content: %q", bt.Text)
	}

	// Second block should be the table.
	if _, ok := last.Blocks[1].(BlockTable); !ok {
		t.Fatalf("block 1: want BlockTable, got %T", last.Blocks[1])
	}
}

func TestFlushBeforeTypedBlock(t *testing.T) {
	m := Model{
		messages:  make([]ChatMessage, 0, 8),
		streamBuf: &strings.Builder{},
	}

	// Start an assistant message.
	m.messages = append(m.messages, ChatMessage{Role: "assistant", Blocks: []ContentBlock{}})

	// Simulate text streaming.
	m.streamBuf.WriteString("before table")

	// Flush stream buffer then append a typed block (simulating TypeTable arrival).
	m.flushStreamBuffer()
	m.appendBlock(BlockTable{
		Headers: []string{"Col"},
		Rows:    [][]string{{"val"}},
	})

	last := m.messages[len(m.messages)-1]
	if len(last.Blocks) != 2 {
		t.Fatalf("expected 2 blocks, got %d", len(last.Blocks))
	}
	if _, ok := last.Blocks[0].(BlockText); !ok {
		t.Errorf("block 0: want BlockText, got %T", last.Blocks[0])
	}
	if _, ok := last.Blocks[1].(BlockTable); !ok {
		t.Errorf("block 1: want BlockTable, got %T", last.Blocks[1])
	}
}
