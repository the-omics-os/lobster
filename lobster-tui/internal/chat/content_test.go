package chat

import (
	"strings"
	"testing"
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
	// Table block should use renderProtocolTable output
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
