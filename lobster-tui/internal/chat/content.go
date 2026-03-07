package chat

import (
	"fmt"
	"strings"
)

// ContentBlock is a sealed interface representing a typed chunk of chat content.
// The unexported blockType() method ensures only types in this package can
// implement the interface.
type ContentBlock interface {
	blockType() string
}

// BlockText holds plain or markdown text.
type BlockText struct {
	Text string
}

func (BlockText) blockType() string { return "text" }

// BlockCode holds a fenced code block with optional language.
type BlockCode struct {
	Language string
	Content  string
}

func (BlockCode) blockType() string { return "code" }

// BlockTable holds structured table data (headers + rows).
type BlockTable struct {
	Headers []string
	Rows    [][]string
}

func (BlockTable) blockType() string { return "table" }

// BlockAlert holds an alert message with severity level.
type BlockAlert struct {
	Level   string
	Message string
}

func (BlockAlert) blockType() string { return "alert" }

// BlockHandoff holds an agent transition record.
type BlockHandoff struct {
	From   string
	To     string
	Reason string
}

func (BlockHandoff) blockType() string { return "handoff" }

// Content returns a backward-compatible flat string representation of the
// message's block list. This allows existing rendering code (which expects a
// single string) to keep working while the internal model uses typed blocks.
func (m ChatMessage) Content() string {
	if len(m.Blocks) == 0 {
		return ""
	}
	var b strings.Builder
	for _, block := range m.Blocks {
		switch v := block.(type) {
		case BlockText:
			b.WriteString(v.Text)
		case BlockCode:
			b.WriteString(fmt.Sprintf("\n```%s\n%s\n```\n", v.Language, v.Content))
		case BlockTable:
			// Backward-compat: pipe-delimited markdown table
			b.WriteString(strings.Join(v.Headers, " | ") + "\n")
			for _, row := range v.Rows {
				b.WriteString(strings.Join(row, " | ") + "\n")
			}
		case BlockAlert:
			b.WriteString(v.Message)
		case BlockHandoff:
			b.WriteString(v.Reason)
		}
	}
	return b.String()
}

// findBlock searches a block slice for the first block of type T and returns
// a pointer to it, or nil if not found. Used by views.go to check for typed
// blocks before falling back to legacy string-based rendering.
func findBlock[T ContentBlock](blocks []ContentBlock) *T {
	for _, b := range blocks {
		if typed, ok := b.(T); ok {
			return &typed
		}
	}
	return nil
}

// appendBlock appends a ContentBlock to the last assistant message in the
// message list. If no assistant message exists, it creates one.
func (m *Model) appendBlock(block ContentBlock) {
	if len(m.messages) > 0 {
		last := &m.messages[len(m.messages)-1]
		if last.Role == "assistant" {
			last.Blocks = append(last.Blocks, block)
			return
		}
	}
	// No existing assistant message — create one.
	m.messages = append(m.messages, ChatMessage{
		Role:   "assistant",
		Blocks: []ContentBlock{block},
	})
}

// textBlocks is a convenience helper that wraps a plain string into a
// single-element ContentBlock slice for constructing ChatMessages.
func textBlocks(s string) []ContentBlock {
	return []ContentBlock{BlockText{Text: s}}
}

// flushStreamBuffer converts any accumulated streaming text in streamBuf into
// a BlockText and appends it to the last assistant message. If the buffer is
// empty, this is a no-op.
func (m *Model) flushStreamBuffer() {
	if m.streamBuf.Len() == 0 {
		return
	}
	text := m.streamBuf.String()
	m.streamBuf.Reset()
	m.appendBlock(BlockText{Text: text})
}

// collectLastAssistantMessage returns a single-element slice containing the
// last assistant message, or nil if none exists. Used by the TypeDone handler
// to feed inlinePrintMessagesCmd after flushStreamBuffer has put text into
// m.messages in-place (which otherwise never gets tea.Println'd).
func (m *Model) collectLastAssistantMessage() []ChatMessage {
	for i := len(m.messages) - 1; i >= 0; i-- {
		if m.messages[i].Role == "assistant" {
			return []ChatMessage{m.messages[i]}
		}
	}
	return nil
}
