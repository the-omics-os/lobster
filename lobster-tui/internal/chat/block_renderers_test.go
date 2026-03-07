package chat

import (
	"strings"
	"testing"

	"github.com/the-omics-os/lobster-tui/internal/theme"
)

func testStyles() theme.Styles {
	return theme.BuildStyles(theme.LobsterDark.Colors)
}

func TestRenderBlockTable(t *testing.T) {
	s := testStyles()
	b := BlockTable{
		Headers: []string{"Name", "Age"},
		Rows:    [][]string{{"Alice", "30"}, {"Bob", "25"}},
	}
	out := renderBlockTable(b, s, 80)

	// Should contain rounded border characters
	if !strings.Contains(out, "╭") && !strings.Contains(out, "┌") {
		t.Error("expected rounded border chars in table output")
	}

	// Should contain all header and cell values
	for _, val := range []string{"Name", "Age", "Alice", "30", "Bob", "25"} {
		if !strings.Contains(out, val) {
			t.Errorf("expected table output to contain %q", val)
		}
	}
}

func TestRenderBlockTable_Empty(t *testing.T) {
	s := testStyles()
	b := BlockTable{Headers: nil, Rows: nil}
	out := renderBlockTable(b, s, 80)
	if out != "" {
		t.Errorf("expected empty string for empty headers, got %q", out)
	}
}

func TestRenderBlockText_Plain(t *testing.T) {
	s := testStyles()
	b := BlockText{Text: "Hello world"}
	out := renderBlockText(b, s, 80, nil)
	if !strings.Contains(out, "Hello world") {
		t.Errorf("expected output to contain 'Hello world', got %q", out)
	}
}

func TestRenderBlockText_Glamour(t *testing.T) {
	// We can't easily create a glamour renderer in tests without more setup,
	// so we just verify the nil path works and the function signature is correct.
	s := testStyles()
	b := BlockText{Text: "**bold** text"}
	out := renderBlockText(b, s, 80, nil)
	if !strings.Contains(out, "bold") {
		t.Errorf("expected output to contain 'bold', got %q", out)
	}
}

func TestRenderBlock_Dispatch(t *testing.T) {
	s := testStyles()

	// BlockTable dispatches to table renderer
	tableBlock := BlockTable{
		Headers: []string{"Col"},
		Rows:    [][]string{{"Val"}},
	}
	tableOut := renderBlock(tableBlock, s, 80, nil)
	if !strings.Contains(tableOut, "Col") || !strings.Contains(tableOut, "Val") {
		t.Error("renderBlock(BlockTable) did not dispatch to table renderer")
	}

	// BlockText dispatches to text renderer
	textBlock := BlockText{Text: "hello"}
	textOut := renderBlock(textBlock, s, 80, nil)
	if !strings.Contains(textOut, "hello") {
		t.Error("renderBlock(BlockText) did not dispatch to text renderer")
	}
}

func TestStreamingNeverCached(t *testing.T) {
	// A message with IsStreaming=true should never populate cache.
	msg := ChatMessage{
		Role:        "assistant",
		Blocks:      []ContentBlock{BlockText{Text: "streaming content"}},
		IsStreaming: true,
	}
	s := testStyles()

	// Render the message
	_ = renderMessage(msg, s, 80, nil, false)

	// Cache should remain empty (not populated for streaming messages)
	_, _, ok := msg.cache.get(80)
	if ok {
		t.Error("streaming message should never be cached")
	}
}

func TestCrushStyleMessages(t *testing.T) {
	s := testStyles()
	msg := ChatMessage{
		Role:   "user",
		Blocks: []ContentBlock{BlockText{Text: "hello"}},
	}
	out := renderMessage(msg, s, 80, nil, false)

	// Crush-style means no full box border (no top/bottom rounded corners).
	// The rounded border top-left "╭" and bottom-left "╰" should NOT appear.
	if strings.Contains(out, "╭") || strings.Contains(out, "╰") {
		t.Error("user message should not have full box border (crush-style)")
	}
}
