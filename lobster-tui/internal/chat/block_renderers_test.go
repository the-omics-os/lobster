package chat

import (
	"regexp"
	"strings"
	"testing"

	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/glamour"

	"github.com/the-omics-os/lobster-tui/internal/theme"
)

var ansiRegexp = regexp.MustCompile(`\x1b\[[0-9;]*m`)

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

func TestRenderBlockTable_ContentFitsWithoutStretching(t *testing.T) {
	s := testStyles()
	b := BlockTable{
		Headers: []string{"#", "Status", "Name"},
		Rows: [][]string{
			{"1", "Loaded", "rna_demo_reference"},
			{"2", "Available", "atac_demo_layout_review"},
		},
	}

	out := renderBlockTable(b, s, 120)

	if w := renderedMaxLineWidth(out); w >= 120 {
		t.Fatalf("expected compact table width below viewport, got %d", w)
	}
}

func TestRenderBlockTable_WrapsWhenViewportIsNarrow(t *testing.T) {
	s := testStyles()
	b := BlockTable{
		Headers: []string{"Command", "Description"},
		Rows: [][]string{
			{"/analysis-dash", "Rich analysis dashboard degraded to /plots plus /metadata in Go mode"},
		},
	}

	out := renderBlockTable(b, s, 48)

	if w := renderedMaxLineWidth(out); w > 48 {
		t.Fatalf("expected table width to stay within viewport, got %d", w)
	}
}

func TestRenderBlockTable_UsesColumnLayoutMetadata(t *testing.T) {
	s := testStyles()
	b := BlockTable{
		Headers: []string{"#", "Dataset"},
		Columns: []BlockTableColumn{
			{Name: "#", Width: 4, Justify: "right", NoWrap: true},
			{Name: "Dataset", MaxWidth: 18, Overflow: "ellipsis"},
		},
		Rows: [][]string{
			{"1", "geo_gse247686_transcriptomics_single_cell_autosave"},
		},
	}

	out := renderBlockTable(b, s, 40)
	clean := ansiRegexp.ReplaceAllString(out, "")

	if !strings.Contains(clean, "…") || strings.Contains(clean, "single_cell_autosave") {
		t.Fatalf("expected dataset cell to be capped by metadata, got %q", clean)
	}
	if w := renderedMaxLineWidth(out); w > 40 {
		t.Fatalf("expected metadata-aware table to stay within viewport, got %d", w)
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

func TestRenderBlockText_PlainSkipsMarkdownRenderer(t *testing.T) {
	s := testStyles()
	renderer, err := glamour.NewTermRenderer(
		glamour.WithAutoStyle(),
		glamour.WithWordWrap(80),
	)
	if err != nil {
		t.Fatalf("failed to build glamour renderer: %v", err)
	}

	b := BlockText{Text: "- research_agent", Markdown: false}
	out := renderBlockText(b, s, 80, renderer)
	if !strings.Contains(out, "research_agent") {
		t.Fatalf("expected underscores to remain literal, got %q", out)
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

func TestRenderBlockCode_Python(t *testing.T) {
	s := testStyles()
	b := BlockCode{Language: "python", Content: "def hello():\n    print('hi')"}
	out := renderBlockCode(b, s, 60)

	if !strings.Contains(out, "python") {
		t.Error("expected output to contain language label 'python'")
	}
	if !strings.Contains(out, "hello") {
		t.Error("expected output to contain code text 'hello'")
	}
	// Syntax highlighting should produce ANSI escape codes (terminal256 formatter)
	if !strings.Contains(out, "\x1b[") {
		t.Error("expected ANSI escape codes from syntax highlighting")
	}
	// Should NOT contain raw triple backticks (that's the stub format)
	if strings.Contains(out, "```") {
		t.Error("should not contain raw backtick fencing (stub format)")
	}
}

func TestRenderBlockCode_UnknownLang(t *testing.T) {
	s := testStyles()
	b := BlockCode{Language: "zzzunknown", Content: "some code"}
	out := renderBlockCode(b, s, 60)

	if !strings.Contains(out, "some code") {
		t.Error("expected fallback to render code content for unknown language")
	}
}

func TestRenderBlockCode_Empty(t *testing.T) {
	s := testStyles()
	b := BlockCode{Language: "go", Content: ""}
	out := renderBlockCode(b, s, 60)

	// Should at minimum contain the label or be minimal
	if strings.Contains(out, "panic") {
		t.Error("empty content should not cause panic")
	}
}

func TestRenderBlockCode_NoLanguage(t *testing.T) {
	s := testStyles()
	b := BlockCode{Language: "", Content: "plain text code"}
	out := renderBlockCode(b, s, 60)

	if !strings.Contains(out, "plain text code") {
		t.Error("expected output to contain code content")
	}
}

func TestRenderBlockCode_WidthConstrained(t *testing.T) {
	s := testStyles()
	b := BlockCode{Language: "python", Content: "def hello():\n    print('hi')"}
	// Should not panic at narrow width
	out := renderBlockCode(b, s, 30)
	if out == "" {
		t.Error("expected non-empty output even at narrow width")
	}
}

func TestRenderInlineAlert_WrappedWarningDoesNotLeakCSIFragments(t *testing.T) {
	s := testStyles()
	out := renderInlineAlert(
		"warning",
		"/analysis-dash renders Rich dashboard panels and is not available in Go TUI. Showing /metadata + /plots fallback.",
		s,
		60,
	)

	clean := ansiRegexp.ReplaceAllString(out, "")
	if strings.Contains(clean, "[m[") || strings.Contains(clean, "[1;") {
		t.Fatalf("expected wrapped alert without leaked CSI fragments, got %q", clean)
	}
	if !strings.Contains(clean, "WARNING") {
		t.Fatalf("expected warning chip in output, got %q", clean)
	}
	if !strings.Contains(clean, "/metadata + /plots") || !strings.Contains(clean, "fallback.") {
		t.Fatalf("expected wrapped alert body in output, got %q", clean)
	}
}

func TestRenderBlockAlert_Error(t *testing.T) {
	s := testStyles()
	b := BlockAlert{Level: "error", Message: "Something failed"}
	out := renderBlockAlert(b, s, 80)

	if !strings.Contains(out, "ERROR") {
		t.Error("expected output to contain 'ERROR' chip")
	}
	if !strings.Contains(out, "Something failed") {
		t.Error("expected output to contain message text")
	}
}

func TestRenderBlockAlert_Warning(t *testing.T) {
	s := testStyles()
	b := BlockAlert{Level: "warning", Message: "Be careful"}
	out := renderBlockAlert(b, s, 80)

	if !strings.Contains(out, "WARNING") {
		t.Error("expected output to contain 'WARNING' chip")
	}
}

func TestRenderBlockAlert_Success(t *testing.T) {
	s := testStyles()
	b := BlockAlert{Level: "success", Message: "All good"}
	out := renderBlockAlert(b, s, 80)

	if !strings.Contains(out, "SUCCESS") {
		t.Error("expected output to contain 'SUCCESS' chip")
	}
}

func TestRenderBlockAlert_Info(t *testing.T) {
	s := testStyles()
	// Default/info level
	b := BlockAlert{Level: "", Message: "FYI message"}
	out := renderBlockAlert(b, s, 80)

	if !strings.Contains(out, "INFO") {
		t.Error("expected output to contain 'INFO' chip for default level")
	}
}

func TestRenderBlockHandoff(t *testing.T) {
	s := testStyles()
	b := BlockHandoff{From: "supervisor", To: "transcriptomics_expert", Reason: "analyzing RNA-seq"}
	out := renderBlockHandoff(b, s, 80)

	if !strings.Contains(out, "transcriptomics_expert") {
		t.Error("expected output to contain To agent name")
	}
	if !strings.Contains(out, "-->") {
		t.Error("expected output to contain handoff arrow prefix")
	}
	if !strings.Contains(out, "analyzing RNA-seq") {
		t.Error("expected output to contain reason")
	}
}

func TestRenderBlockHandoff_EmptyReason(t *testing.T) {
	s := testStyles()
	b := BlockHandoff{From: "supervisor", To: "data_expert", Reason: ""}
	out := renderBlockHandoff(b, s, 80)

	if !strings.Contains(out, "data_expert") {
		t.Error("expected output to contain To agent name")
	}
	if !strings.Contains(out, "-->") {
		t.Error("expected output to contain handoff arrow prefix")
	}
}

func renderedMaxLineWidth(s string) int {
	maxWidth := 0
	for _, line := range strings.Split(s, "\n") {
		clean := ansiRegexp.ReplaceAllString(line, "")
		if w := lipgloss.Width(clean); w > maxWidth {
			maxWidth = w
		}
	}
	return maxWidth
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
