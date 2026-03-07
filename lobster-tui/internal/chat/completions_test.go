package chat

import (
	"io"
	"strings"
	"testing"

	"charm.land/bubbles/v2/textarea"

	"github.com/the-omics-os/lobster-tui/internal/protocol"
)

func TestBuildSuggestionsTopLevel(t *testing.T) {
	m := &Model{}
	got := m.buildSuggestions("/wo")
	if !contains(got, "/workspace") {
		t.Fatalf("expected /workspace in %#v", got)
	}
}

func TestBuildSuggestionsSubcommandsWithTrailingSpace(t *testing.T) {
	m := &Model{}
	got := m.buildSuggestions("/workspace ")
	if len(got) == 0 {
		t.Fatalf("expected workspace subcommands, got %#v", got)
	}
	if got[0] != "/workspace info" {
		t.Fatalf("expected sorted suggestions, got %#v", got)
	}
}

func TestBuildSuggestionsDeepSubcommandsWithTrailingSpace(t *testing.T) {
	m := &Model{}
	got := m.buildSuggestions("/queue clear ")
	want := []string{"/queue clear all", "/queue clear download"}
	if len(got) != len(want) {
		t.Fatalf("expected %#v, got %#v", want, got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("expected %#v, got %#v", want, got)
		}
	}
}

func TestBuildSuggestionsDescribeFromModalities(t *testing.T) {
	m := &Model{
		modalities: []ModalityInfo{
			{Name: "rna"},
			{Name: "atac"},
		},
	}

	got := m.buildSuggestions("/describe ")
	want := []string{"/describe atac", "/describe rna"}
	if len(got) != len(want) {
		t.Fatalf("expected %#v, got %#v", want, got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("expected %#v, got %#v", want, got)
		}
	}
}

func TestBuildSuggestionsQueueListDeepSubcommands(t *testing.T) {
	m := &Model{}
	got := m.buildSuggestions("/queue list ")
	want := []string{"/queue list download", "/queue list publication"}
	if len(got) != len(want) {
		t.Fatalf("expected %#v, got %#v", want, got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("expected %#v, got %#v", want, got)
		}
	}
}

func TestBuildSuggestionsSaveForce(t *testing.T) {
	m := &Model{}
	got := m.buildSuggestions("/save ")
	want := []string{"/save --force"}
	if len(got) != len(want) {
		t.Fatalf("expected %#v, got %#v", want, got)
	}
	if got[0] != want[0] {
		t.Fatalf("expected %#v, got %#v", want, got)
	}
}

func TestBuildSuggestionsConfigProviderIncludesListAndSwitch(t *testing.T) {
	m := &Model{}
	got := m.buildSuggestions("/config provider ")
	if !contains(got, "/config provider list") {
		t.Fatalf("expected /config provider list in %#v", got)
	}
	if !contains(got, "/config provider switch") {
		t.Fatalf("expected /config provider switch in %#v", got)
	}
}

func TestParsePathCompletionContext(t *testing.T) {
	tests := []struct {
		input   string
		ok      bool
		command string
		prefix  string
	}{
		{input: "/read ", ok: true, command: "/read", prefix: ""},
		{input: "/read data/", ok: true, command: "/read", prefix: "data/"},
		{input: "/read My Data/file.csv", ok: true, command: "/read", prefix: "My Data/file.csv"},
		{input: "/open ./res", ok: true, command: "/open", prefix: "./res"},
		{input: "/open My Data/file.csv", ok: true, command: "/open", prefix: "My Data/file.csv"},
		{input: "/open My Data/file.csv ", ok: true, command: "/open", prefix: "My Data/file.csv "},
		{input: `/open "My Data/file.csv"`, ok: true, command: "/open", prefix: `"My Data/file.csv"`},
		{input: "/workspace load ", ok: true, command: "/workspace load", prefix: ""},
		{input: "/workspace load rn", ok: true, command: "/workspace load", prefix: "rn"},
		{input: "/workspace load Project A", ok: true, command: "/workspace load", prefix: "Project A"},
		{input: "/workspace   load Project A", ok: true, command: "/workspace load", prefix: "Project A"},
		{input: "/workspace load  Project A", ok: true, command: "/workspace load", prefix: " Project A"},
		{input: `/workspace load "Project A"`, ok: true, command: "/workspace load", prefix: `"Project A"`},
		{input: "/workspace load", ok: false},
		{input: "/workspace info ", ok: false},
		{input: "/workspace lo", ok: false},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.input, func(t *testing.T) {
			got, ok := parsePathCompletionContext(tc.input)
			if ok != tc.ok {
				t.Fatalf("expected ok=%v, got %v", tc.ok, ok)
			}
			if !ok {
				return
			}
			if got.Command != tc.command || got.Prefix != tc.prefix {
				t.Fatalf("expected (%q, %q), got (%q, %q)", tc.command, tc.prefix, got.Command, got.Prefix)
			}
		})
	}
}

func TestSanitizeSuggestions(t *testing.T) {
	got := sanitizeSuggestions([]string{"  /read foo  ", "", "/read foo", "   ", "/read bar"})
	want := []string{"/read foo", "/read bar"}
	if len(got) != len(want) {
		t.Fatalf("expected %#v, got %#v", want, got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("expected %#v, got %#v", want, got)
		}
	}
}

func TestMergeSuggestionsPreservesStaticThenDynamic(t *testing.T) {
	static := []string{"/workspace", "/read"}
	dynamic := []string{"/read foo", "/read bar", "/read"}
	got := mergeSuggestions(static, dynamic)
	want := []string{"/workspace", "/read", "/read foo", "/read bar"}
	if len(got) != len(want) {
		t.Fatalf("expected %#v, got %#v", want, got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("expected %#v, got %#v", want, got)
		}
	}
}

func TestMaybeRequestProtocolCompletions(t *testing.T) {
	m := &Model{
		handler: protocol.NewHandler(strings.NewReader(""), io.Discard),
	}

	cmd := m.maybeRequestProtocolCompletions("/read data")
	if cmd == nil {
		t.Fatal("expected completion request command")
	}
	if m.completionPendingID == "" {
		t.Fatal("expected pending completion id")
	}
	if got := m.completionRequestInputs[m.completionPendingID]; got != "/read data" {
		t.Fatalf("expected request input to be tracked, got %q", got)
	}

	// Same input while pending should not re-request.
	if next := m.maybeRequestProtocolCompletions("/read data"); next != nil {
		t.Fatal("expected no duplicate request for identical pending input")
	}
}

func TestMaybeRequestProtocolCompletionsUsesCache(t *testing.T) {
	m := &Model{
		handler:                 protocol.NewHandler(strings.NewReader(""), io.Discard),
		completionCache:         map[string][]string{"/read data": {"/read data.csv"}},
		completionRequestInputs: map[string]string{},
	}
	if cmd := m.maybeRequestProtocolCompletions("/read data"); cmd != nil {
		t.Fatal("expected cache hit to skip request")
	}
}

func TestInputHistoryRecallUpDown(t *testing.T) {
	m := &Model{
		input:             textarea.New(),
		inputHistory:      make([]string, 0, maxInputHistory),
		inputHistoryIndex: -1,
	}
	m.pushInputHistory("first")
	m.pushInputHistory("second")
	m.pushInputHistory("second") // duplicate should be ignored

	m.input.SetValue("draft")
	if !m.recallHistoryUp() {
		t.Fatal("expected history up to succeed")
	}
	if got := m.input.Value(); got != "second" {
		t.Fatalf("expected second, got %q", got)
	}
	if !m.recallHistoryUp() {
		t.Fatal("expected second history up to succeed")
	}
	if got := m.input.Value(); got != "first" {
		t.Fatalf("expected first, got %q", got)
	}
	if !m.recallHistoryDown() {
		t.Fatal("expected history down to succeed")
	}
	if got := m.input.Value(); got != "second" {
		t.Fatalf("expected second, got %q", got)
	}
	if !m.recallHistoryDown() {
		t.Fatal("expected history down to restore draft")
	}
	if got := m.input.Value(); got != "draft" {
		t.Fatalf("expected draft, got %q", got)
	}
	if m.inputHistoryIndex != -1 {
		t.Fatalf("expected history index reset, got %d", m.inputHistoryIndex)
	}
	if len(m.inputHistory) != 2 {
		t.Fatalf("expected deduped history size 2, got %d", len(m.inputHistory))
	}
}

func contains(items []string, target string) bool {
	for _, item := range items {
		if item == target {
			return true
		}
	}
	return false
}
