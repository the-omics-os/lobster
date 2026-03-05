package chat

import "testing"

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

func contains(items []string, target string) bool {
	for _, item := range items {
		if item == target {
			return true
		}
	}
	return false
}
