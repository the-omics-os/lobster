package initwizard

import (
	"path/filepath"
	"reflect"
	"testing"
)

func tickedPackages(t *testing.T, saved string) []string {
	t.Helper()
	t.Setenv("LOBSTER_INIT_PACKAGES", saved) // the name go_tui_bridge sends
	m := NewWizardModel("lobster-dark", filepath.Join(t.TempDir(), "result.json"))
	var ticked []string
	for _, item := range m.agentSelect.items {
		if item.selected {
			ticked = append(ticked, item.value)
		}
	}
	return ticked
}

func TestAgentPackagesStartAtTheDefaultsWithoutEnv(t *testing.T) {
	got := tickedPackages(t, "")
	want := []string{"lobster-research", "lobster-transcriptomics"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ticked %v, want %v", got, want)
	}
}

func TestAgentPackagesStartAtTheSavedPackages(t *testing.T) {
	got := tickedPackages(t, "lobster-research,lobster-genomics")
	want := []string{"lobster-research", "lobster-genomics"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ticked %v, want %v", got, want)
	}
}
