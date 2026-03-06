package initwizard

import "testing"

func TestExpandSelectedPackagesExpandsPackageIDs(t *testing.T) {
	agents := expandSelectedPackages([]string{
		"lobster-research",
		"lobster-transcriptomics",
		"lobster-research",
	})

	expected := []string{
		"research_agent",
		"data_expert_agent",
		"transcriptomics_expert",
		"annotation_expert",
		"de_analysis_expert",
	}

	if len(agents) != len(expected) {
		t.Fatalf("expected %d agents, got %d: %#v", len(expected), len(agents), agents)
	}

	for i, agent := range expected {
		if agents[i] != agent {
			t.Fatalf("expected agents[%d] = %q, got %q", i, agent, agents[i])
		}
	}
}

func TestBuildOllamaModelOptionsDeduplicatesDetectedModels(t *testing.T) {
	options := buildOllamaModelOptions(ollamaStatus{
		Installed: true,
		Running:   true,
		Models:    []string{"gpt-oss:20b", "custom-local"},
	})

	counts := map[string]int{}
	for _, option := range options {
		counts[option.Value]++
	}

	if counts["gpt-oss:20b"] != 1 {
		t.Fatalf("expected gpt-oss:20b exactly once, got %d", counts["gpt-oss:20b"])
	}
	if counts["custom-local"] != 1 {
		t.Fatalf("expected custom-local exactly once, got %d", counts["custom-local"])
	}
	if counts[customOllamaModel] != 1 {
		t.Fatalf("expected custom option exactly once, got %d", counts[customOllamaModel])
	}
}

func TestGetSmartStandardizationBeneficiaries(t *testing.T) {
	beneficiaries := getSmartStandardizationBeneficiaries([]string{
		"research_agent",
		"annotation_expert",
		"proteomics_expert",
	})

	expected := []string{"Annotation Expert", "Proteomics Expert"}
	if len(beneficiaries) != len(expected) {
		t.Fatalf("expected %d beneficiaries, got %d: %#v", len(expected), len(beneficiaries), beneficiaries)
	}

	for i, name := range expected {
		if beneficiaries[i] != name {
			t.Fatalf("expected beneficiaries[%d] = %q, got %q", i, name, beneficiaries[i])
		}
	}
}
