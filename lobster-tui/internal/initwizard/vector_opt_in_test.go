package initwizard

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func vectorConsentModel(t *testing.T, packageName string) WizardModel {
	t.Helper()
	t.Setenv("OPENAI_API_KEY", "")
	m := NewWizardModel("lobster-dark", filepath.Join(t.TempDir(), "result.json"))
	for i := range m.agentSelect.items {
		m.agentSelect.items[i].selected = m.agentSelect.items[i].value == packageName
	}
	if len(m.agentSelect.Selected()) != 1 {
		t.Fatalf("fixture package not in actual wizard choices: %q", packageName)
	}
	m.result.Provider = providerOmicsOS
	m.result.APIKey = "omx_synthetic_provider_key"
	m.step = stepOptionalKeys
	m.initOptionalKeys()
	return m
}

func TestVectorDefaultNoEnterDeclines(t *testing.T) {
	for _, pkg := range []string{"lobster-metadata", "lobster-visualization"} {
		t.Run(pkg, func(t *testing.T) {
			m := vectorConsentModel(t, pkg)
			if pkg == "lobster-metadata" && !strings.Contains(m.smartStdConfirm.description, "Helpful for") {
				t.Fatal("beneficiary context was lost")
			}
			if m.smartStdConfirm.value {
				t.Error("vector consent must default No, including benefiting agents")
			}
			model, _ := m.Update(makeKeyPress("enter")) // decline NCBI
			m = model.(WizardModel)
			if m.optionalSub != subStepSmartStdConfirm {
				t.Fatalf("cloud-provider optional path did not reach vector consent: %v", m.optionalSub)
			}
			model, _ = m.Update(makeKeyPress("enter")) // use declared No default
			m = model.(WizardModel)
			if !m.done || m.step != stepDone {
				t.Fatal("Enter did not finish without asking for an embedding key")
			}
			if m.result.SmartStandardizationEnabled || m.result.SmartStandardizationOpenAIKey != "" {
				t.Fatal("default decline enabled vector search or returned a key")
			}
			if m.result.NCBIKey != "" || m.result.CloudKey != "" {
				t.Fatal("decline changed unrelated optional keys")
			}
			if _, err := os.Stat(m.resultPath); !os.IsNotExist(err) {
				t.Fatal("pure model updates unexpectedly wrote the wizard result")
			}
		})
	}
}

func TestVectorExplicitNoDeclines(t *testing.T) {
	m := vectorConsentModel(t, "lobster-metadata")
	m.optionalSub = subStepSmartStdConfirm
	model, _ := m.Update(makeKeyPress("n"))
	m = model.(WizardModel)
	if !m.done || m.result.SmartStandardizationEnabled || m.result.SmartStandardizationOpenAIKey != "" {
		t.Fatal("explicit No must finish with vector search disabled")
	}
}

func TestVectorExplicitYesRequestsKeyAndOptInWorks(t *testing.T) {
	m := vectorConsentModel(t, "lobster-metadata")
	m.optionalSub = subStepSmartStdConfirm
	model, _ := m.Update(makeKeyPress("y"))
	m = model.(WizardModel)
	if m.done || m.optionalSub != subStepSmartStdKey || !m.result.SmartStandardizationEnabled {
		t.Fatal("explicit Yes did not request the required embedding key")
	}
	m.smartStdInput.SetValue("sk-synthetic-embedding-key")
	model, _ = m.Update(makeKeyPress("enter"))
	m = model.(WizardModel)
	if !m.done || m.result.SmartStandardizationOpenAIKey != "sk-synthetic-embedding-key" {
		t.Fatal("explicit opt-in with a key did not complete")
	}
}
