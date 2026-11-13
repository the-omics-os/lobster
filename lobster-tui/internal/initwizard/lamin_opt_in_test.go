package initwizard

import (
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"
)

func laminModel(t *testing.T, ask string) WizardModel {
	t.Helper()
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv(laminEnv, ask)
	m := NewWizardModel("lobster-dark", filepath.Join(t.TempDir(), "result.json"))
	m.result.Provider = providerOmicsOS
	m.result.APIKey = "omx_synthetic_provider_key"
	m.step = stepOptionalKeys
	m.initOptionalKeys()
	m.optionalSub = subStepSmartStdConfirm
	return m
}

func press(t *testing.T, m WizardModel, key string) WizardModel {
	t.Helper()
	model, _ := m.Update(makeKeyPress(key))
	return model.(WizardModel)
}

func TestLaminNotAskedWithoutEnv(t *testing.T) {
	m := press(t, laminModel(t, ""), "n") // decline Smart Standardization
	if !m.done || m.result.LaminEnabled != nil {
		t.Fatal("without LOBSTER_INIT_LAMIN the wizard must finish with no answer")
	}
	raw, _ := json.Marshal(m.result)
	if strings.Contains(string(raw), "lamin_enabled") {
		t.Fatal("an unasked question must leave lamin_enabled out of the result")
	}
}

func TestLaminDefaultNoEnterDeclines(t *testing.T) {
	m := press(t, laminModel(t, "0"), "n")
	if m.done || m.optionalSub != subStepLaminConfirm {
		t.Fatal("the LaminDB question did not follow Smart Standardization")
	}
	if !strings.Contains(m.viewOptionalKeys(), "contacts no server") {
		t.Fatal("the question text is missing")
	}
	m = press(t, m, "enter")
	if !m.done || m.result.LaminEnabled == nil || *m.result.LaminEnabled {
		t.Fatal("Enter on the default must answer No")
	}
}

func TestLaminStartsAtTheSavedAnswer(t *testing.T) {
	m := press(t, press(t, laminModel(t, "1"), "n"), "enter")
	if !m.done || m.result.LaminEnabled == nil || !*m.result.LaminEnabled {
		t.Fatal("Enter must keep the saved Yes")
	}
	raw, _ := json.Marshal(m.result)
	if !strings.Contains(string(raw), `"lamin_enabled":true`) {
		t.Fatalf("result JSON lacks the answer: %s", raw)
	}
}

func TestLaminExplicitYesAndBack(t *testing.T) {
	m := press(t, laminModel(t, "0"), "n")
	m = press(t, m, "esc")
	if m.optionalSub != subStepSmartStdConfirm || m.done {
		t.Fatal("esc must go back to Smart Standardization")
	}
	m = press(t, press(t, m, "n"), "y")
	if !m.done || !*m.result.LaminEnabled {
		t.Fatal("y must answer Yes")
	}
}
