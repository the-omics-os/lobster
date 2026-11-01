package initwizard

import (
	"strings"
	"testing"
)

func TestCloudSetupViewNamesSupportedAPIKeyAndNPMDevicePaths(t *testing.T) {
	m := NewWizardModel("", "")
	providerView := m.providerSelect.View(m.theme)
	if strings.Contains(strings.ToLower(providerView), "login via browser") {
		t.Errorf("Cloud provider choice offers unsupported browser login: %s", providerView)
	}
	model, _ := m.Update(makeKeyPress("enter"))
	m = model.(WizardModel)
	if m.result.Provider != providerOmicsOS || m.step != stepAPIKey {
		t.Fatal("Cloud choice did not enter its credential step")
	}
	view := m.viewAPIKey()
	for _, expected := range []string{"app.omics-os.com", "@omicsos/lobster", "cloud login --device"} {
		if !strings.Contains(view, expected) {
			t.Errorf("Cloud credential view missing supported guidance %q: %s", expected, view)
		}
	}
	if strings.Contains(strings.ToLower(view), "browser login") {
		t.Fatalf("Cloud credential view still offers unsupported browser login: %s", view)
	}
}

func TestCloudSetupEmptyKeyDoesNotStartOAuthOrAdvance(t *testing.T) {
	m := NewWizardModel("", "")
	model, _ := m.Update(makeKeyPress("enter"))
	m = model.(WizardModel)
	model, _ = m.Update(makeKeyPress("enter"))
	m = model.(WizardModel)
	if m.oauthWaiting || m.oauthCancel != nil || m.oauthAuthURL != "" {
		t.Fatal("Empty key started an unsupported OAuth flow")
	}
	if m.step != stepAPIKey || m.result.APIKey != "" || m.result.OAuthAuthenticated {
		t.Fatal("Empty Cloud key was accepted as completed setup")
	}
	view := m.viewAPIKey()
	if !strings.Contains(view, "API key") || !strings.Contains(view, "app.omics-os.com") {
		t.Fatalf("Empty key needs an actionable API-key message: %s", view)
	}
}

func TestCloudSetupManualKeyRemainsSupportedAndIsNotRendered(t *testing.T) {
	m := NewWizardModel("", "")
	model, _ := m.Update(makeKeyPress("enter"))
	m = model.(WizardModel)
	const key = "omk_local_setup_test"
	m.apiKeyInputs[0].SetValue(key)
	if strings.Contains(m.viewAPIKey(), key) {
		t.Fatal("Credential view exposed an API key")
	}
	model, _ = m.Update(makeKeyPress("enter"))
	m = model.(WizardModel)
	if m.result.APIKey != key || m.oauthWaiting || m.result.OAuthAuthenticated {
		t.Fatal("Manual Cloud key did not remain the selected credential")
	}
	if m.step == stepAPIKey {
		t.Fatal("Nonblank manual key did not advance the wizard")
	}
}
