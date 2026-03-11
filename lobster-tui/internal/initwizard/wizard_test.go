package initwizard

import (
	"testing"

	tea "charm.land/bubbletea/v2"
)

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

func TestBuildOllamaSelectItemsDeduplicatesDetectedModels(t *testing.T) {
	items := buildOllamaSelectItems(ollamaStatus{
		Installed: true,
		Running:   true,
		Models:    []string{"gpt-oss:20b", "custom-local"},
	})

	counts := map[string]int{}
	for _, item := range items {
		counts[item.value]++
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

// makeKeyPress creates a tea.KeyPressMsg for testing.
func makeKeyPress(key string) tea.KeyPressMsg {
	switch key {
	case "enter":
		return tea.KeyPressMsg{Code: tea.KeyEnter}
	case "esc":
		return tea.KeyPressMsg{Code: tea.KeyEscape}
	case "space":
		return tea.KeyPressMsg{Code: tea.KeySpace}
	case "up":
		return tea.KeyPressMsg{Code: tea.KeyUp}
	case "down":
		return tea.KeyPressMsg{Code: tea.KeyDown}
	case "tab":
		return tea.KeyPressMsg{Code: tea.KeyTab}
	default:
		// Single character.
		if len(key) == 1 {
			return tea.KeyPressMsg{Code: rune(key[0]), Text: key}
		}
		return tea.KeyPressMsg{}
	}
}

func TestMultiSelectToggleAndConfirm(t *testing.T) {
	ms := multiSelectModel{
		title: "Test",
		items: []multiSelectItem{
			{label: "A", value: "a", selected: false},
			{label: "B", value: "b", selected: false},
			{label: "C", value: "c", selected: true},
		},
	}

	// Enter with no additional selections (only C is selected) should work.
	ms, done, cancelled := ms.Update(makeKeyPress("enter"))
	if !done || cancelled {
		t.Fatal("expected done=true, cancelled=false for pre-selected item")
	}
	if selected := ms.Selected(); len(selected) != 1 || selected[0] != "c" {
		t.Fatalf("expected [c], got %v", selected)
	}

	// Reset and select items manually.
	ms.items[0].selected = false
	ms.items[1].selected = false
	ms.items[2].selected = false
	ms.cursor = 0

	// Toggle first item with space.
	ms, _, _ = ms.Update(makeKeyPress("space"))
	if !ms.items[0].selected {
		t.Fatal("expected item 0 to be selected after space")
	}

	// Move down and toggle second item.
	ms, _, _ = ms.Update(makeKeyPress("down"))
	ms, _, _ = ms.Update(makeKeyPress("space"))
	if !ms.items[1].selected {
		t.Fatal("expected item 1 to be selected after space")
	}

	// Confirm.
	ms, done, cancelled = ms.Update(makeKeyPress("enter"))
	if !done || cancelled {
		t.Fatal("expected done after enter with selections")
	}
	selected := ms.Selected()
	if len(selected) != 2 || selected[0] != "a" || selected[1] != "b" {
		t.Fatalf("expected [a, b], got %v", selected)
	}
}

func TestMultiSelectRejectsEmptySelection(t *testing.T) {
	ms := multiSelectModel{
		title: "Test",
		items: []multiSelectItem{
			{label: "A", value: "a", selected: false},
		},
	}
	ms, done, _ := ms.Update(makeKeyPress("enter"))
	if done {
		t.Fatal("should not advance with no selection")
	}
}

func TestMultiSelectCancel(t *testing.T) {
	ms := multiSelectModel{
		title: "Test",
		items: []multiSelectItem{
			{label: "A", value: "a", selected: true},
		},
	}
	_, _, cancelled := ms.Update(makeKeyPress("esc"))
	if !cancelled {
		t.Fatal("expected cancel on esc")
	}
}

func TestSingleSelectNavigateAndConfirm(t *testing.T) {
	ss := singleSelectModel{
		title: "Test",
		items: []singleSelectItem{
			{label: "A", value: "a"},
			{label: "B", value: "b"},
		},
		cursor: 0,
	}

	// Move down and select.
	ss, _, _, _ = ss.Update(makeKeyPress("down"))
	if ss.cursor != 1 {
		t.Fatalf("expected cursor=1, got %d", ss.cursor)
	}

	ss, val, done, _ := ss.Update(makeKeyPress("enter"))
	if !done || val != "b" {
		t.Fatalf("expected done=true, val=b, got done=%v, val=%s", done, val)
	}
}

func TestSingleSelectCancel(t *testing.T) {
	ss := singleSelectModel{
		title: "Test",
		items: []singleSelectItem{
			{label: "A", value: "a"},
		},
	}
	_, _, _, cancelled := ss.Update(makeKeyPress("esc"))
	if !cancelled {
		t.Fatal("expected cancel on esc")
	}
}

func TestConfirmModelYesNo(t *testing.T) {
	cm := confirmModel{title: "Test?", value: false}

	cm, done, _ := cm.Update(makeKeyPress("y"))
	if !done || !cm.value {
		t.Fatal("expected done=true, value=true after 'y'")
	}

	cm = confirmModel{title: "Test?", value: true}
	cm, done, _ = cm.Update(makeKeyPress("n"))
	if !done || cm.value {
		t.Fatal("expected done=true, value=false after 'n'")
	}
}

func TestConfirmModelCancel(t *testing.T) {
	cm := confirmModel{title: "Test?"}
	_, _, cancelled := cm.Update(makeKeyPress("esc"))
	if !cancelled {
		t.Fatal("expected cancel on esc")
	}
}

func TestWizardStartsAtProvider(t *testing.T) {
	m := NewWizardModel("", "")
	if m.step != stepProvider {
		t.Fatalf("expected stepProvider as initial step, got %d", m.step)
	}
}

func TestWizardModelCancellation(t *testing.T) {
	m := NewWizardModel("", "")

	// Esc on step 1 (provider) should cancel entirely.
	result, _ := m.Update(makeKeyPress("esc"))
	m = result.(WizardModel)
	if m.step != stepDone {
		t.Fatalf("expected stepDone after cancel, got %d", m.step)
	}
	r, err := m.Result()
	if !r.Cancelled {
		t.Fatal("expected Cancelled=true")
	}
	if err == nil {
		t.Fatal("expected non-nil error for cancellation")
	}
}

func TestWizardModelProviderSelectionInitializesAPIStep(t *testing.T) {
	m := NewWizardModel("", "")

	// Select Anthropic (index 1 — Omics-OS Cloud is at 0).
	result, _ := m.Update(makeKeyPress("down"))
	m = result.(WizardModel)
	result, _ = m.Update(makeKeyPress("enter"))
	m = result.(WizardModel)

	if m.step != stepAPIKey {
		t.Fatalf("expected stepAPIKey, got %d", m.step)
	}
	if m.result.Provider != providerAnthropic {
		t.Fatalf("expected provider=anthropic, got %s", m.result.Provider)
	}
	if len(m.apiKeyInputs) != 1 {
		t.Fatalf("expected 1 API key input for anthropic, got %d", len(m.apiKeyInputs))
	}
}

func TestWizardModelBedrockHasTwoKeyInputs(t *testing.T) {
	m := NewWizardModel("", "")

	// Move to Bedrock (index 2 — Omics-OS Cloud is at 0, Anthropic at 1).
	result, _ := m.Update(makeKeyPress("down"))
	m = result.(WizardModel)
	result, _ = m.Update(makeKeyPress("down"))
	m = result.(WizardModel)
	result, _ = m.Update(makeKeyPress("enter"))
	m = result.(WizardModel)

	if m.result.Provider != providerBedrock {
		t.Fatalf("expected bedrock, got %s", m.result.Provider)
	}
	if len(m.apiKeyInputs) != 2 {
		t.Fatalf("expected 2 API key inputs for bedrock, got %d", len(m.apiKeyInputs))
	}
}

func TestWizardModelOllamaUsesSelect(t *testing.T) {
	m := NewWizardModel("", "")

	// Move to Ollama (index 3 — Omics-OS Cloud at 0, Anthropic at 1, Bedrock at 2).
	for i := 0; i < 3; i++ {
		result, _ := m.Update(makeKeyPress("down"))
		m = result.(WizardModel)
	}
	result, _ := m.Update(makeKeyPress("enter"))
	m = result.(WizardModel)

	if m.result.Provider != providerOllama {
		t.Fatalf("expected ollama, got %s", m.result.Provider)
	}
	// Ollama step uses ollamaSelect, not apiKeyInputs.
	if len(m.ollamaSelect.items) == 0 {
		t.Fatal("expected ollama select items to be populated")
	}
}

func TestWizardModelAnthropicShowsProfileStep(t *testing.T) {
	m := NewWizardModel("", "")

	// Select Anthropic (index 1).
	result, _ := m.Update(makeKeyPress("down"))
	m = result.(WizardModel)
	result, _ = m.Update(makeKeyPress("enter"))
	m = result.(WizardModel)

	// Type something in API key and submit.
	for _, ch := range "sk-test" {
		result, _ = m.Update(tea.KeyPressMsg{Code: ch, Text: string(ch)})
		m = result.(WizardModel)
	}
	result, _ = m.Update(makeKeyPress("enter"))
	m = result.(WizardModel)

	if m.step != stepModelSelect {
		t.Fatalf("expected stepModelSelect for anthropic, got %d", m.step)
	}

	// Accept default model selection.
	result, _ = m.Update(makeKeyPress("enter"))
	m = result.(WizardModel)

	if m.step != stepProfile {
		t.Fatalf("expected stepProfile after model select for anthropic, got %d", m.step)
	}
}

func TestWizardModelNonAnthropicSkipsProfile(t *testing.T) {
	m := NewWizardModel("", "")

	// Select Gemini (index 4 — Omics-OS Cloud at 0).
	for i := 0; i < 4; i++ {
		result, _ := m.Update(makeKeyPress("down"))
		m = result.(WizardModel)
	}
	result, _ := m.Update(makeKeyPress("enter"))
	m = result.(WizardModel)

	if m.result.Provider != providerGemini {
		t.Fatalf("expected gemini, got %s", m.result.Provider)
	}

	// Type API key and submit.
	for _, ch := range "AIzaSyTest" {
		result, _ = m.Update(tea.KeyPressMsg{Code: ch, Text: string(ch)})
		m = result.(WizardModel)
	}
	result, _ = m.Update(makeKeyPress("enter"))
	m = result.(WizardModel)

	// Should go to model select (gemini has a model catalog).
	if m.step != stepModelSelect {
		t.Fatalf("expected stepModelSelect for gemini, got %d", m.step)
	}

	// Accept default model and should skip profile → agent packages.
	result, _ = m.Update(makeKeyPress("enter"))
	m = result.(WizardModel)

	if m.step != stepAgentPackages {
		t.Fatalf("expected stepAgentPackages after model select for gemini (non-profile), got %d", m.step)
	}
}

func TestOAuthStartedWithEmptyKey(t *testing.T) {
	m := NewWizardModel("", "")

	// Select Omics-OS Cloud (index 0).
	result, _ := m.Update(makeKeyPress("enter"))
	m = result.(WizardModel)

	if m.step != stepAPIKey {
		t.Fatalf("expected stepAPIKey, got %d", m.step)
	}
	if m.result.Provider != providerOmicsOS {
		t.Fatalf("expected omics-os, got %s", m.result.Provider)
	}

	// Press Enter with empty input → should start OAuth (oauthWaiting = true).
	result, cmd := m.Update(makeKeyPress("enter"))
	m = result.(WizardModel)

	if !m.oauthWaiting {
		t.Fatal("expected oauthWaiting=true after empty Enter on omics-os")
	}
	if cmd == nil {
		t.Fatal("expected non-nil tea.Cmd for OAuth flow")
	}
}

func TestOAuthSucceededAdvancesStep(t *testing.T) {
	m := NewWizardModel("", "")

	// Select omics-os -> stepAPIKey.
	result, _ := m.Update(makeKeyPress("enter"))
	m = result.(WizardModel)

	// Simulate OAuth waiting state.
	m.oauthWaiting = true
	m.oauthCancel = make(chan struct{})

	// Send oauthSucceededMsg (no email/tier — those come from Python gateway validation).
	result, _ = m.Update(oauthSucceededMsg{
		AccessToken: "test-token",
		Endpoint:    "https://app.omics-os.com",
	})
	m = result.(WizardModel)

	if m.oauthWaiting {
		t.Fatal("expected oauthWaiting=false after success")
	}
	if !m.result.OAuthAuthenticated {
		t.Fatal("expected OAuthAuthenticated=true")
	}
	// Omics-OS has no model catalog and no profile, so should go to agent packages.
	if m.step != stepAgentPackages {
		t.Fatalf("expected stepAgentPackages after OAuth success, got %d", m.step)
	}
}

func TestOAuthFailedRefocusesInput(t *testing.T) {
	m := NewWizardModel("", "")

	// Select omics-os -> stepAPIKey.
	result, _ := m.Update(makeKeyPress("enter"))
	m = result.(WizardModel)

	// Simulate OAuth waiting state.
	m.oauthWaiting = true
	m.oauthCancel = make(chan struct{})

	// Send oauthFailedMsg.
	result, _ = m.Update(oauthFailedMsg{Reason: "timeout"})
	m = result.(WizardModel)

	if m.oauthWaiting {
		t.Fatal("expected oauthWaiting=false after failure")
	}
	if m.step != stepAPIKey {
		t.Fatalf("expected stepAPIKey after OAuth failure, got %d", m.step)
	}
}

func TestEscDuringOAuthCancels(t *testing.T) {
	m := NewWizardModel("", "")

	// Select omics-os -> stepAPIKey.
	result, _ := m.Update(makeKeyPress("enter"))
	m = result.(WizardModel)

	// Simulate OAuth waiting state.
	m.oauthWaiting = true
	m.oauthCancel = make(chan struct{})

	// Press Esc.
	result, _ = m.Update(makeKeyPress("esc"))
	m = result.(WizardModel)

	if m.oauthWaiting {
		t.Fatal("expected oauthWaiting=false after esc")
	}
	if m.step != stepProvider {
		t.Fatalf("expected stepProvider after esc during OAuth, got %d", m.step)
	}
}

func TestCloudKeySkippedForOmicsOS(t *testing.T) {
	m := NewWizardModel("", "")

	// Navigate to optional keys with omics-os provider.
	m.result.Provider = providerOmicsOS
	m.initOptionalKeys()
	m.step = stepOptionalKeys

	// We're at NCBI confirm. Say no (n = done immediately).
	result, _ := m.Update(makeKeyPress("n"))
	m = result.(WizardModel)

	// Should skip cloud and go to smart std.
	if m.optionalSub != subStepSmartStdConfirm {
		t.Fatalf("expected subStepSmartStdConfirm for omics-os, got %d", m.optionalSub)
	}
}

func TestNonOmicsEmptyKeyDoesNotTriggerOAuth(t *testing.T) {
	m := NewWizardModel("", "")

	// Select Anthropic (index 1).
	result, _ := m.Update(makeKeyPress("down"))
	m = result.(WizardModel)
	result, _ = m.Update(makeKeyPress("enter"))
	m = result.(WizardModel)

	if m.result.Provider != providerAnthropic {
		t.Fatalf("expected anthropic, got %s", m.result.Provider)
	}

	// Press Enter with empty key — should NOT trigger OAuth.
	result, _ = m.Update(makeKeyPress("enter"))
	m = result.(WizardModel)

	if m.oauthWaiting {
		t.Fatal("non-omics-os provider should NOT trigger OAuth on empty key")
	}
}

func TestCloudKeySkippedForOmicsOSAfterNCBIYes(t *testing.T) {
	m := NewWizardModel("", "")

	// Navigate to optional keys with omics-os provider.
	m.result.Provider = providerOmicsOS
	m.initOptionalKeys()
	m.step = stepOptionalKeys

	// NCBI confirm: say yes.
	result, _ := m.Update(makeKeyPress("y"))
	m = result.(WizardModel)

	if m.optionalSub != subStepNCBIKey {
		t.Fatalf("expected subStepNCBIKey after 'y', got %d", m.optionalSub)
	}

	// Type a key and submit.
	for _, ch := range "test-ncbi-key" {
		result, _ = m.Update(tea.KeyPressMsg{Code: ch, Text: string(ch)})
		m = result.(WizardModel)
	}
	result, _ = m.Update(makeKeyPress("enter"))
	m = result.(WizardModel)

	// Should skip cloud and go to smart std for omics-os.
	if m.optionalSub != subStepSmartStdConfirm {
		t.Fatalf("expected subStepSmartStdConfirm after NCBI key for omics-os, got %d", m.optionalSub)
	}
}

func TestOAuthFieldsResetOnProviderChange(t *testing.T) {
	m := NewWizardModel("", "")

	// Select omics-os (index 0) to get to API key step.
	result, _ := m.Update(makeKeyPress("enter"))
	m = result.(WizardModel)

	// Simulate having authenticated via OAuth.
	m.result.OAuthAuthenticated = true
	m.oauthLastError = "timeout"

	// Go back to provider step.
	result, _ = m.Update(makeKeyPress("esc"))
	m = result.(WizardModel)

	if m.step != stepProvider {
		t.Fatalf("expected stepProvider, got %d", m.step)
	}

	// Select a different provider (Anthropic, index 1).
	result, _ = m.Update(makeKeyPress("down"))
	m = result.(WizardModel)
	result, _ = m.Update(makeKeyPress("enter"))
	m = result.(WizardModel)

	// OAuth fields should be reset on provider change.
	if m.result.OAuthAuthenticated {
		t.Fatal("expected OAuthAuthenticated reset to false on provider change")
	}
	if m.oauthLastError != "" {
		t.Fatal("expected oauthLastError reset to empty on provider change")
	}
}

func TestOAuthSurvivesBackNavFromOptionalKeys(t *testing.T) {
	m := NewWizardModel("", "")

	// Set up as omics-os with OAuth completed.
	m.result.Provider = providerOmicsOS
	m.result.OAuthAuthenticated = true
	m.initOptionalKeys()
	m.step = stepOptionalKeys

	// Back out from NCBI confirm to agent packages step.
	result, _ := m.Update(makeKeyPress("esc"))
	m = result.(WizardModel)

	if m.step != stepAgentPackages {
		t.Fatalf("expected stepAgentPackages after back-nav from optional keys, got %d", m.step)
	}

	// OAuthAuthenticated should survive back-navigation (not reset).
	if !m.result.OAuthAuthenticated {
		t.Fatal("OAuthAuthenticated should survive back-navigation from optional keys")
	}
}

func TestCloudKeyShownForOtherProviders(t *testing.T) {
	m := NewWizardModel("", "")

	// Navigate to optional keys with anthropic provider.
	m.result.Provider = providerAnthropic
	m.initOptionalKeys()
	m.step = stepOptionalKeys

	// We're at NCBI confirm. Say no (n = done immediately).
	result, _ := m.Update(makeKeyPress("n"))
	m = result.(WizardModel)

	// Should go to cloud confirm for non-omics-os.
	if m.optionalSub != subStepCloudConfirm {
		t.Fatalf("expected subStepCloudConfirm for anthropic, got %d", m.optionalSub)
	}
}

func TestResolveDefaultOllamaSelectChoice(t *testing.T) {
	items := []singleSelectItem{
		{label: "A", value: "a"},
		{label: "B", value: defaultOllamaModel},
		{label: "C", value: "c"},
	}
	idx := resolveDefaultOllamaSelectChoice(items)
	if idx != 1 {
		t.Fatalf("expected index 1 for default model, got %d", idx)
	}

	items2 := []singleSelectItem{
		{label: "X", value: "x"},
	}
	idx2 := resolveDefaultOllamaSelectChoice(items2)
	if idx2 != 0 {
		t.Fatalf("expected index 0 when default not found, got %d", idx2)
	}
}
