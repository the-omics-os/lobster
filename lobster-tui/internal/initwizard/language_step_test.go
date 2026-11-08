package initwizard

import (
	"strings"
	"testing"

	"github.com/the-omics-os/lobster-tui/internal/i18n"
)

func viewText(m WizardModel) string {
	return m.View().Content
}

func TestLanguageStepOnlyWhenAsked(t *testing.T) {
	t.Setenv(askLangEnv, "")
	i18n.Set("en")
	m := NewWizardModel("", "")
	if m.step != stepProvider {
		t.Fatalf("without %s the wizard must start at the provider step, got %v", askLangEnv, m.step)
	}
}

func TestLanguageStepSwitchesWizardToChinese(t *testing.T) {
	t.Setenv(askLangEnv, "1")
	i18n.Set("en")
	t.Cleanup(func() { i18n.Set("en") })

	m := NewWizardModel("", "")
	if m.step != stepLanguage {
		t.Fatalf("expected language step first, got %v", m.step)
	}
	view := viewText(m)
	for _, want := range []string{"Language", "English", "简体中文"} {
		if !strings.Contains(view, want) {
			t.Fatalf("language view missing %q:\n%s", want, view)
		}
	}

	model, _ := m.Update(makeKeyPress("down"))
	model, _ = model.(WizardModel).Update(makeKeyPress("enter"))
	m = model.(WizardModel)
	if m.step != stepProvider {
		t.Fatalf("expected provider step after choosing, got %v", m.step)
	}
	if m.result.Lang != "zh" || i18n.Current() != "zh" {
		t.Fatalf("lang not applied: result=%q current=%q", m.result.Lang, i18n.Current())
	}
	if view := viewText(m); !strings.Contains(view, i18n.P("Select the provider for Lobster AI.")) || strings.Contains(view, "Select the provider for Lobster AI.") {
		t.Fatalf("provider view not translated:\n%s", view)
	}
	if label := m.agentSelect.items[0].label; !strings.Contains(label, i18n.T("wizard.pkg.research")) || i18n.T("wizard.pkg.research") == "" {
		t.Fatalf("agent labels not rebuilt in zh: %q", label)
	}

	// Esc on the provider screen returns to the language screen.
	model, _ = m.Update(makeKeyPress("esc"))
	m = model.(WizardModel)
	if m.step != stepLanguage {
		t.Fatalf("esc on provider should return to language, got %v", m.step)
	}
}

func TestLanguageStepEscCancels(t *testing.T) {
	t.Setenv(askLangEnv, "1")
	i18n.Set("en")
	m := NewWizardModel("", "")
	model, _ := m.Update(makeKeyPress("esc"))
	m = model.(WizardModel)
	if _, err := m.Result(); err == nil || m.step != stepDone {
		t.Fatalf("esc on language should cancel; step=%v err=%v", m.step, err)
	}
}

func TestLanguageStepDefaultsToCurrentLanguage(t *testing.T) {
	t.Setenv(askLangEnv, "1")
	i18n.Set("zh")
	t.Cleanup(func() { i18n.Set("en") })
	m := NewWizardModel("", "")
	if got := m.langSelect.items[m.langSelect.cursor].value; got != "zh" {
		t.Fatalf("cursor should start on current language zh, got %s", got)
	}
}
