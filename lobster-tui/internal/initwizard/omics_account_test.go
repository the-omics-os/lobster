package initwizard

import (
	"errors"
	"regexp"
	"strings"
	"testing"

	"charm.land/lipgloss/v2"

	"github.com/the-omics-os/lobster-tui/internal/i18n"
)

// The Omics-OS Cloud choice asks "Do you have an Omics-OS account?" before the key step.
// Yes -> the existing key step. No -> open and print the access-request page, wait for
// Enter, show a message above the provider list. Nothing is saved on the No path.

const (
	accountQuestion = "Do you have an Omics-OS account?"
	accountYes      = "Yes, I have an account"
	accountNo       = "No, request access"
	accessOpening   = "Opening the access request page in your browser:"
	accessCopy      = "If it does not open, copy the address into your browser."
	accessPress     = "Press Enter when you have sent the request."
	accessSent      = "Your access request is on its way. We'll email you when it's approved. " +
		"Until then, choose a different provider to continue."
)

var ansiSeq = regexp.MustCompile("\x1b\\[[0-9;]*m")

// render is the screen as text: styling is stripped so a line can be compared whole.
func render(m WizardModel) string { return ansiSeq.ReplaceAllString(m.View().Content, "") }

func keys(m WizardModel, ks ...string) WizardModel {
	for _, k := range ks {
		model, _ := m.Update(makeKeyPress(k))
		m = model.(WizardModel)
	}
	return m
}

// fakeBrowser replaces the opener for one test and records every URL it is given.
func fakeBrowser(t *testing.T, err error) *[]string {
	t.Helper()
	var urls []string
	orig := browserOpen
	browserOpen = func(u string) error {
		urls = append(urls, u)
		return err
	}
	t.Cleanup(func() { browserOpen = orig })
	return &urls
}

// flat joins a wrapped passage back into one line of single spaces.
func flat(s string) string { return strings.Join(strings.Fields(s), " ") }

func lines(s string) []string { return strings.Split(s, "\n") }

func hasLine(s, want string) bool {
	for _, l := range lines(s) {
		if strings.TrimSpace(l) == want {
			return true
		}
	}
	return false
}

func TestProviderLabelNamesTheAccountNotFreeLLMs(t *testing.T) {
	m := NewWizardModel("", "")
	if got := m.providerSelect.items[0].label; got != "Omics-OS Cloud      -- Omics-OS Cloud account" {
		t.Fatalf("label %q", got)
	}
	if strings.Contains(strings.ToLower(render(m)), "free") {
		t.Fatalf("the provider list still says free: %s", render(m))
	}
}

func TestOmicsChoiceAsksTheAccountQuestionFirst(t *testing.T) {
	m := keys(NewWizardModel("", ""), "enter")
	view := render(m)
	for _, want := range []string{accountQuestion, accountYes, accountNo} {
		if !strings.Contains(view, want) {
			t.Errorf("question screen misses %q: %s", want, view)
		}
	}
	if m.step == stepAPIKey || m.step == stepProvider {
		t.Fatalf("the choice went straight past the question (step %v)", m.step)
	}
	if m.result.Provider != "" {
		t.Fatalf("provider saved before the question was answered: %q", m.result.Provider)
	}
}

func TestAccountYesGoesToTheExistingKeyStep(t *testing.T) {
	urls := fakeBrowser(t, nil)
	m := keys(NewWizardModel("", ""), "enter", "enter")
	if m.step != stepAPIKey || m.result.Provider != providerOmicsOS {
		t.Fatalf("Yes did not reach the key step: step %v provider %q", m.step, m.result.Provider)
	}
	if !strings.Contains(m.viewAPIKey(), "app.omics-os.com") {
		t.Fatalf("key step changed: %s", m.viewAPIKey())
	}
	if len(*urls) != 0 {
		t.Fatalf("Yes opened a page: %v", *urls)
	}
}

func TestAccountNoOpensThePageAndPrintsTheFourLines(t *testing.T) {
	urls := fakeBrowser(t, nil)
	m := keys(NewWizardModel("", ""), "enter", "down", "enter")
	if len(*urls) != 1 || (*urls)[0] != "https://www.omics-os.com/early-access" {
		t.Fatalf("browser calls %v", *urls)
	}
	view := render(m)
	for _, want := range []string{accessOpening, accessCopy, accessPress} {
		if !hasLine(view, want) {
			t.Errorf("missing line %q: %s", want, view)
		}
	}
	if !hasLine(view, "https://www.omics-os.com/early-access") {
		t.Errorf("the URL is not on its own line: %s", view)
	}
	if m.step == stepProvider || m.step == stepAPIKey || m.result.Provider != "" || m.done {
		t.Fatalf("the No path moved on early: step %v provider %q done %v", m.step, m.result.Provider, m.done)
	}
}

func TestAccountNoPrintsTheLinesEvenWhenTheBrowserFails(t *testing.T) {
	urls := fakeBrowser(t, errors.New("no browser"))
	m := keys(NewWizardModel("", ""), "enter", "down", "enter")
	if len(*urls) != 1 {
		t.Fatalf("browser calls %v", *urls)
	}
	view := render(m)
	for _, want := range []string{accessOpening, "https://www.omics-os.com/early-access", accessCopy, accessPress} {
		if !hasLine(view, want) {
			t.Errorf("missing line %q after a browser error: %s", want, view)
		}
	}
	if strings.Contains(strings.ToLower(view), "no browser") || strings.Contains(strings.ToLower(view), "error") {
		t.Errorf("the browser error leaked into the screen: %s", view)
	}
}

func TestAccountNoEnterReturnsToTheProviderListWithTheMessage(t *testing.T) {
	fakeBrowser(t, nil)
	m := keys(NewWizardModel("", ""), "enter", "down", "enter", "enter")
	if m.step != stepProvider || m.result.Provider != "" || m.done {
		t.Fatalf("not back on the list: step %v provider %q done %v", m.step, m.result.Provider, m.done)
	}
	view := render(m)
	if !strings.Contains(flat(view), accessSent) {
		t.Fatalf("message missing: %s", view)
	}
	msgAt, listAt := strings.Index(view, "Your access request is on its way."), strings.Index(view, "Select the provider")
	if listAt < 0 || msgAt > listAt {
		t.Fatalf("the message is not above the provider list: %s", view)
	}
	if !strings.Contains(view, "Anthropic") {
		t.Fatalf("the provider list is gone: %s", view)
	}
	if strings.Contains(view, accessOpening) {
		t.Fatalf("the waiting screen text stayed: %s", view)
	}
}

func TestAccountNoSavesNothing(t *testing.T) {
	fakeBrowser(t, nil)
	fresh := NewWizardModel("", "").result
	m := keys(NewWizardModel("", ""), "enter", "down", "enter", "enter")
	if m.result.Provider != fresh.Provider || m.result.APIKey != "" || m.result.OAuthAuthenticated || m.oauthWaiting {
		t.Fatalf("the No path changed the result: %+v", m.result)
	}
}

func TestEscOnTheQuestionReturnsToTheProviderList(t *testing.T) {
	urls := fakeBrowser(t, nil)
	m := keys(NewWizardModel("", ""), "enter", "esc")
	if m.step != stepProvider || m.result.Provider != "" || m.done {
		t.Fatalf("Esc: step %v provider %q done %v", m.step, m.result.Provider, m.done)
	}
	if len(*urls) != 0 {
		t.Fatalf("Esc opened a page: %v", *urls)
	}
	view := render(m)
	if strings.Contains(flat(view), accessSent) || strings.Contains(view, accountQuestion) {
		t.Fatalf("stale text after Esc: %s", view)
	}
}

func TestTheMessageGoesAtTheNextProviderChoice(t *testing.T) {
	fakeBrowser(t, nil)
	m := keys(NewWizardModel("", ""), "enter", "down", "enter", "enter")
	m = keys(m, "down", "enter") // Anthropic
	if m.step != stepAPIKey || m.result.Provider != providerAnthropic {
		t.Fatalf("could not continue with another provider: step %v provider %q", m.step, m.result.Provider)
	}
	if strings.Contains(flat(render(m)), accessSent) {
		t.Fatalf("the message followed the user into the next step: %s", render(m))
	}
	m = keys(m, "esc") // back to the list
	if strings.Contains(flat(render(m)), accessSent) {
		t.Fatalf("the message came back with the list: %s", render(m))
	}
}

func TestOtherProvidersDoNotGetTheQuestion(t *testing.T) {
	m := keys(NewWizardModel("", ""), "down", "enter")
	if m.step != stepAPIKey || strings.Contains(render(m), accountQuestion) {
		t.Fatalf("Anthropic was asked the account question: step %v: %s", m.step, render(m))
	}
}

func TestAccountScreensFollowTheLanguage(t *testing.T) {
	fakeBrowser(t, nil)
	i18n.Set("zh")
	t.Cleanup(func() { i18n.Set("en") })
	m := keys(NewWizardModel("", ""), "enter")
	if !strings.Contains(render(m), "您已有 Omics-OS 账户吗？") {
		t.Fatalf("question not translated: %s", render(m))
	}
	m = keys(m, "down", "enter")
	for _, en := range []string{accessOpening, accessCopy, accessPress} {
		if strings.Contains(render(m), en) {
			t.Errorf("English left on the zh screen: %q in %s", en, render(m))
		}
	}
	if !hasLine(render(m), "https://www.omics-os.com/early-access") {
		t.Errorf("the URL is not on its own line in zh: %s", render(m))
	}
}

// The renderer cuts a line wider than the window (it does not wrap), so every line of
// the new screens must fit a default 80-column terminal, in English and in Chinese
// (Chinese characters are two columns wide).
func TestAccountScreensFitEightyColumns(t *testing.T) {
	for _, lang := range []string{"en", "zh"} {
		t.Run(lang, func(t *testing.T) {
			fakeBrowser(t, nil)
			i18n.Set(lang)
			t.Cleanup(func() { i18n.Set("en") })
			m := NewWizardModel("", "")
			m = keys(m, "enter")
			screens := map[string]string{"question": render(m)}
			m = keys(m, "down", "enter")
			screens["no screen"] = render(m)
			m = keys(m, "enter")
			screens["list with message"] = render(m)
			for name, screen := range screens {
				for _, line := range lines(screen) {
					if w := lipgloss.Width(line); w > 80 {
						t.Errorf("%s: a line is %d columns wide: %q", name, w, line)
					}
				}
			}
			if !strings.Contains(strings.Join(strings.Fields(screens["list with message"]), ""),
				strings.Join(strings.Fields(i18n.P(accessSent)), "")) {
				t.Errorf("the wrapped message lost words: %s", screens["list with message"])
			}
		})
	}
}
