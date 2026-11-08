package chat

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/the-omics-os/lobster-tui/internal/i18n"
	"github.com/the-omics-os/lobster-tui/internal/protocol"
)

func setLangMsg(t *testing.T, lang string) protocolMsg {
	t.Helper()
	raw, err := json.Marshal(protocol.SetLangPayload{Lang: lang})
	if err != nil {
		t.Fatal(err)
	}
	return protocolMsg{Message: protocol.Message{Type: protocol.TypeSetLang, Payload: raw}}
}

func completionDesc(cmd string) string {
	for _, c := range topLevelCommands() {
		if c.cmd == cmd {
			return c.desc
		}
	}
	return ""
}

func TestSetLangSwitchesUIAtRuntime(t *testing.T) {
	t.Cleanup(func() { i18n.Set("en") })
	i18n.Set("en")
	m := newTestModel()
	m.ready = true

	english := completionDesc("/lang")
	if english != "Switch UI language (en, zh)" {
		t.Fatalf("unexpected English description %q", english)
	}

	updated, _ := m.handleProtocol(setLangMsg(t, "zh"))
	m = updated.(Model)
	if i18n.Current() != "zh" {
		t.Fatalf("expected zh after set_lang, got %q", i18n.Current())
	}
	if got := completionDesc("/lang"); got == english || got == "" {
		t.Fatalf("completion description did not switch: %q", got)
	}
	if got := m.input.Placeholder; !m.inline && strings.Contains(got, "Ask Lobster") {
		t.Fatalf("placeholder did not switch: %q", got)
	}

	updated, _ = m.handleProtocol(setLangMsg(t, "en"))
	m = updated.(Model)
	if i18n.Current() != "en" || completionDesc("/lang") != english {
		t.Fatalf("switching back to English failed: lang=%q", i18n.Current())
	}
}

func TestSetLangUnknownFallsBackToEnglish(t *testing.T) {
	t.Cleanup(func() { i18n.Set("en") })
	m := newTestModel()
	updated, _ := m.handleProtocol(setLangMsg(t, "xx"))
	_ = updated.(Model)
	if i18n.Current() != "en" {
		t.Fatalf("expected English fallback, got %q", i18n.Current())
	}
}
