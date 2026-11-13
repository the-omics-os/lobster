// Package i18n loads the same JSON catalogs as lobster/i18n.
// English is complete. Overlays may be partial. Unknown languages use English.
package i18n

import (
	"embed"
	"encoding/json"
	"os"
	"strings"
)

//go:embed catalogs/*.json
var files embed.FS

var (
	lang     = "en"
	catalogs = map[string]map[string]string{}
	phrases  = map[string]string{}
)

func init() {
	for _, code := range []string{"en", "zh"} {
		raw, err := files.ReadFile("catalogs/" + code + ".json")
		if err != nil {
			continue
		}
		parsed := map[string]string{}
		if json.Unmarshal(raw, &parsed) == nil {
			catalogs[code] = parsed
		}
	}
	if raw, err := files.ReadFile("catalogs/phrases.zh.json"); err == nil {
		parsed := map[string]string{}
		if json.Unmarshal(raw, &parsed) == nil {
			phrases = parsed
		}
	}
	Set(os.Getenv("LOBSTER_LANG"))
}

// Set selects the process language. Unknown codes fall back to English.
func Set(code string) string {
	code = canonicalize(code)
	if _, ok := catalogs[code]; !ok {
		code = "en"
	}
	lang = code
	return lang
}

// Current returns the active language code.
func Current() string {
	return lang
}

func canonicalize(code string) string {
	code = strings.ToLower(strings.TrimSpace(code))
	code = strings.ReplaceAll(code, "_", "-")
	switch code {
	case "", "en", "english":
		return "en"
	case "zh", "zh-cn", "zh-hans", "cn", "chinese", "mandarin":
		return "zh"
	default:
		return code
	}
}

// P translates one English UI sentence. Unknown sentences stay English.
func P(english string) string {
	if lang != "zh" {
		return english
	}
	if text, ok := phrases[english]; ok && text != "" {
		return text
	}
	return english
}

// T returns the active translation, then English, then the key.
func T(key string) string {
	if text, ok := catalogs[lang][key]; ok && text != "" {
		return text
	}
	if text, ok := catalogs["en"][key]; ok {
		return text
	}
	return key
}
