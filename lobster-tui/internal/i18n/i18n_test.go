package i18n

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestFallbackAndMandarin(t *testing.T) {
	if got := Set("zz"); got != "en" {
		t.Fatalf("unknown language = %s", got)
	}
	if T("wizard.pkg.research") != "Literature search & data discovery" {
		t.Fatalf("english fallback = %q", T("wizard.pkg.research"))
	}
	if got := Set("zh-CN"); got != "zh" {
		t.Fatalf("alias = %s", got)
	}
	if T("wizard.pkg.research") != "文献检索和数据查找" {
		t.Fatalf("mandarin = %q", T("wizard.pkg.research"))
	}
	if T("not.a.key") != "not.a.key" {
		t.Fatalf("missing key returned %q", T("not.a.key"))
	}
	Set("en")
}

func TestEmbeddedCatalogsMatchPython(t *testing.T) {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("no caller")
	}
	pythonDir := filepath.Clean(filepath.Join(filepath.Dir(file), "..", "..", "..", "lobster", "i18n"))
	for _, name := range []string{"en.json", "zh.json", "phrases.zh.json"} {
		embedded, err := files.ReadFile("catalogs/" + name)
		if err != nil {
			t.Fatal(err)
		}
		source, err := os.ReadFile(filepath.Join(pythonDir, name))
		if err != nil {
			t.Fatal(err)
		}
		var want, got map[string]string
		if err := json.Unmarshal(source, &want); err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(embedded, &got); err != nil {
			t.Fatal(err)
		}
		if len(want) != len(got) {
			t.Fatalf("%s key count python=%d embed=%d", name, len(want), len(got))
		}
		for key, value := range want {
			if got[key] != value {
				t.Fatalf("%s key %s differs", name, key)
			}
		}
	}
}
