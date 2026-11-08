package i18n

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"testing"
)

// uiRoots are the packages whose user-visible text must be translatable.
var uiRoots = []string{"../chat", "../biocomp", "../initwizard"}

// goFiles returns every non-test Go file below the given roots.
func goFiles(t *testing.T, roots ...string) []string {
	t.Helper()
	var out []string
	for _, root := range roots {
		err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if !d.IsDir() && strings.HasSuffix(path, ".go") && !strings.HasSuffix(path, "_test.go") {
				out = append(out, path)
			}
			return nil
		})
		if err != nil {
			t.Fatalf("walk %s: %v", root, err)
		}
	}
	sort.Strings(out)
	return out
}

func parseFile(t *testing.T, fset *token.FileSet, path string) *ast.File {
	t.Helper()
	f, err := parser.ParseFile(fset, path, nil, 0)
	if err != nil {
		t.Fatalf("parse %s: %v", path, err)
	}
	return f
}

// i18nCall reports the function name when call is i18n.P or i18n.T with a literal argument.
func i18nCall(call *ast.CallExpr) (string, string, bool) {
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok || len(call.Args) != 1 {
		return "", "", false
	}
	pkg, ok := sel.X.(*ast.Ident)
	if !ok || pkg.Name != "i18n" || (sel.Sel.Name != "P" && sel.Sel.Name != "T") {
		return "", "", false
	}
	lit, ok := call.Args[0].(*ast.BasicLit)
	if !ok || lit.Kind != token.STRING {
		return "", "", false
	}
	text, err := strconv.Unquote(lit.Value)
	if err != nil {
		return "", "", false
	}
	return sel.Sel.Name, text, true
}

// Every i18n.P literal needs a Chinese phrase; every i18n.T key needs an English entry.
func TestEveryTranslationCallHasCatalogEntry(t *testing.T) {
	fset := token.NewFileSet()
	roots := append([]string{"../../cmd"}, uiRoots...)
	for _, path := range goFiles(t, roots...) {
		f := parseFile(t, fset, path)
		ast.Inspect(f, func(n ast.Node) bool {
			call, ok := n.(*ast.CallExpr)
			if !ok {
				return true
			}
			name, text, ok := i18nCall(call)
			if !ok {
				return true
			}
			switch name {
			case "P":
				if _, found := phrases[text]; !found {
					t.Errorf("%s: i18n.P(%q) has no entry in phrases.zh.json", fset.Position(call.Pos()), text)
				}
			case "T":
				if _, found := catalogs["en"][text]; !found {
					t.Errorf("%s: i18n.T(%q) has no entry in en.json", fset.Position(call.Pos()), text)
				}
			}
			return true
		})
	}
}

var (
	hasWord    = regexp.MustCompile(`[A-Za-z]{2,}`)
	keepAsIs   = regexp.MustCompile(`^(\s*[/\-]?[a-z0-9_.\-]+\s*|.*%[sdvqf].*|[A-Z]{2,5}|Ctrl\+[A-Z]|\[[a-zA-Z/]+\]|https?://\S+)$`)
	uiSinks    = map[string]bool{"Render": true, "WithHelp": true}
	uiFields   = map[string]bool{"Title": true, "Message": true, "Placeholder": true, "Label": true, "Description": true}
	protocolOK = map[string]bool{"Session: ": true, "Provider: ": true, "Model: ": true}
	// keepEnglish lists intentional English: product names and example credentials.
	keepEnglish = map[string]bool{
		"Lobster AI":           true,
		"AKIAIOSFODNN7EXAMPLE": true,
		"wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY": true,
		"AIzaSy...": true,
	}
)

// looksLikeUIText is a heuristic for English words a user reads.
func looksLikeUIText(text string) bool {
	trimmed := strings.TrimSpace(text)
	if trimmed == "" || protocolOK[text] || keepEnglish[trimmed] || !hasWord.MatchString(trimmed) {
		return false
	}
	return !keepAsIs.MatchString(trimmed)
}

// rawUILiterals lists English literals passed straight to render sinks or UI fields.
func rawUILiterals(t *testing.T) []string {
	fset := token.NewFileSet()
	var found []string
	add := func(pos token.Pos, lit *ast.BasicLit) {
		if lit == nil || lit.Kind != token.STRING {
			return
		}
		text, err := strconv.Unquote(lit.Value)
		if err == nil && looksLikeUIText(text) {
			found = append(found, fset.Position(pos).String()+": "+strconv.Quote(text))
		}
	}
	for _, path := range goFiles(t, uiRoots...) {
		f := parseFile(t, fset, path)
		ast.Inspect(f, func(n ast.Node) bool {
			switch node := n.(type) {
			case *ast.CallExpr:
				if sel, ok := node.Fun.(*ast.SelectorExpr); ok && uiSinks[sel.Sel.Name] {
					for _, arg := range node.Args {
						lit, _ := arg.(*ast.BasicLit)
						add(arg.Pos(), lit)
					}
				}
			case *ast.KeyValueExpr:
				if key, ok := node.Key.(*ast.Ident); ok && uiFields[key.Name] {
					lit, _ := node.Value.(*ast.BasicLit)
					add(node.Value.Pos(), lit)
				}
			case *ast.AssignStmt:
				for i, lhs := range node.Lhs {
					sel, ok := lhs.(*ast.SelectorExpr)
					if !ok || !uiFields[sel.Sel.Name] || i >= len(node.Rhs) {
						continue
					}
					lit, _ := node.Rhs[i].(*ast.BasicLit)
					add(node.Rhs[i].Pos(), lit)
				}
			}
			return true
		})
	}
	return found
}

// TestNoRawUILiterals fails on English literals that bypass i18n.P.
// Set LOBSTER_I18N_STRICT=0 to only report them while translating.
func TestNoRawUILiterals(t *testing.T) {
	found := rawUILiterals(t)
	strict := os.Getenv("LOBSTER_I18N_STRICT") != "0"
	for _, item := range found {
		if strict {
			t.Errorf("untranslated UI literal: %s", item)
		} else {
			t.Logf("untranslated UI literal: %s", item)
		}
	}
	t.Logf("%d untranslated UI literals (strict=%v)", len(found), strict)
}
