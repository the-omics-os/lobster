// Package theme — JSON theme loading.
//
// Themes can be loaded from JSON files and registered into the Available map.
// The active theme can be overridden via the LOBSTER_TUI_THEME environment
// variable, which accepts a theme ID (e.g. "lobster-dark").
//
// JSON schema example: see themes/lobster-dark.json in the repo root.
package theme

import (
	"encoding/json"
	"fmt"
	"image/color"
	"os"
	"path/filepath"
	"strings"

	"charm.land/lipgloss/v2"
)

// ThemeJSON is the serialisation schema for theme files.
type ThemeJSON struct {
	ID          string     `json:"id"`
	Name        string     `json:"name"`
	Description string     `json:"description"`
	Author      string     `json:"author"`
	Version     string     `json:"version"`
	Colors      ColorsJSON `json:"colors"`
}

// ColorsJSON mirrors Colors but uses plain strings for JSON portability.
type ColorsJSON struct {
	Primary    string `json:"primary"`
	Secondary  string `json:"secondary"`
	Background string `json:"background"`
	Surface    string `json:"surface"`
	Overlay    string `json:"overlay"`
	Text       string `json:"text"`
	TextMuted  string `json:"text_muted"`
	TextDim    string `json:"text_dim"`
	Success    string `json:"success"`
	Warning    string `json:"warning"`
	Error      string `json:"error"`
	Info       string `json:"info"`
	Accent1    string `json:"accent1"`
	Accent2    string `json:"accent2"`
	Accent3    string `json:"accent3"`
}

// parseColor converts a hex string (with or without '#') to a color.Color via
// lipgloss.Color(). Returns nil and an error if the string is empty.
func parseColor(s string) (color.Color, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil, fmt.Errorf("empty color value")
	}
	// lipgloss accepts bare hex strings; ensure the '#' prefix is present.
	if !strings.HasPrefix(s, "#") {
		s = "#" + s
	}
	return lipgloss.Color(s), nil
}

// colorsFromJSON converts a ColorsJSON into a Colors, validating every field.
func colorsFromJSON(j ColorsJSON) (Colors, error) {
	var (
		c   Colors
		err error
	)
	parse := func(field string, dest *color.Color) {
		if err != nil {
			return
		}
		*dest, err = parseColor(field)
	}

	parse(j.Primary, &c.Primary)
	parse(j.Secondary, &c.Secondary)
	parse(j.Background, &c.Background)
	parse(j.Surface, &c.Surface)
	parse(j.Overlay, &c.Overlay)
	parse(j.Text, &c.Text)
	parse(j.TextMuted, &c.TextMuted)
	parse(j.TextDim, &c.TextDim)
	parse(j.Success, &c.Success)
	parse(j.Warning, &c.Warning)
	parse(j.Error, &c.Error)
	parse(j.Info, &c.Info)
	parse(j.Accent1, &c.Accent1)
	parse(j.Accent2, &c.Accent2)
	parse(j.Accent3, &c.Accent3)

	return c, err
}

// LoadThemeFromJSON parses raw JSON bytes into a Theme, registers it, and
// returns it. Returns an error if parsing or color validation fails.
func LoadThemeFromJSON(data []byte) (*Theme, error) {
	var tj ThemeJSON
	if err := json.Unmarshal(data, &tj); err != nil {
		return nil, fmt.Errorf("theme JSON decode: %w", err)
	}
	if tj.ID == "" {
		return nil, fmt.Errorf("theme JSON missing required field 'id'")
	}

	colors, err := colorsFromJSON(tj.Colors)
	if err != nil {
		return nil, fmt.Errorf("theme %q invalid color: %w", tj.ID, err)
	}

	t := &Theme{
		ID:          tj.ID,
		Name:        tj.Name,
		Description: tj.Description,
		Author:      tj.Author,
		Version:     tj.Version,
		Colors:      colors,
	}
	Register(t) // builds Styles and adds to Available
	return t, nil
}

// LoadThemeFromFile reads a JSON file at path and delegates to LoadThemeFromJSON.
func LoadThemeFromFile(path string) (*Theme, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read theme file %q: %w", path, err)
	}
	return LoadThemeFromJSON(data)
}

// LoadThemesFromDirectory loads all *.json files in dir as themes. Files that
// fail to parse are skipped with a warning printed to stderr. Returns the
// number of themes successfully loaded.
func LoadThemesFromDirectory(dir string) (int, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return 0, fmt.Errorf("read theme directory %q: %w", dir, err)
	}

	loaded := 0
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		path := filepath.Join(dir, e.Name())
		if _, err := LoadThemeFromFile(path); err != nil {
			fmt.Fprintf(os.Stderr, "warning: skipping theme file %q: %v\n", path, err)
			continue
		}
		loaded++
	}
	return loaded, nil
}

// ApplyEnvOverride checks the LOBSTER_TUI_THEME environment variable and calls
// SetTheme if the variable is set. Call this after registering all themes.
// Returns an error if the env var is set but the theme ID is unknown.
func ApplyEnvOverride() error {
	id := os.Getenv("LOBSTER_TUI_THEME")
	if id == "" {
		return nil
	}
	return SetTheme(id)
}
