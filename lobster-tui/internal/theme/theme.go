// Package theme provides the Lobster TUI theming system.
//
// Usage:
//
//	theme.SetTheme("lobster-dark")
//	s := theme.Current.Styles.AssistantMessage.Render("Hello")
package theme

import (
	"fmt"
	"image/color"

	"charm.land/lipgloss/v2"
)

// Colors holds all semantic color tokens for a theme.
type Colors struct {
	Primary    color.Color
	Secondary  color.Color
	Background color.Color
	Surface    color.Color
	Overlay    color.Color

	Text      color.Color
	TextMuted color.Color
	TextDim   color.Color

	Success color.Color
	Warning color.Color
	Error   color.Color
	Info    color.Color

	Accent1 color.Color
	Accent2 color.Color
	Accent3 color.Color
}

// Styles holds pre-built lipgloss styles derived from a Colors palette.
// All styles are constructed once by BuildStyles and reused at render time.
type Styles struct {
	// Layout
	Header    lipgloss.Style
	Footer    lipgloss.Style
	StatusBar lipgloss.Style
	Sidebar   lipgloss.Style
	MainPanel lipgloss.Style
	Divider   lipgloss.Style

	// Chat messages
	UserMessage      lipgloss.Style
	AssistantMessage lipgloss.Style
	SystemMessage    lipgloss.Style
	ToolMessage      lipgloss.Style

	// Input
	InputField    lipgloss.Style
	InputPrompt   lipgloss.Style
	FormContainer lipgloss.Style
	FormField     lipgloss.Style
	FormLabel     lipgloss.Style

	// Feedback
	AlertSuccess lipgloss.Style
	AlertWarning lipgloss.Style
	AlertError   lipgloss.Style
	AlertInfo    lipgloss.Style

	// Agent / pipeline
	AgentBadge      lipgloss.Style
	AgentTransition lipgloss.Style
	ProgressBar     lipgloss.Style
	SpinnerStyle    lipgloss.Style
	ToolExecution   lipgloss.Style
	ToolRunning     lipgloss.Style
	ToolSuccess     lipgloss.Style
	ToolError       lipgloss.Style
	ModalityLoaded  lipgloss.Style

	// Code / markdown
	CodeBlock  lipgloss.Style
	InlineCode lipgloss.Style

	// General
	Bold   lipgloss.Style
	Muted  lipgloss.Style
	Dimmed lipgloss.Style
	Link   lipgloss.Style
}

// Theme is the top-level descriptor for a complete visual theme.
type Theme struct {
	ID          string
	Name        string
	Description string
	Author      string
	Version     string
	Colors      Colors
	Styles      Styles
}

// BuildStyles constructs all Styles from the given Colors palette.
// Call this once after creating or loading a theme's Colors.
func BuildStyles(c Colors) Styles {
	return Styles{
		// ---- Layout -------------------------------------------------------
		Header: lipgloss.NewStyle().
			Foreground(c.Primary).
			Bold(true).
			Padding(0, 1).
			BorderBottom(true).
			BorderStyle(lipgloss.NormalBorder()).
			BorderForeground(c.Overlay),

		Footer: lipgloss.NewStyle().
			Background(c.Surface).
			Foreground(c.TextMuted).
			Padding(0, 1),

		StatusBar: lipgloss.NewStyle().
			Background(c.Surface).
			Foreground(c.Text).
			Padding(0, 1).
			BorderTop(true).
			BorderStyle(lipgloss.NormalBorder()).
			BorderForeground(c.Overlay),

		Sidebar: lipgloss.NewStyle().
			Background(c.Surface).
			Foreground(c.Text).
			Padding(1, 2),

		MainPanel: lipgloss.NewStyle().
			Background(c.Background).
			Foreground(c.Text).
			Padding(1, 2),

		Divider: lipgloss.NewStyle().
			Foreground(c.TextDim),

		// ---- Chat messages ------------------------------------------------
		UserMessage: lipgloss.NewStyle().
			Foreground(c.Text).
			Background(c.Surface).
			Padding(0, 1).
			MarginRight(1).
			Border(lipgloss.RoundedBorder()).
			BorderForeground(c.Primary).
			BorderStyle(lipgloss.RoundedBorder()),

		AssistantMessage: lipgloss.NewStyle().
			Foreground(c.Text).
			Background(c.Overlay).
			Padding(0, 1).
			MarginRight(1).
			Border(lipgloss.RoundedBorder()).
			BorderForeground(c.Accent1).
			BorderStyle(lipgloss.RoundedBorder()),

		SystemMessage: lipgloss.NewStyle().
			Foreground(c.TextMuted).
			Padding(0, 1).
			BorderLeft(true).
			BorderForeground(c.Overlay),

		ToolMessage: lipgloss.NewStyle().
			Foreground(c.Text).
			PaddingLeft(2).
			BorderLeft(true).
			BorderForeground(c.Info).
			BorderStyle(lipgloss.NormalBorder()),

		// ---- Input --------------------------------------------------------
		InputField: lipgloss.NewStyle().
			Background(c.Surface).
			Foreground(c.Text).
			Padding(0, 1).
			Border(lipgloss.RoundedBorder()).
			BorderForeground(c.Secondary),

		InputPrompt: lipgloss.NewStyle().
			Foreground(c.Primary).
			Bold(true),

		FormContainer: lipgloss.NewStyle().
			Background(c.Surface).
			Padding(1, 2).
			Border(lipgloss.RoundedBorder()).
			BorderForeground(c.Overlay),

		FormField: lipgloss.NewStyle().
			Foreground(c.Text).
			Padding(0, 1),

		FormLabel: lipgloss.NewStyle().
			Foreground(c.TextMuted).
			Bold(true),

		// ---- Feedback -----------------------------------------------------
		AlertSuccess: lipgloss.NewStyle().
			Foreground(c.Success).
			Bold(true).
			Padding(0, 1).
			Background(c.Surface).
			MarginRight(1).
			BorderLeft(true).
			BorderStyle(lipgloss.NormalBorder()).
			BorderForeground(c.Success),

		AlertWarning: lipgloss.NewStyle().
			Foreground(c.Warning).
			Bold(true).
			Padding(0, 1).
			Background(c.Surface).
			MarginRight(1).
			BorderLeft(true).
			BorderStyle(lipgloss.NormalBorder()).
			BorderForeground(c.Warning),

		AlertError: lipgloss.NewStyle().
			Foreground(c.Error).
			Bold(true).
			Padding(0, 1).
			Background(c.Surface).
			MarginRight(1).
			BorderLeft(true).
			BorderStyle(lipgloss.NormalBorder()).
			BorderForeground(c.Error),

		AlertInfo: lipgloss.NewStyle().
			Foreground(c.Info).
			Bold(true).
			Padding(0, 1).
			Background(c.Surface).
			MarginRight(1).
			BorderLeft(true).
			BorderStyle(lipgloss.NormalBorder()).
			BorderForeground(c.Info),

		// ---- Agent / pipeline ---------------------------------------------
		AgentBadge: lipgloss.NewStyle().
			Background(c.Primary).
			Foreground(c.Background).
			Bold(true).
			Padding(0, 1),

		AgentTransition: lipgloss.NewStyle().
			Foreground(c.Accent2).
			Italic(true),

		ProgressBar: lipgloss.NewStyle().
			Foreground(c.Primary),

		SpinnerStyle: lipgloss.NewStyle().
			Foreground(c.Primary),

		ToolExecution: lipgloss.NewStyle().
			Foreground(c.TextMuted).
			Italic(true),

		ToolRunning: lipgloss.NewStyle().
			Foreground(c.Warning).
			Bold(true),

		ToolSuccess: lipgloss.NewStyle().
			Foreground(c.Success).
			Bold(true),

		ToolError: lipgloss.NewStyle().
			Foreground(c.Error).
			Bold(true),

		ModalityLoaded: lipgloss.NewStyle().
			Foreground(c.Accent3).
			Bold(true),

		// ---- Code / markdown ----------------------------------------------
		CodeBlock: lipgloss.NewStyle().
			Background(c.Surface).
			Foreground(c.Text).
			Padding(1, 2).
			Border(lipgloss.RoundedBorder()).
			BorderForeground(c.Overlay),

		InlineCode: lipgloss.NewStyle().
			Background(c.Overlay).
			Foreground(c.Accent2).
			Padding(0, 1),

		// ---- General ------------------------------------------------------
		Bold: lipgloss.NewStyle().
			Foreground(c.Text).
			Bold(true),

		Muted: lipgloss.NewStyle().
			Foreground(c.TextMuted),

		Dimmed: lipgloss.NewStyle().
			Foreground(c.TextDim),

		Link: lipgloss.NewStyle().
			Foreground(c.Info).
			Underline(true),
	}
}

// BuildCleanStyles constructs a low-chrome style profile intended for inline
// terminal usage where full-screen panels feel too heavy.
func BuildCleanStyles(c Colors) Styles {
	return Styles{
		// ---- Layout -------------------------------------------------------
		Header: lipgloss.NewStyle().
			Foreground(c.Primary).
			Bold(true),

		Footer: lipgloss.NewStyle().
			Foreground(c.TextMuted),

		StatusBar: lipgloss.NewStyle().
			Foreground(c.TextMuted),

		Sidebar: lipgloss.NewStyle().
			Foreground(c.TextMuted),

		MainPanel: lipgloss.NewStyle().
			Foreground(c.Text),

		Divider: lipgloss.NewStyle().
			Foreground(c.TextDim),

		// ---- Chat messages ------------------------------------------------
		UserMessage: lipgloss.NewStyle().
			Foreground(c.Text),

		AssistantMessage: lipgloss.NewStyle().
			Foreground(c.Text),

		SystemMessage: lipgloss.NewStyle().
			Foreground(c.TextMuted),

		ToolMessage: lipgloss.NewStyle().
			Foreground(c.Text),

		// ---- Input --------------------------------------------------------
		InputField: lipgloss.NewStyle().
			Foreground(c.Text),

		InputPrompt: lipgloss.NewStyle().
			Foreground(c.Primary).
			Bold(true),

		FormContainer: lipgloss.NewStyle().
			Foreground(c.Text),

		FormField: lipgloss.NewStyle().
			Foreground(c.Text),

		FormLabel: lipgloss.NewStyle().
			Foreground(c.TextMuted).
			Bold(true),

		// ---- Feedback -----------------------------------------------------
		AlertSuccess: lipgloss.NewStyle().
			Foreground(c.Success).
			Bold(true).
			Underline(true),

		AlertWarning: lipgloss.NewStyle().
			Foreground(c.Warning).
			Bold(true).
			Underline(true),

		AlertError: lipgloss.NewStyle().
			Foreground(c.Error).
			Bold(true).
			Underline(true),

		AlertInfo: lipgloss.NewStyle().
			Foreground(c.Info).
			Bold(true),

		// ---- Agent / pipeline ---------------------------------------------
		AgentBadge: lipgloss.NewStyle().
			Foreground(c.Primary).
			Bold(true),

		AgentTransition: lipgloss.NewStyle().
			Foreground(c.Accent2).
			Italic(true),

		ProgressBar: lipgloss.NewStyle().
			Foreground(c.Primary),

		SpinnerStyle: lipgloss.NewStyle().
			Foreground(c.Primary),

		ToolExecution: lipgloss.NewStyle().
			Foreground(c.TextMuted).
			Italic(true),

		ToolRunning: lipgloss.NewStyle().
			Foreground(c.Warning).
			Bold(true),

		ToolSuccess: lipgloss.NewStyle().
			Foreground(c.Success).
			Bold(true),

		ToolError: lipgloss.NewStyle().
			Foreground(c.Error).
			Bold(true),

		ModalityLoaded: lipgloss.NewStyle().
			Foreground(c.Accent3).
			Bold(true),

		// ---- Code / markdown ----------------------------------------------
		CodeBlock: lipgloss.NewStyle().
			Foreground(c.Text),

		InlineCode: lipgloss.NewStyle().
			Foreground(c.Accent2),

		// ---- General ------------------------------------------------------
		Bold: lipgloss.NewStyle().
			Foreground(c.Text).
			Bold(true),

		Muted: lipgloss.NewStyle().
			Foreground(c.TextMuted),

		Dimmed: lipgloss.NewStyle().
			Foreground(c.TextDim),

		Link: lipgloss.NewStyle().
			Foreground(c.Info).
			Underline(true),
	}
}

// ---- Registry -------------------------------------------------------------

// Current is the active theme used by all UI components.
var Current *Theme

// Available is the registry of all known themes, keyed by ID.
var Available = map[string]*Theme{}

// SetTheme activates a theme by ID. Returns an error if the ID is not found.
func SetTheme(id string) error {
	t, ok := Available[id]
	if !ok {
		return fmt.Errorf("theme %q not found (available: %v)", id, listIDs())
	}
	Current = t
	return nil
}

// Register adds a theme to the registry and rebuilds its styles from its
// Colors. Call Register for every theme before calling SetTheme.
func Register(t *Theme) {
	t.Styles = BuildStyles(t.Colors)
	Available[t.ID] = t
}

// listIDs returns all registered theme IDs for error messages.
func listIDs() []string {
	ids := make([]string, 0, len(Available))
	for id := range Available {
		ids = append(ids, id)
	}
	return ids
}
