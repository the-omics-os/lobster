// Completions provides context-aware autocomplete for the chat input.
//
// Uses the built-in textinput.SetSuggestions() API from charmbracelet/bubbles.
// Suggestions are rebuilt on every keystroke to provide context-sensitive
// completions: top-level commands when typing "/", subcommands after a
// command with children, and cached modality names for data commands.
package chat

import (
	"sort"
	"strings"
)

// commandDef describes a slash command for autocompletion.
type commandDef struct {
	cmd  string // e.g. "/workspace list"
	desc string // human description (for future dropdown)
}

// topLevelCommands are the commands shown when the user types "/".
// Order: most-used first for discovery, then alphabetical.
var topLevelCommands = []commandDef{
	{"/data", "Current data summary"},
	{"/help", "Show available commands"},
	{"/input-features", "Show input capabilities"},
	{"/read", "View file contents"},
	{"/workspace", "Workspace info & management"},
	{"/plots", "List generated plots"},
	{"/tokens", "Token usage & costs"},
	{"/session", "Current session status"},
	{"/status", "Subscription tier & agents"},
	{"/status-panel", "System status dashboard"},
	{"/workspace-info", "Workspace overview dashboard"},
	{"/analysis-dash", "Analysis dashboard"},
	{"/progress", "Progress monitor"},
	{"/pipeline", "Pipeline operations"},
	{"/config", "Configuration settings"},
	{"/queue", "Download queue management"},
	{"/metadata", "Metadata overview"},
	{"/modalities", "Detailed modality info"},
	{"/describe", "Describe a specific modality"},
	{"/export", "Export to ZIP"},
	{"/save", "Save state to workspace"},
	{"/files", "List workspace files"},
	{"/tree", "Directory tree view"},
	{"/plot", "Open plot file"},
	{"/open", "Open file in system app"},
	{"/restore", "Restore previous datasets"},
	{"/vector-search", "Search ontology collections"},
	{"/reset", "Reset conversation"},
	{"/clear", "Clear screen"},
	{"/dashboard", "Open classic dashboard"},
	{"/exit", "Exit"},
}

// subcommands maps parent commands to their subcommand completions.
var subcommands = map[string][]string{
	"/workspace": {"list", "load", "info", "remove", "save", "status"},
	"/save":      {"--force"},
	"/config":    {"show", "provider", "model"},
	"/queue":     {"load", "list", "clear", "export", "import"},
	"/metadata":  {"publications", "samples", "workspace", "exports", "list", "clear"},
	"/pipeline":  {"export", "list", "run", "info"},
}

// buildSuggestions returns the appropriate suggestion list based on the
// current input text. Called on every keystroke to keep suggestions fresh.
func (m *Model) buildSuggestions(input string) []string {
	input = strings.TrimLeft(input, " \t")

	// No input or not a command -> no suggestions.
	if input == "" || !strings.HasPrefix(input, "/") {
		return nil
	}

	hasTrailingSpace := strings.HasSuffix(input, " ")
	trimmed := strings.TrimSpace(input)
	if trimmed == "/" {
		return matchTopLevel("/")
	}

	parts := strings.Fields(trimmed)
	if len(parts) == 0 {
		return nil
	}

	cmd := strings.ToLower(parts[0])

	// Case 1: Typing top-level command text, e.g. "/wo".
	if len(parts) == 1 && !hasTrailingSpace {
		return matchTopLevel(parts[0])
	}

	// Case 2: Command + trailing space, e.g. "/workspace " or "/describe ".
	if len(parts) == 1 && hasTrailingSpace {
		if subs, ok := subcommands[cmd]; ok {
			return matchPrefix(subs, "", cmd+" ")
		}
		switch cmd {
		case "/describe":
			return matchPrefix(m.modalityNames(), "", "/describe ")
		}
		return nil
	}

	// Case 3: Command with arguments in progress.
	if subs, ok := subcommands[cmd]; ok {
		// Subcommand completion, e.g. "/workspace li".
		if len(parts) == 2 && !hasTrailingSpace {
			return matchPrefix(subs, parts[1], cmd+" ")
		}
		// Third-level completion, e.g. "/queue clear ".
		if len(parts) == 2 && hasTrailingSpace {
			return matchDeepSubcommands(cmd, parts[1], "")
		}
		if len(parts) >= 3 {
			typed := parts[2]
			if hasTrailingSpace {
				typed = ""
			}
			return matchDeepSubcommands(cmd, parts[1], typed)
		}
		return nil
	}

	// Contextual arg completion, e.g. "/describe rna".
	switch cmd {
	case "/describe":
		if len(parts) == 2 {
			arg := parts[1]
			if hasTrailingSpace {
				arg = ""
			}
			return matchPrefix(m.modalityNames(), arg, "/describe ")
		}
		return nil
	}

	return nil
}

// matchTopLevel returns top-level commands matching the given prefix.
func matchTopLevel(prefix string) []string {
	prefix = strings.ToLower(prefix)
	var matches []string
	for _, c := range topLevelCommands {
		if strings.HasPrefix(strings.ToLower(c.cmd), prefix) {
			matches = append(matches, c.cmd)
		}
	}
	return matches
}

// matchPrefix returns items from candidates that prefix-match the typed text,
// prepending the given command prefix to each result.
func matchPrefix(candidates []string, typed string, cmdPrefix string) []string {
	typed = strings.ToLower(typed)
	var matches []string
	for _, c := range candidates {
		if strings.HasPrefix(strings.ToLower(c), typed) {
			matches = append(matches, cmdPrefix+c)
		}
	}
	sort.Strings(matches)
	return matches
}

// matchDeepSubcommands handles third-level completions like "/queue clear download".
func matchDeepSubcommands(cmd, sub, typed string) []string {
	key := cmd + " " + strings.ToLower(sub)
	var candidates []string
	switch key {
	case "/queue clear":
		candidates = []string{"download", "all"}
	case "/queue list":
		candidates = []string{"publication", "download"}
	case "/metadata clear":
		candidates = []string{"exports", "all"}
	case "/workspace save":
		candidates = []string{"--force"}
	case "/config provider":
		candidates = []string{"list", "switch", "anthropic", "bedrock", "ollama", "gemini", "azure", "openai", "openrouter"}
	case "/config model":
		candidates = []string{"list", "switch"}
	default:
		return nil
	}
	return matchPrefix(candidates, typed, cmd+" "+sub+" ")
}

// modalityNames returns the names of cached modalities for completion.
func (m *Model) modalityNames() []string {
	if len(m.modalities) == 0 {
		return nil
	}
	names := make([]string, len(m.modalities))
	for i, mod := range m.modalities {
		names[i] = mod.Name
	}
	return names
}

// refreshSuggestions updates the textinput suggestion list based on current input.
// Call this after every input update in the key handler.
func (m *Model) refreshSuggestions() {
	suggestions := m.buildSuggestions(m.input.Value())
	m.input.SetSuggestions(suggestions)
}
