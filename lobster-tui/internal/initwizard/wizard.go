// Package initwizard provides the interactive init wizard for lobster-tui.
//
// The wizard collects configuration in five steps:
//  1. Agent package selection (MultiSelect)
//  2. Provider selection (Select)
//  3. API key entry (Input, provider-specific)
//  4. Profile selection — Anthropic and Bedrock only (Select)
//  5. Optional keys — NCBI and Omics-OS Cloud (Confirm + Input)
//
// On completion, a single JSON object is written to stdout and the process
// exits 0. On user cancellation, {"cancelled":true} is written and the
// process exits 1.
package initwizard

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/charmbracelet/huh"
	"github.com/charmbracelet/lipgloss"

	lobsterTheme "github.com/the-omics-os/lobster-tui/internal/theme"
)

// ---------------------------------------------------------------------------
// Output schema
// ---------------------------------------------------------------------------

// WizardResult is the JSON object written to stdout on wizard completion.
type WizardResult struct {
	Provider           string   `json:"provider"`
	APIKey             string   `json:"api_key"`
	APIKeySecondary    string   `json:"api_key_secondary"`
	Profile            string   `json:"profile"`
	Agents             []string `json:"agents"`
	NCBIKey            string   `json:"ncbi_key"`
	CloudKey           string   `json:"cloud_key"`
	OllamaModel        string   `json:"ollama_model"`
	Cancelled          bool     `json:"cancelled"`
}

// ---------------------------------------------------------------------------
// Provider constants
// ---------------------------------------------------------------------------

const (
	providerAnthropic  = "anthropic"
	providerBedrock    = "bedrock"
	providerOllama     = "ollama"
	providerGemini     = "gemini"
	providerAzure      = "azure"
	providerOpenAI     = "openai"
	providerOpenRouter = "openrouter"
)

// providersWithProfile are the providers that expose a profile selection step.
var providersWithProfile = map[string]bool{
	providerAnthropic: true,
	providerBedrock:   true,
}

// ---------------------------------------------------------------------------
// huh theme builder
// ---------------------------------------------------------------------------

// buildHuhTheme derives a huh.Theme from the active Lobster lipgloss theme.
func buildHuhTheme(t *lobsterTheme.Theme) *huh.Theme {
	ht := huh.ThemeBase()

	primary := t.Colors.Primary
	accent := t.Colors.Accent2
	muted := t.Colors.TextMuted
	text := t.Colors.Text
	success := t.Colors.Success
	errColor := t.Colors.Error
	surface := t.Colors.Surface

	// Focused state
	ht.Focused.Base = ht.Focused.Base.BorderForeground(primary)
	ht.Focused.Title = lipgloss.NewStyle().Foreground(primary).Bold(true)
	ht.Focused.Description = lipgloss.NewStyle().Foreground(muted)
	ht.Focused.ErrorIndicator = lipgloss.NewStyle().Foreground(errColor).SetString(" *")
	ht.Focused.ErrorMessage = lipgloss.NewStyle().Foreground(errColor)
	ht.Focused.SelectSelector = lipgloss.NewStyle().Foreground(accent).SetString("> ")
	ht.Focused.NextIndicator = lipgloss.NewStyle().Foreground(accent).MarginLeft(1).SetString("->")
	ht.Focused.PrevIndicator = lipgloss.NewStyle().Foreground(accent).MarginRight(1).SetString("<-")
	ht.Focused.Option = lipgloss.NewStyle().Foreground(text)
	ht.Focused.MultiSelectSelector = lipgloss.NewStyle().Foreground(accent).SetString("> ")
	ht.Focused.SelectedOption = lipgloss.NewStyle().Foreground(success)
	ht.Focused.SelectedPrefix = lipgloss.NewStyle().Foreground(success).SetString("[x] ")
	ht.Focused.UnselectedPrefix = lipgloss.NewStyle().Foreground(muted).SetString("[ ] ")
	ht.Focused.UnselectedOption = lipgloss.NewStyle().Foreground(text)
	ht.Focused.FocusedButton = lipgloss.NewStyle().
		Foreground(t.Colors.Background).
		Background(primary).
		Bold(true).
		Padding(0, 2).
		MarginRight(1)
	ht.Focused.BlurredButton = lipgloss.NewStyle().
		Foreground(muted).
		Background(surface).
		Padding(0, 2).
		MarginRight(1)
	ht.Focused.TextInput.Cursor = lipgloss.NewStyle().Foreground(primary)
	ht.Focused.TextInput.Placeholder = lipgloss.NewStyle().Foreground(muted)
	ht.Focused.TextInput.Prompt = lipgloss.NewStyle().Foreground(accent)
	ht.Focused.TextInput.Text = lipgloss.NewStyle().Foreground(text)

	// Blurred state mirrors focused but with dimmer border
	ht.Blurred = ht.Focused
	ht.Blurred.Base = ht.Focused.Base.BorderStyle(lipgloss.HiddenBorder())
	ht.Blurred.Card = ht.Blurred.Base
	ht.Blurred.NextIndicator = lipgloss.NewStyle()
	ht.Blurred.PrevIndicator = lipgloss.NewStyle()

	// Group title/description
	ht.Group.Title = lipgloss.NewStyle().Foreground(primary).Bold(true).MarginBottom(1)
	ht.Group.Description = lipgloss.NewStyle().Foreground(muted)

	return ht
}

// ---------------------------------------------------------------------------
// Run — public entry point
// ---------------------------------------------------------------------------

// Run launches the init wizard with the given theme name.
// Pass an empty string or "lobster-dark" to use the default dark theme.
// On successful completion it writes a JSON result to stdout and returns nil.
// On cancellation it writes {"cancelled":true} and returns a non-nil sentinel.
func Run(themeName string) error {
	// ---- Theme setup --------------------------------------------------------
	if themeName != "" {
		if err := lobsterTheme.SetTheme(themeName); err != nil {
			// Non-fatal: fall back to whatever Current already is.
			fmt.Fprintf(os.Stderr, "warning: %v\n", err)
		}
	}
	activeTheme := lobsterTheme.Current
	if activeTheme == nil {
		activeTheme = lobsterTheme.LobsterDark
	}
	ht := buildHuhTheme(activeTheme)

	// ---- State --------------------------------------------------------------
	var (
		selectedAgents []string
		provider       string

		apiKey          string
		apiKeySecondary string
		ollamaModel     string

		profile string

		wantNCBI  bool
		ncbiKey   string
		wantCloud bool
		cloudKey  string
	)

	// ---- Step 1 + 2: Agents + Provider in one form -------------------------
	agentOptions := []huh.Option[string]{
		huh.NewOption("lobster-research         (2 agents — PubMed, GEO, data discovery)", "lobster-research").Selected(true),
		huh.NewOption("lobster-transcriptomics  (3 agents — scRNA-seq, bulk RNA-seq analysis)", "lobster-transcriptomics").Selected(true),
		huh.NewOption("lobster-visualization    (1 agent  — Plotly data visualization)", "lobster-visualization"),
		huh.NewOption("lobster-genomics         (2 agents — VCF, GWAS, clinical variants)", "lobster-genomics"),
		huh.NewOption("lobster-proteomics       (3 agents — MS, affinity, biomarkers)", "lobster-proteomics"),
		huh.NewOption("lobster-metabolomics     (1 agent  — LC-MS, GC-MS, NMR analysis)", "lobster-metabolomics"),
	}

	providerOptions := []huh.Option[string]{
		huh.NewOption("Anthropic (Claude)  — Quick testing, development", providerAnthropic),
		huh.NewOption("AWS Bedrock         — Production, enterprise use", providerBedrock),
		huh.NewOption("Ollama (Local)      — Privacy, zero cost, offline", providerOllama),
		huh.NewOption("Google Gemini       — Latest models with thinking", providerGemini),
		huh.NewOption("Azure AI            — Enterprise Azure deployments", providerAzure),
		huh.NewOption("OpenAI              — GPT-4o, reasoning models", providerOpenAI),
		huh.NewOption("OpenRouter          — 600+ models via one API key", providerOpenRouter),
	}

	// Default provider selection to anthropic
	provider = providerAnthropic

	form1 := huh.NewForm(
		huh.NewGroup(
			huh.NewMultiSelect[string]().
				Title("Agent Packages").
				Description("Select the agent packages to install. Use space to toggle.").
				Options(agentOptions...).
				Value(&selectedAgents).
				Validate(func(vals []string) error {
					if len(vals) == 0 {
						return fmt.Errorf("select at least one agent package")
					}
					return nil
				}),
		),
		huh.NewGroup(
			huh.NewSelect[string]().
				Title("LLM Provider").
				Description("Select the provider for Lobster AI.").
				Options(providerOptions...).
				Value(&provider),
		),
	).WithTheme(ht)

	if err := form1.Run(); err != nil {
		return handleAbort(err)
	}

	// ---- Step 3: API key(s) — provider-specific ----------------------------
	var form2 *huh.Form

	switch provider {
	case providerAnthropic:
		form2 = huh.NewForm(
			huh.NewGroup(
				huh.NewInput().
					Title("Anthropic API Key").
					Description("Your Anthropic API key. Find it at console.anthropic.com.").
					Placeholder("sk-ant-...").
					EchoMode(huh.EchoModePassword).
					Value(&apiKey),
			),
		).WithTheme(ht)

	case providerBedrock:
		form2 = huh.NewForm(
			huh.NewGroup(
				huh.NewInput().
					Title("AWS Bedrock Access Key ID").
					Description("AWS access key ID with Bedrock permissions.").
					Placeholder("AKIAIOSFODNN7EXAMPLE").
					Value(&apiKey),
				huh.NewInput().
					Title("AWS Bedrock Secret Access Key").
					Description("AWS secret access key (stored locally, never transmitted).").
					Placeholder("wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY").
					EchoMode(huh.EchoModePassword).
					Value(&apiKeySecondary),
			),
		).WithTheme(ht)

	case providerOllama:
		ollamaModel = "llama3:8b-instruct"
		form2 = huh.NewForm(
			huh.NewGroup(
				huh.NewInput().
					Title("Ollama Model").
					Description("Model name to use with Ollama. Must be pulled locally first.").
					Placeholder("llama3:8b-instruct").
					Value(&ollamaModel),
			),
		).WithTheme(ht)

	case providerGemini:
		form2 = huh.NewForm(
			huh.NewGroup(
				huh.NewInput().
					Title("Google API Key").
					Description("Your Google AI Studio API key. Find it at aistudio.google.com.").
					Placeholder("AIzaSy...").
					EchoMode(huh.EchoModePassword).
					Value(&apiKey),
			),
		).WithTheme(ht)

	case providerAzure:
		form2 = huh.NewForm(
			huh.NewGroup(
				huh.NewInput().
					Title("Azure AI Endpoint URL").
					Description("Your Azure AI Services endpoint URL.").
					Placeholder("https://your-resource.openai.azure.com/").
					Value(&apiKeySecondary),
				huh.NewInput().
					Title("Azure AI API Key").
					Description("Your Azure AI Services API key.").
					Placeholder("your-azure-api-key").
					EchoMode(huh.EchoModePassword).
					Value(&apiKey),
			),
		).WithTheme(ht)

	case providerOpenAI:
		form2 = huh.NewForm(
			huh.NewGroup(
				huh.NewInput().
					Title("OpenAI API Key").
					Description("Your OpenAI API key. Find it at platform.openai.com/api-keys.").
					Placeholder("sk-...").
					EchoMode(huh.EchoModePassword).
					Value(&apiKey),
			),
		).WithTheme(ht)

	case providerOpenRouter:
		form2 = huh.NewForm(
			huh.NewGroup(
				huh.NewInput().
					Title("OpenRouter API Key").
					Description("Your OpenRouter API key. Find it at openrouter.ai/keys.").
					Placeholder("sk-or-...").
					EchoMode(huh.EchoModePassword).
					Value(&apiKey),
			),
		).WithTheme(ht)

	default:
		form2 = huh.NewForm(
			huh.NewGroup(
				huh.NewInput().
					Title("API Key").
					Description("API key for the selected provider.").
					EchoMode(huh.EchoModePassword).
					Value(&apiKey),
			),
		).WithTheme(ht)
	}

	if err := form2.Run(); err != nil {
		return handleAbort(err)
	}

	// ---- Step 4: Profile — Anthropic and Bedrock only ----------------------
	if providersWithProfile[provider] {
		profileOptions := []huh.Option[string]{
			huh.NewOption("Development  — Sonnet 4, fastest and most affordable", "development"),
			huh.NewOption("Production   — Sonnet 4 + Sonnet 4.5 supervisor [Recommended]", "production"),
			huh.NewOption("Performance  — Sonnet 4.5, highest quality", "performance"),
			huh.NewOption("Max          — Opus 4.5 supervisor, most capable, expensive", "max"),
		}

		// Default to production
		profile = "production"

		form3 := huh.NewForm(
			huh.NewGroup(
				huh.NewSelect[string]().
					Title("Performance Profile").
					Description("Choose a profile that balances cost and capability.").
					Options(profileOptions...).
					Value(&profile),
			),
		).WithTheme(ht)

		if err := form3.Run(); err != nil {
			return handleAbort(err)
		}
	}

	// ---- Step 5: Optional keys --------------------------------------------
	form4 := huh.NewForm(
		huh.NewGroup(
			huh.NewConfirm().
				Title("NCBI API Key").
				Description("Add an NCBI API key for enhanced PubMed / literature search? (Recommended for heavy use)").
				Affirmative("Yes, add key").
				Negative("Skip").
				Value(&wantNCBI),
		),
	).WithTheme(ht)

	if err := form4.Run(); err != nil {
		return handleAbort(err)
	}

	if wantNCBI {
		form4b := huh.NewForm(
			huh.NewGroup(
				huh.NewInput().
					Title("NCBI API Key").
					Description("Your NCBI API key. Find it at ncbi.nlm.nih.gov/account/settings.").
					Placeholder("your-ncbi-api-key").
					EchoMode(huh.EchoModePassword).
					Value(&ncbiKey),
			),
		).WithTheme(ht)

		if err := form4b.Run(); err != nil {
			return handleAbort(err)
		}
	}

	form5 := huh.NewForm(
		huh.NewGroup(
			huh.NewConfirm().
				Title("Omics-OS Cloud API Key").
				Description("Add an Omics-OS Cloud API key to unlock premium features?").
				Affirmative("Yes, add key").
				Negative("Skip").
				Value(&wantCloud),
		),
	).WithTheme(ht)

	if err := form5.Run(); err != nil {
		return handleAbort(err)
	}

	if wantCloud {
		form5b := huh.NewForm(
			huh.NewGroup(
				huh.NewInput().
					Title("Omics-OS Cloud API Key").
					Description("Your Omics-OS Cloud key. Find it at app.omics-os.com/settings.").
					Placeholder("omics-...").
					EchoMode(huh.EchoModePassword).
					Value(&cloudKey),
			),
		).WithTheme(ht)

		if err := form5b.Run(); err != nil {
			return handleAbort(err)
		}
	}

	// ---- Output ------------------------------------------------------------
	// Normalise agent names: strip the display description, keep only the
	// package identifier (e.g. "lobster-research").
	cleanAgents := make([]string, len(selectedAgents))
	for i, a := range selectedAgents {
		// Values are already the clean identifiers as set in NewOption.
		cleanAgents[i] = strings.TrimSpace(a)
	}

	result := WizardResult{
		Provider:        provider,
		APIKey:          apiKey,
		APIKeySecondary: apiKeySecondary,
		Profile:         profile,
		Agents:          cleanAgents,
		NCBIKey:         ncbiKey,
		CloudKey:        cloudKey,
		OllamaModel:     ollamaModel,
		Cancelled:       false,
	}

	return writeResult(result)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// handleAbort converts a huh.ErrUserAborted into a canonical cancelled output.
// All other errors are returned as-is.
func handleAbort(err error) error {
	if errors.Is(err, huh.ErrUserAborted) {
		_ = writeResult(WizardResult{Cancelled: true})
		return fmt.Errorf("cancelled")
	}
	return err
}

// writeResult serialises result to stdout as compact JSON followed by a newline.
func writeResult(r WizardResult) error {
	enc := json.NewEncoder(os.Stdout)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(r); err != nil {
		return fmt.Errorf("write result: %w", err)
	}
	return nil
}
