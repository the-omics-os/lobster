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
	"os/exec"
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
	Provider                      string   `json:"provider"`
	APIKey                        string   `json:"api_key"`
	APIKeySecondary               string   `json:"api_key_secondary"`
	Profile                       string   `json:"profile"`
	Agents                        []string `json:"agents"`
	NCBIKey                       string   `json:"ncbi_key"`
	CloudKey                      string   `json:"cloud_key"`
	OllamaModel                   string   `json:"ollama_model"`
	SmartStandardizationEnabled   bool     `json:"smart_standardization_enabled"`
	SmartStandardizationOpenAIKey string   `json:"smart_standardization_openai_key"`
	Cancelled                     bool     `json:"cancelled"`
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
	customOllamaModel  = "__custom_ollama_model__"
	defaultOllamaModel = "gpt-oss:20b"
)

// providersWithProfile are the providers that expose a profile selection step.
var providersWithProfile = map[string]bool{
	providerAnthropic: true,
	providerBedrock:   true,
}

type initAgentPackage struct {
	Name        string
	Description string
	Agents      []string
	Default     bool
}

type ollamaStatus struct {
	Installed bool
	Version   string
	Running   bool
	Models    []string
}

var initAgentPackages = []initAgentPackage{
	{
		Name:        "lobster-research",
		Description: "Literature search & data discovery",
		Agents:      []string{"research_agent", "data_expert_agent"},
		Default:     true,
	},
	{
		Name:        "lobster-transcriptomics",
		Description: "Single-cell & bulk RNA-seq analysis",
		Agents:      []string{"transcriptomics_expert", "annotation_expert", "de_analysis_expert"},
		Default:     true,
	},
	{
		Name:        "lobster-visualization",
		Description: "Data visualization & plotting",
		Agents:      []string{"visualization_expert_agent"},
	},
	{
		Name:        "lobster-genomics",
		Description: "Genomics, GWAS, and clinical variants",
		Agents:      []string{"genomics_expert", "variant_analysis_expert"},
	},
	{
		Name:        "lobster-proteomics",
		Description: "Mass spec, affinity proteomics, and biomarkers",
		Agents:      []string{"proteomics_expert", "proteomics_de_analysis_expert", "biomarker_discovery_expert"},
	},
	{
		Name:        "lobster-metabolomics",
		Description: "LC-MS, GC-MS, and NMR metabolomics",
		Agents:      []string{"metabolomics_expert"},
	},
	{
		Name:        "lobster-ml",
		Description: "Machine learning, feature selection, and survival analysis",
		Agents:      []string{"machine_learning_expert", "feature_selection_expert", "survival_analysis_expert"},
	},
	{
		Name:        "lobster-drug-discovery",
		Description: "Drug discovery, cheminformatics, and translational strategy",
		Agents:      []string{"drug_discovery_expert", "cheminformatics_expert", "clinical_dev_expert", "pharmacogenomics_expert"},
	},
}

var curatedOllamaModels = []struct {
	Name        string
	Description string
}{
	{Name: "qwen3:8b", Description: "Fast local default"},
	{Name: "qwen3:14b", Description: "Stronger reasoning"},
	{Name: "gpt-oss:20b", Description: "Best Lobster default"},
	{Name: "qwen3:30b-a3b", Description: "Highest-quality curated option"},
}

var smartStandardizationAgentLabels = map[string]string{
	"metadata_assistant":     "Metadata Assistant",
	"transcriptomics_expert": "Transcriptomics Expert",
	"annotation_expert":      "Annotation Expert",
	"proteomics_expert":      "Proteomics Expert",
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
// On successful completion it writes a JSON result to stdout or resultPath and returns nil.
// On cancellation it writes {"cancelled":true} and returns a non-nil sentinel.
func Run(themeName string, resultPath string) error {
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
		selectedPackages []string
		provider         string

		apiKey          string
		apiKeySecondary string
		ollamaModel     string
		ollamaChoice    string

		profile string

		wantNCBI                 bool
		ncbiKey                  string
		wantCloud                bool
		cloudKey                 string
		wantSmartStandardization bool
		smartStdOpenAIKey        string
	)

	// ---- Step 1 + 2: Agents + Provider in one form -------------------------
	agentOptions := buildAgentOptions()

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
				Value(&selectedPackages).
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
		return handleAbort(err, resultPath)
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
		status := detectOllamaStatus()
		ollamaOptions := buildOllamaModelOptions(status)
		ollamaChoice = resolveDefaultOllamaChoice(ollamaOptions)
		form2 = huh.NewForm(
			huh.NewGroup(
				huh.NewSelect[string]().
					Title("Ollama Model").
					Description(buildOllamaModelDescription(status)).
					Options(ollamaOptions...).
					Value(&ollamaChoice),
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
		return handleAbort(err, resultPath)
	}

	if provider == providerOllama {
		if ollamaChoice == customOllamaModel {
			ollamaModel = defaultOllamaModel
			customModelForm := huh.NewForm(
				huh.NewGroup(
					huh.NewInput().
						Title("Custom Ollama Model").
						Description("Type the model name exactly as it appears in `ollama list` or `ollama pull`.").
						Placeholder(defaultOllamaModel).
						Value(&ollamaModel).
						Validate(func(value string) error {
							if strings.TrimSpace(value) == "" {
								return fmt.Errorf("enter a model name")
							}
							return nil
						}),
				),
			).WithTheme(ht)

			if err := customModelForm.Run(); err != nil {
				return handleAbort(err, resultPath)
			}
		} else {
			ollamaModel = strings.TrimSpace(ollamaChoice)
		}
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
			return handleAbort(err, resultPath)
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
		return handleAbort(err, resultPath)
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
			return handleAbort(err, resultPath)
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
		return handleAbort(err, resultPath)
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
			return handleAbort(err, resultPath)
		}
	}

	// ---- Step 6: Smart Standardization / vector search --------------------
	selectedAgents := expandSelectedPackages(selectedPackages)
	smartStdDescription := "Map biomedical terms to ontology concepts with OpenAI embeddings + ChromaDB."
	beneficiaries := getSmartStandardizationBeneficiaries(selectedAgents)
	if len(beneficiaries) > 0 {
		smartStdDescription = smartStdDescription + " Helpful for " + strings.Join(beneficiaries, ", ") + "."
	}

	wantSmartStandardization = len(beneficiaries) > 0
	form6 := huh.NewForm(
		huh.NewGroup(
			huh.NewConfirm().
				Title("Smart Standardization").
				Description(smartStdDescription).
				Affirmative("Enable").
				Negative("Skip").
				Value(&wantSmartStandardization),
		),
	).WithTheme(ht)

	if err := form6.Run(); err != nil {
		return handleAbort(err, resultPath)
	}

	if wantSmartStandardization {
		switch {
		case provider == providerOpenAI:
			smartStdOpenAIKey = strings.TrimSpace(apiKey)
		case strings.TrimSpace(os.Getenv("OPENAI_API_KEY")) != "":
			smartStdOpenAIKey = strings.TrimSpace(os.Getenv("OPENAI_API_KEY"))
		default:
			form6b := huh.NewForm(
				huh.NewGroup(
					huh.NewInput().
						Title("OpenAI API Key for Embeddings").
						Description("Required for Smart Standardization / vector search.").
						Placeholder("sk-...").
						EchoMode(huh.EchoModePassword).
						Value(&smartStdOpenAIKey).
						Validate(func(value string) error {
							if strings.TrimSpace(value) == "" {
								return fmt.Errorf("enter an OpenAI API key")
							}
							return nil
						}),
				),
			).WithTheme(ht)

			if err := form6b.Run(); err != nil {
				return handleAbort(err, resultPath)
			}
			smartStdOpenAIKey = strings.TrimSpace(smartStdOpenAIKey)
		}
	}

	// ---- Output ------------------------------------------------------------
	result := WizardResult{
		Provider:                      provider,
		APIKey:                        apiKey,
		APIKeySecondary:               apiKeySecondary,
		Profile:                       profile,
		Agents:                        selectedAgents,
		NCBIKey:                       ncbiKey,
		CloudKey:                      cloudKey,
		OllamaModel:                   strings.TrimSpace(ollamaModel),
		SmartStandardizationEnabled:   wantSmartStandardization,
		SmartStandardizationOpenAIKey: smartStdOpenAIKey,
		Cancelled:                     false,
	}

	return writeResult(result, resultPath)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func buildAgentOptions() []huh.Option[string] {
	options := make([]huh.Option[string], 0, len(initAgentPackages))
	for _, pkg := range initAgentPackages {
		agentCount := len(pkg.Agents)
		agentLabel := "agents"
		if agentCount == 1 {
			agentLabel = "agent"
		}
		label := fmt.Sprintf("%-24s %s (%d %s)", pkg.Name, pkg.Description, agentCount, agentLabel)
		option := huh.NewOption(label, pkg.Name)
		if pkg.Default {
			option = option.Selected(true)
		}
		options = append(options, option)
	}
	return options
}

func expandSelectedPackages(selectedPackages []string) []string {
	seen := map[string]bool{}
	expanded := make([]string, 0, len(selectedPackages))

	for _, selected := range selectedPackages {
		selected = strings.TrimSpace(selected)
		if selected == "" {
			continue
		}

		matchedPackage := false
		for _, pkg := range initAgentPackages {
			if pkg.Name != selected {
				continue
			}
			for _, agent := range pkg.Agents {
				if !seen[agent] {
					expanded = append(expanded, agent)
					seen[agent] = true
				}
			}
			matchedPackage = true
			break
		}
		if matchedPackage {
			continue
		}
		if !seen[selected] {
			expanded = append(expanded, selected)
			seen[selected] = true
		}
	}

	return expanded
}

func getSmartStandardizationBeneficiaries(selectedAgents []string) []string {
	beneficiaries := make([]string, 0, len(selectedAgents))
	for _, agent := range selectedAgents {
		label, ok := smartStandardizationAgentLabels[agent]
		if ok {
			beneficiaries = append(beneficiaries, label)
		}
	}
	return beneficiaries
}

func detectOllamaStatus() ollamaStatus {
	status := ollamaStatus{}

	versionResult, err := exec.Command("ollama", "--version").Output()
	if err != nil {
		return status
	}

	status.Installed = true
	versionText := strings.TrimSpace(string(versionResult))
	if versionText != "" {
		parts := strings.Fields(versionText)
		if len(parts) > 0 {
			status.Version = parts[len(parts)-1]
		}
	}

	listOutput, err := exec.Command("ollama", "list").Output()
	if err != nil {
		return status
	}

	status.Running = true
	for _, line := range strings.Split(string(listOutput), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "NAME") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) == 0 {
			continue
		}
		status.Models = append(status.Models, fields[0])
	}

	return status
}

func buildOllamaModelOptions(status ollamaStatus) []huh.Option[string] {
	options := make([]huh.Option[string], 0, len(status.Models)+len(curatedOllamaModels)+1)
	seen := map[string]bool{}

	for _, model := range status.Models {
		model = strings.TrimSpace(model)
		if model == "" || seen[model] {
			continue
		}
		options = append(options, huh.NewOption(fmt.Sprintf("%-20s detected locally", model), model))
		seen[model] = true
	}

	for _, model := range curatedOllamaModels {
		if seen[model.Name] {
			continue
		}
		options = append(options, huh.NewOption(fmt.Sprintf("%-20s %s", model.Name, model.Description), model.Name))
		seen[model.Name] = true
	}

	options = append(options, huh.NewOption("Other model          Type a custom model name", customOllamaModel))
	return options
}

func resolveDefaultOllamaChoice(options []huh.Option[string]) string {
	for _, option := range options {
		if option.Value == defaultOllamaModel {
			return option.Value
		}
	}
	if len(options) == 0 {
		return defaultOllamaModel
	}
	return options[0].Value
}

func buildOllamaModelDescription(status ollamaStatus) string {
	switch {
	case !status.Installed:
		return "Ollama is not installed yet. Install it from https://ollama.com/download or `curl -fsSL https://ollama.com/install.sh | sh`, then pick a model now or later."
	case !status.Running:
		return "Ollama is installed but not running. Start it with `ollama serve`. You can still save a preferred model now."
	case len(status.Models) > 0 && status.Version != "":
		return fmt.Sprintf("Detected %d local model(s) on Ollama %s. Local models appear first, followed by curated defaults.", len(status.Models), status.Version)
	case len(status.Models) > 0:
		return fmt.Sprintf("Detected %d local model(s). Local models appear first, followed by curated defaults.", len(status.Models))
	default:
		return "No local models were detected yet. Curated models are listed first, plus an option to type another model."
	}
}

// handleAbort converts a huh.ErrUserAborted into a canonical cancelled output.
// All other errors are returned as-is.
func handleAbort(err error, resultPath string) error {
	if errors.Is(err, huh.ErrUserAborted) {
		_ = writeResult(WizardResult{Cancelled: true}, resultPath)
		return fmt.Errorf("cancelled")
	}
	return err
}

// writeResult serialises result to stdout or a result file as compact JSON.
func writeResult(r WizardResult, resultPath string) error {
	writer := os.Stdout
	if resultPath != "" {
		file, err := os.Create(resultPath)
		if err != nil {
			return fmt.Errorf("open result file: %w", err)
		}
		defer file.Close()
		writer = file
	}

	enc := json.NewEncoder(writer)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(r); err != nil {
		return fmt.Errorf("write result: %w", err)
	}
	return nil
}
