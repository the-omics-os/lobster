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
	"fmt"
	"os"
	"os/exec"
	"strings"

	"charm.land/bubbles/v2/textinput"
	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"

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
	ModelID                       string   `json:"model_id"`
	SmartStandardizationEnabled   bool     `json:"smart_standardization_enabled"`
	SmartStandardizationOpenAIKey string   `json:"smart_standardization_openai_key"`
	OAuthAuthenticated            bool     `json:"oauth_authenticated"`
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
	providerOmicsOS    = "omics-os"
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
	Name         string
	Description  string
	Agents       []string
	Default      bool
	Experimental bool
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
		Name:         "lobster-ml",
		Description:  "Machine learning, feature selection, and survival analysis",
		Agents:       []string{"machine_learning_expert", "feature_selection_expert", "survival_analysis_expert"},
		Experimental: true,
	},
	{
		Name:         "lobster-drug-discovery",
		Description:  "Drug discovery, cheminformatics, and translational strategy",
		Agents:       []string{"drug_discovery_expert", "cheminformatics_expert", "clinical_dev_expert", "pharmacogenomics_expert"},
		Experimental: true,
	},
	{
		Name:         "lobster-metadata",
		Description:  "Metadata filtering & standardization",
		Agents:       []string{"metadata_assistant"},
		Experimental: true,
	},
	{
		Name:         "lobster-structural-viz",
		Description:  "Protein structure visualization (PyMOL, PDB)",
		Agents:       []string{"protein_structure_visualization_expert"},
		Experimental: true,
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

// customModelSentinel is the value used for the "Other model" option.
const customModelSentinel = "__custom_model__"

// providerModels lists known models per cloud provider for the model selection step.
// These mirror KNOWN_MODELS from the Python providers — UI catalog only.
var providerModels = map[string][]struct {
	Name        string
	Description string
	IsDefault   bool
}{
	providerAnthropic: {
		{Name: "claude-sonnet-4-20250514", Description: "Claude Sonnet 4 — balanced speed & capability", IsDefault: true},
		{Name: "claude-opus-4-20250514", Description: "Claude Opus 4 — most capable, complex reasoning"},
		{Name: "claude-3-5-sonnet-20241022", Description: "Claude 3.5 Sonnet — previous generation"},
		{Name: "claude-3-5-haiku-20241022", Description: "Claude 3.5 Haiku — fastest, high throughput"},
	},
	providerBedrock: {
		{Name: "us.anthropic.claude-sonnet-4-5-20250929-v1:0", Description: "Claude Sonnet 4.5 — highest quality Sonnet", IsDefault: true},
		{Name: "us.anthropic.claude-sonnet-4-20250514-v1:0", Description: "Claude Sonnet 4 — balanced quality & speed"},
		{Name: "global.anthropic.claude-opus-4-5-20251101-v1:0", Description: "Claude Opus 4.5 — most capable"},
	},
	providerOpenAI: {
		{Name: "gpt-4o", Description: "GPT-4o — most capable GPT model", IsDefault: true},
		{Name: "gpt-4o-mini", Description: "GPT-4o Mini — fast and affordable"},
		{Name: "o3-mini", Description: "o3 Mini — compact reasoning model"},
	},
	providerGemini: {
		{Name: "gemini-3-pro-preview", Description: "Gemini 3 Pro — best balance of speed & capability", IsDefault: true},
		{Name: "gemini-3-flash-preview", Description: "Gemini 3 Flash — fastest, free tier available"},
	},
	providerAzure: {
		{Name: "gpt-4o", Description: "GPT-4o via Azure AI Foundry", IsDefault: true},
		{Name: "deepseek-r1", Description: "DeepSeek R1 reasoning model"},
		{Name: "phi-4", Description: "Microsoft Phi-4 small language model"},
	},
	providerOpenRouter: {
		{Name: "anthropic/claude-sonnet-4-5", Description: "Claude Sonnet 4.5 via OpenRouter", IsDefault: true},
		{Name: "openai/gpt-4o", Description: "GPT-4o via OpenRouter"},
		{Name: "google/gemini-3-pro-preview", Description: "Gemini 3 Pro via OpenRouter"},
	},
}

var smartStandardizationAgentLabels = map[string]string{
	"metadata_assistant":     "Metadata Assistant",
	"transcriptomics_expert": "Transcriptomics Expert",
	"annotation_expert":      "Annotation Expert",
	"proteomics_expert":      "Proteomics Expert",
}

// ---------------------------------------------------------------------------
// Wizard steps
// ---------------------------------------------------------------------------

type wizardStep int

const (
	stepAgentPackages wizardStep = iota
	stepProvider
	stepAPIKey
	stepModelSelect
	stepProfile
	stepOptionalKeys
	stepDone
)

// optionalKeySubStep tracks sub-phases within stepOptionalKeys.
type optionalKeySubStep int

const (
	subStepNCBIConfirm optionalKeySubStep = iota
	subStepNCBIKey
	subStepCloudConfirm
	subStepCloudKey
	subStepSmartStdConfirm
	subStepSmartStdKey
	subStepOptionalDone
)

// ---------------------------------------------------------------------------
// Sub-models: multiSelect
// ---------------------------------------------------------------------------

type multiSelectItem struct {
	label    string
	value    string
	selected bool
}

type multiSelectModel struct {
	title       string
	description string
	items       []multiSelectItem
	cursor      int
}

func (m multiSelectModel) Update(msg tea.Msg) (multiSelectModel, bool, bool) {
	km, ok := msg.(tea.KeyPressMsg)
	if !ok {
		return m, false, false
	}
	switch km.String() {
	case "up", "k":
		if m.cursor > 0 {
			m.cursor--
		}
	case "down", "j":
		if m.cursor < len(m.items)-1 {
			m.cursor++
		}
	case "space":
		m.items[m.cursor].selected = !m.items[m.cursor].selected
	case "enter":
		// Validate at least one selected.
		count := 0
		for _, item := range m.items {
			if item.selected {
				count++
			}
		}
		if count > 0 {
			return m, true, false // done
		}
	case "esc":
		return m, false, true // cancelled
	}
	return m, false, false
}

func (m multiSelectModel) View(th *lobsterTheme.Theme) string {
	titleStyle := lipgloss.NewStyle().Foreground(th.Colors.Primary).Bold(true)
	descStyle := lipgloss.NewStyle().Foreground(th.Colors.TextMuted)
	cursorStyle := lipgloss.NewStyle().Foreground(th.Colors.Accent2)
	selectedStyle := lipgloss.NewStyle().Foreground(th.Colors.Success)
	unselectedStyle := lipgloss.NewStyle().Foreground(th.Colors.Text)

	var sb strings.Builder
	sb.WriteString(titleStyle.Render(m.title) + "\n")
	if m.description != "" {
		sb.WriteString(descStyle.Render(m.description) + "\n")
	}
	sb.WriteString("\n")

	for i, item := range m.items {
		cursor := "  "
		if i == m.cursor {
			cursor = cursorStyle.Render("> ")
		}

		checkbox := "[ ] "
		style := unselectedStyle
		if item.selected {
			checkbox = selectedStyle.Render("[x] ")
			style = selectedStyle
		}

		sb.WriteString(cursor + checkbox + style.Render(item.label) + "\n")
	}

	sb.WriteString("\n" + descStyle.Render("space: toggle  enter: confirm  esc: cancel"))
	return sb.String()
}

func (m multiSelectModel) Selected() []string {
	var result []string
	for _, item := range m.items {
		if item.selected {
			result = append(result, item.value)
		}
	}
	return result
}

// ---------------------------------------------------------------------------
// Sub-models: singleSelect
// ---------------------------------------------------------------------------

type singleSelectItem struct {
	label string
	value string
}

type singleSelectModel struct {
	title       string
	description string
	items       []singleSelectItem
	cursor      int
}

func (m singleSelectModel) Update(msg tea.Msg) (singleSelectModel, string, bool, bool) {
	km, ok := msg.(tea.KeyPressMsg)
	if !ok {
		return m, "", false, false
	}
	switch km.String() {
	case "up", "k":
		if m.cursor > 0 {
			m.cursor--
		}
	case "down", "j":
		if m.cursor < len(m.items)-1 {
			m.cursor++
		}
	case "enter":
		return m, m.items[m.cursor].value, true, false
	case "esc":
		return m, "", false, true
	}
	return m, "", false, false
}

func (m singleSelectModel) View(th *lobsterTheme.Theme) string {
	titleStyle := lipgloss.NewStyle().Foreground(th.Colors.Primary).Bold(true)
	descStyle := lipgloss.NewStyle().Foreground(th.Colors.TextMuted)
	cursorStyle := lipgloss.NewStyle().Foreground(th.Colors.Accent2)
	itemStyle := lipgloss.NewStyle().Foreground(th.Colors.Text)

	var sb strings.Builder
	sb.WriteString(titleStyle.Render(m.title) + "\n")
	if m.description != "" {
		sb.WriteString(descStyle.Render(m.description) + "\n")
	}
	sb.WriteString("\n")

	for i, item := range m.items {
		cursor := "  "
		if i == m.cursor {
			cursor = cursorStyle.Render("> ")
		}
		sb.WriteString(cursor + itemStyle.Render(item.label) + "\n")
	}

	sb.WriteString("\n" + descStyle.Render("enter: select  esc: back"))
	return sb.String()
}

// ---------------------------------------------------------------------------
// Sub-models: confirmModel
// ---------------------------------------------------------------------------

type confirmModel struct {
	title       string
	description string
	value       bool
}

func (m confirmModel) Update(msg tea.Msg) (confirmModel, bool, bool) {
	km, ok := msg.(tea.KeyPressMsg)
	if !ok {
		return m, false, false
	}
	switch km.String() {
	case "y", "Y":
		m.value = true
		return m, true, false
	case "n", "N":
		m.value = false
		return m, true, false
	case "enter":
		return m, true, false
	case "left", "h":
		m.value = true
	case "right", "l":
		m.value = false
	case "esc":
		return m, false, true
	}
	return m, false, false
}

func (m confirmModel) View(th *lobsterTheme.Theme) string {
	titleStyle := lipgloss.NewStyle().Foreground(th.Colors.Primary).Bold(true)
	descStyle := lipgloss.NewStyle().Foreground(th.Colors.TextMuted)
	activeStyle := lipgloss.NewStyle().Foreground(th.Colors.Success).Bold(true)
	inactiveStyle := lipgloss.NewStyle().Foreground(th.Colors.TextMuted)

	var sb strings.Builder
	sb.WriteString(titleStyle.Render(m.title) + "\n")
	if m.description != "" {
		sb.WriteString(descStyle.Render(m.description) + "\n")
	}
	sb.WriteString("\n")

	yes := inactiveStyle.Render("Yes")
	no := inactiveStyle.Render("No")
	if m.value {
		yes = activeStyle.Render("> Yes")
		no = inactiveStyle.Render("  No")
	} else {
		yes = inactiveStyle.Render("  Yes")
		no = activeStyle.Render("> No")
	}
	sb.WriteString(yes + "  " + no + "\n")
	sb.WriteString("\n" + descStyle.Render("y/n: choose  enter: confirm  esc: back"))
	return sb.String()
}

// ---------------------------------------------------------------------------
// WizardModel — main BubbleTea model
// ---------------------------------------------------------------------------

// WizardModel is a BubbleTea model that implements the 5-step init wizard.
type WizardModel struct {
	step    wizardStep
	theme   *lobsterTheme.Theme
	result  WizardResult
	done    bool
	err     error
	resultPath string

	// Step 1: Agent packages
	agentSelect multiSelectModel

	// Step 2: Provider
	providerSelect singleSelectModel

	// Step 3: API key(s)
	apiKeyInputs    []textinput.Model
	apiKeyLabels    []string
	apiKeyFocusIdx  int
	// Ollama model selection
	ollamaSelect    singleSelectModel
	ollamaCustom    textinput.Model
	ollamaIsCustom  bool

	// Cloud provider model selection (shown after API key for non-Ollama)
	modelSelect    singleSelectModel
	modelCustom    textinput.Model
	modelIsCustom  bool

	// Step 4: Profile
	profileSelect singleSelectModel

	// OAuth browser login state (Omics-OS Cloud)
	oauthCancel    chan struct{}
	oauthWaiting   bool
	oauthAuthURL   string // full auth URL for fallback display
	oauthLastError string // last failure reason for display

	// Step 5: Optional keys
	optionalSub    optionalKeySubStep
	ncbiConfirm    confirmModel
	ncbiInput      textinput.Model
	cloudConfirm   confirmModel
	cloudInput     textinput.Model
	smartStdConfirm confirmModel
	smartStdInput   textinput.Model
}

// NewWizardModel creates a WizardModel with the given theme and result path.
func NewWizardModel(themeName string, resultPath string) WizardModel {
	if themeName != "" {
		if err := lobsterTheme.SetTheme(themeName); err != nil {
			fmt.Fprintf(os.Stderr, "warning: %v\n", err)
		}
	}
	activeTheme := lobsterTheme.Current
	if activeTheme == nil {
		_ = lobsterTheme.SetTheme(lobsterTheme.AutoDetect())
		activeTheme = lobsterTheme.Current
	}

	m := WizardModel{
		step:       stepAgentPackages,
		theme:      activeTheme,
		resultPath: resultPath,
	}

	// Build agent multi-select items.
	warningStyle := lipgloss.NewStyle().Foreground(activeTheme.Colors.Warning).Bold(true)
	items := make([]multiSelectItem, 0, len(initAgentPackages))
	for _, pkg := range initAgentPackages {
		agentCount := len(pkg.Agents)
		agentLabel := "agents"
		if agentCount == 1 {
			agentLabel = "agent"
		}
		label := fmt.Sprintf("%-24s %s (%d %s)", pkg.Name, pkg.Description, agentCount, agentLabel)
		if pkg.Experimental {
			label = label + warningStyle.Render(" [experimental]")
		}
		items = append(items, multiSelectItem{
			label:    label,
			value:    pkg.Name,
			selected: pkg.Default,
		})
	}
	m.agentSelect = multiSelectModel{
		title:       "Agent Packages",
		description: "Select the agent packages to install. Use space to toggle.",
		items:       items,
	}

	// Build provider single-select.
	m.providerSelect = singleSelectModel{
		title:       "LLM Provider",
		description: "Select the provider for Lobster AI.",
		items: []singleSelectItem{
			{label: "Omics-OS Cloud      -- Managed Bedrock, login via browser", value: providerOmicsOS},
			{label: "Anthropic (Claude)  -- Quick testing, development", value: providerAnthropic},
			{label: "AWS Bedrock         -- Production, enterprise use", value: providerBedrock},
			{label: "Ollama (Local)      -- Privacy, zero cost, offline", value: providerOllama},
			{label: "Google Gemini       -- Latest models with thinking", value: providerGemini},
			{label: "Azure AI            -- Enterprise Azure deployments", value: providerAzure},
			{label: "OpenAI              -- GPT-4o, reasoning models", value: providerOpenAI},
			{label: "OpenRouter          -- 600+ models via one API key", value: providerOpenRouter},
		},
	}

	return m
}

func (m WizardModel) Init() tea.Cmd {
	return textinput.Blink
}

func (m WizardModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch m.step {
	case stepAgentPackages:
		return m.updateAgentPackages(msg)
	case stepProvider:
		return m.updateProvider(msg)
	case stepAPIKey:
		return m.updateAPIKey(msg)
	case stepModelSelect:
		return m.updateModelSelect(msg)
	case stepProfile:
		return m.updateProfile(msg)
	case stepOptionalKeys:
		return m.updateOptionalKeys(msg)
	case stepDone:
		return m, tea.Quit
	}
	return m, nil
}

func (m WizardModel) View() tea.View {
	var content string
	switch m.step {
	case stepAgentPackages:
		content = m.agentSelect.View(m.theme)
	case stepProvider:
		content = m.providerSelect.View(m.theme)
	case stepAPIKey:
		content = m.viewAPIKey()
	case stepModelSelect:
		content = m.viewModelSelect()
	case stepProfile:
		content = m.profileSelect.View(m.theme)
	case stepOptionalKeys:
		content = m.viewOptionalKeys()
	case stepDone:
		content = ""
	}
	return tea.NewView(content)
}

// ---------------------------------------------------------------------------
// Step 1: Agent packages
// ---------------------------------------------------------------------------

func (m WizardModel) updateAgentPackages(msg tea.Msg) (tea.Model, tea.Cmd) {
	var done, cancelled bool
	m.agentSelect, done, cancelled = m.agentSelect.Update(msg)
	if cancelled {
		return m.cancel()
	}
	if done {
		m.result.Agents = nil // populated at output time
		m.step = stepProvider
	}
	return m, nil
}

// ---------------------------------------------------------------------------
// Step 2: Provider
// ---------------------------------------------------------------------------

func (m WizardModel) updateProvider(msg tea.Msg) (tea.Model, tea.Cmd) {
	var value string
	var done, cancelled bool
	m.providerSelect, value, done, cancelled = m.providerSelect.Update(msg)
	if cancelled {
		m.step = stepAgentPackages
		return m, nil
	}
	if done {
		m.result.Provider = value
		// Reset OAuth state when provider actually changes.
		m.oauthWaiting = false
		m.oauthLastError = ""
		m.oauthAuthURL = ""
		m.result.OAuthAuthenticated = false
		m.initAPIKeyStep()
		m.step = stepAPIKey
		return m, textinput.Blink
	}
	return m, nil
}

// ---------------------------------------------------------------------------
// Step 3: API key(s)
// ---------------------------------------------------------------------------

func (m *WizardModel) initAPIKeyStep() {
	m.apiKeyInputs = nil
	m.apiKeyLabels = nil
	m.apiKeyFocusIdx = 0
	m.ollamaIsCustom = false

	switch m.result.Provider {
	case providerAnthropic:
		ti := textinput.New()
		ti.Placeholder = "sk-ant-..."
		ti.EchoMode = textinput.EchoPassword
		ti.Focus()
		m.apiKeyInputs = []textinput.Model{ti}
		m.apiKeyLabels = []string{"Anthropic API Key"}

	case providerBedrock:
		ti1 := textinput.New()
		ti1.Placeholder = "AKIAIOSFODNN7EXAMPLE"
		ti1.Focus()
		ti2 := textinput.New()
		ti2.Placeholder = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
		ti2.EchoMode = textinput.EchoPassword
		m.apiKeyInputs = []textinput.Model{ti1, ti2}
		m.apiKeyLabels = []string{"AWS Bedrock Access Key ID", "AWS Bedrock Secret Access Key"}

	case providerOllama:
		status := detectOllamaStatus()
		items := buildOllamaSelectItems(status)
		defaultChoice := resolveDefaultOllamaSelectChoice(items)
		m.ollamaSelect = singleSelectModel{
			title:       "Ollama Model",
			description: buildOllamaModelDescription(status),
			items:       items,
			cursor:      defaultChoice,
		}
		// Also prepare custom model input.
		ti := textinput.New()
		ti.Placeholder = defaultOllamaModel
		ti.Focus()
		m.ollamaCustom = ti

	case providerGemini:
		ti := textinput.New()
		ti.Placeholder = "AIzaSy..."
		ti.EchoMode = textinput.EchoPassword
		ti.Focus()
		m.apiKeyInputs = []textinput.Model{ti}
		m.apiKeyLabels = []string{"Google API Key"}

	case providerAzure:
		ti1 := textinput.New()
		ti1.Placeholder = "https://your-resource.openai.azure.com/"
		ti1.Focus()
		ti2 := textinput.New()
		ti2.Placeholder = "your-azure-api-key"
		ti2.EchoMode = textinput.EchoPassword
		m.apiKeyInputs = []textinput.Model{ti1, ti2}
		m.apiKeyLabels = []string{"Azure AI Endpoint URL", "Azure AI API Key"}

	case providerOpenAI:
		ti := textinput.New()
		ti.Placeholder = "sk-..."
		ti.EchoMode = textinput.EchoPassword
		ti.Focus()
		m.apiKeyInputs = []textinput.Model{ti}
		m.apiKeyLabels = []string{"OpenAI API Key"}

	case providerOpenRouter:
		ti := textinput.New()
		ti.Placeholder = "sk-or-..."
		ti.EchoMode = textinput.EchoPassword
		ti.Focus()
		m.apiKeyInputs = []textinput.Model{ti}
		m.apiKeyLabels = []string{"OpenRouter API Key"}

	case providerOmicsOS:
		ti := textinput.New()
		ti.Placeholder = "omk_... (optional — press Enter to login via browser instead)"
		ti.EchoMode = textinput.EchoPassword
		ti.Focus()
		m.apiKeyInputs = []textinput.Model{ti}
		m.apiKeyLabels = []string{"Omics-OS API Key (or press Enter for browser login)"}

	default:
		ti := textinput.New()
		ti.EchoMode = textinput.EchoPassword
		ti.Focus()
		m.apiKeyInputs = []textinput.Model{ti}
		m.apiKeyLabels = []string{"API Key"}
	}
}

func (m WizardModel) updateAPIKey(msg tea.Msg) (tea.Model, tea.Cmd) {
	if m.result.Provider == providerOllama {
		return m.updateOllamaKey(msg)
	}

	// Handle OAuth async messages when waiting for browser login.
	if m.oauthWaiting {
		switch msg := msg.(type) {
		case oauthSucceededMsg:
			m.oauthWaiting = false
			m.oauthLastError = ""
			close(m.oauthCancel)
			// Save credentials now (after UI accepts, not inside tea.Cmd).
			if err := saveOAuthCredentials(msg); err != nil {
				m.oauthLastError = "Failed to save credentials: " + err.Error()
				if len(m.apiKeyInputs) > 0 {
					m.apiKeyInputs[m.apiKeyFocusIdx].Focus()
				}
				return m, textinput.Blink
			}
			m.result.OAuthAuthenticated = true
			// Email is NOT in the browser callback — Python postprocessing
			// validates the token against the gateway and enriches credentials.
			m.advanceFromAPIKey()
			return m, nil
		case oauthFailedMsg:
			m.oauthWaiting = false
			close(m.oauthCancel)
			m.oauthLastError = msg.Reason
			// Refocus the text input for manual key paste.
			if len(m.apiKeyInputs) > 0 {
				m.apiKeyInputs[m.apiKeyFocusIdx].Focus()
			}
			return m, textinput.Blink
		case tea.KeyPressMsg:
			if msg.String() == "esc" || msg.String() == "ctrl+c" {
				close(m.oauthCancel)
				m.oauthWaiting = false
				m.step = stepProvider
				return m, nil
			}
			return m, nil // Ignore other keys while waiting.
		default:
			return m, nil
		}
	}

	km, ok := msg.(tea.KeyPressMsg)
	if ok {
		switch km.String() {
		case "enter":
			if m.apiKeyFocusIdx < len(m.apiKeyInputs)-1 {
				m.apiKeyInputs[m.apiKeyFocusIdx].Blur()
				m.apiKeyFocusIdx++
				m.apiKeyInputs[m.apiKeyFocusIdx].Focus()
				return m, textinput.Blink
			}
			// Submit all inputs.
			m.collectAPIKeys()

			// Omics-OS with empty key → start browser OAuth.
			// Skip if already authenticated (back-navigation guard).
			if m.result.Provider == providerOmicsOS && strings.TrimSpace(m.result.APIKey) == "" {
				if m.result.OAuthAuthenticated {
					m.advanceFromAPIKey()
					return m, textinput.Blink
				}
				// Pre-bind port and generate state in main goroutine so the
				// auth URL is available for display immediately.
				params, reason := prepareOAuthFlow()
				if params == nil {
					m.oauthLastError = reason
					return m, textinput.Blink
				}
				m.oauthCancel = make(chan struct{})
				m.oauthWaiting = true
				m.oauthLastError = ""
				m.oauthAuthURL = params.AuthURL
				return m, startOAuthFlow(params, m.oauthCancel)
			}

			m.advanceFromAPIKey()
			return m, textinput.Blink
		case "tab":
			if m.apiKeyFocusIdx < len(m.apiKeyInputs)-1 {
				m.apiKeyInputs[m.apiKeyFocusIdx].Blur()
				m.apiKeyFocusIdx++
				m.apiKeyInputs[m.apiKeyFocusIdx].Focus()
				return m, textinput.Blink
			}
		case "shift+tab":
			if m.apiKeyFocusIdx > 0 {
				m.apiKeyInputs[m.apiKeyFocusIdx].Blur()
				m.apiKeyFocusIdx--
				m.apiKeyInputs[m.apiKeyFocusIdx].Focus()
				return m, textinput.Blink
			}
		case "esc":
			m.step = stepProvider
			return m, nil
		}
	}

	// Forward to active input.
	var cmd tea.Cmd
	m.apiKeyInputs[m.apiKeyFocusIdx], cmd = m.apiKeyInputs[m.apiKeyFocusIdx].Update(msg)
	return m, cmd
}

func (m *WizardModel) collectAPIKeys() {
	switch m.result.Provider {
	case providerBedrock:
		m.result.APIKey = m.apiKeyInputs[0].Value()
		m.result.APIKeySecondary = m.apiKeyInputs[1].Value()
	case providerAzure:
		m.result.APIKeySecondary = m.apiKeyInputs[0].Value()
		m.result.APIKey = m.apiKeyInputs[1].Value()
	default:
		if len(m.apiKeyInputs) > 0 {
			m.result.APIKey = m.apiKeyInputs[0].Value()
		}
	}
}

func (m *WizardModel) advanceFromAPIKey() {
	// Show model selection for providers with a known model catalog.
	if models, ok := providerModels[m.result.Provider]; ok && len(models) > 0 {
		m.initModelSelectStep(models)
		m.step = stepModelSelect
		return
	}
	m.advanceFromModelSelect()
}

func (m *WizardModel) initModelSelectStep(models []struct {
	Name        string
	Description string
	IsDefault   bool
}) {
	items := make([]singleSelectItem, 0, len(models)+1)
	defaultIdx := 0
	for i, model := range models {
		label := fmt.Sprintf("%-45s %s", model.Name, model.Description)
		items = append(items, singleSelectItem{label: label, value: model.Name})
		if model.IsDefault {
			defaultIdx = i
		}
	}
	items = append(items, singleSelectItem{
		label: "Other model (enter custom ID)",
		value: customModelSentinel,
	})
	m.modelSelect = singleSelectModel{
		title:       "Default Model",
		description: "Select the default model. You can change this later with /config model.",
		items:       items,
		cursor:      defaultIdx,
	}
	ti := textinput.New()
	ti.Placeholder = models[defaultIdx].Name
	ti.Focus()
	m.modelCustom = ti
	m.modelIsCustom = false
}

func (m *WizardModel) advanceFromModelSelect() {
	if providersWithProfile[m.result.Provider] {
		m.profileSelect = singleSelectModel{
			title:       "Performance Profile",
			description: "Choose a profile that balances cost and capability.",
			items: []singleSelectItem{
				{label: "Development  -- Sonnet 4, fastest and most affordable", value: "development"},
				{label: "Production   -- Sonnet 4 + Sonnet 4.5 supervisor [Recommended]", value: "production"},
				{label: "Performance  -- Sonnet 4.5, highest quality", value: "performance"},
				{label: "Max          -- Opus 4.5 supervisor, most capable, expensive", value: "max"},
			},
			cursor: 1, // Default to production
		}
		m.step = stepProfile
	} else {
		m.initOptionalKeys()
		m.step = stepOptionalKeys
	}
}

func (m WizardModel) updateOllamaKey(msg tea.Msg) (tea.Model, tea.Cmd) {
	if m.ollamaIsCustom {
		// Custom model text input.
		km, ok := msg.(tea.KeyPressMsg)
		if ok {
			switch km.String() {
			case "enter":
				val := strings.TrimSpace(m.ollamaCustom.Value())
				if val == "" {
					val = defaultOllamaModel
				}
				m.result.OllamaModel = val
				m.advanceFromAPIKey()
				return m, textinput.Blink
			case "esc":
				m.ollamaIsCustom = false
				return m, nil
			}
		}
		var cmd tea.Cmd
		m.ollamaCustom, cmd = m.ollamaCustom.Update(msg)
		return m, cmd
	}

	var value string
	var done, cancelled bool
	m.ollamaSelect, value, done, cancelled = m.ollamaSelect.Update(msg)
	if cancelled {
		m.step = stepProvider
		return m, nil
	}
	if done {
		if value == customOllamaModel {
			m.ollamaIsCustom = true
			m.ollamaCustom.Focus()
			return m, textinput.Blink
		}
		m.result.OllamaModel = strings.TrimSpace(value)
		m.advanceFromAPIKey()
		return m, textinput.Blink
	}
	return m, nil
}

// ---------------------------------------------------------------------------
// Step 3b: Model selection (non-Ollama providers)
// ---------------------------------------------------------------------------

func (m WizardModel) updateModelSelect(msg tea.Msg) (tea.Model, tea.Cmd) {
	if m.modelIsCustom {
		km, ok := msg.(tea.KeyPressMsg)
		if ok {
			switch km.String() {
			case "enter":
				val := strings.TrimSpace(m.modelCustom.Value())
				if val != "" {
					m.result.ModelID = val
				}
				m.advanceFromModelSelect()
				return m, textinput.Blink
			case "esc":
				m.modelIsCustom = false
				return m, nil
			}
		}
		var cmd tea.Cmd
		m.modelCustom, cmd = m.modelCustom.Update(msg)
		return m, cmd
	}

	var value string
	var done, cancelled bool
	m.modelSelect, value, done, cancelled = m.modelSelect.Update(msg)
	if cancelled {
		m.step = stepAPIKey
		return m, nil
	}
	if done {
		if value == customModelSentinel {
			m.modelIsCustom = true
			m.modelCustom.Focus()
			return m, textinput.Blink
		}
		m.result.ModelID = strings.TrimSpace(value)
		m.advanceFromModelSelect()
		return m, textinput.Blink
	}
	return m, nil
}

func (m WizardModel) viewModelSelect() string {
	if m.modelIsCustom {
		titleStyle := lipgloss.NewStyle().Foreground(m.theme.Colors.Primary).Bold(true)
		descStyle := lipgloss.NewStyle().Foreground(m.theme.Colors.TextMuted)
		var sb strings.Builder
		sb.WriteString(titleStyle.Render("Custom Model ID") + "\n")
		sb.WriteString(descStyle.Render("Enter the model ID exactly as your provider expects it.") + "\n\n")
		sb.WriteString(m.modelCustom.View() + "\n")
		sb.WriteString("\n" + descStyle.Render("enter: confirm  esc: back"))
		return sb.String()
	}
	return m.modelSelect.View(m.theme)
}

func (m WizardModel) viewAPIKey() string {
	if m.oauthWaiting {
		titleStyle := lipgloss.NewStyle().Foreground(m.theme.Colors.Primary).Bold(true)
		descStyle := lipgloss.NewStyle().Foreground(m.theme.Colors.TextMuted)
		accentStyle := lipgloss.NewStyle().Foreground(m.theme.Colors.Accent2)
		var sb strings.Builder
		sb.WriteString(titleStyle.Render("Omics-OS Cloud Login") + "\n\n")
		sb.WriteString(accentStyle.Render("◐") + " Waiting for browser login...\n\n")
		if m.oauthAuthURL != "" {
			sb.WriteString(descStyle.Render("If the browser didn't open, visit:") + "\n")
			sb.WriteString(descStyle.Render(m.oauthAuthURL) + "\n\n")
		}
		sb.WriteString(descStyle.Render("esc: cancel"))
		return sb.String()
	}

	if m.result.Provider == providerOllama {
		if m.ollamaIsCustom {
			titleStyle := lipgloss.NewStyle().Foreground(m.theme.Colors.Primary).Bold(true)
			descStyle := lipgloss.NewStyle().Foreground(m.theme.Colors.TextMuted)
			var sb strings.Builder
			sb.WriteString(titleStyle.Render("Custom Ollama Model") + "\n")
			sb.WriteString(descStyle.Render("Type the model name exactly as it appears in `ollama list` or `ollama pull`.") + "\n\n")
			sb.WriteString(m.ollamaCustom.View() + "\n")
			sb.WriteString("\n" + descStyle.Render("enter: confirm  esc: back"))
			return sb.String()
		}
		return m.ollamaSelect.View(m.theme)
	}

	titleStyle := lipgloss.NewStyle().Foreground(m.theme.Colors.Primary).Bold(true)
	descStyle := lipgloss.NewStyle().Foreground(m.theme.Colors.TextMuted)
	labelStyle := lipgloss.NewStyle().Foreground(m.theme.Colors.Accent2)
	focusIndicator := lipgloss.NewStyle().Foreground(m.theme.Colors.Success)

	var sb strings.Builder
	sb.WriteString(titleStyle.Render("API Keys") + "\n")

	// Show OAuth failure hint when returning from failed browser login.
	if m.oauthLastError != "" && m.result.Provider == providerOmicsOS {
		warnStyle := lipgloss.NewStyle().Foreground(m.theme.Colors.Warning)
		sb.WriteString(warnStyle.Render("Browser login failed: "+m.oauthLastError+". Paste a key instead.") + "\n")
	}
	sb.WriteString("\n")

	for i, label := range m.apiKeyLabels {
		indicator := "  "
		if i == m.apiKeyFocusIdx {
			indicator = focusIndicator.Render("> ")
		}
		sb.WriteString(indicator + labelStyle.Render(label) + "\n")
		sb.WriteString("  " + m.apiKeyInputs[i].View() + "\n\n")
	}

	sb.WriteString(descStyle.Render("enter: confirm  tab: next field  esc: back"))
	return sb.String()
}

// ---------------------------------------------------------------------------
// Step 4: Profile
// ---------------------------------------------------------------------------

func (m WizardModel) updateProfile(msg tea.Msg) (tea.Model, tea.Cmd) {
	var value string
	var done, cancelled bool
	m.profileSelect, value, done, cancelled = m.profileSelect.Update(msg)
	if cancelled {
		// Go back to model select if provider has a model catalog, else API key.
		if _, ok := providerModels[m.result.Provider]; ok {
			m.step = stepModelSelect
		} else {
			m.step = stepAPIKey
			m.initAPIKeyStep()
		}
		return m, textinput.Blink
	}
	if done {
		m.result.Profile = value
		m.initOptionalKeys()
		m.step = stepOptionalKeys
		return m, textinput.Blink
	}
	return m, nil
}

// ---------------------------------------------------------------------------
// Step 5: Optional keys
// ---------------------------------------------------------------------------

func (m *WizardModel) initOptionalKeys() {
	m.optionalSub = subStepNCBIConfirm
	m.ncbiConfirm = confirmModel{
		title:       "NCBI API Key",
		description: "Add an NCBI API key for enhanced PubMed / literature search? (Recommended for heavy use)",
		value:       false,
	}
	ti := textinput.New()
	ti.Placeholder = "your-ncbi-api-key"
	ti.EchoMode = textinput.EchoPassword
	m.ncbiInput = ti

	m.cloudConfirm = confirmModel{
		title:       "Omics-OS Cloud API Key",
		description: "Add an Omics-OS Cloud API key to unlock premium features?",
		value:       false,
	}
	ti2 := textinput.New()
	ti2.Placeholder = "omics-..."
	ti2.EchoMode = textinput.EchoPassword
	m.cloudInput = ti2

	// Smart standardization default depends on selected agents.
	selectedPackages := m.agentSelect.Selected()
	selectedAgents := expandSelectedPackages(selectedPackages)
	beneficiaries := getSmartStandardizationBeneficiaries(selectedAgents)
	desc := "Map biomedical terms to ontology concepts with OpenAI embeddings + ChromaDB."
	if len(beneficiaries) > 0 {
		desc = desc + " Helpful for " + strings.Join(beneficiaries, ", ") + "."
	}
	m.smartStdConfirm = confirmModel{
		title:       "Smart Standardization",
		description: desc,
		value:       len(beneficiaries) > 0,
	}
	ti3 := textinput.New()
	ti3.Placeholder = "sk-..."
	ti3.EchoMode = textinput.EchoPassword
	m.smartStdInput = ti3
}

func (m WizardModel) updateOptionalKeys(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch m.optionalSub {
	case subStepNCBIConfirm:
		var done, cancelled bool
		m.ncbiConfirm, done, cancelled = m.ncbiConfirm.Update(msg)
		if cancelled {
			// Go back to previous step.
			if providersWithProfile[m.result.Provider] {
				m.step = stepProfile
			} else {
				m.step = stepAPIKey
				m.initAPIKeyStep()
			}
			return m, textinput.Blink
		}
		if done {
			if m.ncbiConfirm.value {
				m.optionalSub = subStepNCBIKey
				m.ncbiInput.Focus()
				return m, textinput.Blink
			}
			// Skip cloud key when already authenticated via Omics-OS Cloud.
			if m.result.Provider == providerOmicsOS {
				m.optionalSub = subStepSmartStdConfirm
			} else {
				m.optionalSub = subStepCloudConfirm
			}
		}
		return m, nil

	case subStepNCBIKey:
		km, ok := msg.(tea.KeyPressMsg)
		if ok {
			switch km.String() {
			case "enter":
				m.result.NCBIKey = m.ncbiInput.Value()
				// Skip cloud key when already authenticated via Omics-OS Cloud.
				if m.result.Provider == providerOmicsOS {
					m.optionalSub = subStepSmartStdConfirm
				} else {
					m.optionalSub = subStepCloudConfirm
				}
				return m, nil
			case "esc":
				m.optionalSub = subStepNCBIConfirm
				return m, nil
			}
		}
		var cmd tea.Cmd
		m.ncbiInput, cmd = m.ncbiInput.Update(msg)
		return m, cmd

	case subStepCloudConfirm:
		var done, cancelled bool
		m.cloudConfirm, done, cancelled = m.cloudConfirm.Update(msg)
		if cancelled {
			m.optionalSub = subStepNCBIConfirm
			return m, nil
		}
		if done {
			if m.cloudConfirm.value {
				m.optionalSub = subStepCloudKey
				m.cloudInput.Focus()
				return m, textinput.Blink
			}
			m.optionalSub = subStepSmartStdConfirm
		}
		return m, nil

	case subStepCloudKey:
		km, ok := msg.(tea.KeyPressMsg)
		if ok {
			switch km.String() {
			case "enter":
				m.result.CloudKey = m.cloudInput.Value()
				m.optionalSub = subStepSmartStdConfirm
				return m, nil
			case "esc":
				m.optionalSub = subStepCloudConfirm
				return m, nil
			}
		}
		var cmd tea.Cmd
		m.cloudInput, cmd = m.cloudInput.Update(msg)
		return m, cmd

	case subStepSmartStdConfirm:
		var done, cancelled bool
		m.smartStdConfirm, done, cancelled = m.smartStdConfirm.Update(msg)
		if cancelled {
			// Go back to NCBI when omics-os (cloud step is skipped).
			if m.result.Provider == providerOmicsOS {
				m.optionalSub = subStepNCBIConfirm
			} else {
				m.optionalSub = subStepCloudConfirm
			}
			return m, nil
		}
		if done {
			m.result.SmartStandardizationEnabled = m.smartStdConfirm.value
			if m.smartStdConfirm.value {
				// Check if we already have an OpenAI key.
				switch {
				case m.result.Provider == providerOpenAI:
					m.result.SmartStandardizationOpenAIKey = strings.TrimSpace(m.result.APIKey)
					return m.finishWizard()
				case strings.TrimSpace(os.Getenv("OPENAI_API_KEY")) != "":
					m.result.SmartStandardizationOpenAIKey = strings.TrimSpace(os.Getenv("OPENAI_API_KEY"))
					return m.finishWizard()
				default:
					m.optionalSub = subStepSmartStdKey
					m.smartStdInput.Focus()
					return m, textinput.Blink
				}
			}
			return m.finishWizard()
		}
		return m, nil

	case subStepSmartStdKey:
		km, ok := msg.(tea.KeyPressMsg)
		if ok {
			switch km.String() {
			case "enter":
				val := strings.TrimSpace(m.smartStdInput.Value())
				if val == "" {
					return m, nil // require non-empty
				}
				m.result.SmartStandardizationOpenAIKey = val
				return m.finishWizard()
			case "esc":
				m.optionalSub = subStepSmartStdConfirm
				return m, nil
			}
		}
		var cmd tea.Cmd
		m.smartStdInput, cmd = m.smartStdInput.Update(msg)
		return m, cmd
	}

	return m, nil
}

func (m WizardModel) viewOptionalKeys() string {
	switch m.optionalSub {
	case subStepNCBIConfirm:
		return m.ncbiConfirm.View(m.theme)
	case subStepNCBIKey:
		return m.viewTextInput("NCBI API Key", "Your NCBI API key. Find it at ncbi.nlm.nih.gov/account/settings.", m.ncbiInput)
	case subStepCloudConfirm:
		return m.cloudConfirm.View(m.theme)
	case subStepCloudKey:
		return m.viewTextInput("Omics-OS Cloud API Key", "Your Omics-OS Cloud key. Find it at app.omics-os.com/settings.", m.cloudInput)
	case subStepSmartStdConfirm:
		return m.smartStdConfirm.View(m.theme)
	case subStepSmartStdKey:
		return m.viewTextInput("OpenAI API Key for Embeddings", "Required for Smart Standardization / vector search.", m.smartStdInput)
	}
	return ""
}

func (m WizardModel) viewTextInput(title, desc string, ti textinput.Model) string {
	titleStyle := lipgloss.NewStyle().Foreground(m.theme.Colors.Primary).Bold(true)
	descStyle := lipgloss.NewStyle().Foreground(m.theme.Colors.TextMuted)
	var sb strings.Builder
	sb.WriteString(titleStyle.Render(title) + "\n")
	sb.WriteString(descStyle.Render(desc) + "\n\n")
	sb.WriteString(ti.View() + "\n")
	sb.WriteString("\n" + descStyle.Render("enter: confirm  esc: back"))
	return sb.String()
}

// ---------------------------------------------------------------------------
// Finish / Cancel
// ---------------------------------------------------------------------------

func (m WizardModel) finishWizard() (tea.Model, tea.Cmd) {
	// Expand selected packages to agent names.
	selectedPackages := m.agentSelect.Selected()
	m.result.Agents = expandSelectedPackages(selectedPackages)
	m.result.OllamaModel = strings.TrimSpace(m.result.OllamaModel)
	// For Ollama, copy OllamaModel to ModelID so Python has one field to check.
	if m.result.Provider == providerOllama && m.result.ModelID == "" && m.result.OllamaModel != "" {
		m.result.ModelID = m.result.OllamaModel
	}
	m.result.Cancelled = false
	m.done = true
	m.step = stepDone
	return m, tea.Quit
}

func (m WizardModel) cancel() (tea.Model, tea.Cmd) {
	m.result = WizardResult{Cancelled: true}
	m.done = true
	m.err = fmt.Errorf("cancelled")
	m.step = stepDone
	return m, tea.Quit
}

// Result returns the wizard result and error after the program finishes.
func (m WizardModel) Result() (WizardResult, error) {
	return m.result, m.err
}

// ---------------------------------------------------------------------------
// Run — public entry point
// ---------------------------------------------------------------------------

// Run launches the init wizard with the given theme name.
// Pass an empty string or "lobster-dark" to use the default dark theme.
// On successful completion it writes a JSON result to stdout or resultPath and returns nil.
// On cancellation it writes {"cancelled":true} and returns a non-nil sentinel.
func Run(themeName string, resultPath string) error {
	model := NewWizardModel(themeName, resultPath)

	p := tea.NewProgram(model)
	finalModel, err := p.Run()
	if err != nil {
		return fmt.Errorf("wizard: %w", err)
	}

	wm, ok := finalModel.(WizardModel)
	if !ok {
		return fmt.Errorf("wizard: unexpected model type")
	}

	result, wizErr := wm.Result()
	return writeResult(result, resultPath, wizErr)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

func buildOllamaSelectItems(status ollamaStatus) []singleSelectItem {
	items := make([]singleSelectItem, 0, len(status.Models)+len(curatedOllamaModels)+1)
	seen := map[string]bool{}

	for _, model := range status.Models {
		model = strings.TrimSpace(model)
		if model == "" || seen[model] {
			continue
		}
		items = append(items, singleSelectItem{
			label: fmt.Sprintf("%-20s detected locally", model),
			value: model,
		})
		seen[model] = true
	}

	for _, model := range curatedOllamaModels {
		if seen[model.Name] {
			continue
		}
		items = append(items, singleSelectItem{
			label: fmt.Sprintf("%-20s %s", model.Name, model.Description),
			value: model.Name,
		})
		seen[model.Name] = true
	}

	items = append(items, singleSelectItem{
		label: "Other model          Type a custom model name",
		value: customOllamaModel,
	})
	return items
}

func resolveDefaultOllamaSelectChoice(items []singleSelectItem) int {
	for i, item := range items {
		if item.value == defaultOllamaModel {
			return i
		}
	}
	return 0
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

// writeResult serialises result to stdout or a result file as compact JSON.
func writeResult(r WizardResult, resultPath string, wizErr error) error {
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
	return wizErr
}
