// Package sampleplate adapts Crust's platemap into a BioComp component that
// lets the user assign one of a list of choices to each sample on a plate.
// It draws the ask_user "plate_map" view (a choice field with for_each).
//
// Crust's platemap draws the plate and handles navigation; assignment, the
// submit rule and the reply shape live here, so Crust stays unchanged.
// Number keys 1-9 assign the 1st-9th choice; 0 marks a sample unresolved.
//
// Reply: {"assignments": {sample_id: choice}, "changes": [...]}.
package sampleplate

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"charm.land/bubbles/v2/key"
	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"

	"github.com/the-omics-os/crust"
	"github.com/the-omics-os/crust/platemap"

	"github.com/the-omics-os/lobster-tui/internal/biocomp"
)

const unknown = "unknown"

func init() {
	biocomp.Register("plate_map", func() biocomp.BioComponent {
		return &SamplePlateComponent{}
	})
}

// sampleWell is one sample placed on a well. The JSON tags are required:
// Crust's own structs have none, and Go would not match snake_case keys.
type sampleWell struct {
	Row       int    `json:"row"`
	Col       int    `json:"col"`
	SampleID  string `json:"sample_id"`
	Label     string `json:"label"`
	Condition string `json:"condition"`
	Batch     string `json:"batch"`
	Warning   string `json:"warning"`
	// From a data reference: the row's other columns as stored, and the
	// model's own flag and suggestion (shown as the model's, never as data).
	Details    map[string]string `json:"details"`
	Flag       string            `json:"flag"`
	Suggestion string            `json:"suggestion"`
}

type sampleplateData struct {
	Question string       `json:"question"`
	Title    string       `json:"title"`
	Format   int          `json:"format"`
	Choices  []string     `json:"choices"`
	Wells    []sampleWell `json:"wells"`
}

// SamplePlateComponent wraps a Crust platemap.Model.
type SamplePlateComponent struct {
	plate       platemap.Model
	question    string
	title       string
	format      platemap.PlateFormat
	choices     []string
	samples     []sampleWell      // in the order supplied
	assignments map[string]string // sample_id -> choice or "unknown"
	reviewed    map[string]bool   // flagged samples the user has assigned
	message     string            // feedback line, e.g. why enter was refused
}

func (c *SamplePlateComponent) Init(data json.RawMessage) error {
	var d sampleplateData
	if err := json.Unmarshal(data, &d); err != nil {
		return fmt.Errorf("plate_map: invalid data: %w", err)
	}
	if len(d.Wells) == 0 {
		return fmt.Errorf("plate_map: at least one well is required")
	}

	format := platemap.PlateFormat(d.Format)
	if d.Format == 0 {
		format = platemap.Plate96
	}
	if !format.Valid() {
		return fmt.Errorf("plate_map: format must be 96, 384 or 1536, got %d", d.Format)
	}

	choices := d.Choices
	if len(choices) == 0 || len(choices) > 9 {
		return fmt.Errorf("plate_map: need 1 to 9 choices (one number key each), got %d", len(choices))
	}

	seenIDs := map[string]bool{}
	seenWells := map[[2]int]bool{}
	assignments := map[string]string{}
	for _, w := range d.Wells {
		if w.SampleID == "" {
			return fmt.Errorf("plate_map: every well needs a sample_id")
		}
		if seenIDs[w.SampleID] {
			return fmt.Errorf("plate_map: duplicate sample_id %q", w.SampleID)
		}
		if w.Row < 0 || w.Col < 0 || w.Row >= format.Rows() || w.Col >= format.Cols() {
			return fmt.Errorf("plate_map: sample %q at row %d col %d is outside a %d-well plate",
				w.SampleID, w.Row, w.Col, int(format))
		}
		if seenWells[[2]int{w.Row, w.Col}] {
			return fmt.Errorf("plate_map: two samples share row %d col %d", w.Row, w.Col)
		}
		seenIDs[w.SampleID] = true
		seenWells[[2]int{w.Row, w.Col}] = true
		assignments[w.SampleID] = startingAssignment(w.Condition, choices)
	}

	c.question = d.Question
	c.title = d.Title
	if c.title == "" {
		c.title = "Sample conditions"
	}
	c.format = format
	c.choices = choices
	c.samples = append([]sampleWell(nil), d.Wells...)
	c.assignments = assignments
	c.reviewed = map[string]bool{}
	c.message = ""

	wells := make([]platemap.Well, 0, len(c.samples))
	for _, s := range c.samples {
		wells = append(wells, c.toCrustWell(s))
	}
	c.plate = platemap.New(
		platemap.WithPlate(platemap.PlateData{Format: format, Title: c.title, Wells: wells}),
		platemap.WithViewMode(platemap.ViewHitClass),
		platemap.WithCursor(c.samples[0].Row, c.samples[0].Col),
	)
	return nil
}

// startingAssignment keeps a supplied condition only if it is one of the
// choices; anything else (including "unknown") must be resolved by the user.
func startingAssignment(condition string, choices []string) string {
	cond := strings.ToLower(strings.TrimSpace(condition))
	for _, choice := range choices {
		if cond == choice {
			return choice
		}
	}
	return unknown
}

// toCrustWell maps a sample onto Crust's fixed Well fields. Crust has no field
// for an arbitrary condition, so the current assignment and the row's stored
// columns are shown in Reagent (visible in the plate's footer), and samples
// that still need attention are marked as hits.
func (c *SamplePlateComponent) toCrustWell(s sampleWell) platemap.Well {
	parts := []string{}
	if s.Label != "" {
		parts = append(parts, s.Label)
	}
	if s.Batch != "" {
		parts = append(parts, s.Batch)
	}
	parts = append(parts, "now: "+c.assignments[s.SampleID])
	if s.Warning != "" {
		parts = append(parts, "warning: "+s.Warning)
	}
	keys := make([]string, 0, len(s.Details))
	for k := range s.Details {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		if v := s.Details[k]; v != "" {
			parts = append(parts, k+"="+v)
		}
	}
	return platemap.Well{
		Row:      s.Row,
		Col:      s.Col,
		Control:  platemap.ControlSample,
		SampleID: s.SampleID,
		Reagent:  strings.Join(parts, " · "),
		Hit:      c.needsAttention(s),
	}
}

// needsAttention: the value is unresolved, or the model flagged the sample
// and the user has not yet assigned it.
func (c *SamplePlateComponent) needsAttention(s sampleWell) bool {
	return c.assignments[s.SampleID] == unknown || (s.Flag != "" && !c.reviewed[s.SampleID])
}

func (c *SamplePlateComponent) sampleAtCursor() (sampleWell, bool) {
	row, col := c.plate.Cursor()
	for _, s := range c.samples {
		if s.Row == row && s.Col == col {
			return s, true
		}
	}
	return sampleWell{}, false
}

// keyChoice maps a number key to a choice: 1 is the first choice, 0 resets
// the sample to unresolved.
func (c *SamplePlateComponent) keyChoice(k string) (string, bool) {
	if len(k) != 1 || k[0] < '0' || k[0] > '9' {
		return "", false
	}
	if k == "0" {
		return unknown, true
	}
	i := int(k[0] - '1')
	if i >= len(c.choices) {
		return "", false
	}
	return c.choices[i], true
}

// legend lists the assignment keys, e.g. "1 control · 2 treated · 0 unresolved".
func (c *SamplePlateComponent) legend() string {
	parts := make([]string, 0, len(c.choices)+1)
	for i, choice := range c.choices {
		parts = append(parts, fmt.Sprintf("%d %s", i+1, choice))
	}
	return strings.Join(append(parts, "0 unresolved"), " · ")
}

func (c *SamplePlateComponent) unresolved() []string {
	var ids []string
	for _, s := range c.samples {
		if c.assignments[s.SampleID] == unknown {
			ids = append(ids, s.SampleID)
		}
	}
	return ids
}

func (c *SamplePlateComponent) unreviewed() []string {
	var ids []string
	for _, s := range c.samples {
		if s.Flag != "" && !c.reviewed[s.SampleID] {
			ids = append(ids, s.SampleID)
		}
	}
	return ids
}

func (c *SamplePlateComponent) HandleMsg(msg tea.Msg) (*biocomp.ComponentResult, tea.Cmd) {
	if km, ok := msg.(tea.KeyPressMsg); ok {
		k := km.String()
		if choice, ok := c.keyChoice(k); ok {
			s, found := c.sampleAtCursor()
			if !found {
				c.message = "No sample in this well."
				return nil, nil
			}
			c.assignments[s.SampleID] = choice
			c.reviewed[s.SampleID] = choice != unknown
			c.plate.UpsertWell(c.toCrustWell(s))
			c.message = fmt.Sprintf("%s set to %s.", s.SampleID, choice)
			return nil, nil
		}
		if k == "enter" {
			// Crust would submit the focused well; we submit all assignments.
			if ids := c.unresolved(); len(ids) > 0 {
				c.message = "Resolve before submitting: " + strings.Join(ids, ", ")
				return nil, nil
			}
			if ids := c.unreviewed(); len(ids) > 0 {
				c.message = "Review flagged samples (press a number on each): " + strings.Join(ids, ", ")
				return nil, nil
			}
			return c.submitResult(), nil
		}
	}

	next, cmd := c.plate.Update(msg)
	c.plate = next.(platemap.Model)
	if cmd == nil {
		return nil, nil
	}
	// Safe to run immediately: platemap's commands only return a message.
	if cancel, ok := cmd().(crust.CancelMsg); ok {
		return &biocomp.ComponentResult{
			Action: "cancel",
			Data:   map[string]any{"reason": cancel.Reason},
		}, nil
	}
	return nil, nil
}

// submitResult returns every assignment plus the samples whose supplied
// condition was changed, so the original is never silently replaced.
func (c *SamplePlateComponent) submitResult() *biocomp.ComponentResult {
	assignments := map[string]any{}
	changes := []any{}
	reviewed := []any{}
	for _, s := range c.samples {
		now := c.assignments[s.SampleID]
		assignments[s.SampleID] = now
		if supplied := strings.ToLower(strings.TrimSpace(s.Condition)); supplied != now {
			changes = append(changes, map[string]any{
				"sample_id": s.SampleID,
				"supplied":  s.Condition,
				"assigned":  now,
			})
		}
		if s.Flag != "" {
			reviewed = append(reviewed, map[string]any{
				"sample_id": s.SampleID,
				"flag":      s.Flag,
				"assigned":  now,
			})
		}
	}
	return &biocomp.ComponentResult{
		Action: "submit",
		Data:   map[string]any{"assignments": assignments, "changes": changes, "reviewed_flags": reviewed},
	}
}

func (c *SamplePlateComponent) View(width, height int) string {
	questionStyle := lipgloss.NewStyle().Bold(true).Width(width)
	dimStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("240"))
	focusStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("63")).Bold(true)
	messageStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("214"))

	var top []string
	if c.question != "" {
		top = append(top, questionStyle.Render(c.question))
	}
	top = append(top, dimStyle.Render(c.legend()+" · enter submit all"))

	if s, ok := c.sampleAtCursor(); ok {
		supplied := s.Condition
		if strings.TrimSpace(supplied) == "" {
			supplied = "(blank)"
		}
		line := fmt.Sprintf("%s  %s", c.plate.Coordinate(), s.SampleID)
		if s.Label != "" {
			line += " · " + s.Label
		}
		line += fmt.Sprintf(" · data: %s · now: %s", supplied, c.assignments[s.SampleID])
		if s.Warning != "" {
			line += " · warning: " + s.Warning
		}
		if s.Flag != "" {
			line += " · AI flag: " + s.Flag
		}
		if s.Suggestion != "" {
			line += " · AI suggests: " + s.Suggestion
		}
		top = append(top, focusStyle.Render(line))
	} else {
		top = append(top, dimStyle.Render(c.plate.Coordinate()+"  (empty well)"))
	}

	top = append(top, dimStyle.Render(c.countsLine()))
	if c.message != "" {
		top = append(top, messageStyle.Render(c.message))
	}

	plateHeight := height - len(top)
	if plateHeight < 6 {
		plateHeight = 6
	}
	c.plate.SetWidth(width)
	c.plate.SetHeight(plateHeight)
	return lipgloss.JoinVertical(lipgloss.Left, append(top, c.plate.Render())...)
}

func (c *SamplePlateComponent) countsLine() string {
	counts := map[string]int{}
	for _, v := range c.assignments {
		counts[v]++
	}
	keys := append([]string(nil), c.choices...)
	sort.Strings(keys)
	parts := make([]string, 0, len(keys)+1)
	for _, k := range keys {
		parts = append(parts, fmt.Sprintf("%s %d", k, counts[k]))
	}
	parts = append(parts, fmt.Sprintf("unresolved %d", counts[unknown]))
	parts = append(parts, fmt.Sprintf("flagged to review %d", len(c.unreviewed())))
	return strings.Join(parts, " · ")
}

func (c *SamplePlateComponent) SetData(data json.RawMessage) error {
	return c.Init(data)
}

func (c *SamplePlateComponent) InitCmd() tea.Cmd { return nil }
func (c *SamplePlateComponent) Name() string     { return "plate_map" }
func (c *SamplePlateComponent) Mode() string     { return "overlay" }

func (c *SamplePlateComponent) KeyBindings() []key.Binding {
	return []key.Binding{
		key.NewBinding(key.WithKeys("up", "down", "left", "right"), key.WithHelp("arrows", "move")),
		key.NewBinding(key.WithKeys("1", "2", "3", "4", "5", "6", "7", "8", "9", "0"), key.WithHelp("1-9/0", "assign")),
		key.NewBinding(key.WithKeys("tab"), key.WithHelp("tab", "view")),
		key.NewBinding(key.WithKeys("enter"), key.WithHelp("enter", "submit all")),
		key.NewBinding(key.WithKeys("esc"), key.WithHelp("esc", "back/cancel")),
	}
}

func (c *SamplePlateComponent) ChangeEvent() map[string]any { return nil }
