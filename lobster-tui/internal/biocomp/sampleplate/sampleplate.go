// Package sampleplate adapts Crust's platemap into a BioComp component that
// lets the user assign one of a list of choices to each sample on a plate.
// It draws the ask_user "plate_map" view (a choice field with for_each).
//
// Crust's platemap draws the plate and handles navigation; assignment, the
// submit rule and the reply shape live here, so Crust stays unchanged.
// Number keys 1-9 assign the 1st-9th choice; 0 marks a sample unresolved.
// esc asks first: a second esc in a row cancels the question.
//
// Samples listed under "unavailable" are not on the plate: they are shown
// under it with their reason and are never part of the reply.
//
// Reply: {"assignments": {sample_id: choice}, "changes": [...]}.
package sampleplate

import (
	"encoding/json"
	"fmt"
	"slices"
	"sort"
	"strings"

	"charm.land/bubbles/v2/key"
	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"

	"github.com/the-omics-os/crust"
	"github.com/the-omics-os/crust/platemap"

	"github.com/the-omics-os/lobster-tui/internal/biocomp"
)

// unresolved is the assignment of a sample the user has not resolved yet. It
// is "", which can never be a choice (Init refuses an empty one), so an
// option literally called "unknown" can be assigned and sent like any other.
// A reply never contains it: enter is refused while any sample has it.
const unresolved = ""

// unresolvedText is how an unresolved assignment is shown on screen.
const unresolvedText = "unknown"

// escPrompt is shown after a first esc; only a second esc in a row cancels.
const escPrompt = "press esc again to cancel this question; nothing will be sent"

// plateMinHeight is the fewest lines Crust's platemap draws (its own minimum
// render height), whatever height it is given.
const plateMinHeight = 9

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

// unavailableSample is a sample that can't be assigned, with the reason.
type unavailableSample struct {
	SampleID string `json:"sample_id"`
	Label    string `json:"label"`
	Reason   string `json:"reason"`
}

type sampleplateData struct {
	Question    string              `json:"question"`
	Title       string              `json:"title"`
	Format      int                 `json:"format"`
	Choices     []string            `json:"choices"`
	Wells       []sampleWell        `json:"wells"`
	Unavailable []unavailableSample `json:"unavailable,omitempty"`
}

// SamplePlateComponent wraps a Crust platemap.Model.
type SamplePlateComponent struct {
	plate       platemap.Model
	question    string
	format      platemap.PlateFormat
	choices     []string
	samples     []sampleWell // in the order supplied
	unavailable []unavailableSample
	assignments map[string]string // sample_id -> choice, or unresolved ("")
	reviewed    map[string]bool   // flagged samples the user has assigned
	message     string            // feedback line, e.g. why enter was refused
	escArmed    bool              // the last key was an esc that would cancel
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
	for i, choice := range choices {
		if choice == unresolved {
			return fmt.Errorf("plate_map: choice %d is empty", i+1)
		}
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
	seenUnavailable := map[string]bool{}
	for _, u := range d.Unavailable {
		switch {
		case u.SampleID == "":
			return fmt.Errorf("plate_map: every unavailable sample needs a sample_id")
		case seenIDs[u.SampleID]:
			return fmt.Errorf("plate_map: sample %q is both on the plate and unavailable", u.SampleID)
		case seenUnavailable[u.SampleID]:
			return fmt.Errorf("plate_map: duplicate unavailable sample %q", u.SampleID)
		}
		seenUnavailable[u.SampleID] = true
	}

	c.question = d.Question
	c.format = format
	c.choices = choices
	c.samples = append([]sampleWell(nil), d.Wells...)
	c.unavailable = append([]unavailableSample(nil), d.Unavailable...)
	c.assignments = assignments
	c.reviewed = map[string]bool{}
	c.message = ""
	c.escArmed = false

	wells := make([]platemap.Well, 0, len(c.samples))
	for _, s := range c.samples {
		wells = append(wells, c.toCrustWell(s))
	}
	// No Crust title: the question, which holds it, is drawn above the plate,
	// and a title only makes Crust's header line wider than the screen.
	c.plate = platemap.New(
		platemap.WithPlate(platemap.PlateData{Format: format, Wells: wells}),
		platemap.WithViewMode(platemap.ViewHitClass),
		platemap.WithCursor(c.samples[0].Row, c.samples[0].Col),
	)
	return nil
}

// startingAssignment keeps a supplied condition only if it is exactly one of
// the choices (after trimming spaces; case matters, as in Prefill, because
// the choices are option ids); anything else must be resolved by the user.
func startingAssignment(condition string, choices []string) string {
	cond := strings.TrimSpace(condition)
	for _, choice := range choices {
		if cond == choice {
			return choice
		}
	}
	return unresolved
}

// shown is an assignment as displayed: unresolved reads "unknown".
func shown(assignment string) string {
	if assignment == unresolved {
		return unresolvedText
	}
	return assignment
}

// toCrustWell maps a sample onto Crust's fixed Well fields. Crust has no field
// for an arbitrary condition, so the current assignment is shown in Reagent
// (in the plate's footer), and samples that still need attention are marked
// as hits. Crust cuts that footer line at the width, so the warning and the
// row's stored details go only on the plate's own wrapped focus line (View).
func (c *SamplePlateComponent) toCrustWell(s sampleWell) platemap.Well {
	parts := []string{}
	if s.Label != "" {
		parts = append(parts, s.Label)
	}
	if s.Batch != "" {
		parts = append(parts, s.Batch)
	}
	parts = append(parts, "now: "+shown(c.assignments[s.SampleID]))
	return platemap.Well{
		Row:      s.Row,
		Col:      s.Col,
		Control:  platemap.ControlSample,
		SampleID: s.SampleID,
		Reagent:  strings.Join(parts, " · "),
		Hit:      c.needsAttention(s),
	}
}

// detailsText is the row's stored columns as "key: value", sorted by key.
func detailsText(details map[string]string) string {
	keys := make([]string, 0, len(details))
	for k, v := range details {
		if v != "" {
			keys = append(keys, k)
		}
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, k+": "+details[k])
	}
	return strings.Join(parts, " · ")
}

// needsAttention: the value is unresolved, or the model flagged the sample
// and the user has not yet assigned it.
func (c *SamplePlateComponent) needsAttention(s sampleWell) bool {
	return c.assignments[s.SampleID] == unresolved || (s.Flag != "" && !c.reviewed[s.SampleID])
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
		return unresolved, true
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

func (c *SamplePlateComponent) unresolvedSamples() []string {
	var ids []string
	for _, s := range c.samples {
		if c.assignments[s.SampleID] == unresolved {
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
	km, isKey := msg.(tea.KeyPressMsg)
	isEsc := isKey && km.String() == "esc"
	if isKey {
		k := km.String()
		if !isEsc && c.escArmed {
			c.disarm() // any other key: the next esc asks again
		}
		if choice, ok := c.keyChoice(k); ok {
			s, found := c.sampleAtCursor()
			if !found {
				c.message = "No sample in this well."
				return nil, nil
			}
			c.assignments[s.SampleID] = choice
			c.reviewed[s.SampleID] = choice != unresolved
			c.plate.UpsertWell(c.toCrustWell(s))
			c.message = fmt.Sprintf("%s set to %s.", s.SampleID, shown(choice))
			return nil, nil
		}
		if k == "ctrl+d" {
			return &biocomp.ComponentResult{Action: "defer", Data: map[string]any{}}, nil
		}
		if k == "enter" {
			// Crust would submit the focused well; we submit all assignments.
			if ids := c.unresolvedSamples(); len(ids) > 0 {
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

	// esc still goes to Crust first: it closes Crust's help or a row/column
	// selection. Only when Crust would cancel is the cancel held back here,
	// so one esc never throws the whole plate away.
	next, cmd := c.plate.Update(msg)
	c.plate = next.(platemap.Model)
	var cancel crust.CancelMsg
	isCancel := false
	if cmd != nil {
		// Safe to run immediately: platemap's commands only return a message.
		cancel, isCancel = cmd().(crust.CancelMsg)
	}
	switch {
	case isCancel && c.escArmed:
		return &biocomp.ComponentResult{
			Action: "cancel",
			Data:   map[string]any{"reason": cancel.Reason},
		}, nil
	case isCancel:
		c.escArmed, c.message = true, escPrompt
	case isEsc && c.escArmed:
		c.disarm() // Crust used this esc to go back, not to cancel
	}
	return nil, nil
}

// disarm forgets a first esc and clears its prompt.
func (c *SamplePlateComponent) disarm() {
	c.escArmed = false
	if c.message == escPrompt {
		c.message = ""
	}
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
		if supplied := strings.TrimSpace(s.Condition); supplied != now {
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

// hang renders text wrapped to the width, after a prefix; continuation lines
// are indented under the text, never cut.
func hang(prefix, text string, width int, style lipgloss.Style) string {
	w := max(width-lipgloss.Width(prefix), 1)
	return lipgloss.JoinHorizontal(lipgloss.Top, prefix, style.Width(w).Render(text))
}

// clipLines keeps at most n lines of block; when lines are dropped, the last
// one kept ends in "…" so it is clear there is more.
// clipWidth cuts every line of block to width cells.
func clipWidth(block string, width int) string {
	if width <= 0 {
		return block
	}
	lines := strings.Split(block, "\n")
	for i, line := range lines {
		lines[i] = ansi.Truncate(line, width, "…")
	}
	return strings.Join(lines, "\n")
}

func clipLines(block string, n, width int) string {
	lines := strings.Split(block, "\n")
	if len(lines) <= n {
		return block
	}
	lines = lines[:n]
	lines[n-1] = ansi.Truncate(lines[n-1], max(width-1, 1), "") + "…"
	return strings.Join(lines, "\n")
}

func (c *SamplePlateComponent) View(width, height int) string {
	dimStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("240"))
	focusStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("63")).Bold(true)
	messageStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("214"))
	detailStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("63"))

	var top []string
	details, focusAt := "", 0 // the focused sample's stored details go under its line
	if c.question != "" {
		top = append(top, hang("", c.question, width, lipgloss.NewStyle().Bold(true)))
	}
	top = append(top, hang("", c.legend()+" · enter submit all", width, dimStyle))

	if s, ok := c.sampleAtCursor(); ok {
		supplied := s.Condition
		if strings.TrimSpace(supplied) == "" {
			supplied = "(none)"
		}
		line := fmt.Sprintf("%s  %s", c.plate.Coordinate(), s.SampleID)
		if s.Label != "" {
			line += " · " + s.Label
		}
		if s.Batch != "" {
			line += " · batch: " + s.Batch
		}
		// The well's condition is the supervisor's default, not the data file.
		line += fmt.Sprintf(" · proposed: %s · now: %s", supplied, shown(c.assignments[s.SampleID]))
		if s.Warning != "" {
			line += " · warning: " + s.Warning
		}
		if s.Flag != "" {
			line += " · AI flag: " + s.Flag
		}
		if s.Suggestion != "" {
			line += " · AI suggests: " + s.Suggestion
		}
		top = append(top, hang("", line, width, focusStyle))
		focusAt = len(top) - 1
		if d := detailsText(s.Details); d != "" {
			details = hang("  ", d, width, detailStyle)
		}
	} else {
		top = append(top, dimStyle.Render(c.plate.Coordinate()+"  (empty well)"))
	}

	top = append(top, hang("", c.countsLine(), width, dimStyle))
	if c.message != "" {
		top = append(top, hang("", c.message, width, messageStyle))
	}

	var below []string
	if len(c.unavailable) > 0 {
		below = append(below, dimStyle.Render("Not on the plate (can't be assigned):"))
		for _, u := range c.unavailable {
			text := u.SampleID
			if u.Label != "" {
				text += " " + u.Label
			}
			if u.Reason != "" {
				text += " — " + u.Reason
			}
			below = append(below, hang("✕ ", text, width, dimStyle))
		}
	}

	rest := lipgloss.Height(strings.Join(top, "\n"))
	if len(below) > 0 {
		rest += lipgloss.Height(strings.Join(below, "\n"))
	}
	if details != "" {
		// The details take room from the plate first; once the plate is at
		// its minimum they are clipped, so they never push it off screen.
		details = clipLines(details, max(height-rest-plateMinHeight, 1), width)
		top = slices.Insert(top, focusAt+1, details)
		rest += lipgloss.Height(details)
	}
	c.plate.SetWidth(width)
	c.plate.SetHeight(max(height-rest, plateMinHeight))
	// Crust's header and inspector lines can be wider than the screen
	// (about 66 cells, more with long labels): cut them, so the terminal
	// never wraps the plate. The focused well is in full on our own line.
	parts := append(top, clipWidth(c.plate.Render(), width))
	return lipgloss.JoinVertical(lipgloss.Left, append(parts, below...)...)
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
	parts = append(parts, fmt.Sprintf("unresolved %d", counts[unresolved]))
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
		key.NewBinding(key.WithKeys("ctrl+d"), key.WithHelp("ctrl+d", "defer")),
		// Help key stays "esc" so a dashboard can leave it out (it has its own).
		key.NewBinding(key.WithKeys("esc"), key.WithHelp("esc", "twice: cancel")),
	}
}

func (c *SamplePlateComponent) ChangeEvent() map[string]any { return nil }

// Draft returns the assignments on screen in the reply's shape, so a
// dashboard can tell whether they differ from the answer it recorded.
func (c *SamplePlateComponent) Draft() map[string]any {
	assignments := make(map[string]any, len(c.samples))
	for _, s := range c.samples {
		assignments[s.SampleID] = c.assignments[s.SampleID]
	}
	return map[string]any{"assignments": assignments}
}

// Prefill restores the assignments of an earlier reply (a dashboard section
// that was already answered). The supplied conditions are kept, so the reply's
// "changes" still compare against the data. Unknown samples and values that
// are not one of the choices are ignored; flagged samples listed in the
// reply's "reviewed_flags" count as reviewed.
func (c *SamplePlateComponent) Prefill(reply map[string]any) {
	assignments, _ := reply["assignments"].(map[string]any)
	reviewed := map[string]bool{}
	if flags, ok := reply["reviewed_flags"].([]any); ok {
		for _, f := range flags {
			if m, ok := f.(map[string]any); ok {
				reviewed[fmt.Sprint(m["sample_id"])] = true
			}
		}
	}
	for _, s := range c.samples {
		v, ok := assignments[s.SampleID].(string)
		if !ok {
			continue
		}
		for _, choice := range c.choices {
			if v == choice {
				c.assignments[s.SampleID] = v
				c.reviewed[s.SampleID] = reviewed[s.SampleID]
				c.plate.UpsertWell(c.toCrustWell(s))
			}
		}
	}
}
