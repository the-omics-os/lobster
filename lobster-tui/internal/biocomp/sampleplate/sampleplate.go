// Package sampleplate adapts Crust's platemap into a BioComp component that
// lets the user put each sample on a plate into one of a list of groups. It
// draws the ask_user "plate_map" view (a choice field with for_each).
//
// CRUST DEPENDENCY: this needs Crust's platemap Groups view (ViewGroups,
// Well.Group/Noted/Flagged/Excluded, PlateData.Groups, WithGridOnly,
// GroupColor). It is proposed upstream on the Crust branch
// platemap-groups-view and is NOT in Crust v0.1.0, the version go.mod
// requires. Until Crust releases it and go.mod is bumped, lobster-tui builds
// only with a local, uncommitted go.work that uses that Crust branch (see
// docs/dynamic-ui/design.md, "Crust dependency").
//
// Crust draws the grid: each sample's group as a dot in the group's colour.
// Everything else lives here, so the screen reads as one: the question, the
// legend with each option's number key, the current-well line under the
// plate, the reason box, the submit rules and the reply shape.
//
// Keys: number keys 1-9 put the current sample in the 1st-9th option; 0 marks
// it unresolved. An option that asks a reason opens a one-line reason box on
// the current-well line (enter saves, esc puts the sample back); r edits a
// saved reason. n switches the wells between coloured dots and each option's
// number. esc asks first: a second esc in a row cancels the question.
//
// Samples marked "excluded" (QC pools, blanks) sit greyed in their wells,
// with the reason on the current-well line; they are never in the reply.
//
// Reply: {"assignments": {sample_id: option}, "reasons": {sample_id: text},
// "changes": [...], "reviewed_flags": [...]}.
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
// is "", which can never be an option (Init refuses an empty one), so an
// option literally called "unknown" can be assigned and sent like any other.
// A reply never contains it: enter is refused while any sample has it.
const unresolved = ""

// escPrompt is shown after a first esc; only a second esc in a row cancels.
const escPrompt = "press esc again to cancel this question; nothing will be sent"

// listOrderNote is shown when the samples have no positions of their own.
const listOrderNote = "⚠ No positions were given: samples are placed in list order. " +
	"These are not the physical plate positions."

// plateMinHeight is the fewest lines the grid is given: the column numbers
// and at least eight rows (Crust's own minimum render height).
const plateMinHeight = 9

// Colours from Crust's platemap DefaultTheme, so text and grid agree.
var (
	attentionColor = lipgloss.Color("214") // Theme.Missing: needs the user
	mutedColor     = lipgloss.Color("240") // Theme.Empty
	focusColor     = lipgloss.Color("63")
)

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
	Condition string `json:"condition"` // the proposed option, "" = none
	Reason    string `json:"reason"`    // the proposed option's reason, if any
	Batch     string `json:"batch"`
	// Warning is offered as a greyed hint in the reason box (tab inserts
	// it); it is never filled in by itself.
	Warning string `json:"warning"`
	// Excluded: the reason the sample can't be assigned (a QC pool, a
	// blank); it is drawn greyed and is never in the reply.
	Excluded string `json:"excluded"`
	// From a data reference: the row's other columns as stored, and the
	// model's own flag and suggestion (shown as the model's, never as data).
	Details    map[string]string `json:"details"`
	Flag       string            `json:"flag"`
	Suggestion string            `json:"suggestion"`
}

// reasonSpec is the one required text field an option asks for.
type reasonSpec struct {
	Key      string `json:"key"`
	Question string `json:"question"`
}

type plateOption struct {
	ID     string      `json:"id"`
	Label  string      `json:"label"`
	Reason *reasonSpec `json:"reason"`
}

type sampleplateData struct {
	Question string        `json:"question"`
	Title    string        `json:"title"`
	Format   int           `json:"format"`
	Options  []plateOption `json:"options"`
	// Choices is the older form of Options: bare ids, no reasons.
	Choices []string `json:"choices"`
	// Positions is "given" when the wells are the samples' own positions,
	// or "list_order" when they were filled in list order.
	Positions string       `json:"positions"`
	Wells     []sampleWell `json:"wells"`
}

// reasonBox is the one-line reason editor on the current-well line.
type reasonBox struct {
	open       bool
	sample     string // sample_id being edited
	text       string
	prevOption string // what esc puts back
	prevReason string
	prevReview bool
}

// SamplePlateComponent wraps a Crust platemap.Model.
type SamplePlateComponent struct {
	plate       platemap.Model
	question    string
	format      platemap.PlateFormat
	options     []plateOption
	listOrder   bool
	samples     []sampleWell      // in the order supplied
	assignments map[string]string // sample_id -> option id, or unresolved ("")
	reasons     map[string]string // sample_id -> saved reason
	reviewed    map[string]bool   // flagged samples the user has assigned
	box         reasonBox
	message     string // feedback line, e.g. why enter was refused
	escArmed    bool   // the last key was an esc that would cancel
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

	options := d.Options
	if len(options) == 0 {
		for _, id := range d.Choices {
			options = append(options, plateOption{ID: id})
		}
	}
	if len(options) == 0 || len(options) > 9 {
		return fmt.Errorf("plate_map: need 1 to 9 options (one number key each), got %d", len(options))
	}
	seenOptions := map[string]bool{}
	for i := range options {
		o := &options[i]
		switch {
		case o.ID == unresolved:
			return fmt.Errorf("plate_map: option %d is empty", i+1)
		case seenOptions[o.ID]:
			return fmt.Errorf("plate_map: duplicate option %q", o.ID)
		case o.Reason != nil && o.Reason.Key == "":
			return fmt.Errorf("plate_map: option %q asks a reason without a key", o.ID)
		}
		seenOptions[o.ID] = true
		if o.Label == "" {
			o.Label = o.ID
		}
	}

	seenIDs := map[string]bool{}
	seenWells := map[[2]int]bool{}
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
	}

	c.question = d.Question
	c.format = format
	c.options = options
	c.listOrder = d.Positions == "list_order"
	c.samples = append([]sampleWell(nil), d.Wells...)
	c.assignments = map[string]string{}
	c.reasons = map[string]string{}
	c.reviewed = map[string]bool{}
	c.box = reasonBox{}
	c.message = ""
	c.escArmed = false
	for _, s := range c.samples {
		if s.Excluded != "" {
			continue
		}
		option := startingAssignment(s.Condition, options)
		c.assignments[s.SampleID] = option
		if o, ok := c.option(option); ok && o.Reason != nil {
			c.reasons[s.SampleID] = strings.TrimSpace(s.Reason)
		}
	}

	groups := make([]string, len(options))
	for i, o := range options {
		groups[i] = o.ID
	}
	wells := make([]platemap.Well, 0, len(c.samples))
	for _, s := range c.samples {
		wells = append(wells, c.toCrustWell(s))
	}
	start := c.samples[0]
	for _, s := range c.samples {
		if s.Excluded == "" {
			start = s
			break
		}
	}
	// Only the grid: the question, legend and current-well line are drawn
	// here, so they wrap at the width instead of being cut.
	c.plate = platemap.New(
		platemap.WithPlate(platemap.PlateData{Format: format, Wells: wells, Groups: groups}),
		platemap.WithViewMode(platemap.ViewGroups),
		platemap.WithGridOnly(true),
		platemap.WithCursor(start.Row, start.Col),
	)
	return nil
}

// startingAssignment keeps a proposed option only if it is exactly one of the
// option ids (after trimming spaces; case matters, as in Prefill, because the
// options are ids); anything else must be resolved by the user.
func startingAssignment(condition string, options []plateOption) string {
	cond := strings.TrimSpace(condition)
	for _, o := range options {
		if cond == o.ID {
			return o.ID
		}
	}
	return unresolved
}

func (c *SamplePlateComponent) option(id string) (plateOption, bool) {
	for _, o := range c.options {
		if o.ID == id {
			return o, true
		}
	}
	return plateOption{}, false
}

// asksReason reports whether the sample's option asks for a reason.
func (c *SamplePlateComponent) asksReason(sampleID string) bool {
	o, ok := c.option(c.assignments[sampleID])
	return ok && o.Reason != nil
}

// shown is an assignment as displayed: the option's label, or "unresolved".
func (c *SamplePlateComponent) shown(assignment string) string {
	if o, ok := c.option(assignment); ok {
		return o.Label
	}
	return "unresolved"
}

// toCrustWell maps a sample onto Crust's Well: its group is the option id
// (Crust colours it by the option's place in the list), a saved reason is a
// note, and a missing reason or an unreviewed model flag is a flag.
func (c *SamplePlateComponent) toCrustWell(s sampleWell) platemap.Well {
	w := platemap.Well{
		Row:      s.Row,
		Col:      s.Col,
		Control:  platemap.ControlSample,
		SampleID: s.SampleID,
		Excluded: s.Excluded != "",
	}
	if w.Excluded {
		return w
	}
	w.Group = c.assignments[s.SampleID]
	w.Noted = c.reasons[s.SampleID] != ""
	w.Flagged = c.needsReason(s.SampleID) || (s.Flag != "" && !c.reviewed[s.SampleID])
	return w
}

func (c *SamplePlateComponent) needsReason(sampleID string) bool {
	return c.asksReason(sampleID) && c.reasons[sampleID] == ""
}

func (c *SamplePlateComponent) redraw(s sampleWell) {
	c.plate.UpsertWell(c.toCrustWell(s))
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

func (c *SamplePlateComponent) sampleAtCursor() (sampleWell, bool) {
	row, col := c.plate.Cursor()
	for _, s := range c.samples {
		if s.Row == row && s.Col == col {
			return s, true
		}
	}
	return sampleWell{}, false
}

func (c *SamplePlateComponent) sample(id string) sampleWell {
	for _, s := range c.samples {
		if s.SampleID == id {
			return s
		}
	}
	return sampleWell{}
}

// keyOption maps a number key to an option: 1 is the first option, 0 resets
// the sample to unresolved.
func (c *SamplePlateComponent) keyOption(k string) (string, bool) {
	if len(k) != 1 || k[0] < '0' || k[0] > '9' {
		return "", false
	}
	if k == "0" {
		return unresolved, true
	}
	i := int(k[0] - '1')
	if i >= len(c.options) {
		return "", false
	}
	return c.options[i].ID, true
}

// coordinate is a well's name, e.g. "D1".
func coordinate(s sampleWell) string {
	return fmt.Sprintf("%s%d", rowName(s.Row), s.Col+1)
}

func rowName(row int) string {
	if row < 26 {
		return string(rune('A' + row))
	}
	return string(rune('A'+row/26-1)) + string(rune('A'+row%26))
}

// labelled is "id · label", or the label alone when it already holds the id.
func labelled(s sampleWell) string {
	switch {
	case s.Label == "":
		return s.SampleID
	case strings.Contains(s.Label, s.SampleID):
		return s.Label
	}
	return s.SampleID + " · " + s.Label
}

// named lists samples as "D1 TW.M01", at most six, then "and N more".
func named(samples []sampleWell) string {
	parts := []string{}
	for i, s := range samples {
		if i == 6 {
			parts = append(parts, fmt.Sprintf("and %d more", len(samples)-6))
			break
		}
		parts = append(parts, coordinate(s)+" "+s.SampleID)
	}
	return strings.Join(parts, ", ")
}

func (c *SamplePlateComponent) open() []sampleWell {
	var out []sampleWell
	for _, s := range c.samples {
		if s.Excluded == "" {
			out = append(out, s)
		}
	}
	return out
}

func (c *SamplePlateComponent) unresolvedSamples() []sampleWell {
	var out []sampleWell
	for _, s := range c.open() {
		if c.assignments[s.SampleID] == unresolved {
			out = append(out, s)
		}
	}
	return out
}

func (c *SamplePlateComponent) missingReasons() []sampleWell {
	var out []sampleWell
	for _, s := range c.open() {
		if c.needsReason(s.SampleID) {
			out = append(out, s)
		}
	}
	return out
}

func (c *SamplePlateComponent) unreviewed() []sampleWell {
	var out []sampleWell
	for _, s := range c.open() {
		if s.Flag != "" && !c.reviewed[s.SampleID] {
			out = append(out, s)
		}
	}
	return out
}

// IsCapturingText reports whether the reason box is open. A dashboard checks
// it to give esc and tab to the box.
func (c *SamplePlateComponent) IsCapturingText() bool { return c.box.open }

func (c *SamplePlateComponent) HandleMsg(msg tea.Msg) (*biocomp.ComponentResult, tea.Cmd) {
	if c.box.open {
		return c.boxMsg(msg)
	}
	km, isKey := msg.(tea.KeyPressMsg)
	isEsc := isKey && km.String() == "esc"
	if isKey {
		k := km.String()
		if !isEsc && c.escArmed {
			c.disarm() // any other key: the next esc asks again
		}
		if option, ok := c.keyOption(k); ok {
			c.pick(option)
			return nil, nil
		}
		switch k {
		case "r":
			c.editReason()
			return nil, nil
		case "n":
			// Numbers for a terminal without colour, or a reader who can't
			// tell the colours apart.
			c.plate.SetGroupNumbers(!c.plate.GroupNumbers())
			return nil, nil
		case "tab", "shift+tab":
			return nil, nil // Crust's other views mean nothing here
		case "ctrl+d":
			return &biocomp.ComponentResult{Action: "defer", Data: map[string]any{}}, nil
		case "enter":
			// Crust would submit the focused well; we submit all assignments.
			return c.submit(), nil
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

// pick puts the current sample in option (or unresolved). An option that
// asks a reason opens the reason box; esc there puts the sample back.
func (c *SamplePlateComponent) pick(option string) {
	s, found := c.sampleAtCursor()
	switch {
	case !found:
		c.message = "No sample in this well."
		return
	case s.Excluded != "":
		c.message = fmt.Sprintf("%s can't be assigned: %s.", s.SampleID, s.Excluded)
		return
	}
	id := s.SampleID
	before, beforeReason, beforeReview := c.assignments[id], c.reasons[id], c.reviewed[id]
	c.assignments[id] = option
	c.reviewed[id] = option != unresolved
	if o, ok := c.option(option); ok && o.Reason != nil {
		if before != option {
			delete(c.reasons, id) // a reason belongs to the option it was given for
		}
		c.openBox(s, before, beforeReason, beforeReview)
		return
	}
	delete(c.reasons, id)
	c.redraw(s)
	c.message = fmt.Sprintf("%s set to %s.", id, c.shown(option))
}

// editReason (r) reopens the box for the current sample's reason.
func (c *SamplePlateComponent) editReason() {
	s, found := c.sampleAtCursor()
	if !found || s.Excluded != "" || !c.asksReason(s.SampleID) {
		c.message = "r edits a reason; this well's option doesn't take one."
		return
	}
	c.openBox(s, c.assignments[s.SampleID], c.reasons[s.SampleID], c.reviewed[s.SampleID])
}

func (c *SamplePlateComponent) openBox(s sampleWell, prevOption, prevReason string, prevReview bool) {
	c.box = reasonBox{
		open:       true,
		sample:     s.SampleID,
		text:       c.reasons[s.SampleID],
		prevOption: prevOption,
		prevReason: prevReason,
		prevReview: prevReview,
	}
	c.message = ""
	c.redraw(s)
}

func (c *SamplePlateComponent) boxMsg(msg tea.Msg) (*biocomp.ComponentResult, tea.Cmd) {
	if paste, ok := msg.(tea.PasteMsg); ok {
		c.box.text += strings.Join(strings.Fields(paste.Content), " ")
		return nil, nil
	}
	km, ok := msg.(tea.KeyPressMsg)
	if !ok {
		return nil, nil
	}
	id := c.box.sample
	s := c.sample(id)
	switch km.String() {
	case "enter":
		text := strings.TrimSpace(c.box.text)
		if text == "" {
			c.message = fmt.Sprintf("A reason is needed for %s: type one, or esc to undo.", c.shown(c.assignments[id]))
			return nil, nil
		}
		c.reasons[id] = text
		c.box = reasonBox{}
		c.message = fmt.Sprintf("%s: %s, reason saved.", id, c.shown(c.assignments[id]))
	case "esc":
		c.assignments[id] = c.box.prevOption
		c.reviewed[id] = c.box.prevReview
		if c.box.prevReason != "" {
			c.reasons[id] = c.box.prevReason
		} else {
			delete(c.reasons, id)
		}
		c.box = reasonBox{}
		c.message = fmt.Sprintf("%s back to %s.", id, c.shown(c.assignments[id]))
	case "tab":
		c.box.text += s.Warning // a hint, inserted only when asked
	case "backspace":
		if r := []rune(c.box.text); len(r) > 0 {
			c.box.text = string(r[:len(r)-1])
		}
	case "ctrl+d":
		return &biocomp.ComponentResult{Action: "defer", Data: map[string]any{}}, nil
	default:
		if km.Text != "" {
			c.box.text += km.Text
		}
		return nil, nil
	}
	c.redraw(s)
	return nil, nil
}

// submit refuses while a sample is unresolved, lacks its reason or has an
// unreviewed flag, naming the wells and moving to the first; else replies.
func (c *SamplePlateComponent) submit() *biocomp.ComponentResult {
	for _, check := range []struct {
		samples []sampleWell
		why     string
	}{
		{c.unresolvedSamples(), "Resolve before submitting (press a number): "},
		{c.missingReasons(), "Needs a reason (press r): "},
		{c.unreviewed(), "Review flagged samples (press a number on each): "},
	} {
		if len(check.samples) > 0 {
			c.message = check.why + named(check.samples)
			c.plate.SetCursor(check.samples[0].Row, check.samples[0].Col)
			return nil
		}
	}
	return c.submitResult()
}

// disarm forgets a first esc and clears its prompt.
func (c *SamplePlateComponent) disarm() {
	c.escArmed = false
	if c.message == escPrompt {
		c.message = ""
	}
}

// submitResult returns every assignment and reason, plus the samples whose
// proposed option was changed, so the original is never silently replaced.
func (c *SamplePlateComponent) submitResult() *biocomp.ComponentResult {
	data := c.Draft()
	changes := []any{}
	reviewed := []any{}
	for _, s := range c.open() {
		now := c.assignments[s.SampleID]
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
	data["changes"] = changes
	data["reviewed_flags"] = reviewed
	return &biocomp.ComponentResult{Action: "submit", Data: data}
}

// ---------------------------------------------------------------------------
// Drawing
// ---------------------------------------------------------------------------

// hang renders text wrapped to the width, after a prefix; continuation lines
// are indented under the text, never cut.
func hang(prefix, text string, width int, style lipgloss.Style) string {
	w := max(width-lipgloss.Width(prefix), 1)
	return lipgloss.JoinHorizontal(lipgloss.Top, prefix, style.Width(w).Render(text))
}

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

// clipLines keeps at most n lines of block; when lines are dropped, the last
// one kept ends in "…" so it is clear there is more.
func clipLines(block string, n, width int) string {
	lines := strings.Split(block, "\n")
	if len(lines) <= n {
		return block
	}
	lines = lines[:n]
	lines[n-1] = ansi.Truncate(lines[n-1], max(width-1, 1), "") + "…"
	return strings.Join(lines, "\n")
}

// legend pairs each number key with its option's dot, in the option's
// colour, e.g. "1 ● vehicle   2 ● TDX-9   3 ● leave out (asks a reason)".
func (c *SamplePlateComponent) legend() string {
	parts := make([]string, 0, len(c.options)+1)
	for i, o := range c.options {
		fg, _ := c.plate.GroupColor(o.ID)
		text := fmt.Sprintf("%d %s %s", i+1, lipgloss.NewStyle().Foreground(fg).Render("●"), o.Label)
		if o.Reason != nil {
			text += " (asks a reason)"
		}
		parts = append(parts, text)
	}
	dot := lipgloss.NewStyle().Foreground(attentionColor).Bold(true).Render("○")
	return strings.Join(append(parts, "0 "+dot+" unresolved"), "   ")
}

// marks explains the other symbols, only those that can appear here.
func (c *SamplePlateComponent) marks() string {
	parts := []string{}
	for _, o := range c.options {
		if o.Reason != nil {
			parts = append(parts, "◉ has a reason", "! needs a reason (r)")
			break
		}
	}
	for _, s := range c.samples {
		if s.Flag != "" {
			parts = append(parts, "! flagged: review it")
			break
		}
	}
	for _, s := range c.samples {
		if s.Excluded != "" {
			parts = append(parts, "◌ can't be assigned")
			break
		}
	}
	parts = append(parts, "· no sample")
	if c.plate.GroupNumbers() {
		return strings.Join(append(parts, "n dots"), " · ")
	}
	return strings.Join(append(parts, "n numbers"), " · ")
}

// currentWell is the line under the plate about the sample under the cursor:
// who it is, what was proposed, what it is now and its reason; or the reason
// box while one is being typed. Its details follow on their own lines.
func (c *SamplePlateComponent) currentWell(width int) (line string, details []string) {
	focusStyle := lipgloss.NewStyle().Foreground(focusColor).Bold(true)
	dimStyle := lipgloss.NewStyle().Foreground(mutedColor)
	detailStyle := lipgloss.NewStyle().Foreground(focusColor)
	warnStyle := lipgloss.NewStyle().Foreground(attentionColor)

	s, ok := c.sampleAtCursor()
	if !ok {
		return dimStyle.Render(c.plate.Coordinate() + "  (no sample)"), nil
	}
	name := coordinate(s) + "  " + labelled(s)
	if s.Excluded != "" {
		return hang("", name+" · can't be assigned: "+s.Excluded, width, dimStyle), nil
	}
	if d := detailsText(s.Details); d != "" {
		details = append(details, hang("    ", d, width, detailStyle))
	}
	if s.Warning != "" {
		details = append(details, hang("    ", "warning: "+s.Warning, width, warnStyle))
	}

	if c.box.open && c.box.sample == s.SampleID {
		o, _ := c.option(c.assignments[s.SampleID])
		label := "Reason"
		if o.Reason != nil && o.Reason.Question != "" {
			label = strings.TrimRight(o.Reason.Question, ":? ")
		}
		box := focusStyle.Render(fmt.Sprintf("%s → %s · %s: ", name, o.Label, label)) + c.box.text + "█"
		if c.box.text == "" && s.Warning != "" {
			box += " " + dimStyle.Render(s.Warning+"  ← tab inserts")
		}
		help := dimStyle.Render("    enter save · esc undo")
		return box + "\n" + help, details
	}

	if s.Batch != "" {
		name += " · batch: " + s.Batch
	}
	proposed := s.Condition
	if strings.TrimSpace(proposed) == "" {
		proposed = "(none)"
	}
	// The proposed option is the supervisor's default, not the data file.
	text := fmt.Sprintf("%s · proposed: %s · now: %s", name, proposed, c.shown(c.assignments[s.SampleID]))
	switch {
	case c.reasons[s.SampleID] != "":
		text += " · reason: " + c.reasons[s.SampleID]
	case c.needsReason(s.SampleID):
		text += " · needs a reason (press r)"
	}
	if s.Flag != "" {
		text += " · AI flag: " + s.Flag
	}
	if s.Suggestion != "" {
		text += " · AI suggests: " + s.Suggestion
	}
	return hang("", text, width, focusStyle), details
}

func (c *SamplePlateComponent) View(width, height int) string {
	dimStyle := lipgloss.NewStyle().Foreground(mutedColor)
	messageStyle := lipgloss.NewStyle().Foreground(attentionColor)

	var top []string
	if c.question != "" {
		top = append(top, hang("", c.question, width, lipgloss.NewStyle().Bold(true)))
	}
	if c.listOrder {
		top = append(top, hang("", listOrderNote, width, messageStyle.Bold(true)))
	}
	if len(top) > 0 && biocomp.Roomy(height) {
		top = append(top, "") // a gap between the question and the plate
	}

	below := []string{
		"", // a gap between the plate and the lines about it
		hang("", c.legend(), width, lipgloss.NewStyle()),
		hang("", c.marks(), width, dimStyle),
		hang("", c.countsLine(), width, biocomp.StatStyle),
	}
	line, details := c.currentWell(width)
	below = append(below, line)
	detailsAt := len(below)
	if c.message != "" {
		below = append(below, hang("", c.message, width, messageStyle))
	}

	rest := lipgloss.Height(strings.Join(append(append([]string{}, top...), below...), "\n"))
	if len(details) > 0 {
		// The details take room from the plate first; once the plate is at
		// its minimum they are clipped, so they never push it off screen.
		block := clipLines(strings.Join(details, "\n"), max(height-rest-plateMinHeight, 1), width)
		below = slices.Insert(below, detailsAt, block)
		rest += lipgloss.Height(block)
	}
	gridLines := c.format.Rows() + 1
	c.plate.SetWidth(width)
	c.plate.SetHeight(min(gridLines, max(height-rest, plateMinHeight)))
	parts := append(top, clipWidth(c.plate.Render(), width))
	return lipgloss.JoinVertical(lipgloss.Left, append(parts, below...)...)
}

func (c *SamplePlateComponent) countsLine() string {
	counts := map[string]int{}
	excluded := 0
	for _, s := range c.samples {
		if s.Excluded != "" {
			excluded++
			continue
		}
		counts[c.assignments[s.SampleID]]++
	}
	parts := make([]string, 0, len(c.options)+4)
	for _, o := range c.options {
		parts = append(parts, fmt.Sprintf("%s (%d)", o.Label, counts[o.ID]))
	}
	parts = append(parts, fmt.Sprintf("unresolved (%d)", counts[unresolved]))
	if n := len(c.missingReasons()); n > 0 {
		parts = append(parts, fmt.Sprintf("needs a reason (%d)", n))
	}
	if n := len(c.unreviewed()); n > 0 {
		parts = append(parts, fmt.Sprintf("flagged to review (%d)", n))
	}
	if excluded > 0 {
		parts = append(parts, fmt.Sprintf("can't be assigned (%d)", excluded))
	}
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
		key.NewBinding(key.WithKeys("r"), key.WithHelp("r", "reason")),
		key.NewBinding(key.WithKeys("n"), key.WithHelp("n", "numbers")),
		key.NewBinding(key.WithKeys("enter"), key.WithHelp("enter", "submit all")),
		key.NewBinding(key.WithKeys("ctrl+d"), key.WithHelp("ctrl+d", "defer")),
		// Help key stays "esc" so a dashboard can leave it out (it has its own).
		key.NewBinding(key.WithKeys("esc"), key.WithHelp("esc", "twice: cancel")),
	}
}

func (c *SamplePlateComponent) ChangeEvent() map[string]any { return nil }

// Hint is the instruction under the screen (see biocomp.Guide).
func (c *SamplePlateComponent) Hint() string {
	if c.box.open {
		return biocomp.HintReason
	}
	return biocomp.HintPlate
}

// Draft returns the assignments and reasons on screen in the reply's shape,
// so a dashboard can tell whether they differ from the answer it recorded.
func (c *SamplePlateComponent) Draft() map[string]any {
	assignments := map[string]any{}
	reasons := map[string]any{}
	for _, s := range c.open() {
		assignments[s.SampleID] = c.assignments[s.SampleID]
		if r := c.reasons[s.SampleID]; r != "" && c.asksReason(s.SampleID) {
			reasons[s.SampleID] = r
		}
	}
	draft := map[string]any{"assignments": assignments}
	if len(reasons) > 0 { // left out when empty, so older answers still match
		draft["reasons"] = reasons
	}
	return draft
}

// Prefill restores the assignments and reasons of an earlier reply (a
// dashboard section that was already answered). The proposed options are
// kept, so the reply's "changes" still compare against them. Unknown samples
// and values that are not one of the options are ignored; flagged samples
// listed in the reply's "reviewed_flags" count as reviewed.
func (c *SamplePlateComponent) Prefill(reply map[string]any) {
	assignments, _ := reply["assignments"].(map[string]any)
	reasons, _ := reply["reasons"].(map[string]any)
	reviewed := map[string]bool{}
	for _, f := range asMaps(reply["reviewed_flags"]) {
		reviewed[fmt.Sprint(f["sample_id"])] = true
	}
	for _, s := range c.open() {
		v, ok := assignments[s.SampleID].(string)
		if !ok {
			continue
		}
		if _, known := c.option(v); !known {
			continue
		}
		c.assignments[s.SampleID] = v
		c.reviewed[s.SampleID] = reviewed[s.SampleID]
		delete(c.reasons, s.SampleID)
		if r, ok := reasons[s.SampleID].(string); ok && c.asksReason(s.SampleID) && strings.TrimSpace(r) != "" {
			c.reasons[s.SampleID] = strings.TrimSpace(r)
		}
		c.redraw(s)
	}
}

func asMaps(v any) []map[string]any {
	list, _ := v.([]any)
	var out []map[string]any
	for _, item := range list {
		if m, ok := item.(map[string]any); ok {
			out = append(out, m)
		}
	}
	return out
}

// ---------------------------------------------------------------------------
// Dashboard review: one line, and the answer in full
// ---------------------------------------------------------------------------

// Summary is one line for a dashboard section: the count in each option.
func Summary(data json.RawMessage, answer map[string]any) string {
	d := parseData(data)
	assignments, _ := answer["assignments"].(map[string]any)
	counts := map[string]int{}
	for _, v := range assignments {
		counts[fmt.Sprint(v)]++
	}
	var parts []string
	for _, o := range d.options() {
		if n := counts[o.ID]; n > 0 {
			parts = append(parts, fmt.Sprintf("%s (%d)", o.Label, n))
		}
		delete(counts, o.ID)
	}
	var others []string
	for v := range counts {
		others = append(others, v)
	}
	sort.Strings(others)
	for _, v := range others {
		parts = append(parts, fmt.Sprintf("%s (%d)", v, counts[v]))
	}
	return strings.Join(parts, " · ")
}

// Describe lists a plate answer in full for a dashboard's review screen: the
// samples in each option, every reason, what changed from the proposal, and
// the flags that were reviewed.
func Describe(data json.RawMessage, answer map[string]any) []string {
	d := parseData(data)
	assignments, _ := answer["assignments"].(map[string]any)
	reasons, _ := answer["reasons"].(map[string]any)
	var order []string
	inData := map[string]bool{}
	for _, w := range d.Wells {
		if _, ok := assignments[w.SampleID]; ok {
			order = append(order, w.SampleID)
			inData[w.SampleID] = true
		}
	}
	var extra []string
	for id := range assignments {
		if !inData[id] {
			extra = append(extra, id)
		}
	}
	sort.Strings(extra)
	order = append(order, extra...)
	groups := map[string][]string{}
	for _, id := range order {
		v := fmt.Sprint(assignments[id])
		groups[v] = append(groups[v], id)
	}
	labels := map[string]string{}
	var values []string
	for _, o := range d.options() {
		values = append(values, o.ID)
		labels[o.ID] = o.Label
	}
	var others []string
	for v := range groups {
		if _, ok := labels[v]; !ok {
			others = append(others, v)
		}
	}
	sort.Strings(others)
	var lines []string
	for _, v := range append(values, others...) {
		if ids := groups[v]; len(ids) > 0 {
			name := v
			if l := labels[v]; l != "" && l != v {
				name = l + " (" + v + ")"
			}
			lines = append(lines, fmt.Sprintf("%s (%d): %s", name, len(ids), strings.Join(ids, ", ")))
		}
	}
	for _, id := range order {
		if r, ok := reasons[id]; ok && fmt.Sprint(r) != "" {
			v := fmt.Sprint(assignments[id])
			name := labels[v]
			if name == "" {
				name = v
			}
			lines = append(lines, fmt.Sprintf("reason: %s → %s: %v", id, name, r))
		}
	}
	for _, ch := range asMaps(answer["changes"]) {
		supplied := fmt.Sprint(ch["supplied"])
		if strings.TrimSpace(supplied) == "" {
			supplied = "(blank)"
		}
		lines = append(lines, fmt.Sprintf("changed from the data: %v %s → %v", ch["sample_id"], supplied, ch["assigned"]))
	}
	for _, f := range asMaps(answer["reviewed_flags"]) {
		lines = append(lines, fmt.Sprintf("flag reviewed: %v (%v) → %v", f["sample_id"], f["flag"], f["assigned"]))
	}
	if len(lines) == 0 {
		return []string{"no assignments"}
	}
	return lines
}

func parseData(data json.RawMessage) sampleplateData {
	var d sampleplateData
	_ = json.Unmarshal(data, &d)
	return d
}

// options is Options, or the older Choices as options.
func (d sampleplateData) options() []plateOption {
	if len(d.Options) > 0 {
		out := append([]plateOption(nil), d.Options...)
		for i := range out {
			if out[i].Label == "" {
				out[i].Label = out[i].ID
			}
		}
		return out
	}
	out := make([]plateOption, 0, len(d.Choices))
	for _, id := range d.Choices {
		out = append(out, plateOption{ID: id, Label: id})
	}
	return out
}
