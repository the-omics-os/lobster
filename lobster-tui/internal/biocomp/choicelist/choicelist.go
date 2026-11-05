// Package choicelist implements the "choice_list" component: one screen with
// one row per item, and one option picked for each row (for example: retain,
// exclude or defer for every flagged sample).
//
// Keys: ↑/↓ move between rows (unavailable rows are skipped), ←/→ or 1-9 pick
// an option for the row, enter submits once every row is answered, and esc
// asks first: a second esc in a row cancels (any other key in between starts
// over). Picking an option that needs a reason opens a small text box under
// the row; while it is open it takes every printable key, enter saves the
// reason (an empty one is refused), esc discards the change and ←/→ still
// change the option.
//
// Reply: {"answers": {item_id: {"id": option_id, "fields": {key: text}}}}.
// "fields" is only present for options that need a reason. Unavailable items
// are never part of the reply.
package choicelist

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"charm.land/bubbles/v2/key"
	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"

	"github.com/the-omics-os/lobster-tui/internal/biocomp"
)

func init() {
	biocomp.Register("choice_list", func() biocomp.BioComponent {
		return &Component{}
	})
}

type item struct {
	ID          string `json:"id"`
	Label       string `json:"label"`
	Details     string `json:"details,omitempty"`
	Unavailable string `json:"unavailable,omitempty"`
}

// reasonSpec says that an option needs a reason, stored under Key.
type reasonSpec struct {
	Key      string `json:"key"`
	Question string `json:"question"`
}

type option struct {
	ID          string      `json:"id"`
	Label       string      `json:"label"`
	Details     string      `json:"details,omitempty"`
	Unavailable string      `json:"unavailable,omitempty"`
	Reason      *reasonSpec `json:"reason,omitempty"`
}

type answer struct {
	ID     string            `json:"id"`
	Fields map[string]string `json:"fields,omitempty"`
}

type fieldData struct {
	Question string            `json:"question"`
	Details  string            `json:"details,omitempty"`
	Items    []item            `json:"items"`
	Options  []option          `json:"options"`
	Default  map[string]answer `json:"default,omitempty"`
}

// reasonBox is the inline text box for the row being given a reason.
type reasonBox struct {
	open       bool
	item       int
	text       string
	prevOption int    // the row's option before the box opened (for esc)
	prevReason string // the row's reason before the box opened
}

// Component shows the items and the option picked for each.
type Component struct {
	question string
	details  string
	items    []item
	options  []option
	picked   []int    // picked[i] is the index of item i's option, or -1
	reasons  []string // reasons[i] is item i's saved reason
	cursor   int      // always an available item
	offset   int      // first visible row
	box      reasonBox
	message  string
	escArmed bool // the last key was a first esc; the next esc cancels
}

// escPrompt is shown after a first esc; only a second esc in a row cancels.
const escPrompt = "press esc again to cancel this question; nothing will be sent"

func (c *Component) Init(data json.RawMessage) error {
	d, err := parse(data)
	if err != nil {
		return err
	}
	c.question, c.details = d.Question, d.Details
	c.items, c.options = d.Items, d.Options
	c.picked = make([]int, len(c.items))
	c.reasons = make([]string, len(c.items))
	for i := range c.picked {
		c.picked[i] = -1
	}
	c.box, c.message, c.offset, c.escArmed = reasonBox{}, "", 0, false
	c.cursor = c.nextItem(-1, 1)

	optIndex := map[string]int{}
	for j, o := range c.options {
		optIndex[o.ID] = j
	}
	for i, it := range c.items {
		def, ok := d.Default[it.ID]
		if !ok || it.Unavailable != "" {
			continue // unknown or unavailable ids are ignored, never picked silently
		}
		j, ok := optIndex[def.ID]
		if !ok || c.options[j].Unavailable != "" {
			continue
		}
		c.picked[i] = j
		if r := c.options[j].Reason; r != nil {
			c.reasons[i] = strings.TrimSpace(def.Fields[r.Key])
		}
	}
	return nil
}

// parse reads and checks the data; Summary and Describe use it too.
func parse(data json.RawMessage) (fieldData, error) {
	var d fieldData
	if err := json.Unmarshal(data, &d); err != nil {
		return d, fmt.Errorf("choice_list: invalid data: %w", err)
	}
	if d.Question == "" {
		return d, fmt.Errorf("choice_list: question is required")
	}
	if len(d.Items) == 0 || len(d.Options) == 0 {
		return d, fmt.Errorf("choice_list: at least one item and one option are required")
	}
	seen := map[string]bool{}
	available := 0
	for i, it := range d.Items {
		if it.ID == "" {
			return d, fmt.Errorf("choice_list: item %d has no id", i)
		}
		if seen[it.ID] {
			return d, fmt.Errorf("choice_list: duplicate item id %q", it.ID)
		}
		seen[it.ID] = true
		if it.Label == "" {
			d.Items[i].Label = it.ID
		}
		if it.Unavailable == "" {
			available++
		}
	}
	if available == 0 {
		return d, fmt.Errorf("choice_list: every item is unavailable, nothing can be answered")
	}
	seen = map[string]bool{}
	available = 0
	for j, o := range d.Options {
		if o.ID == "" {
			return d, fmt.Errorf("choice_list: option %d has no id", j)
		}
		if seen[o.ID] {
			return d, fmt.Errorf("choice_list: duplicate option id %q", o.ID)
		}
		seen[o.ID] = true
		if o.Label == "" {
			d.Options[j].Label = o.ID
		}
		if o.Reason != nil {
			if o.Reason.Key == "" {
				return d, fmt.Errorf("choice_list: option %q asks for a reason without a key", o.ID)
			}
			if o.Reason.Question == "" {
				d.Options[j].Reason.Question = "Reason?"
			}
		}
		if o.Unavailable == "" {
			available++
		}
	}
	if available == 0 {
		return d, fmt.Errorf("choice_list: every option is unavailable")
	}
	return d, nil
}

// nextItem returns the next available item from i in direction step, or i
// when there is none.
func (c *Component) nextItem(i, step int) int {
	for j := i + step; j >= 0 && j < len(c.items); j += step {
		if c.items[j].Unavailable == "" {
			return j
		}
	}
	return i
}

// nextOption returns the next available option from j in direction step
// (from -1 going left means "the last one"), or j when there is none.
func (c *Component) nextOption(j, step int) int {
	if j < 0 && step < 0 {
		j = len(c.options)
	}
	for k := j + step; k >= 0 && k < len(c.options); k += step {
		if c.options[k].Unavailable == "" {
			return k
		}
	}
	return j
}

// optionOf returns the id of item i's option, or "".
func (c *Component) optionOf(i int) string {
	if c.picked[i] < 0 {
		return ""
	}
	return c.options[c.picked[i]].ID
}

// IsCapturingText reports whether the reason box is open. A dashboard checks
// it so that esc and tab reach the box instead of switching sections.
func (c *Component) IsCapturingText() bool { return c.box.open }

// oneLine turns pasted line breaks and tabs into spaces: a reason is one line.
var oneLine = strings.NewReplacer("\r\n", " ", "\n", " ", "\r", " ", "\t", " ")

func (c *Component) HandleMsg(msg tea.Msg) (*biocomp.ComponentResult, tea.Cmd) {
	if paste, ok := msg.(tea.PasteMsg); ok && c.box.open {
		c.box.text += oneLine.Replace(paste.Content)
		return nil, nil
	}
	km, ok := msg.(tea.KeyPressMsg)
	if !ok {
		return nil, nil
	}
	k := km.String()
	armed := c.escArmed && k == "esc"
	c.escArmed = false // any key but a second esc starts over
	if c.box.open {
		c.handleBox(km)
		return nil, nil
	}
	c.message = ""
	switch k {
	case "up", "k":
		c.cursor = c.nextItem(c.cursor, -1)
	case "down", "j":
		c.cursor = c.nextItem(c.cursor, 1)
	case "right":
		c.pick(c.cursor, c.nextOption(c.picked[c.cursor], 1))
	case "left":
		c.pick(c.cursor, c.nextOption(c.picked[c.cursor], -1))
	case "enter":
		if missing := c.missing(); missing != "" {
			c.message = missing
			return nil, nil
		}
		return &biocomp.ComponentResult{Action: "submit", Data: c.reply()}, nil
	case "esc":
		if armed {
			return &biocomp.ComponentResult{Action: "cancel", Data: map[string]any{}}, nil
		}
		c.escArmed, c.message = true, escPrompt
	case "ctrl+d":
		return &biocomp.ComponentResult{Action: "defer", Data: map[string]any{}}, nil
	default:
		if len(k) == 1 && k[0] >= '1' && k[0] <= '9' {
			if j := int(k[0] - '1'); j < len(c.options) {
				c.pick(c.cursor, j)
			}
		}
	}
	return nil, nil
}

// handleBox: every printable key is typed; enter saves, esc discards and
// ←/→ change the option.
func (c *Component) handleBox(km tea.KeyPressMsg) {
	i := c.box.item
	switch km.String() {
	case "enter":
		text := strings.TrimSpace(c.box.text)
		if text == "" {
			c.message = fmt.Sprintf("A reason is needed for %s (%s): type it and press enter, or esc to go back.",
				c.options[c.picked[i]].Label, c.items[i].ID)
			return
		}
		c.reasons[i] = text
		c.box = reasonBox{}
		c.message = ""
	case "esc":
		c.picked[i], c.reasons[i] = c.box.prevOption, c.box.prevReason
		c.box = reasonBox{}
		c.message = ""
	case "backspace":
		if r := []rune(c.box.text); len(r) > 0 {
			c.box.text = string(r[:len(r)-1])
		}
	case "right":
		c.pick(i, c.nextOption(c.picked[i], 1))
	case "left":
		c.pick(i, c.nextOption(c.picked[i], -1))
	default:
		if km.Text != "" && km.Mod&(tea.ModCtrl|tea.ModAlt) == 0 {
			c.box.text += km.Text
		}
	}
}

// pick sets item i's option to j. An option that needs a reason opens the box.
func (c *Component) pick(i, j int) {
	if j < 0 {
		return
	}
	o := c.options[j]
	if o.Unavailable != "" {
		c.message = fmt.Sprintf("%s can't be picked: %s", o.Label, o.Unavailable)
		return
	}
	inBox := c.box.open && c.box.item == i
	if !inBox {
		c.box.prevOption, c.box.prevReason = c.picked[i], c.reasons[i]
	}
	if j != c.picked[i] {
		c.reasons[i] = ""
	}
	if o.Reason == nil {
		c.picked[i] = j
		c.box = reasonBox{}
		return
	}
	if !inBox {
		c.box.text = c.reasons[i]
	}
	c.picked[i] = j
	c.box.open, c.box.item = true, i
}

// missing explains what stops a submit, or returns "".
func (c *Component) missing() string {
	var noOption, noReason []string
	for i, it := range c.items {
		if it.Unavailable != "" {
			continue
		}
		switch {
		case c.picked[i] < 0:
			noOption = append(noOption, it.ID)
		case c.options[c.picked[i]].Reason != nil && c.reasons[i] == "":
			noReason = append(noReason, it.ID)
		}
	}
	var parts []string
	if len(noOption) > 0 {
		parts = append(parts, "choose an option for "+listIDs(noOption))
	}
	if len(noReason) > 0 {
		parts = append(parts, "give a reason for "+listIDs(noReason))
	}
	if len(parts) == 0 {
		return ""
	}
	return "Not finished: " + strings.Join(parts, "; ") + "."
}

// listIDs names up to 10 ids, then how many more.
func listIDs(ids []string) string {
	if len(ids) <= 10 {
		return strings.Join(ids, ", ")
	}
	return fmt.Sprintf("%s and %d more", strings.Join(ids[:10], ", "), len(ids)-10)
}

// answers returns the reply's answers for every row that has an option.
func (c *Component) answers() map[string]any {
	out := map[string]any{}
	for i, it := range c.items {
		if it.Unavailable != "" || c.picked[i] < 0 {
			continue
		}
		o := c.options[c.picked[i]]
		a := map[string]any{"id": o.ID}
		if o.Reason != nil && c.reasons[i] != "" {
			a["fields"] = map[string]any{o.Reason.Key: c.reasons[i]}
		}
		out[it.ID] = a
	}
	return out
}

func (c *Component) reply() map[string]any { return map[string]any{"answers": c.answers()} }

// Draft returns what is on screen in the reply's shape, so a dashboard can
// tell whether it differs from the answer it recorded.
func (c *Component) Draft() map[string]any { return c.reply() }

// summary is one line of counts per option, e.g.
// "Retain 60 · Exclude 4 · Defer 2 · undecided 6".
func (c *Component) summary() string {
	ids := make(map[string]string, len(c.items))
	for i := range c.items {
		ids[c.items[i].ID] = c.optionOf(i)
	}
	return summaryLine(c.items, c.options, ids)
}

func summaryLine(items []item, options []option, picked map[string]string) string {
	counts := map[string]int{}
	undecided, unavailable := 0, 0
	for _, it := range items {
		switch {
		case it.Unavailable != "":
			unavailable++
		case picked[it.ID] == "":
			undecided++
		default:
			counts[picked[it.ID]]++
		}
	}
	var parts []string
	for _, o := range options {
		if o.Unavailable == "" {
			parts = append(parts, fmt.Sprintf("%s %d", o.Label, counts[o.ID]))
		}
	}
	parts = append(parts, fmt.Sprintf("undecided %d", undecided))
	if unavailable > 0 {
		parts = append(parts, fmt.Sprintf("unavailable %d", unavailable))
	}
	return strings.Join(parts, " · ")
}

// replyAnswers reads {"answers": {item: {"id", "fields"}}} from a reply.
func replyAnswers(reply map[string]any) map[string]answer {
	raw, _ := json.Marshal(reply["answers"])
	var out map[string]answer
	_ = json.Unmarshal(raw, &out)
	return out
}

// Summary is the summary line for a reply to the given data (the dashboard
// shows it next to the section title).
func Summary(data json.RawMessage, reply map[string]any) string {
	d, err := parse(data)
	if err != nil {
		return "answered"
	}
	picked := map[string]string{}
	for id, a := range replyAnswers(reply) {
		picked[id] = a.ID
	}
	return summaryLine(d.Items, d.Options, picked)
}

// Describe lists every answer in a reply, one line per item, in the data's
// item order: "Donor A (S01) → Exclude (exclude) · reason: low depth".
func Describe(data json.RawMessage, reply map[string]any) []string {
	d, _ := parse(data)
	answers := replyAnswers(reply)
	itemLabel := map[string]string{}
	var order []string
	for _, it := range d.Items {
		itemLabel[it.ID] = it.Label
		if _, ok := answers[it.ID]; ok {
			order = append(order, it.ID)
		}
	}
	var extra []string
	for id := range answers {
		if _, ok := itemLabel[id]; !ok {
			extra = append(extra, id)
		}
	}
	sort.Strings(extra)
	optLabel := map[string]string{}
	for _, o := range d.Options {
		optLabel[o.ID] = o.Label
	}
	var lines []string
	for _, id := range append(order, extra...) {
		a := answers[id]
		line := withID(itemLabel[id], id) + " → " + withID(optLabel[a.ID], a.ID)
		keys := make([]string, 0, len(a.Fields))
		for k := range a.Fields {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			line += fmt.Sprintf(" · %s: %s", k, a.Fields[k])
		}
		lines = append(lines, line)
	}
	if len(lines) == 0 {
		lines = append(lines, "nothing answered")
	}
	return lines
}

// withID is "label (id)"; just the id when there is no label, and just the
// label when it already holds the id.
func withID(label, id string) string {
	switch {
	case label == "":
		return id
	case strings.Contains(label, id): // the label already names it
		return label
	}
	return label + " (" + id + ")"
}

var (
	cursorStyle = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("63"))
	normalStyle = lipgloss.NewStyle()
	dimStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color("240"))
	warnStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color("214"))
	boxStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color("63"))
)

// hang renders text wrapped to the width, after a prefix; continuation lines
// are indented under the text, never cut.
func hang(prefix, text string, width int, style lipgloss.Style) string {
	w := max(width-lipgloss.Width(prefix), 1)
	return lipgloss.JoinHorizontal(lipgloss.Top, prefix, style.Width(w).Render(text))
}

func (c *Component) View(width, height int) string {
	top := []string{hang("", c.question, width, lipgloss.NewStyle().Bold(true))}
	if c.details != "" {
		top = append(top, hang("", c.details, width, dimStyle))
	}
	for j, o := range c.options {
		top = append(top, c.optionLine(j, o, width))
	}
	top = append(top, "")

	bottom := []string{"", hang("", c.summary(), width, dimStyle)}
	if c.message != "" {
		bottom = append(bottom, hang("", c.message, width, warnStyle))
	}

	rows := make([]string, len(c.items))
	heights := make([]int, len(c.items))
	for i := range c.items {
		rows[i] = c.row(i, width)
		heights[i] = lipgloss.Height(rows[i])
	}
	avail := height - lipgloss.Height(strings.Join(top, "\n")) - lipgloss.Height(strings.Join(bottom, "\n"))
	start, end := c.window(heights, max(avail, 3))
	if start > 0 || end < len(rows) {
		// Some rows are out of view: keep a line above and below to say so.
		start, end = c.window(heights, max(avail-2, 3))
	}

	lines := append([]string{}, top...)
	if start > 0 || end < len(rows) {
		lines = append(lines, moreLine("↑", start))
	}
	lines = append(lines, rows[start:end]...)
	if start > 0 || end < len(rows) {
		lines = append(lines, moreLine("↓", len(rows)-end))
	}
	lines = append(lines, bottom...)
	return strings.Join(lines, "\n")
}

// moreLine says how many rows are out of view in one direction ("" when none).
func moreLine(arrow string, n int) string {
	if n == 0 {
		return ""
	}
	return dimStyle.Render(fmt.Sprintf("  %s %d more", arrow, n))
}

// window picks the rows to show so the cursor row is fully visible, keeping
// the scroll position stable while it is.
func (c *Component) window(heights []int, avail int) (int, int) {
	if c.cursor < c.offset {
		c.offset = c.cursor
	}
	used := func(from, to int) int {
		n := 0
		for i := from; i <= to; i++ {
			n += heights[i]
		}
		return n
	}
	for c.offset < c.cursor && used(c.offset, c.cursor) > avail {
		c.offset++
	}
	end, lines := c.offset, 0
	for end < len(heights) && (end == c.offset || lines+heights[end] <= avail) {
		lines += heights[end]
		end++
	}
	return c.offset, end
}

func (c *Component) optionLine(j int, o option, width int) string {
	prefix := "  · "
	if j < 9 {
		prefix = fmt.Sprintf("  %d ", j+1)
	}
	text := withID(o.Label, o.ID)
	if o.Details != "" {
		text += " — " + o.Details
	}
	if o.Reason != nil {
		text += " · needs a reason"
	}
	if o.Unavailable != "" {
		return hang("  ✕ ", text+" · unavailable: "+o.Unavailable, width, dimStyle)
	}
	return hang(prefix, text, width, dimStyle)
}

func (c *Component) row(i, width int) string {
	it := c.items[i]
	pointer, style := "  ", normalStyle
	if i == c.cursor {
		pointer, style = "> ", cursorStyle
	}
	w := max(width-4, 1)
	label := withID(it.Label, it.ID)
	if it.Unavailable != "" {
		parts := []string{dimStyle.Width(w).Render(label)}
		if it.Details != "" {
			parts = append(parts, dimStyle.Width(w).Render(it.Details))
		}
		parts = append(parts, dimStyle.Width(w).Render("✕ unavailable: "+it.Unavailable))
		return lipgloss.JoinHorizontal(lipgloss.Top, "    ", strings.Join(parts, "\n"))
	}

	mark := "○ "
	if c.picked[i] >= 0 {
		mark = "● "
	}
	parts := []string{style.Width(w).Render(label)}
	if it.Details != "" {
		parts = append(parts, dimStyle.Width(w).Render(it.Details))
	}
	switch {
	case c.picked[i] < 0:
		parts = append(parts, dimStyle.Width(w).Render("→ not decided yet"))
	default:
		o := c.options[c.picked[i]]
		choice := "→ " + o.Label
		switch {
		case o.Reason == nil:
			parts = append(parts, normalStyle.Width(w).Render(choice))
		case c.reasons[i] != "" && !(c.box.open && c.box.item == i):
			parts = append(parts, normalStyle.Width(w).Render(choice+" · "+o.Reason.Key+": "+c.reasons[i]))
		case c.box.open && c.box.item == i:
			parts = append(parts, normalStyle.Width(w).Render(choice))
		default:
			parts = append(parts, warnStyle.Width(w).Render(choice+" · reason needed"))
		}
	}
	if c.box.open && c.box.item == i {
		q := c.options[c.picked[i]].Reason.Question
		parts = append(parts,
			boxStyle.Width(w).Render(q),
			hang("┃ ", c.box.text+"▏", w, normalStyle),
			dimStyle.Width(w).Render("enter save · esc discard"))
	}
	return lipgloss.JoinHorizontal(lipgloss.Top, pointer+mark, strings.Join(parts, "\n"))
}

func (c *Component) SetData(data json.RawMessage) error {
	return fmt.Errorf("choice_list: live updates are not supported")
}

func (c *Component) InitCmd() tea.Cmd            { return nil }
func (c *Component) Name() string                { return "choice_list" }
func (c *Component) Mode() string                { return "overlay" }
func (c *Component) ChangeEvent() map[string]any { return nil }

func (c *Component) KeyBindings() []key.Binding {
	if c.box.open {
		return []key.Binding{
			key.NewBinding(key.WithKeys("enter"), key.WithHelp("enter", "save reason")),
			key.NewBinding(key.WithKeys("esc"), key.WithHelp("esc", "discard")),
			key.NewBinding(key.WithKeys("left", "right"), key.WithHelp("←/→", "option")),
		}
	}
	return []key.Binding{
		key.NewBinding(key.WithKeys("up", "down"), key.WithHelp("↑/↓", "row")),
		key.NewBinding(key.WithKeys("left", "right"), key.WithHelp("←/→/1-9", "option")),
		key.NewBinding(key.WithKeys("enter"), key.WithHelp("enter", "submit")),
		key.NewBinding(key.WithKeys("ctrl+d"), key.WithHelp("ctrl+d", "defer")),
		// Help key stays "esc" so a dashboard can leave it out (it has its own).
		key.NewBinding(key.WithKeys("esc"), key.WithHelp("esc", "twice: cancel")),
	}
}
