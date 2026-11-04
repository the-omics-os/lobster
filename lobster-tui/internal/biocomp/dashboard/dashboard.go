// Package dashboard implements the ask_user "dashboard" view: several linked
// decisions on one screen. Each section hosts an ordinary BioComp component
// (choice, choice_list, threshold_slider, text_input, plate_map), created from
// the registry once, so switching sections keeps unsaved work.
//
// Sections list the keys they depend on. When a section's answer changes,
// every answered section that depends on it (directly or through others) is
// marked stale with the reason, and the dashboard can't be submitted until
// each stale section is checked again. The dashboard knows nothing about the
// science: dependencies come from the request, not from domain rules.
//
// A re-asked dashboard can carry "problems" (shown above the sections) and,
// per section, "problems" (shown under it; the section starts unanswered) and
// "answer" (the previous reply of that section: without problems it starts
// answered; with problems it only pre-fills the section and is the baseline
// for staleness, so recording the same value again changes nothing).
//
// Keys on the sections: tab / shift+tab move between sections, ctrl+s opens
// the review screen, ctrl+d defers (nothing is applied), esc twice cancels.
// While the active section is typing text (a reason box, or any text_input
// section), esc and tab go to it. On the review screen: enter sends,
// b / esc / shift+tab go back, ↑/↓ scroll.
//
// Reply: {"answers": {section key: that section component's own reply}}.
package dashboard

import (
	"encoding/json"
	"fmt"
	"reflect"
	"sort"
	"strings"

	"charm.land/bubbles/v2/key"
	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"

	"github.com/the-omics-os/lobster-tui/internal/biocomp"
	"github.com/the-omics-os/lobster-tui/internal/biocomp/choicelist"
)

func init() {
	biocomp.Register("dashboard", func() biocomp.BioComponent {
		return &Component{}
	})
}

// Optional interfaces a section component may implement. They are checked
// with type assertions, so upstream components need none of them.
type (
	// capturer: the component is typing text, so esc and tab belong to it.
	capturer interface{ IsCapturingText() bool }
	// drafter: what is on screen, in the component's reply shape.
	drafter interface{ Draft() map[string]any }
	// prefiller: restore an earlier reply that the data can't express.
	prefiller interface{ Prefill(answer map[string]any) }
)

const escPrompt = "press esc again to cancel everything; nothing will be applied"

type sectionData struct {
	Key       string          `json:"key"`
	Title     string          `json:"title"`
	DependsOn []string        `json:"depends_on"`
	Component string          `json:"component"`
	Data      json.RawMessage `json:"data"`
	Problems  []string        `json:"problems,omitempty"`
	Answer    map[string]any  `json:"answer,omitempty"`
}

type dashboardData struct {
	Question string        `json:"question"`
	Problems []string      `json:"problems,omitempty"`
	Sections []sectionData `json:"sections"`
}

type status int

const (
	todo status = iota
	answered
	stale
)

type section struct {
	sectionData
	child        biocomp.BioComponent
	status       status
	answer       map[string]any // the recorded reply (JSON-normalised)
	summary      string         // short text for the answer, e.g. "SRC-A"
	staleWhy     string         // why it needs checking again
	dependents   []int          // sections that list this one in depends_on
	showProblems bool           // problems from the request, until re-answered
	// baseline is the previous answer of a section re-asked with problems:
	// recording it again unchanged makes nothing stale.
	baseline        map[string]any
	baselineSummary string
}

// Component is the dashboard: the sections, each with its own component, and
// the review screen.
type Component struct {
	question string
	problems []string
	sections []*section
	active   int
	initCmd  tea.Cmd
	message  string
	escArmed bool

	reviewing    bool
	pending      map[string]any // the reply shown on the review screen
	reviewOffset int
}

func (c *Component) Init(data json.RawMessage) error {
	var d dashboardData
	if err := json.Unmarshal(data, &d); err != nil {
		return fmt.Errorf("dashboard: invalid data: %w", err)
	}
	if len(d.Sections) < 2 {
		return fmt.Errorf("dashboard: at least 2 sections are required")
	}
	index := map[string]int{}
	*c = Component{question: d.Question, problems: d.Problems}
	var cmds []tea.Cmd
	for i, sd := range d.Sections {
		if sd.Key == "" {
			return fmt.Errorf("dashboard: section %d has no key", i)
		}
		if _, dup := index[sd.Key]; dup {
			return fmt.Errorf("dashboard: duplicate section key %q", sd.Key)
		}
		if sd.Component == "dashboard" || biocomp.Get(sd.Component) == nil {
			return fmt.Errorf("dashboard: section %q: unknown component %q", sd.Key, sd.Component)
		}
		// Every section must be drawable before the dashboard is shown.
		child := biocomp.Get(sd.Component)()
		if err := child.Init(sd.Data); err != nil {
			return fmt.Errorf("dashboard: section %q: %w", sd.Key, err)
		}
		if sd.Title == "" {
			sd.Title = sd.Key
		}
		s := &section{sectionData: sd, child: child, showProblems: len(sd.Problems) > 0}
		if sd.Answer != nil {
			answer := normalize(sd.Answer)
			s.child = prefilled(s, answer)
			if len(sd.Problems) == 0 {
				s.answer, s.status = answer, answered
				s.summary = summarize(s.sectionData, answer)
			} else {
				s.baseline, s.baselineSummary = answer, summarize(s.sectionData, answer)
			}
		}
		cmds = append(cmds, s.child.InitCmd())
		index[sd.Key] = i
		c.sections = append(c.sections, s)
	}
	for i, s := range c.sections {
		for _, dep := range s.DependsOn {
			j, ok := index[dep]
			if !ok || j == i {
				return fmt.Errorf("dashboard: section %q depends on unknown section %q", s.Key, dep)
			}
			c.sections[j].dependents = append(c.sections[j].dependents, i)
		}
	}
	c.initCmd = tea.Batch(cmds...)
	for i, s := range c.sections {
		if s.status != answered {
			c.active = i
			break
		}
	}
	return nil
}

// prefilled returns the section's component showing an earlier answer.
// Choice, choice_list and slider take it as their data's default; a plate
// restores it itself; a text box gets the text typed in. If the pre-filled
// data is refused, the plain component is kept (its draft then differs from
// the answer, which the submit check reports).
func prefilled(s *section, answer map[string]any) biocomp.BioComponent {
	if p, ok := s.child.(prefiller); ok {
		p.Prefill(answer)
		return s.child
	}
	if s.Component == "text_input" {
		if text, ok := answer["answer"].(string); ok {
			s.child.HandleMsg(tea.PasteMsg{Content: text})
		}
		return s.child
	}
	var d map[string]any
	if err := json.Unmarshal(s.Data, &d); err != nil {
		return s.child
	}
	switch s.Component {
	case "choice":
		d["default"] = selectedIDs(answer)
	case "choice_list":
		d["default"] = answer["answers"]
	case "threshold_slider":
		d["default"] = answer["value"]
	default:
		return s.child
	}
	raw, err := json.Marshal(d)
	if err != nil {
		return s.child
	}
	fresh := biocomp.Get(s.Component)()
	if err := fresh.Init(raw); err != nil {
		return s.child
	}
	return fresh
}

// normalize gives a reply the shape JSON decoding gives it ([]any,
// map[string]any, float64), so answers from Python and from Go compare
// equal, and drops the "action" key some components add.
func normalize(v map[string]any) map[string]any {
	out := map[string]any{}
	raw, err := json.Marshal(v)
	if err != nil {
		return v
	}
	_ = json.Unmarshal(raw, &out)
	delete(out, "action")
	return out
}

func (c *Component) InitCmd() tea.Cmd {
	cmd := c.initCmd
	c.initCmd = nil
	return cmd
}

func (c *Component) child() biocomp.BioComponent { return c.sections[c.active].child }

// capturing reports whether the active section is typing text. A text_input
// section always is (it is upstream and has no way to say so).
func (c *Component) capturing() bool {
	child := c.child()
	if child.Name() == "text_input" {
		return true
	}
	if t, ok := child.(capturer); ok {
		return t.IsCapturingText()
	}
	return false
}

func (c *Component) HandleMsg(msg tea.Msg) (*biocomp.ComponentResult, tea.Cmd) {
	km, isKey := msg.(tea.KeyPressMsg)
	if c.reviewing {
		if isKey {
			return c.handleReview(km), nil
		}
		_, cmd := c.child().HandleMsg(msg) // e.g. cursor blink; nothing to record
		return nil, cmd
	}
	if isKey {
		k := km.String()
		capturing := c.capturing()
		if c.escArmed && (k != "esc" || capturing) {
			c.escArmed, c.message = false, ""
		}
		switch {
		case k == "tab" && !capturing:
			c.move(1)
			return nil, nil
		case k == "shift+tab":
			c.move(-1)
			return nil, nil
		case k == "ctrl+s":
			if blockers := c.blockers(); len(blockers) > 0 {
				c.message = "Can't submit yet: " + strings.Join(blockers, "; ")
				return nil, nil
			}
			c.reviewing, c.pending, c.reviewOffset, c.message = true, c.reply(), 0, ""
			return nil, nil
		case k == "ctrl+d":
			return &biocomp.ComponentResult{Action: "defer", Data: map[string]any{}}, nil
		case k == "esc" && !capturing:
			if c.escArmed {
				return &biocomp.ComponentResult{Action: "cancel", Data: map[string]any{}}, nil
			}
			c.escArmed, c.message = true, escPrompt
			return nil, nil
		}
		c.message = ""
	}
	result, cmd := c.child().HandleMsg(msg)
	if result == nil || result.Action != "submit" {
		return nil, cmd // still answering (a child's own cancel is ignored)
	}
	c.record(c.active, result.Data)
	if next := c.nextOpen(); next >= 0 {
		c.active = next
	} else {
		c.message = "Every section is answered. Press ctrl+s to review and submit."
	}
	return nil, cmd
}

// handleReview: enter sends what is shown, b / esc / shift+tab go back.
func (c *Component) handleReview(km tea.KeyPressMsg) *biocomp.ComponentResult {
	switch km.String() {
	case "enter":
		return &biocomp.ComponentResult{Action: "submit", Data: c.pending}
	case "b", "esc", "shift+tab":
		c.reviewing, c.pending, c.message = false, nil, ""
	case "up", "k":
		c.reviewOffset--
	case "down", "j":
		c.reviewOffset++
	case "pgup":
		c.reviewOffset -= 10
	case "pgdown", "space":
		c.reviewOffset += 10
	case "ctrl+d":
		return &biocomp.ComponentResult{Action: "defer", Data: map[string]any{}}
	}
	return nil
}

// move switches to the next/previous section, wrapping around. The sections'
// components are kept, so unsaved work stays where it was.
func (c *Component) move(step int) {
	n := len(c.sections)
	c.active = ((c.active+step)%n + n) % n
	c.message = ""
}

// record stores section i's answer. If it changed, every answered section
// that depends on i, directly or through others, becomes stale.
func (c *Component) record(i int, data map[string]any) {
	s := c.sections[i]
	before, beforeSummary := s.answer, s.summary
	if before == nil && s.baseline != nil {
		before, beforeSummary = s.baseline, s.baselineSummary
	}
	answer := normalize(data)
	s.answer, s.status, s.staleWhy, s.showProblems = answer, answered, "", false
	s.summary = summarize(s.sectionData, answer)
	if sameAnswer(before, answer) {
		return
	}
	why := fmt.Sprintf("%q was answered", s.Title)
	if before != nil {
		why = fmt.Sprintf("%q changed: %s → %s", s.Title, beforeSummary, s.summary)
	}
	seen := map[int]bool{i: true}
	queue := append([]int(nil), s.dependents...)
	for len(queue) > 0 {
		j := queue[0]
		queue = queue[1:]
		if seen[j] {
			continue
		}
		seen[j] = true
		if d := c.sections[j]; d.status != todo {
			d.status, d.staleWhy = stale, why
		}
		queue = append(queue, c.sections[j].dependents...)
	}
}

// sameAnswer reports whether a newly recorded reply keeps the earlier answer.
// Only the keys the earlier answer has are compared: an answer sent with the
// request may leave out what the component works out from it and its data
// (a plate's "changes" and "reviewed_flags" follow from its "assignments"),
// so the full reply to the same decision still counts as the same. A key the
// earlier answer has and the new reply lacks counts as a change.
func sameAnswer(before, after map[string]any) bool {
	if len(before) == 0 {
		return reflect.DeepEqual(before, after)
	}
	for k, v := range before {
		if !reflect.DeepEqual(v, after[k]) {
			return false
		}
	}
	return true
}

// nextOpen is the first section after the active one that isn't answered,
// wrapping around, or -1 when all are answered.
func (c *Component) nextOpen() int {
	n := len(c.sections)
	for step := 1; step <= n; step++ {
		if j := (c.active + step) % n; c.sections[j].status != answered {
			return j
		}
	}
	return -1
}

// unsaved reports whether an answered section shows something other than
// its recorded answer. Components without a draft never count.
func (c *Component) unsaved(s *section) bool {
	d, ok := s.child.(drafter)
	if !ok || s.answer == nil {
		return false
	}
	draft := d.Draft()
	if draft == nil {
		return false
	}
	for k, v := range normalize(draft) {
		if !reflect.DeepEqual(v, s.answer[k]) {
			return true
		}
	}
	return false
}

// blockers explains what stops a submit: unanswered, stale and unsaved
// sections.
func (c *Component) blockers() []string {
	var out []string
	for _, s := range c.sections {
		switch {
		case s.status == todo:
			out = append(out, fmt.Sprintf("%q is not answered", s.Title))
		case s.status == stale:
			out = append(out, fmt.Sprintf("%q needs re-checking (%s)", s.Title, s.staleWhy))
		case c.unsaved(s):
			out = append(out, fmt.Sprintf("unsaved changes in %q: press enter in that section to keep them", s.Title))
		}
	}
	return out
}

func (c *Component) reply() map[string]any {
	answers := map[string]any{}
	for _, s := range c.sections {
		answers[s.Key] = s.answer
	}
	return map[string]any{"answers": answers}
}

var (
	titleStyle  = lipgloss.NewStyle().Bold(true)
	activeStyle = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("63"))
	doneStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color("42"))
	dimStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color("240"))
	warnStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color("214"))
	plainStyle  = lipgloss.NewStyle()
)

// hang renders text wrapped to the width, after a prefix; continuation lines
// are indented under the text, never cut.
func hang(prefix, text string, width int, style lipgloss.Style) string {
	w := max(width-lipgloss.Width(prefix), 1)
	return lipgloss.JoinHorizontal(lipgloss.Top, prefix, style.Width(w).Render(text))
}

func (c *Component) View(width, height int) string {
	if c.reviewing {
		return c.reviewView(width, height)
	}
	top := []string{hang("", c.question, width, titleStyle)}
	for _, p := range c.problems {
		top = append(top, hang("⚠ ", p, width, warnStyle))
	}
	for i, s := range c.sections {
		top = append(top, c.sectionLine(i, s, width))
		if s.status == stale {
			top = append(top, hang("     ⚠ ", "Re-check: "+s.staleWhy, width, warnStyle))
		}
		if s.showProblems {
			for _, p := range s.Problems {
				top = append(top, hang("     ⚠ ", p, width, warnStyle))
			}
		}
	}
	top = append(top, dimStyle.Render(strings.Repeat("─", max(width, 1))))

	bottom := []string{dimStyle.Render(c.progress())}
	if c.message != "" {
		bottom = append(bottom, hang("", c.message, width, warnStyle))
	}

	childHeight := height - lipgloss.Height(strings.Join(top, "\n")) - lipgloss.Height(strings.Join(bottom, "\n"))
	if childHeight < 5 {
		childHeight = 5
	}
	body := c.child().View(width, childHeight)
	return strings.Join(append(append(top, body), bottom...), "\n")
}

func (c *Component) sectionLine(i int, s *section, width int) string {
	icon, style := "○", plainStyle
	switch s.status {
	case answered:
		icon, style = "✓", doneStyle
	case stale:
		icon, style = "⚠", warnStyle
	}
	pointer := "  "
	if i == c.active {
		pointer, style = "▸ ", activeStyle
	}
	text := style.Render(s.Title)
	if s.summary != "" {
		text += dimStyle.Render("  " + s.summary)
	}
	return hang(style.Render(fmt.Sprintf("%s%s %d. ", pointer, icon, i+1)), text, width, plainStyle)
}

// reviewView shows the reply that enter will send, section by section, in
// full, wrapped and scrollable.
func (c *Component) reviewView(width, height int) string {
	var content []string
	content = append(content,
		hang("", "Review before sending: "+c.question, width, titleStyle),
		dimStyle.Width(max(width, 1)).Render("This is exactly what will be sent."),
		"")
	answers, _ := c.pending["answers"].(map[string]any)
	for i, s := range c.sections {
		content = append(content, hang(fmt.Sprintf("%d. ", i+1), s.Title, width, titleStyle))
		answer, _ := answers[s.Key].(map[string]any)
		for _, line := range describe(s.sectionData, answer) {
			content = append(content, hang("     ", line, width, plainStyle))
		}
	}
	lines := strings.Split(strings.Join(content, "\n"), "\n")

	avail := max(height-2, 1)
	c.reviewOffset = min(max(c.reviewOffset, 0), max(len(lines)-avail, 0))
	end := min(c.reviewOffset+avail, len(lines))
	footer := fmt.Sprintf("enter send · b back · ↑/↓ scroll · lines %d–%d of %d",
		c.reviewOffset+1, end, len(lines))
	out := append([]string{}, lines[c.reviewOffset:end]...)
	out = append(out, dimStyle.Render(strings.Repeat("─", max(width, 1))), hang("", footer, width, dimStyle))
	return strings.Join(out, "\n")
}

func (c *Component) progress() string {
	done, stales := 0, 0
	for _, s := range c.sections {
		switch s.status {
		case answered:
			done++
		case stale:
			stales++
		}
	}
	text := fmt.Sprintf("%d/%d answered", done, len(c.sections))
	if stales > 0 {
		text += fmt.Sprintf(" · %d to re-check", stales)
	}
	return text
}

func (c *Component) SetData(data json.RawMessage) error {
	return fmt.Errorf("dashboard: live updates are not supported")
}

func (c *Component) Name() string { return "dashboard" }
func (c *Component) Mode() string { return "overlay" }

func (c *Component) KeyBindings() []key.Binding {
	if c.reviewing {
		return []key.Binding{
			key.NewBinding(key.WithKeys("enter"), key.WithHelp("enter", "send")),
			key.NewBinding(key.WithKeys("b", "esc", "shift+tab"), key.WithHelp("b/esc", "back")),
			key.NewBinding(key.WithKeys("up", "down"), key.WithHelp("↑/↓", "scroll")),
			key.NewBinding(key.WithKeys("ctrl+d"), key.WithHelp("ctrl+d", "defer")),
		}
	}
	if c.capturing() {
		own := []key.Binding{
			key.NewBinding(key.WithKeys("shift+tab"), key.WithHelp("shift+tab", "section")),
			key.NewBinding(key.WithKeys("ctrl+s"), key.WithHelp("ctrl+s", "review")),
			key.NewBinding(key.WithKeys("ctrl+d"), key.WithHelp("ctrl+d", "defer")),
		}
		return append(own, c.child().KeyBindings()...)
	}
	own := []key.Binding{
		key.NewBinding(key.WithKeys("tab", "shift+tab"), key.WithHelp("tab", "section")),
		key.NewBinding(key.WithKeys("ctrl+s"), key.WithHelp("ctrl+s", "review")),
		key.NewBinding(key.WithKeys("ctrl+d"), key.WithHelp("ctrl+d", "defer")),
		key.NewBinding(key.WithKeys("esc"), key.WithHelp("esc esc", "cancel")),
	}
	for _, b := range c.child().KeyBindings() {
		if h := b.Help().Key; h != "esc" && h != "tab" {
			own = append(own, b)
		}
	}
	return own
}

func (c *Component) ChangeEvent() map[string]any { return nil }

// ---------------------------------------------------------------------------
// Per-component helpers: summaries for the section list, full descriptions
// for the review screen
// ---------------------------------------------------------------------------

// summarize describes an answer in a few words for the section list.
func summarize(s sectionData, answer map[string]any) string {
	switch s.Component {
	case "choice":
		labels := optionLabels(s.Data)
		var picked []string
		for _, id := range selectedIDs(answer) {
			picked = append(picked, withID(labels[id], id))
		}
		if len(picked) == 0 {
			return "none picked"
		}
		return strings.Join(picked, ", ")
	case "choice_list":
		return choicelist.Summary(s.Data, answer)
	case "threshold_slider":
		return sliderText(s.Data, answer)
	case "plate_map":
		counts := map[string]int{}
		if assignments, ok := answer["assignments"].(map[string]any); ok {
			for _, v := range assignments {
				counts[fmt.Sprint(v)]++
			}
		}
		keys := make([]string, 0, len(counts))
		for k := range counts {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		parts := make([]string, 0, len(keys))
		for _, k := range keys {
			parts = append(parts, fmt.Sprintf("%s %d", k, counts[k]))
		}
		return strings.Join(parts, " · ")
	case "text_input":
		text := []rune(fmt.Sprint(answer["answer"]))
		if len(text) > 40 { // a summary: the review screen shows it in full
			text = append(text[:39], '…')
		}
		return fmt.Sprintf("%q", string(text))
	}
	return "answered"
}

// describe lists a section's answer in full for the review screen.
func describe(s sectionData, answer map[string]any) []string {
	if answer == nil {
		return []string{"(no answer)"}
	}
	switch s.Component {
	case "choice":
		labels := optionLabels(s.Data)
		var lines []string
		for _, id := range selectedIDs(answer) {
			lines = append(lines, withID(labels[id], id))
		}
		if len(lines) == 0 {
			return []string{"none picked"}
		}
		return lines
	case "choice_list":
		return choicelist.Describe(s.Data, answer)
	case "threshold_slider":
		return []string{sliderText(s.Data, answer)}
	case "plate_map":
		return describePlate(s.Data, answer)
	case "text_input":
		if text := fmt.Sprint(answer["answer"]); text != "" {
			return []string{text}
		}
		return []string{"(empty)"}
	}
	raw, _ := json.Marshal(answer)
	return []string{string(raw)}
}

// describePlate groups the assignments by choice, in the plate's choice
// order and sample order, then lists what changed from the data and the
// flags that were reviewed.
func describePlate(data json.RawMessage, answer map[string]any) []string {
	var d struct {
		Choices []string `json:"choices"`
		Wells   []struct {
			SampleID string `json:"sample_id"`
		} `json:"wells"`
	}
	_ = json.Unmarshal(data, &d)
	assignments, _ := answer["assignments"].(map[string]any)
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
	groups := map[string][]string{}
	for _, id := range append(order, extra...) {
		v := fmt.Sprint(assignments[id])
		groups[v] = append(groups[v], id)
	}
	values := append([]string{}, d.Choices...)
	var others []string
	for v := range groups {
		if !contains(d.Choices, v) {
			others = append(others, v)
		}
	}
	sort.Strings(others)
	var lines []string
	for _, v := range append(values, others...) {
		if ids := groups[v]; len(ids) > 0 {
			lines = append(lines, fmt.Sprintf("%s (%d): %s", v, len(ids), strings.Join(ids, ", ")))
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

func sliderText(data json.RawMessage, answer map[string]any) string {
	var d struct {
		Unit string `json:"unit"`
	}
	_ = json.Unmarshal(data, &d)
	return strings.TrimSpace(fmt.Sprintf("%v %s", answer["value"], d.Unit))
}

func asMaps(v any) []map[string]any {
	var out []map[string]any
	if list, ok := v.([]any); ok {
		for _, e := range list {
			if m, ok := e.(map[string]any); ok {
				out = append(out, m)
			}
		}
	}
	return out
}

func contains(list []string, v string) bool {
	for _, x := range list {
		if x == v {
			return true
		}
	}
	return false
}

// withID is "label (id)", or just the id when there is no other label.
func withID(label, id string) string {
	if label == "" || label == id {
		return id
	}
	return label + " (" + id + ")"
}

func selectedIDs(answer map[string]any) []string {
	var ids []string
	switch picks := answer["selected"].(type) {
	case []map[string]any:
		for _, p := range picks {
			ids = append(ids, fmt.Sprint(p["id"]))
		}
	case []any:
		for _, p := range picks {
			if m, ok := p.(map[string]any); ok {
				ids = append(ids, fmt.Sprint(m["id"]))
			}
		}
	}
	return ids
}

func optionLabels(data json.RawMessage) map[string]string {
	var d struct {
		Options []struct {
			ID    string `json:"id"`
			Label string `json:"label"`
		} `json:"options"`
	}
	_ = json.Unmarshal(data, &d)
	labels := map[string]string{}
	for _, o := range d.Options {
		labels[o.ID] = o.Label
	}
	return labels
}
