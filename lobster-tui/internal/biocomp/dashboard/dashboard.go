// Package dashboard implements the ask_user "dashboard" view: several linked
// decisions on one screen. Each section hosts an ordinary BioComp component
// (choice, threshold_slider, text_input, plate_map), created from the registry.
//
// Sections list the keys they depend on. When a section's answer changes,
// every answered section that depends on it (directly or through others) is
// marked stale with the reason, and the dashboard can't be submitted until
// each stale section is checked again. The dashboard knows nothing about the
// science: dependencies come from the request, not from domain rules.
//
// Keys: tab / shift+tab move between sections, ctrl+s submits everything,
// ctrl+d defers (nothing is applied), esc cancels.
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
)

func init() {
	biocomp.Register("dashboard", func() biocomp.BioComponent {
		return &Component{}
	})
}

type sectionData struct {
	Key       string          `json:"key"`
	Title     string          `json:"title"`
	DependsOn []string        `json:"depends_on"`
	Component string          `json:"component"`
	Data      json.RawMessage `json:"data"`
}

type dashboardData struct {
	Question string        `json:"question"`
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
	status     status
	answer     map[string]any // the section component's last reply
	summary    string         // short text for the answer, e.g. "SRC-A"
	staleWhy   string         // why it needs checking again
	dependents []int          // sections that list this one in depends_on
}

// Component is the dashboard: a list of sections and the active section's
// component.
type Component struct {
	question string
	sections []*section
	active   int
	child    biocomp.BioComponent
	childCmd tea.Cmd
	message  string
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
	c.sections = nil
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
		if err := biocomp.Get(sd.Component)().Init(sd.Data); err != nil {
			return fmt.Errorf("dashboard: section %q: %w", sd.Key, err)
		}
		if sd.Title == "" {
			sd.Title = sd.Key
		}
		index[sd.Key] = i
		c.sections = append(c.sections, &section{sectionData: sd})
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
	c.question = d.Question
	c.message = ""
	return c.open(0)
}

// open makes section i active with a fresh component, pre-filled with the
// section's last answer so revisiting keeps the user's work.
func (c *Component) open(i int) error {
	s := c.sections[i]
	child := biocomp.Get(s.Component)()
	if err := child.Init(prefill(s)); err != nil {
		return fmt.Errorf("dashboard: section %q: %w", s.Key, err)
	}
	c.active = i
	c.child = child
	c.childCmd = child.InitCmd()
	return nil
}

func (c *Component) InitCmd() tea.Cmd {
	cmd := c.childCmd
	c.childCmd = nil
	return cmd
}

func (c *Component) HandleMsg(msg tea.Msg) (*biocomp.ComponentResult, tea.Cmd) {
	if km, ok := msg.(tea.KeyPressMsg); ok {
		switch km.String() {
		case "tab":
			return nil, c.move(1)
		case "shift+tab":
			return nil, c.move(-1)
		case "ctrl+s":
			if blockers := c.blockers(); len(blockers) > 0 {
				c.message = "Can't submit yet: " + strings.Join(blockers, "; ")
				return nil, nil
			}
			return &biocomp.ComponentResult{Action: "submit", Data: c.reply()}, nil
		case "ctrl+d":
			return &biocomp.ComponentResult{Action: "defer", Data: map[string]any{}}, nil
		case "esc":
			return &biocomp.ComponentResult{Action: "cancel", Data: map[string]any{}}, nil
		}
		c.message = ""
	}
	result, cmd := c.child.HandleMsg(msg)
	if result == nil || result.Action != "submit" {
		return nil, cmd // still answering (a child's own cancel is ignored)
	}
	c.record(c.active, result.Data)
	next := c.nextOpen()
	if next < 0 {
		c.message = "Every section is answered. Press ctrl+s to submit."
		next = c.active
	}
	if err := c.open(next); err != nil {
		c.message = err.Error()
	}
	return nil, tea.Batch(cmd, c.InitCmd())
}

// move switches to the next/previous section, wrapping around.
func (c *Component) move(step int) tea.Cmd {
	n := len(c.sections)
	if err := c.open(((c.active+step)%n + n) % n); err != nil {
		c.message = err.Error()
		return nil
	}
	c.message = ""
	return c.InitCmd()
}

// record stores section i's answer. If it changed, every answered section
// that depends on i, directly or through others, becomes stale.
func (c *Component) record(i int, data map[string]any) {
	s := c.sections[i]
	before, beforeSummary := s.answer, s.summary
	answer := make(map[string]any, len(data))
	for k, v := range data {
		if k != "action" {
			answer[k] = v
		}
	}
	s.answer, s.status, s.staleWhy = answer, answered, ""
	s.summary = summarize(s.sectionData, answer)
	if reflect.DeepEqual(before, answer) {
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

// blockers explains what stops a submit: unanswered and stale sections.
func (c *Component) blockers() []string {
	var out []string
	for _, s := range c.sections {
		switch s.status {
		case todo:
			out = append(out, fmt.Sprintf("%q is not answered", s.Title))
		case stale:
			out = append(out, fmt.Sprintf("%q needs re-checking (%s)", s.Title, s.staleWhy))
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
)

func (c *Component) View(width, height int) string {
	top := []string{titleStyle.Width(width).Render(c.question)}
	for i, s := range c.sections {
		top = append(top, c.sectionLine(i, s, width))
		if s.status == stale {
			top = append(top, warnStyle.Width(width).Render("     ⚠ Re-check: "+s.staleWhy))
		}
	}
	top = append(top, dimStyle.Render(strings.Repeat("─", max(width, 1))))

	bottom := []string{dimStyle.Render(c.progress())}
	if c.message != "" {
		bottom = append(bottom, warnStyle.Width(width).Render(c.message))
	}

	childHeight := height - lipgloss.Height(strings.Join(top, "\n")) - len(bottom)
	if childHeight < 5 {
		childHeight = 5
	}
	body := c.child.View(width, childHeight)
	return strings.Join(append(append(top, body), bottom...), "\n")
}

func (c *Component) sectionLine(i int, s *section, width int) string {
	icon, style := "○", lipgloss.NewStyle()
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
	line := style.Render(fmt.Sprintf("%s%s %d. %s", pointer, icon, i+1, s.Title))
	if s.summary != "" {
		line += dimStyle.Render("  " + s.summary)
	}
	return lipgloss.NewStyle().MaxWidth(width).Render(line)
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
	own := []key.Binding{
		key.NewBinding(key.WithKeys("tab", "shift+tab"), key.WithHelp("tab", "section")),
		key.NewBinding(key.WithKeys("ctrl+s"), key.WithHelp("ctrl+s", "submit all")),
		key.NewBinding(key.WithKeys("ctrl+d"), key.WithHelp("ctrl+d", "defer")),
		key.NewBinding(key.WithKeys("esc"), key.WithHelp("esc", "cancel")),
	}
	for _, b := range c.child.KeyBindings() {
		if h := b.Help().Key; h != "esc" && h != "tab" {
			own = append(own, b)
		}
	}
	return own
}

func (c *Component) ChangeEvent() map[string]any { return nil }

// ---------------------------------------------------------------------------
// Per-component helpers: summaries and pre-fill from a previous answer
// ---------------------------------------------------------------------------

// summarize describes an answer in a few words for the section list.
func summarize(s sectionData, answer map[string]any) string {
	switch s.Component {
	case "choice":
		labels := optionLabels(s.Data)
		var picked []string
		for _, id := range selectedIDs(answer) {
			if label := labels[id]; label != "" && label != id {
				picked = append(picked, fmt.Sprintf("%s (%s)", label, id))
			} else {
				picked = append(picked, id)
			}
		}
		if len(picked) == 0 {
			return "none picked"
		}
		return strings.Join(picked, ", ")
	case "threshold_slider":
		var d struct {
			Unit string `json:"unit"`
		}
		_ = json.Unmarshal(s.Data, &d)
		return strings.TrimSpace(fmt.Sprintf("%v %s", answer["value"], d.Unit))
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
		text := fmt.Sprint(answer["answer"])
		if len(text) > 40 {
			text = text[:39] + "…"
		}
		return fmt.Sprintf("%q", text)
	}
	return "answered"
}

// prefill returns the section's data with its last answer filled in.
func prefill(s *section) json.RawMessage {
	if s.answer == nil {
		return s.Data
	}
	var d map[string]any
	if err := json.Unmarshal(s.Data, &d); err != nil {
		return s.Data
	}
	switch s.Component {
	case "choice":
		d["default"] = selectedIDs(s.answer)
	case "threshold_slider":
		d["default"] = s.answer["value"]
	case "plate_map":
		if assignments, ok := s.answer["assignments"].(map[string]any); ok {
			if wells, ok := d["wells"].([]any); ok {
				for _, w := range wells {
					if well, ok := w.(map[string]any); ok {
						if v, ok := assignments[fmt.Sprint(well["sample_id"])]; ok {
							well["condition"] = v
						}
					}
				}
			}
		}
	default:
		return s.Data // text keeps its own data; the summary shows the last answer
	}
	raw, err := json.Marshal(d)
	if err != nil {
		return s.Data
	}
	return raw
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
