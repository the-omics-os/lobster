// Package choice implements the "choice" field of the ask_user decision
// contract: pick options from a list, by stable id.
//
// With min = max = 1 it behaves like a radio list (Enter picks the highlighted
// option). Otherwise it is a checklist: Space toggles, Enter submits once the
// number of picks is between min and max. An option with "unavailable" is
// shown greyed out with its reason and can't be picked.
//
// Reply: {"selected": [{"id": "..."}, ...]} in the order the options were given.
package choice

import (
	"encoding/json"
	"fmt"
	"strings"

	"charm.land/bubbles/v2/key"
	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"

	"github.com/the-omics-os/lobster-tui/internal/biocomp"
)

func init() {
	biocomp.Register("choice", func() biocomp.BioComponent {
		return &Component{}
	})
}

type option struct {
	ID          string `json:"id"`
	Label       string `json:"label"`
	Details     string `json:"details,omitempty"`
	Unavailable string `json:"unavailable,omitempty"`
}

// fieldData is the data the Python adapter sends. Min and Max are pointers
// so a missing value can fall back to 1.
type fieldData struct {
	Question string   `json:"question"`
	Details  string   `json:"details,omitempty"`
	Options  []option `json:"options"`
	Min      *int     `json:"min,omitempty"`
	Max      *int     `json:"max,omitempty"`
	Default  []string `json:"default,omitempty"`
}

// Component shows a list of options and returns the picked ids.
type Component struct {
	question string
	details  string
	options  []option
	min, max int
	picked   []bool // picked[i] is true when options[i] is selected
	cursor   int
	offset   int    // first visible row, for long lists
	problem  string // why the last action was refused
}

func (c *Component) Init(data json.RawMessage) error {
	var d fieldData
	if err := json.Unmarshal(data, &d); err != nil {
		return fmt.Errorf("choice: invalid data: %w", err)
	}
	if d.Question == "" {
		return fmt.Errorf("choice: question is required")
	}
	if len(d.Options) == 0 {
		return fmt.Errorf("choice: at least one option is required")
	}
	seen := make(map[string]int, len(d.Options))
	available := 0
	for i, it := range d.Options {
		if it.ID == "" {
			return fmt.Errorf("choice: option %d has no id", i)
		}
		if _, dup := seen[it.ID]; dup {
			return fmt.Errorf("choice: duplicate option id %q", it.ID)
		}
		seen[it.ID] = i
		if it.Label == "" {
			d.Options[i].Label = it.ID
		}
		if it.Unavailable == "" {
			available++
		}
	}
	c.min, c.max = 1, 1
	if d.Min != nil {
		c.min = *d.Min
	}
	if d.Max != nil {
		c.max = *d.Max
	}
	if c.min < 0 || c.max < 1 || c.min > c.max || c.min > available {
		return fmt.Errorf("choice: min %d / max %d can't be met with %d available options",
			c.min, c.max, available)
	}

	c.question = d.Question
	c.details = d.Details
	c.options = d.Options
	c.picked = make([]bool, len(d.Options))
	c.cursor = c.firstAvailable()
	for _, id := range d.Default {
		i, ok := seen[id]
		if !ok || d.Options[i].Unavailable != "" {
			continue // a stale default is ignored, never picked silently
		}
		if c.single() {
			c.cursor = i
			break
		}
		c.picked[i] = true
	}
	return nil
}

// single reports whether this is a pick-exactly-one radio list.
func (c *Component) single() bool { return c.min == 1 && c.max == 1 }

func (c *Component) firstAvailable() int {
	for i, it := range c.options {
		if it.Unavailable == "" {
			return i
		}
	}
	return 0
}

func (c *Component) HandleMsg(msg tea.Msg) (*biocomp.ComponentResult, tea.Cmd) {
	km, ok := msg.(tea.KeyPressMsg)
	if !ok {
		return nil, nil
	}
	c.problem = ""

	switch km.String() {
	case "up", "k":
		if c.cursor > 0 {
			c.cursor--
		}
	case "down", "j":
		if c.cursor < len(c.options)-1 {
			c.cursor++
		}
	case "space":
		if !c.single() {
			c.toggle(c.cursor)
		}
	case "enter":
		if c.single() {
			return c.pickOne(c.cursor), nil
		}
		if n := c.count(); n < c.min {
			c.problem = fmt.Sprintf("Pick at least %d (%d picked).", c.min, n)
			return nil, nil
		}
		return c.submit(c.pickedIndexes()), nil
	case "esc":
		return &biocomp.ComponentResult{Action: "cancel", Data: map[string]any{}}, nil
	default:
		// Number keys 1-9 pick directly in a short radio list.
		if c.single() && len(c.options) <= 9 && len(km.Text) == 1 {
			if r := km.Text[0]; r >= '1' && int(r-'1') < len(c.options) {
				c.cursor = int(r - '1')
				return c.pickOne(c.cursor), nil
			}
		}
	}
	return nil, nil
}

// pickOne submits option i in a radio list, unless it is unavailable.
func (c *Component) pickOne(i int) *biocomp.ComponentResult {
	if reason := c.options[i].Unavailable; reason != "" {
		c.problem = fmt.Sprintf("%s can't be picked: %s", c.options[i].Label, reason)
		return nil
	}
	return c.submit([]int{i})
}

func (c *Component) toggle(i int) {
	if reason := c.options[i].Unavailable; reason != "" {
		c.problem = fmt.Sprintf("%s can't be picked: %s", c.options[i].Label, reason)
		return
	}
	if !c.picked[i] && c.count() >= c.max {
		c.problem = fmt.Sprintf("You can pick at most %d.", c.max)
		return
	}
	c.picked[i] = !c.picked[i]
}

func (c *Component) count() int {
	n := 0
	for _, p := range c.picked {
		if p {
			n++
		}
	}
	return n
}

func (c *Component) pickedIndexes() []int {
	var out []int
	for i, p := range c.picked {
		if p {
			out = append(out, i)
		}
	}
	return out
}

func (c *Component) submit(indexes []int) *biocomp.ComponentResult {
	selected := make([]map[string]any, 0, len(indexes))
	for _, i := range indexes {
		selected = append(selected, map[string]any{"id": c.options[i].ID})
	}
	return &biocomp.ComponentResult{
		Action: "submit",
		Data:   map[string]any{"selected": selected},
	}
}

var (
	cursorStyle = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("63"))
	normalStyle = lipgloss.NewStyle()
	dimStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color("240"))
	warnStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color("214"))
)

func (c *Component) View(width, height int) string {
	top := []string{lipgloss.NewStyle().Bold(true).Width(width).Render(c.question)}
	if c.details != "" {
		top = append(top, dimStyle.Width(width).Render(c.details))
	}
	top = append(top, "")

	bottom := []string{"", dimStyle.Render(c.status())}
	if c.problem != "" {
		bottom = append(bottom, warnStyle.Width(width).Render(c.problem))
	}

	rows := height - lipgloss.Height(strings.Join(top, "\n")) - len(bottom)
	if rows < 3 {
		rows = 3
	}
	c.scrollTo(rows)
	end := min(c.offset+rows, len(c.options))

	lines := append([]string{}, top...)
	for i := c.offset; i < end; i++ {
		lines = append(lines, c.row(i, width))
	}
	lines = append(lines, bottom...)
	return strings.Join(lines, "\n")
}

// scrollTo keeps the cursor inside the visible window of the given height.
func (c *Component) scrollTo(rows int) {
	if c.cursor < c.offset {
		c.offset = c.cursor
	}
	if c.cursor >= c.offset+rows {
		c.offset = c.cursor - rows + 1
	}
}

func (c *Component) row(i, width int) string {
	it := c.options[i]
	mark := "[ ]"
	if c.picked[i] {
		mark = "[x]"
	}
	if c.single() {
		mark = "( )"
		if i == c.cursor {
			mark = "(●)"
		}
	}
	pointer, style := "  ", normalStyle
	if i == c.cursor {
		pointer, style = "> ", cursorStyle
	}

	// Show the id when it differs from the label, so repeated labels
	// ("Donor A" twice) stay distinguishable.
	var extra []string
	if it.ID != it.Label {
		extra = append(extra, it.ID)
	}
	if it.Details != "" {
		extra = append(extra, it.Details)
	}
	if it.Unavailable != "" {
		mark = " ✕ "
		style = dimStyle
		extra = append(extra, "unavailable: "+it.Unavailable)
	}
	line := style.Render(pointer + mark + " " + it.Label)
	if len(extra) > 0 {
		line += dimStyle.Render("  " + strings.Join(extra, " · "))
	}
	return lipgloss.NewStyle().MaxWidth(width).Render(line)
}

// status describes position and how many picks are allowed.
func (c *Component) status() string {
	pos := fmt.Sprintf("%d/%d", c.cursor+1, len(c.options))
	if c.single() {
		return pos
	}
	var rule string
	switch {
	case c.min == c.max:
		rule = fmt.Sprintf("pick exactly %d", c.min)
	case c.min == 0:
		rule = fmt.Sprintf("pick up to %d", c.max)
	default:
		rule = fmt.Sprintf("pick %d–%d", c.min, c.max)
	}
	return fmt.Sprintf("%s · %d picked · %s", pos, c.count(), rule)
}

func (c *Component) SetData(data json.RawMessage) error {
	return fmt.Errorf("choice: live updates are not supported")
}

func (c *Component) InitCmd() tea.Cmd { return nil }
func (c *Component) Name() string     { return "choice" }
func (c *Component) Mode() string     { return "overlay" }

func (c *Component) KeyBindings() []key.Binding {
	if c.single() {
		return []key.Binding{
			key.NewBinding(key.WithKeys("up", "down"), key.WithHelp("↑/↓", "move")),
			key.NewBinding(key.WithKeys("enter"), key.WithHelp("enter", "pick")),
			key.NewBinding(key.WithKeys("esc"), key.WithHelp("esc", "cancel")),
		}
	}
	return []key.Binding{
		key.NewBinding(key.WithKeys("up", "down"), key.WithHelp("↑/↓", "move")),
		key.NewBinding(key.WithKeys("space"), key.WithHelp("space", "toggle")),
		key.NewBinding(key.WithKeys("enter"), key.WithHelp("enter", "submit")),
		key.NewBinding(key.WithKeys("esc"), key.WithHelp("esc", "cancel")),
	}
}

func (c *Component) ChangeEvent() map[string]any { return nil }
