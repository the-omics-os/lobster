// Package ontologyterm implements the "ontology_browser" view of the
// ask_user decision contract: pick one ontology term, by its id.
//
// Python fills the terms from a public ontology (the supervisor's suggested
// terms and the terms around them); each term names its parent, so they form
// a tree. The screen opens on a short list: the suggested terms first, then
// "Browse the ontology", which opens Crust's ontology browser on the whole
// tree, every branch open (search with / or tab). In the browser, space picks the highlighted
// term (Crust's enter only opens a branch), enter picks a term with no
// children or a search match, and esc goes back to the list.
//
// Reply: {"selected": [{"id": "..."}]}, the shape of "choice".
package ontologyterm

import (
	"encoding/json"
	"fmt"
	"strings"

	"charm.land/bubbles/v2/key"
	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"

	"github.com/the-omics-os/crust"
	crustob "github.com/the-omics-os/crust/ontologybrowser"

	"github.com/the-omics-os/lobster-tui/internal/biocomp"
)

func init() {
	biocomp.Register("ontology_browser", func() biocomp.BioComponent {
		return &Component{}
	})
}

type term struct {
	ID          string `json:"id"`
	Label       string `json:"label"`
	Parent      string `json:"parent,omitempty"`
	Details     string `json:"details,omitempty"`
	Unavailable string `json:"unavailable,omitempty"`
}

type fieldData struct {
	Question  string   `json:"question"`
	Details   string   `json:"details,omitempty"`
	Ontology  string   `json:"ontology,omitempty"`
	Options   []term   `json:"options"`
	Suggested []string `json:"suggested,omitempty"`
	Default   string   `json:"default,omitempty"`
}

// Component lists the suggested terms and opens the browser on request.
type Component struct {
	question string
	details  string
	ontology string
	terms    map[string]term
	order    []string // term ids as Python sent them
	// list holds the suggested term ids, then "" for "Browse the ontology".
	list      []string
	defaultID string
	cursor    int
	lastTerm  string // the term the list last highlighted

	browsing  bool
	browser   crustob.Model
	searching bool // mirrors Crust's search pane focus (space types there)
	help      bool // mirrors Crust's help screen

	problem  string
	escArmed bool
}

func (c *Component) Init(data json.RawMessage) error {
	var d fieldData
	if err := json.Unmarshal(data, &d); err != nil {
		return fmt.Errorf("ontology_browser: invalid data: %w", err)
	}
	if d.Question == "" {
		return fmt.Errorf("ontology_browser: question is required")
	}
	if len(d.Options) == 0 {
		return fmt.Errorf("ontology_browser: at least one term is required")
	}
	c.terms = make(map[string]term, len(d.Options))
	for i, t := range d.Options {
		if t.ID == "" {
			return fmt.Errorf("ontology_browser: term %d has no id", i)
		}
		if _, dup := c.terms[t.ID]; dup {
			return fmt.Errorf("ontology_browser: duplicate term id %q", t.ID)
		}
		if t.Label == "" {
			t.Label = t.ID
		}
		c.terms[t.ID] = t
		c.order = append(c.order, t.ID)
	}
	c.question, c.details, c.ontology = d.Question, d.Details, d.Ontology
	if c.ontology == "" {
		c.ontology = "ontology"
	}
	c.defaultID = ""
	if _, ok := c.terms[d.Default]; ok {
		c.defaultID = d.Default
	}
	seen := map[string]bool{}
	add := func(id string) {
		if _, ok := c.terms[id]; ok && !seen[id] {
			seen[id] = true
			c.list = append(c.list, id)
		}
	}
	add(c.defaultID) // a default that isn't a suggestion is listed too
	for _, id := range d.Suggested {
		add(id)
	}
	c.list = append(c.list, "") // Browse the ontology
	c.cursor, c.lastTerm = 0, c.list[0]
	c.browser = expandAll(crustob.New(crustob.WithRoots(c.tree())))
	if len(c.list) == 1 { // nothing suggested: open on the browser
		c.browsing = true
	}
	return nil
}

// tree builds Crust nodes from the terms' parents, every node loaded (the
// whole tree is here: nothing is fetched while browsing).
func (c *Component) tree() []crustob.OntologyNode {
	children := map[string][]string{}
	var roots []string
	for _, id := range c.order {
		parent := c.terms[id].Parent
		if _, ok := c.terms[parent]; parent == "" || !ok {
			roots = append(roots, id)
			continue
		}
		children[parent] = append(children[parent], id)
	}
	var build func(id string, depth int) crustob.OntologyNode
	build = func(id string, depth int) crustob.OntologyNode {
		t := c.terms[id]
		node := crustob.OntologyNode{
			ID: t.ID, Name: c.name(t), Description: definition(t), Loaded: true,
		}
		if depth < 64 { // the tree has no loops (Python checks), but be safe
			for _, child := range children[id] {
				node.Children = append(node.Children, build(child, depth+1))
			}
		}
		return node
	}
	nodes := make([]crustob.OntologyNode, 0, len(roots))
	for _, id := range roots {
		nodes = append(nodes, build(id, 0))
	}
	return nodes
}

// expandAll opens every branch of the browser, so its search (which only
// looks at terms on screen) finds any term, and goes back to the top. Crust
// has no call for it, so it walks the tree with its own keys: right opens
// the highlighted branch (on a term with no children it would pick it; that
// command is dropped), down moves on, until down no longer moves.
func expandAll(m crustob.Model) crustob.Model {
	step := func(k tea.KeyPressMsg) {
		updated, _ := m.Update(k)
		m = updated.(crustob.Model)
	}
	for i := 0; i < 10000; i++ { // more than any question's terms
		before := m.Selected()
		step(tea.KeyPressMsg{Code: tea.KeyRight})
		step(tea.KeyPressMsg{Code: tea.KeyDown})
		after := m.Selected()
		if before == nil || after == nil || after.ID == before.ID {
			break
		}
	}
	step(tea.KeyPressMsg{Code: tea.KeyHome})
	return m
}

// definition is the term's details without Python's "suggested" marker
// (the star shows that).
func definition(t term) string {
	return strings.TrimPrefix(strings.TrimPrefix(t.Details, "suggested"), " · ")
}

// name is the term's label, starred when it was suggested.
func (c *Component) name(t term) string {
	for _, id := range c.list {
		if id == t.ID {
			return "★ " + t.Label
		}
	}
	return t.Label
}

func (c *Component) HandleMsg(msg tea.Msg) (*biocomp.ComponentResult, tea.Cmd) {
	km, ok := msg.(tea.KeyPressMsg)
	if !ok {
		return nil, nil
	}
	c.problem = ""
	if km.String() == "ctrl+d" {
		return &biocomp.ComponentResult{Action: "defer", Data: map[string]any{}}, nil
	}
	if c.browsing {
		return c.browse(km), nil
	}
	armed := c.escArmed && km.String() == "esc"
	c.escArmed = false
	switch km.String() {
	case "up", "k":
		if c.cursor > 0 {
			c.cursor--
		}
	case "down", "j":
		if c.cursor < len(c.list)-1 {
			c.cursor++
		}
	}
	if id := c.list[c.cursor]; id != "" {
		c.lastTerm = id
	}
	switch km.String() {
	case "enter":
		if c.list[c.cursor] == "" {
			c.openBrowser()
			return nil, nil
		}
		return c.pick(c.list[c.cursor]), nil
	case "esc":
		if armed {
			return &biocomp.ComponentResult{Action: "cancel", Data: map[string]any{}}, nil
		}
		c.escArmed, c.problem = true, escPrompt
	}
	return nil, nil
}

// openBrowser shows the browser on the term the list last highlighted (or
// the default), so it opens where the user was looking, not at the top.
func (c *Component) openBrowser() {
	c.browsing = true
	c.browser = moveTo(c.browser, c.lastTerm)
}

// moveTo moves the browser's cursor onto the term “id“ (every branch is
// open, see expandAll), from the top, with Crust's own keys.
func moveTo(m crustob.Model, id string) crustob.Model {
	step := func(k tea.KeyPressMsg) {
		updated, _ := m.Update(k)
		m = updated.(crustob.Model)
	}
	step(tea.KeyPressMsg{Code: tea.KeyHome})
	for i := 0; i < 10000 && id != ""; i++ {
		before := m.Selected()
		if before == nil || before.ID == id {
			break
		}
		step(tea.KeyPressMsg{Code: tea.KeyDown})
		if after := m.Selected(); after == nil || after.ID == before.ID {
			step(tea.KeyPressMsg{Code: tea.KeyHome}) // not in the tree
			break
		}
	}
	return m
}

// browse passes a key to Crust's browser; space picks the highlighted term,
// and Crust's cancel (esc in the tree) goes back to the list.
func (c *Component) browse(km tea.KeyPressMsg) *biocomp.ComponentResult {
	k := km.String()
	if k == "ctrl+c" {
		return nil // the chat screen handles ctrl+c
	}
	if k == "space" && !c.searching && !c.help {
		if node := c.browser.Selected(); node != nil {
			return c.pick(node.ID)
		}
		return nil
	}
	// Mirror Crust's focus rules (ontologybrowser/model.go updateKey).
	switch {
	case k == "?":
		c.help = !c.help
	case k == "esc" && c.help:
		c.help = false
	case k == "esc" && c.searching:
		c.searching = false
	case c.help:
	case k == "tab":
		c.searching = !c.searching
	case k == "/" && !c.searching:
		c.searching = true
	}
	updated, cmd := c.browser.Update(km)
	c.browser = updated.(crustob.Model)
	if cmd == nil {
		return nil
	}
	switch m := cmd().(type) {
	case crust.SubmitMsg:
		if id, ok := m.Data["id"].(string); ok {
			c.searching = false
			return c.pick(id)
		}
	case crust.CancelMsg:
		c.browsing, c.searching, c.help = false, false, false
	}
	return nil
}

func (c *Component) pick(id string) *biocomp.ComponentResult {
	t, ok := c.terms[id]
	if !ok {
		return nil
	}
	if t.Unavailable != "" {
		c.problem = fmt.Sprintf("%s can't be picked: %s", t.Label, t.Unavailable)
		return nil
	}
	return &biocomp.ComponentResult{
		Action: "submit",
		Data:   map[string]any{"selected": []map[string]any{{"id": id}}},
	}
}

const escPrompt = "press esc again to cancel this question; nothing will be sent"

var (
	boldStyle   = lipgloss.NewStyle().Bold(true)
	cursorStyle = lipgloss.NewStyle().Bold(true).Foreground(biocomp.CursorColor)
	dimStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color("240"))
	warnStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color("214"))
)

func wrap(text string, width int, style lipgloss.Style) string {
	return style.Width(max(width, 1)).Render(text)
}

func (c *Component) View(width, height int) string {
	top := []string{wrap(c.question, width, boldStyle)}
	if biocomp.Roomy(height) && c.details != "" {
		top = append(top, "") // a gap between the question and its details
	}
	if c.details != "" {
		top = append(top, wrap(c.details, width, dimStyle))
	}
	var bottom []string
	if c.problem != "" {
		bottom = append(bottom, wrap(c.problem, width, warnStyle))
	}
	head := strings.Join(top, "\n")
	room := height - lipgloss.Height(head) - len(bottom) - 1
	var body string
	if c.browsing {
		// Crust draws a little taller than asked; shrink it by the overflow.
		c.browser.SetWidth(width)
		c.browser.SetHeight(max(room, 16))
		body = c.browser.Render()
		if over := lipgloss.Height(body) - room; over > 0 {
			c.browser.SetHeight(max(room-over, 16))
			body = c.browser.Render()
		}
	} else {
		body = c.listView(width, room)
	}
	return strings.Join(append([]string{head, "", body}, bottom...), "\n")
}

func (c *Component) listView(width, room int) string {
	lines := []string{dimStyle.Render("Suggested terms:")}
	for i, id := range c.list {
		pointer, style := "  ", lipgloss.NewStyle()
		if i == c.cursor {
			pointer, style = "> ", cursorStyle
		}
		if id == "" {
			text := fmt.Sprintf("Browse nearby %s terms (%d) …", c.ontology, len(c.order))
			lines = append(lines, "", pointer+style.Render(text))
			continue
		}
		t := c.terms[id]
		mark := "( )"
		if id == c.defaultID {
			mark = "(●)"
		}
		line := pointer + mark + " " + style.Render(t.Label) + dimStyle.Render("  "+t.ID)
		if t.Unavailable != "" {
			line += dimStyle.Render(" · unavailable: " + t.Unavailable)
		}
		lines = append(lines, line)
	}
	if id := c.list[c.cursor]; id != "" {
		if d := definition(c.terms[id]); d != "" {
			lines = append(lines, "", wrap(d, width, dimStyle))
		}
	}
	return strings.Join(lines, "\n")
}

func (c *Component) SetData(data json.RawMessage) error {
	return fmt.Errorf("ontology_browser: live updates are not supported")
}

func (c *Component) InitCmd() tea.Cmd { return nil }
func (c *Component) Name() string     { return "ontology_browser" }
func (c *Component) Mode() string     { return "overlay" }

func (c *Component) KeyBindings() []key.Binding {
	if c.browsing {
		return []key.Binding{
			key.NewBinding(key.WithKeys("up", "down"), key.WithHelp("↑/↓", "move")),
			key.NewBinding(key.WithKeys("right", "left"), key.WithHelp("→/←", "open/close")),
			key.NewBinding(key.WithKeys("space"), key.WithHelp("space", "pick")),
			key.NewBinding(key.WithKeys("/"), key.WithHelp("/", "search")),
			key.NewBinding(key.WithKeys("esc"), key.WithHelp("esc", "back")),
			key.NewBinding(key.WithKeys("ctrl+d"), key.WithHelp("ctrl+d", "defer")),
		}
	}
	return []key.Binding{
		key.NewBinding(key.WithKeys("up", "down"), key.WithHelp("↑/↓", "move")),
		key.NewBinding(key.WithKeys("enter"), key.WithHelp("enter", "pick")),
		key.NewBinding(key.WithKeys("ctrl+d"), key.WithHelp("ctrl+d", "defer")),
		key.NewBinding(key.WithKeys("esc"), key.WithHelp("esc", "twice: cancel")),
	}
}

func (c *Component) ChangeEvent() map[string]any { return nil }

// Hint is the instruction under the screen (see biocomp.Guide).
func (c *Component) Hint() string {
	if c.browsing {
		return biocomp.HintBrowse
	}
	return biocomp.HintSuggested
}
