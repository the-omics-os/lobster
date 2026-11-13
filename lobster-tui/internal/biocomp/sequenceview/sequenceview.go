// Package sequenceview implements the "sequence" view of the ask_user
// decision contract: pick one region of a sequence, by id.
//
// Python reads the letters from the workspace FASTA file; each option is a
// region with a 1-based, inclusive start and end. The screen shows Crust's
// sequence viewer with every region marked in its own colour, and the list
// of regions under it: ↑/↓ move between regions (their letters are shown),
// pgup/pgdn scroll the sequence, tab changes how it is coloured, enter picks.
// Crust's viewer only displays; picking is this list.
//
// Reply: {"selected": [{"id": "..."}]}, the shape of "choice".
package sequenceview

import (
	"encoding/json"
	"fmt"
	"image/color"
	"strings"

	"charm.land/bubbles/v2/key"
	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"

	"github.com/the-omics-os/crust/sequenceviewer"

	"github.com/the-omics-os/lobster-tui/internal/biocomp"
)

func init() {
	biocomp.Register("sequence_view", func() biocomp.BioComponent {
		return &Component{}
	})
}

type region struct {
	ID          string `json:"id"`
	Label       string `json:"label"`
	Details     string `json:"details,omitempty"`
	Unavailable string `json:"unavailable,omitempty"`
	Start       int    `json:"start"`
	End         int    `json:"end"`
}

type fieldData struct {
	Question string   `json:"question"`
	Details  string   `json:"details,omitempty"`
	Name     string   `json:"name,omitempty"`
	Kind     string   `json:"kind,omitempty"`
	Sequence string   `json:"sequence"`
	Options  []region `json:"options"`
	Default  string   `json:"default,omitempty"`
}

// The regions' colours, in order, repeating after the last.
var palette = []string{"212", "45", "214", "120", "141", "203", "51", "229"}

// Component shows the sequence and a list of its regions.
type Component struct {
	question, details, name string
	letters                 string
	regions                 []region
	colors                  []color.Color
	defaultAt               int
	cursor                  int
	viewer                  sequenceviewer.Model
	viewerW, viewerH        int
	problem                 string
	escArmed                bool
}

func (c *Component) Init(data json.RawMessage) error {
	var d fieldData
	if err := json.Unmarshal(data, &d); err != nil {
		return fmt.Errorf("sequence_view: invalid data: %w", err)
	}
	if d.Question == "" {
		return fmt.Errorf("sequence_view: question is required")
	}
	if d.Sequence == "" {
		return fmt.Errorf("sequence_view: sequence is required")
	}
	if len(d.Options) == 0 {
		return fmt.Errorf("sequence_view: at least one region is required")
	}
	seen := map[string]bool{}
	available := 0
	for i, r := range d.Options {
		if r.ID == "" {
			return fmt.Errorf("sequence_view: region %d has no id", i)
		}
		if seen[r.ID] {
			return fmt.Errorf("sequence_view: duplicate region id %q", r.ID)
		}
		seen[r.ID] = true
		if r.Unavailable == "" {
			available++
			if r.Start < 1 || r.End < r.Start || r.End > len(d.Sequence) {
				return fmt.Errorf("sequence_view: region %q (%d–%d) is outside the "+
					"sequence (1–%d)", r.ID, r.Start, r.End, len(d.Sequence))
			}
		}
		if r.Label == "" {
			d.Options[i].Label = r.ID
		}
	}
	if available == 0 {
		return fmt.Errorf("sequence_view: every region is unavailable")
	}
	c.question, c.details, c.name = d.Question, d.Details, d.Name
	c.letters, c.regions = d.Sequence, d.Options
	c.defaultAt = -1
	annotations := make([]sequenceviewer.Annotation, 0, len(c.regions))
	for i, r := range c.regions {
		col := lipgloss.Color(palette[i%len(palette)])
		c.colors = append(c.colors, col)
		if r.ID == d.Default && r.Unavailable == "" {
			c.cursor, c.defaultAt = i, i
		}
		if r.Start >= 1 && r.End >= r.Start {
			annotations = append(annotations, sequenceviewer.Annotation{
				Name: r.Label, Start: r.Start, End: r.End, Color: col,
			})
		}
	}
	c.viewer = sequenceviewer.New(
		sequenceviewer.WithSequence(d.Sequence, sequenceviewer.ParseSequenceType(d.Kind)),
		sequenceviewer.WithAnnotations(annotations),
	)
	return nil
}

func (c *Component) HandleMsg(msg tea.Msg) (*biocomp.ComponentResult, tea.Cmd) {
	km, ok := msg.(tea.KeyPressMsg)
	if !ok {
		return nil, nil
	}
	c.problem = ""
	armed := c.escArmed && km.String() == "esc"
	c.escArmed = false
	switch km.String() {
	case "up", "k":
		if c.cursor > 0 {
			c.cursor--
		}
	case "down", "j":
		if c.cursor < len(c.regions)-1 {
			c.cursor++
		}
	case "pgup", "pgdown", "home", "end", "tab", "c", "C":
		// Scrolling and colouring belong to the viewer (Crust maps the keys).
		updated, _ := c.viewer.Update(km)
		c.viewer = updated.(sequenceviewer.Model)
	case "enter":
		r := c.regions[c.cursor]
		if r.Unavailable != "" {
			c.problem = fmt.Sprintf("%s can't be picked: %s", r.Label, r.Unavailable)
			return nil, nil
		}
		return &biocomp.ComponentResult{
			Action: "submit",
			Data:   map[string]any{"selected": []map[string]any{{"id": r.ID}}},
		}, nil
	case "esc":
		if armed {
			return &biocomp.ComponentResult{Action: "cancel", Data: map[string]any{}}, nil
		}
		c.escArmed, c.problem = true, escPrompt
	case "ctrl+d":
		return &biocomp.ComponentResult{Action: "defer", Data: map[string]any{}}, nil
	}
	return nil, nil
}

const escPrompt = "press esc again to cancel this question; nothing will be sent"

var (
	boldStyle   = lipgloss.NewStyle().Bold(true)
	cursorStyle = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("63"))
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
	list := c.listLines(width)
	bottom := []string{}
	if c.problem != "" {
		bottom = append(bottom, wrap(c.problem, width, warnStyle))
	}
	head := strings.Join(top, "\n")
	room := height - lipgloss.Height(head) - len(list) - len(bottom) - 2
	if w, h := width, max(room, 8); w != c.viewerW || h != c.viewerH {
		c.viewerW, c.viewerH = w, h
		updated, _ := c.viewer.Update(tea.WindowSizeMsg{Width: w, Height: h})
		c.viewer = updated.(sequenceviewer.Model)
	}
	parts := append([]string{head, c.viewerView(), ""}, list...)
	return strings.Join(append(parts, bottom...), "\n")
}

// viewerView is Crust's viewer without its blank padding and its own key
// line ("Arrows: scroll"): here the arrows move between regions, and this
// screen's keys are in the help bar.
func (c *Component) viewerView() string {
	var kept []string
	for _, line := range strings.Split(c.viewer.Render(), "\n") {
		plain := strings.TrimSpace(ansi.Strip(line))
		if plain == "" || strings.Contains(plain, "Arrows: scroll") {
			continue
		}
		kept = append(kept, line)
	}
	return strings.Join(kept, "\n")
}

// listLines is one line per region (its colour, label, span, letters), then
// the highlighted region's details.
func (c *Component) listLines(width int) []string {
	lines := make([]string, 0, len(c.regions)+2)
	for i, r := range c.regions {
		pointer, style := "  ", lipgloss.NewStyle()
		if i == c.cursor {
			pointer, style = "> ", cursorStyle
		}
		mark := "( )"
		if i == c.defaultAt {
			mark = "(●)"
		}
		swatch := lipgloss.NewStyle().Foreground(c.colors[i]).Render("■")
		span := fmt.Sprintf("%d–%d", r.Start, r.End)
		text := style.Render(r.Label)
		if r.Unavailable != "" {
			mark, text = " ✕ ", dimStyle.Render(r.Label+" · unavailable: "+r.Unavailable)
		}
		line := pointer + mark + " " + swatch + " " + text + dimStyle.Render("  "+span)
		lines = append(lines, lipgloss.NewStyle().MaxWidth(width).Render(line))
	}
	r := c.regions[c.cursor]
	if r.Start >= 1 && r.End <= len(c.letters) {
		part := c.letters[r.Start-1 : r.End]
		if room := width - 4; len(part) > room && room > 3 {
			part = part[:room-1] + "…"
		}
		lines = append(lines, "", dimStyle.Render("  "+part))
	}
	if r.Details != "" {
		lines = append(lines, wrap("  "+r.Details, width, dimStyle))
	}
	return lines
}

func (c *Component) SetData(data json.RawMessage) error {
	return fmt.Errorf("sequence_view: live updates are not supported")
}

func (c *Component) InitCmd() tea.Cmd { return nil }
func (c *Component) Name() string     { return "sequence_view" }
func (c *Component) Mode() string     { return "overlay" }

func (c *Component) KeyBindings() []key.Binding {
	return []key.Binding{
		key.NewBinding(key.WithKeys("up", "down"), key.WithHelp("↑/↓", "region")),
		key.NewBinding(key.WithKeys("pgup", "pgdown"), key.WithHelp("pgup/pgdn", "scroll")),
		key.NewBinding(key.WithKeys("tab"), key.WithHelp("tab", "colours")),
		key.NewBinding(key.WithKeys("enter"), key.WithHelp("enter", "pick")),
		key.NewBinding(key.WithKeys("ctrl+d"), key.WithHelp("ctrl+d", "defer")),
		key.NewBinding(key.WithKeys("esc"), key.WithHelp("esc", "twice: cancel")),
	}
}

func (c *Component) ChangeEvent() map[string]any { return nil }

// Hint is the instruction under the screen (see biocomp.Guide).
func (c *Component) Hint() string { return biocomp.HintRegion }
