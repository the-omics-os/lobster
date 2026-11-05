package sampleplate

import (
	"fmt"
	"reflect"
	"slices"
	"strings"
	"testing"

	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"
)

// Look-alike samples for a synthetic example.
const lookAlike = `{
  "question": "Assign each sample to control or treated, or leave it out.",
  "title": "Sample conditions (synthetic)",
  "format": 96,
  "choices": ["control", "treated", "leave_out"],
  "wells": [
    {"row": 0, "col": 0, "sample_id": "X01", "label": "Line 1", "condition": "control", "batch": "B1", "warning": ""},
    {"row": 0, "col": 1, "sample_id": "X02", "label": "Line 2", "condition": "treated", "batch": "B1", "warning": "Low read depth"},
    {"row": 0, "col": 2, "sample_id": "X03", "label": "Line 3", "condition": "unknown", "batch": "B2", "warning": "Condition missing"}
  ]
}`

func press(s string) tea.KeyPressMsg {
	switch s {
	case "enter":
		return tea.KeyPressMsg{Code: tea.KeyEnter}
	case "esc":
		return tea.KeyPressMsg{Code: tea.KeyEscape}
	case "right":
		return tea.KeyPressMsg{Code: tea.KeyRight}
	}
	return tea.KeyPressMsg{Code: rune(s[0]), Text: s}
}

func newComp(t *testing.T) *SamplePlateComponent {
	t.Helper()
	c := &SamplePlateComponent{}
	if err := c.Init([]byte(lookAlike)); err != nil {
		t.Fatalf("Init: %v", err)
	}
	return c
}

func TestInitRejectsBadData(t *testing.T) {
	cases := map[string]string{
		"no wells":      `{"wells": []}`,
		"bad format":    `{"format": 100, "wells": [{"row":0,"col":0,"sample_id":"X01"}]}`,
		"off plate":     `{"wells": [{"row":8,"col":0,"sample_id":"X01"}]}`,
		"duplicate id":  `{"wells": [{"row":0,"col":0,"sample_id":"X01"},{"row":0,"col":1,"sample_id":"X01"}]}`,
		"shared well":   `{"wells": [{"row":0,"col":0,"sample_id":"X01"},{"row":0,"col":0,"sample_id":"X02"}]}`,
		"missing id":    `{"wells": [{"row":0,"col":0}]}`,
		"not an object": `"X01"`,
	}
	for name, data := range cases {
		if err := (&SamplePlateComponent{}).Init([]byte(data)); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
}

func TestSnakeCaseFieldsArePopulated(t *testing.T) {
	c := newComp(t)
	s := c.samples[2]
	if s.SampleID != "X03" || s.Label != "Line 3" || s.Batch != "B2" || s.Warning != "Condition missing" {
		t.Fatalf("fields not populated: %+v", s)
	}
	if c.assignments["X03"] != unresolved {
		t.Fatalf("unknown condition should start unresolved, got %q", c.assignments["X03"])
	}
}

func TestEnterRefusedWhileUnresolved(t *testing.T) {
	c := newComp(t)
	if res, _ := c.HandleMsg(press("enter")); res != nil {
		t.Fatalf("enter should be refused, got %+v", res)
	}
	if !strings.Contains(c.message, "X03") {
		t.Fatalf("message should name the unresolved sample, got %q", c.message)
	}
}

func TestAssignAndSubmit(t *testing.T) {
	c := newComp(t)
	c.HandleMsg(press("right"))
	c.HandleMsg(press("right")) // cursor on X03 (A3)
	c.HandleMsg(press("2"))     // 2 = the second choice, "treated"
	if c.assignments["X03"] != "treated" {
		t.Fatalf("X03 = %q, want treated", c.assignments["X03"])
	}

	res, _ := c.HandleMsg(press("enter"))
	if res == nil || res.Action != "submit" {
		t.Fatalf("expected submit, got %+v", res)
	}
	got := res.Data["assignments"].(map[string]any)
	want := map[string]string{"X01": "control", "X02": "treated", "X03": "treated"}
	for id, v := range want {
		if got[id] != v {
			t.Errorf("%s = %v, want %s", id, got[id], v)
		}
	}
	changes := res.Data["changes"].([]any)
	if len(changes) != 1 || changes[0].(map[string]any)["sample_id"] != "X03" {
		t.Fatalf("changes should list only X03, got %v", changes)
	}
}

const wantEscPrompt = "press esc again to cancel this question; nothing will be sent"

// One esc only warns (Crust's own esc would cancel at once); the second esc
// in a row cancels, with Crust's reason, as before.
func TestEscAsksOnceThenCancels(t *testing.T) {
	c := newComp(t)
	c.HandleMsg(press("2")) // some work that one esc must not throw away
	if res, _ := c.HandleMsg(press("esc")); res != nil {
		t.Fatalf("a single esc cancelled: %+v", res)
	}
	if c.message != wantEscPrompt || !strings.Contains(ansi.Strip(c.View(90, 30)), wantEscPrompt) {
		t.Fatalf("prompt not shown: %q", c.message)
	}
	res, _ := c.HandleMsg(press("esc"))
	if res == nil || res.Action != "cancel" || res.Data["reason"] != "user cancelled" {
		t.Fatalf("esc esc: expected cancel, got %+v", res)
	}
}

func TestAnotherKeyBetweenEscsStartsOver(t *testing.T) {
	c := newComp(t)
	c.HandleMsg(press("esc"))
	if res, _ := c.HandleMsg(press("right")); res != nil || c.message != "" {
		t.Fatalf("another key should disarm and clear the prompt: %+v %q", res, c.message)
	}
	if res, _ := c.HandleMsg(press("esc")); res != nil {
		t.Fatalf("esc, right, esc cancelled: %+v", res)
	}
	if res, _ := c.HandleMsg(press("esc")); res == nil || res.Action != "cancel" {
		t.Fatalf("then esc esc should cancel, got %+v", res)
	}
}

// Crust uses esc to close its help first; that esc neither cancels nor arms.
func TestEscThatClosesCrustsHelpDoesNotArm(t *testing.T) {
	c := newComp(t)
	c.HandleMsg(press("?"))
	if res, _ := c.HandleMsg(press("esc")); res != nil || c.escArmed {
		t.Fatalf("esc closing the help: %+v, armed %v", res, c.escArmed)
	}
	if res, _ := c.HandleMsg(press("esc")); res != nil || !c.escArmed {
		t.Fatalf("first esc after the help should only arm: %+v", res)
	}
	if res, _ := c.HandleMsg(press("esc")); res == nil || res.Action != "cancel" {
		t.Fatalf("second esc should cancel, got %+v", res)
	}
}

func TestEscHelpSaysTwice(t *testing.T) {
	for _, b := range newComp(t).KeyBindings() {
		if b.Help().Key == "esc" {
			if b.Help().Desc != "twice: cancel" {
				t.Fatalf("esc help = %q", b.Help().Desc)
			}
			return
		}
	}
	t.Fatal("no esc help")
}

// Option ids are compared exactly (as Prefill does): capitalised ids keep
// their defaults, and nothing unchanged is listed under "changes". Review
// finding 1.
func TestCapitalisedChoicesKeepTheirDefaults(t *testing.T) {
	raw := `{"question": "Group?", "choices": ["Control", "KO", "leave_out"], "wells": [
	  {"row": 0, "col": 0, "sample_id": "X01", "condition": "Control"},
	  {"row": 0, "col": 1, "sample_id": "X02", "condition": " KO "},
	  {"row": 0, "col": 2, "sample_id": "X03", "condition": "leave_out"}]}`
	c := &SamplePlateComponent{}
	if err := c.Init([]byte(raw)); err != nil {
		t.Fatal(err)
	}
	want := map[string]any{"X01": "Control", "X02": "KO", "X03": "leave_out"}
	if got := c.Draft()["assignments"]; !reflect.DeepEqual(got, want) {
		t.Fatalf("starting assignments = %v, want %v", got, want)
	}
	res, _ := c.HandleMsg(press("enter"))
	if res == nil || res.Action != "submit" {
		t.Fatalf("expected submit, got %+v (%q)", res, c.message)
	}
	if got := res.Data["assignments"]; !reflect.DeepEqual(got, want) {
		t.Fatalf("assignments = %v", got)
	}
	if changes := res.Data["changes"].([]any); len(changes) != 0 {
		t.Fatalf("nothing changed, but changes = %v", changes)
	}
	// A case-only mismatch is not one of the choices: it is not guessed.
	c2 := &SamplePlateComponent{}
	if err := c2.Init([]byte(strings.Replace(raw, `"condition": "Control"`, `"condition": "control"`, 1))); err != nil {
		t.Fatal(err)
	}
	if c2.assignments["X01"] != unresolved {
		t.Fatalf("X01 = %q, want unresolved", c2.assignments["X01"])
	}
}

// An option literally called "unknown" is an ordinary choice: it can be the
// supplied condition, be assigned and be sent. Unresolved is a separate
// value that never reaches the reply. Review finding 5.
func TestAnOptionCalledUnknownCanBeSent(t *testing.T) {
	raw := `{"question": "Group?", "choices": ["control", "unknown"], "wells": [
	  {"row": 0, "col": 0, "sample_id": "X01", "condition": "unknown"},
	  {"row": 0, "col": 1, "sample_id": "X02", "condition": "missing"}]}`
	c := &SamplePlateComponent{}
	if err := c.Init([]byte(raw)); err != nil {
		t.Fatal(err)
	}
	if c.assignments["X01"] != "unknown" || c.assignments["X02"] != unresolved {
		t.Fatalf("assignments = %q", c.assignments)
	}
	if view := ansi.Strip(c.View(90, 30)); !strings.Contains(view, "unknown 1 · unresolved 1") {
		t.Fatalf("counts should tell the option from unresolved:\n%s", view)
	}
	if res, _ := c.HandleMsg(press("enter")); res != nil || !strings.Contains(c.message, "X02") {
		t.Fatalf("enter with X02 unresolved: %+v %q", res, c.message)
	}
	c.HandleMsg(press("0")) // X01 back to unresolved: not sendable either
	if res, _ := c.HandleMsg(press("enter")); res != nil {
		t.Fatalf("enter with X01 unresolved sent %+v", res)
	}
	c.HandleMsg(press("2")) // X01 -> the option "unknown"
	c.HandleMsg(press("right"))
	c.HandleMsg(press("2")) // X02 -> the option "unknown"
	res, _ := c.HandleMsg(press("enter"))
	if res == nil || res.Action != "submit" {
		t.Fatalf("expected submit, got %+v (%q)", res, c.message)
	}
	want := map[string]any{"X01": "unknown", "X02": "unknown"}
	if got := res.Data["assignments"]; !reflect.DeepEqual(got, want) {
		t.Fatalf("assignments = %v, want %v", got, want)
	}
	changes := res.Data["changes"].([]any)
	if len(changes) != 1 || changes[0].(map[string]any)["sample_id"] != "X02" {
		t.Fatalf("changes should list only X02, got %v", changes)
	}
}

func TestInitRefusesAnEmptyChoice(t *testing.T) {
	raw := strings.Replace(lookAlike, `["control", "treated", "leave_out"]`, `["control", ""]`, 1)
	if err := (&SamplePlateComponent{}).Init([]byte(raw)); err == nil {
		t.Fatal("an empty choice would be the unresolved value; expected an error")
	}
}

func TestViewShowsFocusAndPlate(t *testing.T) {
	c := newComp(t)
	out := ansi.Strip(c.View(90, 30))
	for _, want := range []string{"X01", "Line 1", "unresolved 1", "   1   2   3", "A  ●   ●   ○", "H  ·"} {
		if !strings.Contains(out, want) {
			t.Errorf("view missing %q", want)
		}
	}
}

func TestNumberKeysFollowTheChoices(t *testing.T) {
	c := newComp(t) // choices: control, treated, leave_out; cursor on X01
	for key, want := range map[string]string{"3": "leave_out", "1": "control", "0": unresolved} {
		c.HandleMsg(press(key))
		if c.assignments["X01"] != want {
			t.Fatalf("key %s: X01 = %q, want %q", key, c.assignments["X01"], want)
		}
	}
	c.HandleMsg(press("1"))
	c.HandleMsg(press("4")) // no fourth choice: ignored
	if c.assignments["X01"] != "control" {
		t.Fatalf("key 4 changed X01 to %q", c.assignments["X01"])
	}
	if out := ansi.Strip(c.View(90, 30)); !strings.Contains(out, "1 ● control   2 ● treated   3 ● leave_out   0 ○ unresolved") {
		t.Fatalf("legend missing:\n%s", out)
	}
}

func TestInitNeedsOneToNineChoices(t *testing.T) {
	for _, choices := range []string{`[]`, `["1","2","3","4","5","6","7","8","9","10"]`} {
		raw := strings.Replace(lookAlike, `["control", "treated", "leave_out"]`, choices, 1)
		if err := (&SamplePlateComponent{}).Init([]byte(raw)); err == nil {
			t.Errorf("choices %s: expected an error", choices)
		}
	}
}

func TestExcludedSamplesSitGreyedInTheirWellsAndAreNotAnswered(t *testing.T) {
	raw := strings.Replace(lookAlike, `"wells": [`,
		`"wells": [{"row": 0, "col": 3, "sample_id": "QC1", "label": "Pool", "excluded": "pooled QC sample"},`, 1)
	c := &SamplePlateComponent{}
	if err := c.Init([]byte(raw)); err != nil {
		t.Fatal(err)
	}
	if row, col := c.plate.Cursor(); row != 0 || col != 0 {
		t.Fatalf("the cursor should start on the first sample that can be assigned, got %d,%d", row, col)
	}
	view := ansi.Strip(c.View(90, 30))
	if !strings.Contains(view, "A  ●   ●   ○   ◌") || !strings.Contains(view, "can't be assigned 1") ||
		!strings.Contains(view, "◌ can't be assigned") {
		t.Fatalf("excluded well not drawn greyed:\n%s", view)
	}
	for i := 0; i < 3; i++ {
		c.HandleMsg(press("right"))
	}
	if view := ansi.Strip(c.View(90, 30)); !strings.Contains(view, "A4  QC1 · Pool · can't be assigned: pooled QC sample") {
		t.Fatalf("current-well line should give the reason:\n%s", view)
	}
	c.HandleMsg(press("1"))
	if c.message != "QC1 can't be assigned: pooled QC sample." {
		t.Fatalf("message = %q", c.message)
	}
	c.HandleMsg(press("left"))
	c.HandleMsg(press("1")) // X03 -> control
	res, _ := c.HandleMsg(press("enter"))
	if res == nil || res.Action != "submit" {
		t.Fatalf("expected submit, got %+v (%q)", res, c.message)
	}
	if _, ok := res.Data["assignments"].(map[string]any)["QC1"]; ok {
		t.Fatal("an excluded sample is in the reply")
	}
}

func seventyTwo(t *testing.T) *SamplePlateComponent {
	t.Helper()
	var wells []string
	for i := 0; i < 72; i++ {
		cond := "control"
		if i%2 == 1 {
			cond = "treated"
		}
		if i == 71 {
			cond = ""
		}
		wells = append(wells, fmt.Sprintf(`{"row": %d, "col": %d, "sample_id": "S%02d", "label": "Donor %d", "condition": %q}`,
			i/12, i%12, i+1, i+1, cond))
	}
	raw := `{"question": "Group of each sample?", "format": 96, "choices": ["control", "treated", "leave_out"],
	  "wells": [` + strings.Join(wells, ",") + `,
	    {"row": 6, "col": 0, "sample_id": "S73", "label": "Donor 73", "excluded": "failed library"}]}`
	c := &SamplePlateComponent{}
	if err := c.Init([]byte(raw)); err != nil {
		t.Fatal(err)
	}
	return c
}

func TestSeventyTwoSamplesOnA96Plate(t *testing.T) {
	c := seventyTwo(t)
	view := ansi.Strip(c.View(100, 40))
	for _, want := range []string{"control 36", "treated 35", "unresolved 1", "can't be assigned 1", "G  ◌", "H  ·"} {
		if !strings.Contains(view, want) {
			t.Fatalf("view missing %q:\n%s", want, view)
		}
	}
	if res, _ := c.HandleMsg(press("enter")); res != nil || !strings.Contains(c.message, "F12 S72") {
		t.Fatalf("enter should name S72 as unresolved: %+v %q", res, c.message)
	}
	if row, col := c.plate.Cursor(); row != 5 || col != 11 {
		t.Fatalf("the refused enter should move to S72 (F12), got %d,%d", row, col)
	}
	c.HandleMsg(press("3")) // S72 (F12) -> leave_out
	res, _ := c.HandleMsg(press("enter"))
	if res == nil || res.Action != "submit" {
		t.Fatalf("expected submit, got %+v (%q)", res, c.message)
	}
	if got := len(res.Data["assignments"].(map[string]any)); got != 72 {
		t.Fatalf("%d assignments, want 72", got)
	}
}

func TestLongQuestionWraps(t *testing.T) {
	long := "Assign each of these samples to the group it belongs to so that the comparison is right"
	raw := strings.Replace(lookAlike, "Assign each sample to control or treated, or leave it out.", long, 1)
	c := &SamplePlateComponent{}
	if err := c.Init([]byte(raw)); err != nil {
		t.Fatal(err)
	}
	view := ansi.Strip(c.View(40, 40))
	words := strings.Fields(view)
	for _, w := range strings.Fields(long) {
		found := false
		for _, x := range words {
			found = found || x == w
		}
		if !found {
			t.Fatalf("word %q missing:\n%s", w, view)
		}
	}
}

func TestDraftAndPrefill(t *testing.T) {
	c := newComp(t)
	c.HandleMsg(press("2")) // X01 -> treated
	if got := c.Draft()["assignments"].(map[string]any); got["X01"] != "treated" || got["X03"] != unresolved {
		t.Fatalf("draft = %v", got)
	}

	// Pre-filling from a previous reply restores the assignments, and the
	// supplied conditions stay as they were, so "changes" is unchanged.
	first := newComp(t)
	first.HandleMsg(press("right"))
	first.HandleMsg(press("right"))
	first.HandleMsg(press("2"))
	res, _ := first.HandleMsg(press("enter"))
	again := newComp(t)
	again.Prefill(res.Data)
	res2, _ := again.HandleMsg(press("enter"))
	if res2 == nil || !reflect.DeepEqual(res.Data, res2.Data) {
		t.Fatalf("prefill changed the reply:\n%v\n%v", res.Data, res2)
	}
}

// The selected sample's details (from a data reference) wrap on their own
// lines under the current-well line, never cut. E2E_REPORT #2.
const longNote = "Condition label unresolved; a second manifest suggests treated (suggestion only)"

func withNote(t *testing.T) *SamplePlateComponent {
	t.Helper()
	raw := strings.Replace(lookAlike, `"warning": ""}`,
		`"warning": "", "details": {"note": "`+longNote+`"}}`, 1)
	c := &SamplePlateComponent{}
	if err := c.Init([]byte(raw)); err != nil {
		t.Fatal(err)
	}
	return c
}

func TestLongDetailsWrapUnderTheFocusLine(t *testing.T) {
	c := withNote(t) // cursor on X01, the sample with the note
	for _, width := range []int{50, 60} {
		view := ansi.Strip(c.View(width, 40))
		for _, l := range strings.Split(view, "\n") {
			if w := ansi.StringWidth(strings.TrimRight(l, " ")); w > width {
				t.Errorf("width %d: line is %d wide: %q", width, w, l)
			}
		}
		focus := strings.Index(view, "A1  X01")
		if focus < 0 || focus < strings.Index(view, "H  ·") {
			t.Fatalf("width %d: the current-well line should be under the plate:\n%s", width, view)
		}
		words := strings.Fields(view[focus:])
		for _, w := range strings.Fields("note: " + longNote) {
			if !slices.Contains(words, w) {
				t.Errorf("width %d: word %q missing:\n%s", width, w, view)
			}
		}
	}
}

func TestLongDetailsNeverPushThePlateOffScreen(t *testing.T) {
	c := withNote(t)
	full := ansi.Strip(c.View(50, 60))
	noteLines := 0
	for _, l := range strings.Split(full, "\n") {
		if strings.HasPrefix(l, "    ") {
			noteLines++
		}
	}
	if noteLines < 2 {
		t.Fatalf("the note should take several lines at width 50:\n%s", full)
	}
	height := lipgloss.Height(full) - 1 // one line short: the note gives way
	view := ansi.Strip(c.View(50, height))
	if got := lipgloss.Height(view); got > height {
		t.Fatalf("view is %d lines, given %d:\n%s", got, height, view)
	}
	for _, want := range []string{"A  ●", "H  ·", "A1  X01", "note: Condition", "…"} {
		if !strings.Contains(view, want) {
			t.Errorf("view missing %q:\n%s", want, view)
		}
	}
}

// Crust draws only the grid: no header, footer or title of its own, and no
// line of the screen is wider than the screen (C-08).
func TestOnlyCrustsGridAndNoLineIsWiderThanTheScreen(t *testing.T) {
	raw := strings.Replace(lookAlike, `"Sample conditions (synthetic)"`,
		`"Which group is each sample in? Assign each one, or leave it out."`, 1)
	c := &SamplePlateComponent{}
	if err := c.Init([]byte(raw)); err != nil {
		t.Fatal(err)
	}
	for _, width := range []int{52, 60, 80} {
		view := ansi.Strip(c.View(width, 40))
		for _, l := range strings.Split(view, "\n") {
			if w := ansi.StringWidth(l); w > width {
				t.Errorf("width %d: line is %d wide: %q", width, w, l)
			}
		}
		for _, chrome := range []string{"96-Well Plate", "View:", "Cursor:", "[A1]"} {
			if strings.Contains(view, chrome) {
				t.Errorf("width %d: Crust's %q is drawn:\n%s", width, chrome, view)
			}
		}
	}
}
