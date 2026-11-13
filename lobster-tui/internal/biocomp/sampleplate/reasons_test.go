package sampleplate

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	tea "charm.land/bubbletea/v2"
	"github.com/charmbracelet/x/ansi"
)

// A plate whose third option asks a reason, as Python sends it. X02 has a
// warning (offered as a hint); X03 was proposed "leave out" without a reason;
// X04 was proposed "leave out" with one (a re-ask or an earlier answer).
const withReasons = `{
  "question": "Which group is each sample in?",
  "format": 96,
  "positions": "given",
  "options": [
    {"id": "vehicle", "label": "vehicle"},
    {"id": "tdx9", "label": "TDX-9"},
    {"id": "leave_out", "label": "leave out", "reason": {"key": "reason", "question": "Why leave it out?"}}
  ],
  "wells": [
    {"row": 3, "col": 0, "sample_id": "X01", "condition": "vehicle"},
    {"row": 3, "col": 1, "sample_id": "X02", "condition": "tdx9", "warning": "low RNA yield (RIN 4.1)"},
    {"row": 3, "col": 2, "sample_id": "X03", "condition": "leave_out"},
    {"row": 3, "col": 3, "sample_id": "X04", "condition": "leave_out", "reason": "tube cracked"}
  ]
}`

func reasonPlate(t *testing.T) *SamplePlateComponent {
	t.Helper()
	c := &SamplePlateComponent{}
	if err := c.Init([]byte(withReasons)); err != nil {
		t.Fatalf("Init: %v", err)
	}
	return c
}

func view(c *SamplePlateComponent) string { return ansi.Strip(c.View(100, 40)) }

func typeText(c *SamplePlateComponent, text string) {
	for _, r := range text {
		c.HandleMsg(tea.KeyPressMsg{Code: r, Text: string(r)})
	}
}

func TestGroupsAreDotsWithMarksAndTheLegendNamesTheKeys(t *testing.T) {
	c := reasonPlate(t)
	v := view(c)
	for _, want := range []string{
		"D  ●   ●   ●!  ◉",
		"1 ● vehicle   2 ● TDX-9   3 ● leave out (asks a reason)   0 ○ unresolved",
		"◉ has a reason · ! needs a reason (r)",
		"needs a reason (1)",
	} {
		if !strings.Contains(v, want) {
			t.Fatalf("view lacks %q:\n%s", want, v)
		}
	}
	if strings.Contains(v, "⚠ No positions") {
		t.Fatalf("positions were given, but the list-order note is shown:\n%s", v)
	}
	raw := c.View(100, 40)
	for _, colour := range []string{"38;5;39m", "38;5;77m", "38;5;204m", "38;5;214m"} {
		if !strings.Contains(raw, colour) {
			t.Fatalf("the plate lacks Crust's colour %q", colour)
		}
	}
}

func TestListOrderIsSaidOnScreen(t *testing.T) {
	c := &SamplePlateComponent{}
	if err := c.Init([]byte(strings.Replace(withReasons, `"given"`, `"list_order"`, 1))); err != nil {
		t.Fatal(err)
	}
	if v := view(c); !strings.Contains(v, "No positions were given: samples are placed in list order") {
		t.Fatalf("list-order note missing:\n%s", v)
	}
}

func TestAReasonOptionOpensABoxOnTheCurrentWellLine(t *testing.T) {
	c := reasonPlate(t)
	c.HandleMsg(press("right")) // X02, which has a warning
	c.HandleMsg(press("3"))
	if !c.IsCapturingText() {
		t.Fatal("the reason box did not open")
	}
	v := view(c)
	box := "D2  X02 → leave out · Why leave it out: █ low RNA yield (RIN 4.1)  ← tab inserts"
	if !strings.Contains(v, box) || !strings.Contains(v, "enter save · esc undo") {
		t.Fatalf("box not on the current-well line:\n%s", v)
	}
	if strings.Index(v, box) < strings.Index(v, "H  ·") {
		t.Fatalf("the box should be under the plate, which stays drawn:\n%s", v)
	}
	if c.box.text != "" {
		t.Fatalf("the warning was filled in by itself: %q", c.box.text)
	}
}

func TestABlankReasonIsRefused(t *testing.T) {
	c := reasonPlate(t)
	c.HandleMsg(press("3"))
	typeText(c, "   ")
	c.HandleMsg(press("enter"))
	if !c.IsCapturingText() || c.message != "A reason is needed for leave out: type one, or esc to undo." {
		t.Fatalf("blank reason: open %v, message %q", c.IsCapturingText(), c.message)
	}
}

func TestEscInTheBoxPutsTheWellBack(t *testing.T) {
	c := reasonPlate(t)
	c.HandleMsg(press("3")) // X01: vehicle -> leave out
	typeText(c, "half")
	if r, _ := c.HandleMsg(press("esc")); r != nil {
		t.Fatalf("esc in the box replied: %+v", r)
	}
	if c.IsCapturingText() || c.assignments["X01"] != "vehicle" || c.reasons["X01"] != "" {
		t.Fatalf("after esc: open %v, X01 %q, reason %q", c.IsCapturingText(), c.assignments["X01"], c.reasons["X01"])
	}
	if c.message != "X01 back to vehicle." || c.escArmed {
		t.Fatalf("message %q, armed %v", c.message, c.escArmed)
	}
}

func TestTabInsertsTheWarningAndEnterSavesIt(t *testing.T) {
	c := reasonPlate(t)
	c.HandleMsg(press("right"))
	c.HandleMsg(press("3"))
	c.HandleMsg(tea.KeyPressMsg{Code: tea.KeyTab})
	c.HandleMsg(press("enter"))
	if c.IsCapturingText() || c.reasons["X02"] != "low RNA yield (RIN 4.1)" {
		t.Fatalf("open %v, reason %q", c.IsCapturingText(), c.reasons["X02"])
	}
	if v := view(c); !strings.Contains(v, "D  ●   ◉   ●!  ◉") ||
		!strings.Contains(v, "now: leave out · reason: low RNA yield (RIN 4.1)") {
		t.Fatalf("saved reason not shown:\n%s", v)
	}
}

func TestPasteGoesIntoTheBox(t *testing.T) {
	c := reasonPlate(t)
	c.HandleMsg(press("3"))
	c.HandleMsg(tea.PasteMsg{Content: "spilled\nduring  extraction"})
	c.HandleMsg(press("enter"))
	if c.reasons["X01"] != "spilled during extraction" {
		t.Fatalf("reason = %q", c.reasons["X01"])
	}
}

func TestREditsASavedReason(t *testing.T) {
	c := reasonPlate(t)
	c.HandleMsg(press("r")) // X01 is vehicle: no reason to edit
	if c.IsCapturingText() || !strings.Contains(c.message, "doesn't take one") {
		t.Fatalf("r on vehicle: open %v, %q", c.IsCapturingText(), c.message)
	}
	for i := 0; i < 3; i++ {
		c.HandleMsg(press("right")) // X04: leave out, "tube cracked"
	}
	c.HandleMsg(press("r"))
	if !c.IsCapturingText() || c.box.text != "tube cracked" {
		t.Fatalf("r should open the box with the reason, got %v %q", c.IsCapturingText(), c.box.text)
	}
	typeText(c, " at spin")
	c.HandleMsg(press("enter"))
	if c.reasons["X04"] != "tube cracked at spin" {
		t.Fatalf("reason = %q", c.reasons["X04"])
	}
	c.HandleMsg(press("r"))
	typeText(c, "xx")
	c.HandleMsg(press("esc")) // undo the edit: the saved reason stays
	if c.reasons["X04"] != "tube cracked at spin" || c.assignments["X04"] != "leave_out" {
		t.Fatalf("esc after r: %q %q", c.assignments["X04"], c.reasons["X04"])
	}
}

func TestEnterIsRefusedWhileAReasonIsMissing(t *testing.T) {
	c := reasonPlate(t)
	if r, _ := c.HandleMsg(press("enter")); r != nil {
		t.Fatalf("submitted with X03's reason missing: %+v", r)
	}
	if c.message != "Needs a reason (press r): D3 X03" {
		t.Fatalf("message = %q", c.message)
	}
	if row, col := c.plate.Cursor(); row != 3 || col != 2 {
		t.Fatalf("enter should move to X03 (D3), got %d,%d", row, col)
	}
	c.HandleMsg(press("r"))
	typeText(c, "contaminated")
	c.HandleMsg(press("enter"))
	r, _ := c.HandleMsg(press("enter"))
	if r == nil || r.Action != "submit" {
		t.Fatalf("expected submit, got %+v (%q)", r, c.message)
	}
	want := map[string]any{"X03": "contaminated", "X04": "tube cracked"}
	if got := r.Data["reasons"]; !reflect.DeepEqual(got, want) {
		t.Fatalf("reasons = %v, want %v", got, want)
	}
}

func TestPickingAnotherOptionDropsTheReason(t *testing.T) {
	c := reasonPlate(t)
	for i := 0; i < 3; i++ {
		c.HandleMsg(press("right"))
	}
	c.HandleMsg(press("1")) // X04: leave out -> vehicle
	if _, ok := c.reasons["X04"]; ok {
		t.Fatal("a reason stayed on a sample whose option takes none")
	}
	c.HandleMsg(press("3")) // back to leave out: asked afresh
	if c.box.text != "" {
		t.Fatalf("the dropped reason came back: %q", c.box.text)
	}
}

func TestReasonsSurviveADraftAndPrefill(t *testing.T) {
	c := reasonPlate(t)
	c.HandleMsg(press("3"))
	typeText(c, "spilled")
	c.HandleMsg(press("enter"))
	c.HandleMsg(press("right"))
	c.HandleMsg(press("right"))
	c.HandleMsg(press("r"))
	typeText(c, "contaminated")
	c.HandleMsg(press("enter"))
	first, _ := c.HandleMsg(press("enter"))
	if first == nil {
		t.Fatalf("not sent: %q", c.message)
	}

	again := reasonPlate(t)
	again.Prefill(first.Data)
	if !reflect.DeepEqual(again.Draft(), c.Draft()) {
		t.Fatalf("prefill draft = %v, want %v", again.Draft(), c.Draft())
	}
	second, _ := again.HandleMsg(press("enter"))
	if second == nil || !reflect.DeepEqual(first.Data, second.Data) {
		t.Fatalf("prefill changed the reply:\n%v\n%v", first.Data, second)
	}
}

func TestDraftLeavesReasonsOutWhenThereAreNone(t *testing.T) {
	c := newComp(t)
	if _, ok := c.Draft()["reasons"]; ok {
		t.Fatalf("draft = %v", c.Draft())
	}
}

func TestInitRefusesBadOptions(t *testing.T) {
	for name, options := range map[string]string{
		"duplicate":     `[{"id": "a"}, {"id": "a"}]`,
		"reason no key": `[{"id": "a", "reason": {"key": ""}}]`,
		"empty id":      `[{"id": ""}]`,
		"ten options":   `[{"id":"1"},{"id":"2"},{"id":"3"},{"id":"4"},{"id":"5"},{"id":"6"},{"id":"7"},{"id":"8"},{"id":"9"},{"id":"10"}]`,
	} {
		raw := `{"options": ` + options + `, "wells": [{"row": 0, "col": 0, "sample_id": "X"}]}`
		if err := (&SamplePlateComponent{}).Init([]byte(raw)); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
}

func TestDescribeListsEveryReasonForTheReview(t *testing.T) {
	answer := map[string]any{
		"assignments": map[string]any{"X01": "vehicle", "X02": "tdx9", "X03": "leave_out", "X04": "leave_out"},
		"reasons":     map[string]any{"X03": "contaminated", "X04": "tube cracked"},
	}
	got := Describe(json.RawMessage(withReasons), answer)
	want := []string{
		"vehicle (1): X01",
		"TDX-9 (tdx9) (1): X02",
		"leave out (leave_out) (2): X03, X04",
		"reason: X03 → leave out: contaminated",
		"reason: X04 → leave out: tube cracked",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Describe = %q\nwant %q", got, want)
	}
	if s := Summary(json.RawMessage(withReasons), answer); s != "vehicle (1) · TDX-9 (1) · leave out (2)" {
		t.Fatalf("Summary = %q", s)
	}
}

func TestNSwitchesTheWellsToNumbersAndBack(t *testing.T) {
	c := reasonPlate(t)
	c.HandleMsg(press("n"))
	if v := view(c); !strings.Contains(v, "D  1   2   3!  3") || !strings.Contains(v, "n dots") {
		t.Fatalf("numbers not drawn:\n%s", v)
	}
	c.HandleMsg(press("n"))
	if v := view(c); !strings.Contains(v, "D  ●   ●   ●!  ◉") || !strings.Contains(v, "n numbers") {
		t.Fatalf("dots not back:\n%s", v)
	}
}

func TestAGapSeparatesThePlateFromTheLinesUnderIt(t *testing.T) {
	lines := strings.Split(view(reasonPlate(t)), "\n")
	for i, l := range lines {
		if strings.HasPrefix(l, "H  ") {
			if strings.TrimSpace(lines[i+1]) != "" || !strings.HasPrefix(lines[i+2], "1 ● vehicle") {
				t.Fatalf("no gap under the plate:\n%s", strings.Join(lines, "\n"))
			}
			return
		}
	}
	t.Fatal("no row H")
}

func TestALabelHoldingTheIdIsNotRepeated(t *testing.T) {
	c := &SamplePlateComponent{}
	raw := `{"options": [{"id": "a"}], "wells": [
	  {"row": 0, "col": 0, "sample_id": "R01", "label": "R01 Line A"},
	  {"row": 0, "col": 1, "sample_id": "R02", "label": "Line B"}]}`
	if err := c.Init([]byte(raw)); err != nil {
		t.Fatal(err)
	}
	if v := view(c); !strings.Contains(v, "A1  R01 Line A · proposed") {
		t.Fatalf("id repeated:\n%s", v)
	}
	c.HandleMsg(press("right"))
	if v := view(c); !strings.Contains(v, "A2  R02 · Line B · proposed") {
		t.Fatalf("a label without the id should keep it:\n%s", v)
	}
}
