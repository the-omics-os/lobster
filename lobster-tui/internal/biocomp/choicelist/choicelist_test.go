package choicelist

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"testing"

	tea "charm.land/bubbletea/v2"
	"github.com/charmbracelet/x/ansi"

	"github.com/the-omics-os/lobster-tui/internal/biocomp"
)

// Synthetic QC review: four samples, one of them not answerable, and three
// options of which one needs a reason.
var qcOptions = []map[string]any{
	{"id": "retain", "label": "Retain"},
	{"id": "exclude", "label": "Exclude", "details": "left out of the analysis",
		"reason": map[string]any{"key": "reason", "question": "Why exclude it?"}},
	{"id": "defer", "label": "Defer"},
}

var qcItems = []map[string]any{
	{"id": "S01", "label": "Donor A", "details": "Low library size"},
	{"id": "S02", "label": "Donor B"},
	{"id": "S03", "label": "Donor C", "unavailable": "No counts in this source"},
	{"id": "S05", "label": "Donor A"},
}

func listData(items, options []map[string]any, extra map[string]any) json.RawMessage {
	d := map[string]any{"question": "Review each flagged sample", "items": items, "options": options}
	for k, v := range extra {
		d[k] = v
	}
	raw, _ := json.Marshal(d)
	return raw
}

func newList(t *testing.T, raw json.RawMessage) *Component {
	t.Helper()
	c := &Component{}
	if err := c.Init(raw); err != nil {
		t.Fatal(err)
	}
	return c
}

func keyMsg(k string) tea.Msg {
	switch k {
	case "up":
		return tea.KeyPressMsg{Code: tea.KeyUp}
	case "down":
		return tea.KeyPressMsg{Code: tea.KeyDown}
	case "left":
		return tea.KeyPressMsg{Code: tea.KeyLeft}
	case "right":
		return tea.KeyPressMsg{Code: tea.KeyRight}
	case "enter":
		return tea.KeyPressMsg{Code: tea.KeyEnter}
	case "esc":
		return tea.KeyPressMsg{Code: tea.KeyEscape}
	case "tab":
		return tea.KeyPressMsg{Code: tea.KeyTab}
	case "backspace":
		return tea.KeyPressMsg{Code: tea.KeyBackspace}
	case "space":
		return tea.KeyPressMsg{Code: tea.KeySpace, Text: " "}
	}
	return tea.KeyPressMsg{Code: rune(k[0]), Text: k}
}

func press(c *Component, keys ...string) *biocomp.ComponentResult {
	var result *biocomp.ComponentResult
	for _, k := range keys {
		result, _ = c.HandleMsg(keyMsg(k))
	}
	return result
}

// typeText sends each character as its own key press.
func typeText(c *Component, text string) {
	for _, r := range text {
		if r == ' ' {
			c.HandleMsg(keyMsg("space"))
			continue
		}
		c.HandleMsg(tea.KeyPressMsg{Code: r, Text: string(r)})
	}
}

func answers(t *testing.T, r *biocomp.ComponentResult) map[string]any {
	t.Helper()
	if r == nil || r.Action != "submit" {
		t.Fatalf("expected submit, got %+v", r)
	}
	raw, _ := json.Marshal(r.Data)
	var out struct {
		Answers map[string]any `json:"answers"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatal(err)
	}
	return out.Answers
}

func plain(s string) string { return ansi.Strip(s) }

func TestRegistered(t *testing.T) {
	if biocomp.Get("choice_list") == nil {
		t.Fatal(`"choice_list" is not registered`)
	}
}

func TestInitRejectsUnanswerableData(t *testing.T) {
	cases := map[string]json.RawMessage{
		"no question":       listData(qcItems, qcOptions, map[string]any{"question": ""}),
		"no items":          listData(nil, qcOptions, nil),
		"no options":        listData(qcItems, nil, nil),
		"item without id":   listData([]map[string]any{{"label": "x"}}, qcOptions, nil),
		"duplicate item":    listData([]map[string]any{{"id": "a"}, {"id": "a"}}, qcOptions, nil),
		"duplicate option":  listData(qcItems, []map[string]any{{"id": "a"}, {"id": "a"}}, nil),
		"option without id": listData(qcItems, []map[string]any{{"label": "x"}}, nil),
		"reason without key": listData(qcItems, []map[string]any{
			{"id": "a", "reason": map[string]any{"question": "why?"}}}, nil),
		"every item unavailable": listData([]map[string]any{{"id": "a", "unavailable": "gone"}}, qcOptions, nil),
		"every option unavailable": listData(qcItems, []map[string]any{
			{"id": "a", "unavailable": "no"}}, nil),
		"not json": json.RawMessage(`{"items": "S01"}`),
	}
	for name, raw := range cases {
		if err := (&Component{}).Init(raw); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
}

func TestArrowsAndNumberKeysPickAnOptionForTheRow(t *testing.T) {
	c := newList(t, listData(qcItems, qcOptions, nil))
	press(c, "right") // S01: nothing -> retain
	if got := c.optionOf(0); got != "retain" {
		t.Fatalf("right: S01 = %q", got)
	}
	press(c, "down", "3") // S02: defer
	if got := c.optionOf(1); got != "defer" {
		t.Fatalf("3: S02 = %q", got)
	}
	press(c, "left") // S02: defer -> exclude (needs a reason: box opens)
	if got := c.optionOf(1); got != "exclude" || !c.IsCapturingText() {
		t.Fatalf("left: S02 = %q, box open = %v", got, c.IsCapturingText())
	}
	press(c, "left") // in the box, left still changes the option: -> retain, box closes
	if got := c.optionOf(1); got != "retain" || c.IsCapturingText() {
		t.Fatalf("left in box: S02 = %q, box open = %v", got, c.IsCapturingText())
	}
}

func TestReasonIsTypedInlineAndSent(t *testing.T) {
	c := newList(t, listData(qcItems, qcOptions, nil))
	press(c, "2") // S01: exclude -> box opens
	if !c.IsCapturingText() {
		t.Fatal("the reason box did not open")
	}
	if view := plain(c.View(80, 30)); !strings.Contains(view, "Why exclude it?") {
		t.Fatalf("the box doesn't ask its question:\n%s", view)
	}
	typeText(c, "low depth")
	press(c, "enter")
	if c.IsCapturingText() {
		t.Fatal("enter did not save the reason")
	}
	press(c, "down", "1", "down", "3", "enter")
	got := answers(t, press(c, "enter"))
	want := map[string]any{
		"S01": map[string]any{"id": "exclude", "fields": map[string]any{"reason": "low depth"}},
		"S02": map[string]any{"id": "retain"},
		"S05": map[string]any{"id": "defer"},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("answers = %v\nwant      %v", got, want)
	}
}

func TestEmptyReasonIsRefused(t *testing.T) {
	c := newList(t, listData(qcItems, qcOptions, nil))
	press(c, "2", "enter")
	if !c.IsCapturingText() || !strings.Contains(c.message, "reason is needed") {
		t.Fatalf("empty reason accepted (box open %v, message %q)", c.IsCapturingText(), c.message)
	}
	typeText(c, "   ")
	press(c, "enter")
	if !c.IsCapturingText() {
		t.Fatal("a reason of only spaces was accepted")
	}
}

func TestTheBoxTakesEveryPrintableKey(t *testing.T) {
	c := newList(t, listData(qcItems, qcOptions, nil))
	press(c, "2")
	typeText(c, "1 j k q 3")
	press(c, "tab", "up", "down") // not typed, and they don't move the cursor
	if c.box.text != "1 j k q 3" || c.cursor != 0 || c.optionOf(0) != "exclude" {
		t.Fatalf("box = %q, cursor = %d, option = %q", c.box.text, c.cursor, c.optionOf(0))
	}
	press(c, "backspace")
	c.HandleMsg(tea.PasteMsg{Content: " pasted"})
	if c.box.text != "1 j k q  pasted" {
		t.Fatalf("box = %q", c.box.text)
	}
}

func TestEscInTheBoxDiscardsTheChange(t *testing.T) {
	c := newList(t, listData(qcItems, qcOptions, nil))
	press(c, "1", "2") // retain, then exclude (box)
	typeText(c, "abc")
	if r := press(c, "esc"); r != nil {
		t.Fatalf("esc in the box cancelled the question: %+v", r)
	}
	if c.IsCapturingText() || c.optionOf(0) != "retain" {
		t.Fatalf("esc should restore retain, got %q (box open %v)", c.optionOf(0), c.IsCapturingText())
	}
	// Closing the box is not the first of two escs: two more are needed.
	if r := press(c, "esc"); r != nil {
		t.Fatalf("one esc after closing the box cancelled: %+v", r)
	}
	if r := press(c, "esc"); r == nil || r.Action != "cancel" {
		t.Fatalf("esc esc outside the box should cancel, got %+v", r)
	}
}

// One esc only warns; a second esc in a row cancels; any other key in
// between starts over. The typed reasons and picks survive a single esc.
func TestEscAsksOnceBeforeCancelling(t *testing.T) {
	c := newList(t, listData(qcItems, qcOptions, nil))
	press(c, "2")
	typeText(c, "low depth")
	press(c, "enter", "down", "1") // S01 exclude + reason, S02 retain
	if r := press(c, "esc"); r != nil {
		t.Fatalf("a single esc cancelled: %+v", r)
	}
	if c.message != "press esc again to cancel this question; nothing will be sent" {
		t.Fatalf("message = %q", c.message)
	}
	if !strings.Contains(ansi.Strip(c.View(100, 30)), "press esc again to cancel this question") {
		t.Fatal("the prompt is not on screen")
	}
	if r := press(c, "down"); r != nil || c.message != "" {
		t.Fatalf("another key should disarm and clear the prompt: %+v %q", r, c.message)
	}
	if r := press(c, "esc"); r != nil {
		t.Fatalf("esc after another key cancelled at once: %+v", r)
	}
	if r := press(c, "esc"); r == nil || r.Action != "cancel" {
		t.Fatalf("second esc in a row should cancel, got %+v", r)
	}
	if c.optionOf(0) != "exclude" || c.reasons[0] != "low depth" || c.optionOf(1) != "retain" {
		t.Fatalf("answers changed: %q %q %q", c.optionOf(0), c.reasons[0], c.optionOf(1))
	}
}

func TestEscHelpSaysTwice(t *testing.T) {
	c := newList(t, listData(qcItems, qcOptions, nil))
	for _, b := range c.KeyBindings() {
		if b.Help().Key == "esc" {
			if b.Help().Desc != "twice: cancel" {
				t.Fatalf("esc help = %q", b.Help().Desc)
			}
			return
		}
	}
	t.Fatal("no esc help")
}

func TestUnavailableItemIsSkippedAndNotAnswered(t *testing.T) {
	c := newList(t, listData(qcItems, qcOptions, nil))
	press(c, "down", "down") // S01 -> S02 -> S05 (S03 skipped)
	if c.items[c.cursor].ID != "S05" {
		t.Fatalf("cursor on %s, want S05", c.items[c.cursor].ID)
	}
	press(c, "up")
	if c.items[c.cursor].ID != "S02" {
		t.Fatalf("cursor on %s, want S02", c.items[c.cursor].ID)
	}
	view := plain(c.View(100, 30))
	if !strings.Contains(view, "✕ unavailable: No counts in this source") {
		t.Fatalf("unavailable reason not shown:\n%s", view)
	}
	press(c, "up", "1", "down", "1", "down", "1")
	got := answers(t, press(c, "enter"))
	if _, ok := got["S03"]; ok || len(got) != 3 {
		t.Fatalf("answers = %v, want S01, S02, S05 only", got)
	}
}

func TestUnavailableOptionIsRefused(t *testing.T) {
	opts := append([]map[string]any{}, qcOptions...)
	opts = append(opts, map[string]any{"id": "rerun", "label": "Re-run", "unavailable": "No raw reads"})
	c := newList(t, listData(qcItems, opts, nil))
	press(c, "4")
	if c.optionOf(0) != "" || !strings.Contains(c.message, "No raw reads") {
		t.Fatalf("option = %q, message = %q", c.optionOf(0), c.message)
	}
	press(c, "left") // from nothing, left goes to the last available option
	if c.optionOf(0) != "defer" {
		t.Fatalf("left skipped to %q, want defer", c.optionOf(0))
	}
}

func TestSubmitIsRefusedUntilCompleteAndNamesWhatIsMissing(t *testing.T) {
	c := newList(t, listData(qcItems, qcOptions, map[string]any{
		"default": map[string]any{"S01": map[string]any{"id": "exclude"}},
	}))
	if r := press(c, "enter"); r != nil {
		t.Fatalf("submitted an incomplete list: %+v", r)
	}
	for _, want := range []string{"S02", "S05", "reason", "S01"} {
		if !strings.Contains(c.message, want) {
			t.Fatalf("message %q does not mention %s", c.message, want)
		}
	}
	if strings.Contains(c.message, "S03") {
		t.Fatalf("message asks for the unavailable S03: %q", c.message)
	}
}

func TestDefaultsPreselectAndUnknownIDsAreIgnored(t *testing.T) {
	c := newList(t, listData(qcItems, qcOptions, map[string]any{
		"default": map[string]any{
			"S01": map[string]any{"id": "exclude", "fields": map[string]any{"reason": "low depth"}},
			"S02": map[string]any{"id": "no-such-option"},
			"S03": map[string]any{"id": "retain"}, // unavailable item: never answered
			"S99": map[string]any{"id": "retain"}, // unknown item
			"S05": map[string]any{"id": "defer"},
		},
	}))
	if c.optionOf(0) != "exclude" || c.reasons[0] != "low depth" || c.optionOf(1) != "" || c.optionOf(3) != "defer" {
		t.Fatalf("defaults: %q/%q, %q, %q", c.optionOf(0), c.reasons[0], c.optionOf(1), c.optionOf(3))
	}
	if c.optionOf(2) != "" {
		t.Fatal("an unavailable item was pre-filled")
	}
	if c.IsCapturingText() {
		t.Fatal("a default opened the reason box")
	}
	press(c, "down", "1")
	got := answers(t, press(c, "enter"))
	if got["S01"].(map[string]any)["fields"].(map[string]any)["reason"] != "low depth" {
		t.Fatalf("default reason lost: %v", got)
	}
}

func TestRepickingAReasonOptionKeepsTheSavedReason(t *testing.T) {
	c := newList(t, listData(qcItems, qcOptions, map[string]any{
		"default": map[string]any{"S01": map[string]any{"id": "exclude", "fields": map[string]any{"reason": "low depth"}}},
	}))
	press(c, "2")
	if c.box.text != "low depth" {
		t.Fatalf("box should start with the saved reason, got %q", c.box.text)
	}
}

func TestRepeatedLabelsShowTheirIDs(t *testing.T) {
	view := plain(newList(t, listData(qcItems, qcOptions, nil)).View(100, 30))
	if !strings.Contains(view, "Donor A (S01)") || !strings.Contains(view, "Donor A (S05)") {
		t.Fatalf("ids missing for repeated labels:\n%s", view)
	}
}

func TestSummaryCountsEachOption(t *testing.T) {
	c := newList(t, listData(qcItems, qcOptions, nil))
	press(c, "1", "down", "3")
	want := "Retain 1 · Exclude 0 · Defer 1 · undecided 1 · unavailable 1"
	if got := c.summary(); got != want {
		t.Fatalf("summary = %q, want %q", got, want)
	}
	if view := plain(c.View(100, 30)); !strings.Contains(view, want) {
		t.Fatalf("summary not shown:\n%s", view)
	}
	reply := map[string]any{"answers": map[string]any{"S01": map[string]any{"id": "exclude"}}}
	if got := Summary(listData(qcItems, qcOptions, nil), reply); got != "Retain 0 · Exclude 1 · Defer 0 · undecided 2 · unavailable 1" {
		t.Fatalf("Summary = %q", got)
	}
}

func TestLongTextWrapsAndNothingIsLost(t *testing.T) {
	long := "this sentence is deliberately much longer than forty columns so it must wrap"
	items := []map[string]any{
		{"id": "S01", "label": "first sample label " + long, "details": "detail words " + long},
		{"id": "S02", "label": "second", "unavailable": "reason words " + long},
	}
	opts := []map[string]any{
		{"id": "a", "label": "option label " + long, "details": "option details " + long},
		{"id": "b", "label": "other", "unavailable": "option unavailable " + long},
	}
	c := newList(t, listData(items, opts, map[string]any{"details": "question details " + long}))
	c.question = "question text " + long
	view := plain(c.View(40, 200))
	for _, line := range strings.Split(view, "\n") {
		if w := ansi.StringWidth(line); w > 40 {
			t.Fatalf("line wider than 40 (%d): %q", w, line)
		}
	}
	all := []string{c.question, c.details}
	for _, it := range c.items {
		all = append(all, it.Label, it.Details, it.Unavailable)
	}
	for _, o := range c.options {
		all = append(all, o.Label, o.Details, o.Unavailable)
	}
	words := strings.Fields(view)
	for _, text := range all {
		for _, w := range strings.Fields(text) {
			if !containsWord(words, w) {
				t.Fatalf("word %q missing from the view:\n%s", w, view)
			}
		}
	}
}

func containsWord(words []string, w string) bool {
	for _, x := range words {
		if strings.Contains(x, w) {
			return true
		}
	}
	return false
}

func TestSeventyTwoRowsScrollKeepsTheCursorRowVisible(t *testing.T) {
	var items []map[string]any
	for i := 1; i <= 72; i++ {
		it := map[string]any{"id": fmt.Sprintf("S%02d", i), "label": "Donor with a fairly long name that wraps"}
		if i%10 == 0 {
			it["unavailable"] = "not in this source"
		}
		items = append(items, it)
	}
	c := newList(t, listData(items, qcOptions, nil))
	const width, height = 30, 24
	check := func(step int) {
		t.Helper()
		view := plain(c.View(width, height))
		lines := strings.Split(view, "\n")
		if len(lines) > height {
			t.Fatalf("step %d: %d lines for height %d", step, len(lines), height)
		}
		id := c.items[c.cursor].ID
		if !strings.Contains(view, "("+id+")") {
			t.Fatalf("step %d: cursor row %s not visible:\n%s", step, id, view)
		}
	}
	for step := 0; step < 80; step++ {
		check(step)
		press(c, "1", "down")
	}
	if c.items[c.cursor].ID != "S72" {
		t.Fatalf("cursor stopped at %s", c.items[c.cursor].ID)
	}
	for step := 0; step < 80; step++ {
		press(c, "up")
		check(step)
	}
	got := answers(t, press(c, "enter"))
	if len(got) != 65 { // 72 minus the 7 unavailable
		t.Fatalf("%d answers, want 65", len(got))
	}
}

func TestDraftIsTheReplyShapeOfWhatIsOnScreen(t *testing.T) {
	c := newList(t, listData(qcItems, qcOptions, nil))
	if d := c.Draft(); len(d["answers"].(map[string]any)) != 0 {
		t.Fatalf("empty list draft = %v", d)
	}
	press(c, "1")
	d := c.Draft()["answers"].(map[string]any)
	if !reflect.DeepEqual(d["S01"], map[string]any{"id": "retain"}) {
		t.Fatalf("draft = %v", d)
	}
}

// ctrl+d defers the whole question (v3 C-02): nothing is sent as an answer.
func TestCtrlDDefers(t *testing.T) {
	c := newList(t, listData(qcItems, qcOptions, nil))
	press(c, "1")
	r, _ := c.HandleMsg(tea.KeyPressMsg{Code: 'd', Mod: tea.ModCtrl})
	if r == nil || r.Action != "defer" || len(r.Data) != 0 {
		t.Fatalf("ctrl+d: got %+v", r)
	}
	found := false
	for _, b := range c.KeyBindings() {
		found = found || (b.Help().Key == "ctrl+d" && b.Help().Desc == "defer")
	}
	if !found {
		t.Fatal("ctrl+d is not in the help")
	}
}
