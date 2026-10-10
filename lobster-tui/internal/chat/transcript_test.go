package chat

import (
	"fmt"
	"strings"
	"testing"

	tea "charm.land/bubbletea/v2"

	"github.com/the-omics-os/lobster-tui/internal/protocol"
)

// Scenario tests for inline scrollback delivery. Each drives the real
// Model.Update loop and asserts the ordered transcript of tea.Println bodies.

func TestTranscriptCompletedAnswerIsPrintedOnce(t *testing.T) {
	h := newTranscriptHarness(t)
	h.user("question one")
	h.text("ANSWER-ONE")
	h.done()
	h.settle()

	h.expectOnce("question one", "ANSWER-ONE")
	h.order("question one", "ANSWER-ONE")
	if s := h.m.printStats; s.committed != s.dispatched || len(h.m.outbox) != 0 || len(h.m.inflightPrints) != 0 {
		t.Fatalf("every dispatched segment should be acknowledged: %+v", s)
	}
	if h.m.activeTurn != activeTurnNone || h.m.isStreaming {
		t.Fatal("completed turn must be inactive")
	}
	h.statsBalanced()
}

func TestTranscriptTwoTurnsKeepEveryAnswerInOrder(t *testing.T) {
	h := newTranscriptHarness(t)
	h.user("q1")
	h.text("ANSWER-ONE")
	h.done()
	h.settle()
	h.user("q2")
	h.text("ANSWER-TWO")
	h.done()
	h.settle()

	h.expectOnce("ANSWER-ONE", "ANSWER-TWO")
	h.order("q1", "ANSWER-ONE", "q2", "ANSWER-TWO")
	h.statsBalanced()
}

func TestTranscriptNextTurnStartsBeforeTickStillPrintsPreviousAnswer(t *testing.T) {
	h := newTranscriptHarness(t)
	h.holdTic = true
	h.user("q1")
	h.text("ANSWER-ONE")
	h.done()
	h.user("q2") // submitted before the deferral tick fires
	h.text("ANSWER-TWO")
	h.done()
	h.settle()

	h.expectOnce("ANSWER-ONE", "ANSWER-TWO")
	// q2 was queued behind the deferred first answer, so order is preserved.
	h.order("q1", "ANSWER-ONE", "q2", "ANSWER-TWO")
	h.statsBalanced()
}

func TestTranscriptEmptyOrRepeatedDoneNeverReprintsEarlierAnswer(t *testing.T) {
	h := newTranscriptHarness(t)
	h.user("q1")
	h.text("ANSWER-ONE")
	h.done()
	h.settle()

	h.done() // repeated
	h.settle()
	h.user("/status") // slash turn that only produces an alert
	h.alert("warning", "ALERT-ONLY")
	h.done()
	h.settle()
	h.user("q3") // tool-only turn: no text at all
	h.done()
	h.settle()

	h.expectOnce("ANSWER-ONE", "ALERT-ONLY")
	h.statsBalanced()
}

func TestTranscriptHITLInterruptResumePrintsEachPartOnce(t *testing.T) {
	h := newTranscriptHarness(t)
	h.user("analyze")
	h.text("QUESTION-PART")
	h.doneWith("interrupt")
	h.settle()
	if h.m.activeTurn != activeTurnChat {
		t.Fatal("an interrupt pauses the turn; it must stay active")
	}
	h.text("RESUMED-PART")
	h.done()
	h.settle()

	h.expectOnce("QUESTION-PART", "RESUMED-PART")
	h.order("analyze", "QUESTION-PART", "RESUMED-PART")
	if h.m.activeTurn != activeTurnNone {
		t.Fatal("final done must close the turn")
	}
	h.statsBalanced()
}

func TestTranscriptClearBeforeTickDropsStaleAnswerOnly(t *testing.T) {
	h := newTranscriptHarness(t)
	h.holdTic = true
	h.user("q1")
	h.text("STALE-ANSWER")
	h.done()
	h.m.applyClearTarget("output") // /clear while the deferred print is pending
	h.user("q2")
	h.text("FRESH-ANSWER")
	h.done()
	h.settle() // stale tick is released first, then the fresh one

	h.expectNever("STALE-ANSWER")
	h.expectOnce("FRESH-ANSWER")
	h.statsBalanced()
	if h.m.printStats.dropped == 0 {
		t.Fatal("expected the cleared segment to be counted as dropped")
	}
}

func TestTranscriptStatusClearKeepsCompletedAnswer(t *testing.T) {
	h := newTranscriptHarness(t)
	h.holdTic = true
	h.user("q1")
	h.text("KEEP-ME")
	h.done()
	h.m.applyClearTarget("status")
	h.settle()
	h.expectOnce("KEEP-ME")
}

func TestTranscriptCancelDiscardsPartialAndNeverResurrectsIt(t *testing.T) {
	h := newTranscriptHarness(t)
	h.user("q1")
	h.text("PARTIAL-TEXT")
	h.proto(protocol.TypeCode, protocol.CodePayload{Language: "python", Content: "UNFINISHED-CODE"})
	h.doneWith("cancelled")
	h.settle()
	h.expectNever("PARTIAL-TEXT", "UNFINISHED-CODE")

	h.user("q2")
	h.text("ANSWER-TWO")
	h.done()
	h.settle()

	h.expectOnce("ANSWER-TWO")
	h.expectNever("PARTIAL-TEXT", "UNFINISHED-CODE")
	h.statsBalanced()
}

func TestTranscriptCancellingALaterTurnKeepsEarlierCompletedAnswer(t *testing.T) {
	h := newTranscriptHarness(t)
	h.holdTic = true
	h.user("q1")
	h.text("ANSWER-ONE")
	h.done()
	h.user("q2")
	h.text("PARTIAL-TWO")
	h.m.isCanceling = true // Ctrl+C on turn two while answer one is still pending
	h.doneWith("cancelled")
	h.settle()

	h.expectOnce("ANSWER-ONE")
	h.expectNever("PARTIAL-TWO")
}

func TestTranscriptAlertBetweenDoneAndTickCannotOvertakeAnswer(t *testing.T) {
	h := newTranscriptHarness(t)
	h.holdTic = true
	h.user("q1")
	h.text("ANSWER-ONE")
	h.done()
	h.alert("warning", "LATE-WARNING") // arrives before the 50ms tick
	h.settle()

	h.order("ANSWER-ONE", "LATE-WARNING")
	h.expectOnce("ANSWER-ONE", "LATE-WARNING")
}

func TestTranscriptErrorDoneKeepsPartialMarkedIncompleteAndClosesTurn(t *testing.T) {
	h := newTranscriptHarness(t)
	h.user("q1")
	h.text("PARTIAL-ANSWER")
	h.alert("error", "BACKEND-FAILED")
	h.doneWith("error")
	h.settle()

	h.expectOnce("PARTIAL-ANSWER", "BACKEND-FAILED")
	h.expectOnce("Response incomplete")
	h.order("PARTIAL-ANSWER", "Response incomplete")
	if h.m.activeTurn != activeTurnNone || h.m.isStreaming || h.m.streamBuf.Len() != 0 {
		t.Fatalf("error done must close the turn: turn=%v streaming=%v buf=%q",
			h.m.activeTurn, h.m.isStreaming, h.m.streamBuf.String())
	}

	h.user("q2")
	h.text("CLEAN-ANSWER")
	h.done()
	h.settle()
	h.expectOnce("CLEAN-ANSWER")
	if strings.Contains(h.joined(), "PARTIAL-ANSWERCLEAN") || strings.Contains(h.joined(), "PARTIAL-ANSWER CLEAN") {
		t.Fatal("next turn must not be concatenated onto the failed turn")
	}
	h.statsBalanced()
}

func TestTranscriptErrorDoneWithoutOutputPrintsNothingExtra(t *testing.T) {
	h := newTranscriptHarness(t)
	h.user("q1")
	h.text("ANSWER-ONE")
	h.done()
	h.settle()
	h.user("q2")
	h.alert("error", "FAILED-EARLY")
	h.doneWith("error")
	h.settle()

	h.expectOnce("ANSWER-ONE", "FAILED-EARLY")
	h.expectNever("Response incomplete")
}

func TestTranscriptBackendExitFlushesPendingAnswerBeforeQuit(t *testing.T) {
	h := newTranscriptHarness(t)
	h.holdTic = true
	h.user("q1")
	h.text("LAST-WORDS")
	h.done()
	h.send(protocolEOF{}) // backend exits right after done, tick still pending

	h.expectOnce("LAST-WORDS")
	if !h.quit {
		t.Fatal("EOF must quit")
	}
	if h.quitAfter < len(h.prints) {
		t.Fatal("answer must be printed before quit")
	}
}

func TestTranscriptBackendExitMidStreamPrintsPartialAsIncomplete(t *testing.T) {
	h := newTranscriptHarness(t)
	h.user("q1")
	h.text("HALF-AN-ANSWER")
	h.send(protocolEOF{})

	h.expectOnce("HALF-AN-ANSWER", "Response incomplete")
	h.order("HALF-AN-ANSWER", "Response incomplete")
}

func TestTranscriptMixedBlocksPrintOnceInOrder(t *testing.T) {
	h := newTranscriptHarness(t)
	h.user("q1")
	h.text("INTRO-TEXT")
	h.proto(protocol.TypeCode, protocol.CodePayload{Language: "python", Content: "CODE-BLOCK"})
	h.proto(protocol.TypeTable, protocol.TablePayload{Headers: []string{"COL-A"}, Rows: [][]string{{"CELL-1"}}})
	h.proto(protocol.TypeAgentTransition, protocol.AgentTransitionPayload{To: "de_expert", Reason: "HANDOFF-TASK"})
	h.text("CLOSING-TEXT")
	h.done()
	h.settle()

	h.expectOnce("INTRO-TEXT", "CODE-BLOCK", "CELL-1", "HANDOFF-TASK", "CLOSING-TEXT")
	// Scrollback order follows transcript order, including the handoff that
	// arrived between the table and the closing text.
	h.order("INTRO-TEXT", "CODE-BLOCK", "CELL-1", "HANDOFF-TASK", "CLOSING-TEXT")
	h.statsBalanced()
}

func TestTranscriptHandoffFlushDoesNotReprintAtDone(t *testing.T) {
	h := newTranscriptHarness(t)
	h.user("q1")
	h.text("SUPERVISOR-LINE")
	h.proto(protocol.TypeAgentTransition, protocol.AgentTransitionPayload{To: "de_expert", Reason: "DELEGATED-TASK"})
	h.text("SPECIALIST-LINE") // flushes the pending segment (handoff badge)
	h.done()
	h.settle()

	h.expectOnce("SUPERVISOR-LINE", "DELEGATED-TASK", "SPECIALIST-LINE")
	h.order("SUPERVISOR-LINE", "DELEGATED-TASK", "SPECIALIST-LINE")
}

func TestTranscriptSlashCommandOutputIsImmediate(t *testing.T) {
	h := newTranscriptHarness(t)
	h.holdTic = true
	h.user("/workspace")
	h.proto(protocol.TypeTable, protocol.TablePayload{Headers: []string{"H"}, Rows: [][]string{{"SLASH-ROW"}}})
	h.done()

	h.expectOnce("SLASH-ROW")
	if len(h.held) != 0 {
		t.Fatal("slash command output has no live stream and must not wait for a tick")
	}
}

func TestTranscriptStaleSegmentIDsAreNeverReused(t *testing.T) {
	h := newTranscriptHarness(t)
	h.holdTic = true
	h.user("q1")
	h.text("OLD")
	h.done()
	oldHeld := append([]inlinePrintReady(nil), h.held...)
	if len(oldHeld) != 1 {
		t.Fatalf("expected one pending deferral, got %d", len(oldHeld))
	}
	h.m.applyClearTarget("all")
	h.user("q2")
	h.text("NEW")
	h.done()
	if h.held[len(h.held)-1].id <= oldHeld[0].id {
		t.Fatal("segment IDs must be monotonic across clear")
	}
	// A late ack for the dropped segment must not touch accounting.
	before := h.m.printStats
	h.send(inlinePrintComplete{id: oldHeld[0].id})
	if h.m.printStats.committed != before.committed {
		t.Fatal("stale ack committed a segment it never dispatched")
	}
	h.settle()
	h.expectNever("OLD")
	h.expectOnce("NEW")
}

func TestTranscriptBannerIsNotConsumedByDroppedSegment(t *testing.T) {
	h := newTranscriptHarness(t)
	h.m.inlineBannerPrinted = false
	h.holdTic = true
	h.user("q1") // user message is dispatched immediately and carries the banner
	if !h.m.inlineBannerPrinted {
		t.Fatal("banner should be consumed by the first dispatched print")
	}

	h2 := newTranscriptHarness(t)
	h2.m.inlineBannerPrinted = false
	h2.m.ready = true
	h2.m.inflightPrints = nil
	// Prepare a deferred first segment, then clear before it is dispatched.
	h2.m.activeTurn = activeTurnChat
	h2.m.isStreaming = true
	h2.m.streamBuf.WriteString("never shown")
	h2.holdTic = true
	h2.done()
	if h2.m.inlineBannerPrinted {
		t.Fatal("preparing a deferred print must not consume the banner")
	}
	h2.m.applyClearTarget("output")
	h2.settle()
	if h2.m.inlineBannerPrinted {
		t.Fatal("a dropped segment must not consume the banner")
	}
	h2.expectNever("never shown")
}

func TestTranscriptDroppedCommandCannotLoseQueuedOutput(t *testing.T) {
	// Handlers that queue output and discard their own return value (as
	// renderDataSummary-style helpers do) must not lose it: Update drains.
	h := newTranscriptHarness(t)
	h.m.appendMessage(ChatMessage{Role: "system", Blocks: textBlocks("QUEUED-BY-HELPER")}, false)
	h.send(tea.WindowSizeMsg{Width: 80, Height: 24})
	h.expectOnce("QUEUED-BY-HELPER")
}

func TestTranscriptUserQuitFlushesPendingAnswer(t *testing.T) {
	h := newTranscriptHarness(t)
	h.holdTic = true
	h.user("q1")
	h.text("PARTING-ANSWER")
	h.done() // deferred print still waiting for its tick
	h.send(tea.KeyPressMsg{Code: 'c', Mod: tea.ModCtrl})

	h.expectOnce("PARTING-ANSWER")
	if !h.quit || h.quitAfter < len(h.prints) {
		t.Fatal("quit must happen after the pending answer is written")
	}
}

func TestTranscriptLongAnswerPrintsInTerminalSizedPieces(t *testing.T) {
	// A streamed answer taller than the terminal must be written in pieces
	// that fit next to the live frame; one oversized insert pushes the prompt,
	// footer and blank rows into scrollback above the answer.
	h := newTranscriptHarness(t)
	h.send(tea.WindowSizeMsg{Width: 100, Height: 40})
	h.user("which sub-agents do you have?")
	var body strings.Builder
	for i := 0; i < 30; i++ {
		fmt.Fprintf(&body, "LINE-%02d lorem ipsum dolor sit amet\n\n", i)
	}
	for _, chunk := range strings.SplitAfter(body.String(), "\n") {
		h.text(chunk)
	}
	h.done()
	h.settle()

	frame := strings.Count(h.m.View().Content, "\n") + 1
	for i, p := range h.prints {
		if rows := strings.Count(p, "\n") + 1; rows+frame > h.m.height {
			t.Fatalf("print %d is %d rows; with a %d-row frame it overflows a %d-row terminal", i, rows, frame, h.m.height)
		}
	}
	if len(h.prints) < 3 {
		t.Fatalf("expected the long answer to be split, got %d prints", len(h.prints))
	}
	needles := make([]string, 0, 30)
	for i := 0; i < 30; i++ {
		needles = append(needles, fmt.Sprintf("LINE-%02d", i))
	}
	h.expectOnce(needles...)
	h.order(needles...)
	h.statsBalanced()
}
