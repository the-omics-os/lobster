package chat

import (
	"strings"
	"time"

	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
)

// Inline scrollback outbox.
//
// In inline-flow mode finished transcript content is written to the terminal
// scrollback with tea.Println. That write used to be decided from transient
// model state (active turn, cancel flag, "last assistant message") at the moment
// a delayed command fired, which let completed answers disappear or be printed
// twice. The outbox makes delivery explicit:
//
//   - Selection: every ChatMessage tracks how many of its blocks were already
//     handed to scrollback (printed). Only blocks beyond that cursor are ever
//     selected, so repeated or empty completions are naturally idempotent.
//   - Capture: selected blocks are copied into an immutable printSegment, so
//     later appends to (or merges into) m.messages cannot change what is
//     delivered.
//   - Ordering: segments are delivered strictly FIFO. A segment that must wait
//     one frame for the viewport to clear ("deferred") holds back everything
//     queued behind it.
//   - Invalidation: clearing the transcript bumps inlinePrintEpoch and drops
//     queued segments. Segment IDs are never reused, so a late tick or ack for a
//     dropped segment can never affect newer output.
//   - Single egress: segments leave the outbox only from Model.Update (see
//     drainOutbox), so no call site can mark output delivered and then lose the
//     command that delivers it.
//
// The acknowledgement (inlinePrintComplete) is application-level: it proves the
// Println message was handed to the renderer, not that the terminal painted it.

// printDeferDelay is how long a deferred segment waits so BubbleTea can flush
// the cleared live viewport before the final text is written above it.
const printDeferDelay = 50 * time.Millisecond

// incompleteResponseNotice follows partial output of a turn that ended
// abnormally, so a truncated answer is never mistaken for a finished one.
const incompleteResponseNotice = "Response incomplete: the turn ended before the answer finished."

// printSegment is one immutable unit of scrollback output.
type printSegment struct {
	id    uint64
	epoch uint64
	// parts are the rendered, non-empty message renderings in order.
	parts    []string
	deferred bool // wait one frame tick before delivery
	ticked   bool // the deferral tick was scheduled
	ready    bool // the deferral tick fired
}

// printStats are content-free counters describing the outbox lifecycle. They
// exist for tests and diagnostics and never contain transcript text.
type printStats struct {
	enqueued   uint64
	dispatched uint64
	committed  uint64
	dropped    uint64
}

// cloneBlocks copies a block slice so a captured segment is independent of the
// mutable message it came from.
func cloneBlocks(blocks []ContentBlock) []ContentBlock {
	return append([]ContentBlock(nil), blocks...)
}

// hasUnprintedAssistant reports whether any assistant block has not yet been
// handed to scrollback.
func (m *Model) hasUnprintedAssistant() bool {
	for i := m.printScan; i < len(m.messages); i++ {
		msg := &m.messages[i]
		if msg.Role == "assistant" && msg.printed < len(msg.Blocks) {
			return true
		}
	}
	return false
}

// takeUnprinted returns copies of every block that has not yet been handed to
// scrollback and advances the per-message cursors. It is the only selector for
// turn output, replacing "latest assistant message" lookups.
func (m *Model) takeUnprinted() []ChatMessage {
	var out []ChatMessage
	for i := m.printScan; i < len(m.messages); i++ {
		msg := &m.messages[i]
		if msg.printed >= len(msg.Blocks) {
			continue
		}
		out = append(out, ChatMessage{
			Role:   msg.Role,
			Blocks: cloneBlocks(msg.Blocks[msg.printed:]),
			Agent:  msg.Agent,
		})
		msg.printed = len(msg.Blocks)
	}
	m.advancePrintScan()
	return out
}

// discardUnprinted consumes every unprinted block without printing it. Used
// when a turn is cancelled: its unfinished output must neither appear now nor
// resurface with a later turn.
func (m *Model) discardUnprinted() {
	for i := m.printScan; i < len(m.messages); i++ {
		m.messages[i].printed = len(m.messages[i].Blocks)
	}
	m.advancePrintScan()
}

// advancePrintScan moves the low-water mark past the leading run of fully
// consumed messages. The last message always stays in range because blocks can
// still be appended to it.
func (m *Model) advancePrintScan() {
	for m.printScan < len(m.messages)-1 {
		msg := &m.messages[m.printScan]
		if msg.printed < len(msg.Blocks) {
			return
		}
		m.printScan++
	}
}

// markMessagesPrinted records that freshly appended messages are being
// delivered as a whole (user prompts, alerts, standalone handoffs).
func (m *Model) markMessagesPrinted(from int) {
	for i := from; i < len(m.messages); i++ {
		m.messages[i].printed = len(m.messages[i].Blocks)
	}
	m.advancePrintScan()
}

// renderPrintParts renders messages for scrollback, dropping empty renderings.
func (m *Model) renderPrintParts(msgs []ChatMessage) []string {
	if !m.inlineFlowMode() {
		return nil
	}
	renderer := m.getMarkdownRenderer()
	parts := make([]string, 0, len(msgs))
	for _, msg := range msgs {
		rendered := strings.TrimRight(renderMessage(msg, m.styles, m.width, renderer, true), "\n")
		if strings.TrimSpace(rendered) == "" {
			continue
		}
		parts = append(parts, rendered)
	}
	return parts
}

// enqueuePrint captures msgs as an immutable segment. Nothing is written until
// Model.Update drains the outbox.
func (m *Model) enqueuePrint(msgs []ChatMessage, deferred bool) {
	parts := m.renderPrintParts(msgs)
	if len(parts) == 0 {
		return
	}
	m.nextPrintID++
	m.outbox = append(m.outbox, printSegment{
		id:       m.nextPrintID,
		epoch:    m.inlinePrintEpoch,
		parts:    parts,
		deferred: deferred,
	})
	m.printStats.enqueued++
}

// invalidatePrints drops every queued segment. Called when the transcript is
// cleared; already-dispatched segments cannot be recalled.
func (m *Model) invalidatePrints() {
	m.inlinePrintEpoch++
	m.printStats.dropped += uint64(len(m.outbox))
	m.outbox = nil
}

// segmentBody joins a segment's parts, prefixing the one-time banner the first
// time anything is actually dispatched (not when it is merely prepared).
func (m *Model) segmentBody(seg printSegment) string {
	if m.inlineBannerPrinted {
		return strings.Join(seg.parts, "\n")
	}
	m.inlineBannerPrinted = true
	parts := make([]string, 0, len(seg.parts)+2)
	if header := strings.TrimSpace(renderHeader(*m)); header != "" {
		parts = append(parts, header)
	}
	if summary := strings.TrimSpace(renderRuntimeSummary(*m)); summary != "" {
		parts = append(parts, summary)
	}
	parts = append(parts, seg.parts...)
	return strings.Join(parts, "\n")
}

// splitForScrollback splits a scrollback body into pieces that each fit in the
// terminal together with the live frame.
//
// Bubble Tea's inline insert-above makes room by scrolling the screen by the
// height of the printed text and then moving back up over the frame. When the
// text plus the frame is taller than the terminal, that cursor move is clamped
// at the top row and the frame (prompt, footer) and the blank rows left by a
// shrunken live view are pushed into the scrollback as a ghost. A long answer
// streamed into a full-height live view hits exactly that case on completion.
// Printing it in terminal-sized pieces keeps every insert within bounds.
func (m *Model) splitForScrollback(body string) []string {
	if m.height <= 0 {
		return []string{body}
	}
	frame := strings.Count(m.View().Content, "\n") + 1
	budget := m.height - frame - 1
	if budget < 1 {
		budget = 1
	}
	lines := strings.Split(body, "\n")
	var chunks []string
	start, rows := 0, 0
	for i, line := range lines {
		r := 1
		if w := lipgloss.Width(line); m.width > 0 && w > m.width {
			r += w / m.width
		}
		if rows > 0 && rows+r > budget {
			chunks = append(chunks, strings.Join(lines[start:i], "\n"))
			start, rows = i, 0
		}
		rows += r
	}
	return append(chunks, strings.Join(lines[start:], "\n"))
}

// markPrintReady releases a deferred segment once its tick fires. Ticks for
// dropped or unknown segments are ignored.
func (m *Model) markPrintReady(id, epoch uint64) {
	if epoch != m.inlinePrintEpoch {
		return
	}
	for i := range m.outbox {
		if m.outbox[i].id == id {
			m.outbox[i].ready = true
			return
		}
	}
}

// commitPrint records the application-level acknowledgement of a dispatched
// segment. Unknown or repeated IDs are ignored.
func (m *Model) commitPrint(id uint64) {
	if id == 0 {
		return
	}
	if _, ok := m.inflightPrints[id]; ok {
		delete(m.inflightPrints, id)
		m.printStats.committed++
	}
}

// drainOutbox schedules deferral ticks and dispatches every segment at the head
// of the queue that is ready, in order. When force is set deferred segments are
// released immediately (shutdown).
func (m *Model) drainOutbox(force bool) tea.Cmd {
	var cmds []tea.Cmd
	for i := range m.outbox {
		seg := &m.outbox[i]
		if force {
			seg.ready = true
		}
		if seg.deferred && !seg.ticked && !seg.ready {
			seg.ticked = true
			id, epoch := seg.id, seg.epoch
			cmds = append(cmds, tea.Tick(printDeferDelay, func(time.Time) tea.Msg {
				return inlinePrintReady{id: id, epoch: epoch}
			}))
		}
	}

	var seq []tea.Cmd
	n := 0
	for n < len(m.outbox) {
		seg := m.outbox[n]
		if seg.deferred && !seg.ready {
			break
		}
		id := seg.id
		for _, chunk := range m.splitForScrollback(m.segmentBody(seg)) {
			seq = append(seq, tea.Println(chunk))
		}
		seq = append(seq, func() tea.Msg { return inlinePrintComplete{id: id} })
		if m.inflightPrints == nil {
			m.inflightPrints = make(map[uint64]struct{}, 4)
		}
		m.inflightPrints[id] = struct{}{}
		m.printStats.dispatched++
		n++
	}
	if n > 0 {
		m.outbox = append([]printSegment(nil), m.outbox[n:]...)
	}
	if len(seq) > 0 {
		cmds = append(cmds, tea.Sequence(seq...))
	}
	return tea.Batch(cmds...)
}
