// Package protocol — async read/write handler over stdin/stdout streams.
//
// Handler implements a non-blocking, channel-based I/O bridge between the
// BubbleTea event loop and the Lobster AI Python process. The Python process
// writes newline-delimited JSON to the TUI's stdin; the TUI writes the same
// format to the Python process's stdin.
//
// Concurrency model:
//   - A single goroutine owns the reader (ReadLoop).
//   - A single goroutine owns the writer (WriteLoop).
//   - BubbleTea's Update function sends messages to the writeCh channel and
//     receives decoded Messages from the incoming Msgs() channel.
package protocol

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sync"
)

// Handler manages bidirectional newline-delimited JSON communication.
type Handler struct {
	reader  io.Reader
	writer  io.Writer
	writeMu sync.Mutex

	inCh  chan Message
	errCh chan error
	done  chan struct{}
}

// NewHandler creates a Handler reading from r and writing to w.
// Call StartReadLoop() to begin consuming messages from r.
func NewHandler(r io.Reader, w io.Writer) *Handler {
	return &Handler{
		reader: r,
		writer: w,
		inCh:   make(chan Message, 128),
		errCh:  make(chan error, 8),
		done:   make(chan struct{}),
	}
}

// maxMessageBytes bounds one message line (large markdown blocks or code cells
// exceed bufio's default 64 KB).
const maxMessageBytes = 4 * 1024 * 1024 // 4 MB

// ErrMessageTooLarge is reported on Errs() for a message line of maxMessageBytes
// or more, which is skipped.
var ErrMessageTooLarge = errors.New("a message over 4 MB was skipped")

// StartReadLoop launches a goroutine that continuously reads newline-delimited
// JSON messages from the underlying reader and sends them to the Msgs() channel.
// The loop exits on EOF or a read error; the error is forwarded to Errs(). A
// line that does not parse or is too long is skipped with an error on Errs().
// Call Close() to signal shutdown.
func (h *Handler) StartReadLoop() {
	go func() {
		reader := bufio.NewReader(h.reader)

		for {
			select {
			case <-h.done:
				return
			default:
			}

			line, tooLong, err := readLine(reader)
			if tooLong {
				h.errCh <- fmt.Errorf("protocol read: %w", ErrMessageTooLarge)
			}
			if err != nil {
				if err != io.EOF {
					h.errCh <- fmt.Errorf("protocol read: %w", err)
				}
				// EOF — the Python process exited.
				close(h.inCh)
				return
			}

			line = bytes.TrimSuffix(line, []byte("\r"))
			if len(line) == 0 {
				continue
			}

			var msg Message
			if err := json.Unmarshal(line, &msg); err != nil {
				// Malformed line — emit error and keep running.
				h.errCh <- fmt.Errorf("protocol parse: %w (line: %q)", err, truncate(string(line), 120))
				continue
			}

			select {
			case h.inCh <- msg:
			case <-h.done:
				return
			}
		}
	}()
}

// readLine reads one line without its newline. A line of maxMessageBytes or
// more is read to its end and dropped (tooLong); the last line may lack a
// newline.
func readLine(r *bufio.Reader) (line []byte, tooLong bool, err error) {
	for {
		chunk, err := r.ReadSlice('\n')
		if !tooLong {
			line = append(line, chunk...)
			tooLong = len(bytes.TrimSuffix(line, []byte("\n"))) >= maxMessageBytes
			if tooLong {
				line = nil
			}
		}
		switch {
		case err == bufio.ErrBufferFull:
			continue
		case err == io.EOF && len(line) > 0:
			return line, tooLong, nil
		case err != nil:
			return nil, tooLong, err
		}
		return bytes.TrimSuffix(line, []byte("\n")), tooLong, nil
	}
}

// Send encodes msg as a single JSON line and writes it to the underlying writer.
// Send is safe for concurrent use.
func (h *Handler) Send(msg Message) error {
	if msg.Version == 0 {
		msg.Version = Version
	}

	data, err := json.Marshal(msg)
	if err != nil {
		return fmt.Errorf("protocol marshal: %w", err)
	}

	h.writeMu.Lock()
	defer h.writeMu.Unlock()

	if _, err := h.writer.Write(append(data, '\n')); err != nil {
		return fmt.Errorf("protocol write: %w", err)
	}
	return nil
}

// SendTyped is a convenience wrapper that encodes payload as the Payload field.
func (h *Handler) SendTyped(msgType string, payload any, id string) error {
	raw, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("protocol marshal payload: %w", err)
	}
	return h.Send(Message{
		Version: Version,
		Type:    msgType,
		Payload: raw,
		ID:      id,
	})
}

// Msgs returns the channel on which decoded incoming Messages are delivered.
// The channel is closed when the read loop exits (EOF or Close).
func (h *Handler) Msgs() <-chan Message {
	return h.inCh
}

// Errs returns the channel on which read/parse errors are delivered.
// Errors are non-fatal; the loop continues unless the stream is closed.
func (h *Handler) Errs() <-chan error {
	return h.errCh
}

// Close signals the read loop to exit gracefully.
func (h *Handler) Close() {
	select {
	case <-h.done:
		// Already closed.
	default:
		close(h.done)
	}
}

// DecodePayload unmarshals a Message's Payload field into dst.
// Use this helper in BubbleTea Update() to avoid repeated json.Unmarshal calls.
func DecodePayload(msg Message, dst any) error {
	if err := json.Unmarshal(msg.Payload, dst); err != nil {
		return fmt.Errorf("decode %q payload: %w", msg.Type, err)
	}
	return nil
}

// truncate shortens s to at most n runes for use in error messages.
func truncate(s string, n int) string {
	runes := []rune(s)
	if len(runes) <= n {
		return s
	}
	return string(runes[:n]) + "…"
}
