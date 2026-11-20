package protocol

import (
	"bytes"
	"strings"
	"testing"
	"time"
)

// A line longer than the reader's 4 MB buffer ends the read loop: the messages
// after it never arrive and the TUI loses the Python process. Python keeps each
// message well below it (a list is sent 1,000 lines at a time).
func TestALineOverTheReaderBufferEndsTheLink(t *testing.T) {
	long := `{"version":1,"type":"text","payload":{"content":"` +
		strings.Repeat("x", 5*1024*1024) + `"}}` + "\n"
	after := `{"version":1,"type":"text","payload":{"content":"after"}}` + "\n"
	h := NewHandler(strings.NewReader(long+after), &bytes.Buffer{})
	h.StartReadLoop()
	defer h.Close()

	select {
	case err := <-h.Errs():
		if !strings.Contains(err.Error(), "token too long") {
			t.Fatalf("read error = %v, want token too long", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("no read error for the long line")
	}
	if msg, ok := <-h.Msgs(); ok {
		t.Fatalf("a message after the long line arrived: %s", msg.Payload)
	}
}
