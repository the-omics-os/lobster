package protocol

import (
	"bytes"
	"strings"
	"testing"
	"time"
)

func textLine(content string) string {
	return `{"version":1,"type":"text","payload":{"content":"` + content + `"}}`
}

func readAll(t *testing.T, h *Handler) (contents []string, errs []string) {
	t.Helper()
	timeout := time.After(10 * time.Second)
	for {
		select {
		case msg, ok := <-h.Msgs():
			if !ok {
				for {
					select {
					case err := <-h.Errs():
						errs = append(errs, err.Error())
					default:
						return contents, errs
					}
				}
			}
			var p TextPayload
			if err := DecodePayload(msg, &p); err != nil {
				t.Fatalf("decode: %v", err)
			}
			contents = append(contents, p.Content)
		case err := <-h.Errs():
			errs = append(errs, err.Error())
		case <-timeout:
			t.Fatal("the read loop did not end")
		}
	}
}

// A line longer than the reader's 4 MB limit is skipped with an error; the
// messages after it still arrive, so the TUI keeps the Python process.
func TestALineOverTheReaderLimitIsSkippedAndTheLinkKept(t *testing.T) {
	long := textLine(strings.Repeat("x", 5*1024*1024))
	input := textLine("before") + "\n" + long + "\n" + textLine("after") + "\n"
	h := NewHandler(strings.NewReader(input), &bytes.Buffer{})
	h.StartReadLoop()
	defer h.Close()

	contents, errs := readAll(t, h)

	if strings.Join(contents, ",") != "before,after" {
		t.Fatalf("messages = %q, want before,after", contents)
	}
	if len(errs) != 1 || !strings.Contains(errs[0], "skipped") {
		t.Fatalf("errors = %q, want one about the skipped message", errs)
	}
}

// Lines just under the limit, CRLF endings, blank lines and a last line without
// a newline read as before.
func TestLinesWithinTheLimitReadAsBefore(t *testing.T) {
	const limit = 4 * 1024 * 1024
	fill := limit - 1 - len(textLine(""))
	near := textLine(strings.Repeat("y", fill))
	input := textLine("a") + "\r\n\n" + near + "\n" + textLine("b")
	h := NewHandler(strings.NewReader(input), &bytes.Buffer{})
	h.StartReadLoop()
	defer h.Close()

	contents, errs := readAll(t, h)

	if len(errs) != 0 {
		t.Fatalf("errors = %q", errs)
	}
	if len(contents) != 3 || contents[0] != "a" || len(contents[1]) != fill || contents[2] != "b" {
		t.Fatalf("got %d messages", len(contents))
	}
}
