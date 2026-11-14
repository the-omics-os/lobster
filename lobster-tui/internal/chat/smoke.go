package chat

import (
	"fmt"
	"io"
	"strings"

	"github.com/the-omics-os/lobster-tui/internal/protocol"
	"github.com/the-omics-os/lobster-tui/internal/theme"
)

// Smoke builds the chat model against an inert protocol handler and renders
// one frame to a string, without a TTY. It backs `lobster-tui --smoke`.
func Smoke(version string) (string, error) {
	handler := protocol.NewHandler(strings.NewReader(""), io.Discard)
	model := NewModel(handler, theme.BuildCleanStyles(theme.LobsterDark.Colors), 80, 24, false, false, version)
	frame := model.View().Content
	if err := checkSmokeFrame(frame); err != nil {
		return "", err
	}
	return frame, nil
}

// smokeLandmark is the header the first frame always shows.
const smokeLandmark = "Lobster AI"

func checkSmokeFrame(frame string) error {
	if !strings.Contains(frame, smokeLandmark) {
		return fmt.Errorf("frame lacks %q", smokeLandmark)
	}
	return nil
}
