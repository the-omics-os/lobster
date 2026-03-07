package biocomp

import (
	"strings"
	"testing"

	"charm.land/bubbles/v2/key"
)

func TestOverlaySizeSmall(t *testing.T) {
	t.Parallel()
	w, h := OverlaySize(100, 40, "small")
	if w != 60 {
		t.Fatalf("expected width 60, got %d", w)
	}
	if h != 20 {
		t.Fatalf("expected height 20, got %d", h)
	}
}

func TestOverlaySizeMedium(t *testing.T) {
	t.Parallel()
	w, h := OverlaySize(100, 40, "medium")
	if w != 70 {
		t.Fatalf("expected width 70, got %d", w)
	}
	if h != 24 {
		t.Fatalf("expected height 24, got %d", h)
	}
}

func TestOverlaySizeLarge(t *testing.T) {
	t.Parallel()
	w, h := OverlaySize(100, 40, "large")
	if w != 80 {
		t.Fatalf("expected width 80, got %d", w)
	}
	if h != 28 {
		t.Fatalf("expected height 28, got %d", h)
	}
}

func TestOverlaySizeFullscreenOnSmallTerminal(t *testing.T) {
	t.Parallel()
	// Width too small.
	w, h := OverlaySize(60, 30, "medium")
	if w != 60 || h != 30 {
		t.Fatalf("expected fullscreen 60x30, got %dx%d", w, h)
	}

	// Height too small.
	w, h = OverlaySize(100, 15, "medium")
	if w != 100 || h != 15 {
		t.Fatalf("expected fullscreen 100x15, got %dx%d", w, h)
	}
}

func TestOverlaySizeMinimumClamp(t *testing.T) {
	t.Parallel()
	// Very large terminal but small percentage should still be >= 40x10.
	w, h := OverlaySize(80, 22, "small")
	if w < 40 {
		t.Fatalf("expected width >= 40, got %d", w)
	}
	if h < 10 {
		t.Fatalf("expected height >= 10, got %d", h)
	}
}

func TestRenderFrameNonEmpty(t *testing.T) {
	t.Parallel()
	result := RenderFrame("Test Title", "Hello content", "esc quit", 60, 20)
	if result == "" {
		t.Fatal("expected non-empty frame output")
	}
	if !strings.Contains(result, "Test Title") {
		t.Fatal("expected frame to contain title")
	}
}

func TestRenderHelpBarEmpty(t *testing.T) {
	t.Parallel()
	result := RenderHelpBar(nil, 80)
	if result != "" {
		t.Fatalf("expected empty string for nil bindings, got %q", result)
	}
}

func TestRenderHelpBarWithBindings(t *testing.T) {
	t.Parallel()
	bindings := []key.Binding{
		key.NewBinding(key.WithKeys("enter"), key.WithHelp("enter", "confirm")),
		key.NewBinding(key.WithKeys("esc"), key.WithHelp("esc", "cancel")),
	}
	result := RenderHelpBar(bindings, 80)
	if result == "" {
		t.Fatal("expected non-empty help bar")
	}
}
