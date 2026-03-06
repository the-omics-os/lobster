package chat

import "testing"

func TestShouldEnableMouseCaptureDefaults(t *testing.T) {
	t.Setenv("LOBSTER_TUI_MOUSE_CAPTURE", "")

	if shouldEnableMouseCapture(true) {
		t.Fatalf("expected mouse capture disabled by default in inline mode")
	}
	if !shouldEnableMouseCapture(false) {
		t.Fatalf("expected mouse capture enabled by default in fullscreen mode")
	}
}

func TestShouldEnableMouseCaptureEnvOverrideTrue(t *testing.T) {
	t.Setenv("LOBSTER_TUI_MOUSE_CAPTURE", "true")

	if !shouldEnableMouseCapture(true) {
		t.Fatalf("expected env override to force mouse capture on in inline mode")
	}
	if !shouldEnableMouseCapture(false) {
		t.Fatalf("expected env override to keep mouse capture on in fullscreen mode")
	}
}

func TestShouldEnableMouseCaptureEnvOverrideFalse(t *testing.T) {
	t.Setenv("LOBSTER_TUI_MOUSE_CAPTURE", "off")

	if shouldEnableMouseCapture(true) {
		t.Fatalf("expected env override to force mouse capture off in inline mode")
	}
	if shouldEnableMouseCapture(false) {
		t.Fatalf("expected env override to force mouse capture off in fullscreen mode")
	}
}

func TestShouldShowInlineIntroDefaults(t *testing.T) {
	t.Setenv("LOBSTER_TUI_NO_INTRO", "")

	if !shouldShowInlineIntro(true) {
		t.Fatalf("expected inline intro enabled by default in inline mode")
	}
	if shouldShowInlineIntro(false) {
		t.Fatalf("expected inline intro disabled in fullscreen mode")
	}
}

func TestShouldShowInlineIntroEnvOverrideTrue(t *testing.T) {
	t.Setenv("LOBSTER_TUI_NO_INTRO", "true")

	if shouldShowInlineIntro(true) {
		t.Fatalf("expected env override to disable inline intro")
	}
}
