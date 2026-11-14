package chat

import "testing"

func TestSmokeRendersFrame(t *testing.T) {
	if _, err := Smoke("test"); err != nil {
		t.Fatalf("Smoke: %v", err)
	}
}

func TestSmokeRejectsFrameWithoutLandmark(t *testing.T) {
	if err := checkSmokeFrame("some other non-empty frame"); err == nil {
		t.Fatal("a frame without the header landmark must fail the smoke check")
	}
}
