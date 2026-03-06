package chat

import (
	"regexp"
	"strings"
	"testing"
	"time"
)

var ansiEscapeRe = regexp.MustCompile(`\x1b\[[0-9;]*m`)

func TestRenderInlineIntroActiveShowsTaglines(t *testing.T) {
	m := newTestModel()
	m.inline = true
	m.width = 120
	m.welcomeActive = true
	m.welcomeStart = time.Now()
	m.welcomeDNA = "ATGCATGCATGCATGCATGCATGC"
	m.welcomeSporadicCell = -1

	intro := stripANSI(renderInlineIntro(m))
	if !strings.Contains(intro, welcomeTagline) {
		t.Fatalf("expected tagline in intro, got:\n%s", intro)
	}
	if !strings.Contains(intro, welcomeSubTagline) {
		t.Fatalf("expected sub-tagline in intro, got:\n%s", intro)
	}
}

func TestRenderInlineIntroInactiveKeepsLogoAndText(t *testing.T) {
	m := newTestModel()
	m.inline = true
	m.width = 120
	m.welcomeActive = false
	m.welcomeStart = time.Now().Add(-10 * time.Second)
	m.welcomeDNA = "ATGCATGCATGCATGCATGCATGC"
	m.welcomeSporadicCell = -1

	intro := stripANSI(renderInlineIntro(m))
	if !strings.Contains(intro, welcomeTagline) {
		t.Fatalf("expected tagline to remain visible after animation, got:\n%s", intro)
	}
	if !strings.Contains(intro, "▗▖    ▗▄▖") {
		t.Fatalf("expected static logo to remain visible after animation, got:\n%s", intro)
	}
}

func TestRenderAnimatedWelcomeTitleInitialScrambleDiffersFromIdle(t *testing.T) {
	lines := welcomeTitleLinesForWidth(120)
	fadeTotal := time.Duration(welcomeFadeSteps) * welcomeFadeStepDuration

	scramble := newTestModel()
	scramble.inline = true
	scramble.width = 120
	scramble.welcomeActive = true
	scramble.welcomeDNA = "ATGCATGCATGCATGCATGCATGC"
	scramble.welcomeSporadicCell = -1
	scramble.welcomeFrame = 8
	scramble.welcomeStart = time.Now().Add(-(fadeTotal + 5*welcomeAnimInterval))

	idle := scramble
	idle.welcomeFrame = 32
	idle.welcomeSporadicCell = -1
	idle.welcomeSporadicTick = 0
	idle.welcomeStart = time.Now().Add(-(fadeTotal + time.Duration(welcomeInitialScrambleFrames)*welcomeAnimInterval + 100*time.Millisecond))

	scrambleTitle := stripANSI(renderAnimatedWelcomeTitle(scramble, lines))
	idleTitle := stripANSI(renderAnimatedWelcomeTitle(idle, lines))
	if scrambleTitle == idleTitle {
		t.Fatalf("expected initial scramble frame to differ from idle title")
	}
}

func TestRenderAnimatedWelcomeTitleSporadicDiffersFromIdle(t *testing.T) {
	lines := welcomeTitleLinesForWidth(120)
	base := newTestModel()
	base.inline = true
	base.width = 120
	base.welcomeActive = true
	base.welcomeDNA = "ATGCATGCATGCATGCATGCATGC"
	base.welcomeFrame = 40
	base.welcomeStart = time.Now().Add(-(welcomeSporadicDelay + 300*time.Millisecond))
	base.welcomeSporadicCell = -1
	base.welcomeSporadicTick = 0

	sporadic := base
	sporadic.welcomeSporadicCell = 3
	sporadic.welcomeSporadicTick = 2

	baseTitle := stripANSI(renderAnimatedWelcomeTitle(base, lines))
	sporadicTitle := stripANSI(renderAnimatedWelcomeTitle(sporadic, lines))
	if baseTitle == sporadicTitle {
		t.Fatalf("expected sporadic scramble frame to differ from idle title")
	}
}

func TestRenderInlineIntroNarrowFallbackUsesPlainTitle(t *testing.T) {
	m := newTestModel()
	m.inline = true
	m.width = 40
	m.welcomeActive = true
	m.welcomeStart = time.Now()
	m.welcomeSporadicCell = -1

	intro := stripANSI(renderInlineIntro(m))
	if !strings.Contains(intro, welcomeTitleText) {
		t.Fatalf("expected plain title in compact intro, got:\n%s", intro)
	}
	if strings.Contains(intro, "▗▖    ▗▄▖") {
		t.Fatalf("did not expect figlet title in compact intro, got:\n%s", intro)
	}
}

func stripANSI(s string) string {
	return ansiEscapeRe.ReplaceAllString(s, "")
}
