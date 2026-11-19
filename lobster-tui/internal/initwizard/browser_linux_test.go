//go:build linux

package initwizard

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

// fakeXdgOpen puts an xdg-open first on PATH that records that it ran, its process id and
// its process table entry (field 6 after the name is the session id).
func fakeXdgOpen(t *testing.T) (marker string) {
	t.Helper()
	dir := t.TempDir()
	marker = filepath.Join(dir, "ran")
	script := "#!/bin/sh\necho $$ > " + marker + "\ncat /proc/$$/stat > " + marker + ".stat\n"
	if err := os.WriteFile(filepath.Join(dir, "xdg-open"), []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", dir+string(os.PathListSeparator)+os.Getenv("PATH"))
	return marker
}

// With no display, xdg-open falls back to a console browser (w3m, lynx) that can open the
// controlling terminal and draw over the wizard. Never start it; the URL is on screen.
func TestOpenBrowserOnLinuxWithoutADisplayStartsNothing(t *testing.T) {
	marker := fakeXdgOpen(t)
	t.Setenv("DISPLAY", "")
	t.Setenv("WAYLAND_DISPLAY", "")

	if err := openBrowser("https://www.omics-os.com/early-access"); err == nil {
		t.Fatal("no display: want an error so the caller prints the address")
	}
	time.Sleep(300 * time.Millisecond)
	if _, err := os.Stat(marker); err == nil {
		t.Fatal("xdg-open was started without a display")
	}
}

// With a display the browser is started in its own session, so nothing it launches can
// reach the wizard's controlling terminal.
func TestOpenBrowserOnLinuxStartsInItsOwnSession(t *testing.T) {
	for _, env := range []string{"DISPLAY", "WAYLAND_DISPLAY"} {
		t.Run(env, func(t *testing.T) {
			marker := fakeXdgOpen(t)
			t.Setenv("DISPLAY", "")
			t.Setenv("WAYLAND_DISPLAY", "")
			t.Setenv(env, ":0")

			if err := openBrowser("https://www.omics-os.com/early-access"); err != nil {
				t.Fatalf("openBrowser: %v", err)
			}
			var stat []byte
			for i := 0; i < 50 && len(stat) == 0; i++ {
				stat, _ = os.ReadFile(marker + ".stat")
				time.Sleep(100 * time.Millisecond)
			}
			if len(stat) == 0 {
				t.Fatal("xdg-open did not run")
			}
			pid, _ := os.ReadFile(marker)
			// /proc/<pid>/stat: "pid (comm) state ppid pgrp session ..."
			rest := strings.Fields(string(stat)[strings.LastIndex(string(stat), ")")+2:])
			session := rest[3]
			if session != strings.TrimSpace(string(pid)) {
				t.Fatalf("session %s, pid %s: the browser shares our session", session, strings.TrimSpace(string(pid)))
			}
			if own := strconv.Itoa(os.Getpid()); session == own {
				t.Fatalf("session %s is the test's own", session)
			}
		})
	}
}
