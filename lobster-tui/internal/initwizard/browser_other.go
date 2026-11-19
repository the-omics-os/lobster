//go:build !linux

package initwizard

import "os/exec"

// detachFromTerminal is needed only where xdg-open can fall back to a console browser.
func detachFromTerminal(*exec.Cmd) {}
