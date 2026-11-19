//go:build linux

package initwizard

import (
	"os/exec"
	"syscall"
)

// detachFromTerminal starts the command in its own session, so nothing it launches can
// reach the wizard's controlling terminal.
func detachFromTerminal(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
}
