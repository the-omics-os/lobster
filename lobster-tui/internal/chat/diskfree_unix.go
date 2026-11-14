//go:build !windows

package chat

import "syscall"

// freeDiskBytes reports the bytes available to unprivileged users at path.
func freeDiskBytes(path string) (uint64, bool) {
	var stat syscall.Statfs_t
	if err := syscall.Statfs(path, &stat); err != nil {
		return 0, false
	}
	return stat.Bavail * uint64(stat.Bsize), true
}
