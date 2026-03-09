package theme

// AutoDetect returns the default theme ID. With ANSI adaptive colors, there
// is only one theme that works on both light and dark terminals.
func AutoDetect() string {
	return "lobster-default"
}
