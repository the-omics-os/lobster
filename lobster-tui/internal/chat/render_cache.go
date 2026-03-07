package chat

// renderCache stores a width-keyed rendered string for a finalized message.
// Streaming messages must never be cached.
type renderCache struct {
	rendered string
	width    int
	height   int
	valid    bool
}

// get returns the cached render if width matches. Returns ("", 0, false) on miss.
func (c *renderCache) get(width int) (string, int, bool) {
	if c.valid && c.width == width {
		return c.rendered, c.height, true
	}
	return "", 0, false
}

// set stores a rendered string for the given width.
func (c *renderCache) set(rendered string, width, height int) {
	c.rendered = rendered
	c.width = width
	c.height = height
	c.valid = true
}

// clear invalidates the cache.
func (c *renderCache) clear() {
	c.rendered = ""
	c.width = 0
	c.height = 0
	c.valid = false
}
