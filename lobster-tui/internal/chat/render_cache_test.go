package chat

import (
	"testing"
)

func TestRenderCache_HitOnSameWidth(t *testing.T) {
	var c renderCache
	c.set("hello", 80, 1)
	got, h, ok := c.get(80)
	if !ok {
		t.Fatal("expected cache hit at width=80")
	}
	if got != "hello" {
		t.Errorf("got %q, want %q", got, "hello")
	}
	if h != 1 {
		t.Errorf("height = %d, want 1", h)
	}
}

func TestRenderCache_MissOnDifferentWidth(t *testing.T) {
	var c renderCache
	c.set("hello", 80, 1)
	_, _, ok := c.get(60)
	if ok {
		t.Fatal("expected cache miss at width=60 after setting width=80")
	}
}

func TestRenderCache_Clear(t *testing.T) {
	var c renderCache
	c.set("hello", 80, 1)
	c.clear()
	_, _, ok := c.get(80)
	if ok {
		t.Fatal("expected cache miss after clear()")
	}
}
