package biocomp

// Factory creates a new uninitialized component instance.
type Factory func() BioComponent

// registry maps component names to factories.
var registry = map[string]Factory{}

// Register adds a factory. Called from init() in each component package.
func Register(name string, factory Factory) {
	registry[name] = factory
}

// Get returns a factory by name, or nil if unknown.
func Get(name string) Factory {
	return registry[name]
}

// Names returns all registered component names (for testing).
func Names() []string {
	names := make([]string, 0, len(registry))
	for n := range registry {
		names = append(names, n)
	}
	return names
}
