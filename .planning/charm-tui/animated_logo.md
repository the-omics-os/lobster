# Animated Logo — Lobster AI Title (CLI ASCII Translation)

Research into the web hero animation for translation into a terminal UI with Charm (Bubble Tea / Lip Gloss).

---

## Web Source: What the Hero Looks Like

### Text

```
Lobster AI
```

Displayed as an `<h1>` at `text-5xl md:text-7xl` (48px → 72px on desktop).

### Font

- **Primary:** JetBrains Mono
- **Fallbacks:** Fira Code → Consolas → Monaco → monospace
- **Weight:** `font-bold` (700)
- **Tracking:** `tracking-tight` (letter-spacing: -0.025em)
- **Style:** monospace only — no serif or sans-serif

### Color Palette (dark theme)

| Role | Hex | HSL |
|------|-----|-----|
| Background | `#050505` | `0 0% 2%` |
| Foreground (text) | `#f5f5f5` | `0 0% 96%` |
| Accent / near-white | `#e5e5e5` | `0 0% 90%` |
| Muted | `#888888` | `0 0% 53%` |
| Subtle | `#555555` | `0 0% 33%` |

**There is no teal or color accent in the title itself.** It renders in near-white `#f5f5f5` on near-black `#050505`. The monochromatic palette is intentional.

---

## Web Animation: TextScramble

The title uses a custom `TextScramble` React component. There are **two animation phases**:

### Phase 1 — Initial Scramble (on mount)

- Character set: `"ATCG"` (DNA nucleotides only)
- Duration: `1.2s`
- Speed (interval): `0.05s` per frame = **50ms tick rate**
- Behavior: left-to-right reveal. At each tick, characters to the left of `progress * textLength` are locked to their final value; characters to the right display a random `A/T/C/G`.
- Total frames: `duration / speed = 1.2 / 0.05 = 24 frames`
- Color during scramble: each nucleotide letter gets a unique color:
  - `A` → `#22c55e` (green — Adenine)
  - `T` → `#ef4444` (red — Thymine)
  - `C` → `#3b82f6` (blue — Cytosine)
  - `G` → `#eab308` (yellow — Guanine)
- Once a character is resolved to its final value, color returns to `inherit` (foreground white)
- Transition: `color 0.1s ease` per character

### Phase 2 — Sporadic Idle Scramble (repeating forever)

- Activates after `duration * 1000 + 1000ms = 2.2s` from mount
- Timer: every `sporadicInterval + rand * sporadicInterval` = between `2000ms` and `4000ms`
- Chance to fire: `sporadicChance = 0.7` (70% of ticks actually trigger)
- On trigger: picks **one random non-space character** and cycles it through 8 frames of random nucleotides at **50ms per frame** (400ms total), then snaps back
- Same nucleotide coloring as Phase 1 during the 8 frames

### Entry Fade-In (Framer Motion wrapper)

The entire hero section enters with:
- `initial: { opacity: 0, y: 20 }`
- `animate: { opacity: 1, y: 0 }`
- `transition: { duration: 0.6 }`

---

## ASCII Art Translation

### Figlet / Big Text Options

For a large CLI logo, use `figlet` with a monospace-friendly font. Recommended fonts (available in standard figlet):

| Font | Character | Style |
|------|-----------|-------|
| `Big` | Wide, clean | Close to bold block |
| `Standard` | Readable, moderate size | Default |
| `Slant` | Italic feel | Less accurate |
| `Small` | Compact | Fits narrow terminals |
| `ANSI Shadow` | Shadowed block letters | Most dramatic |

For "Lobster AI", `Big` or `ANSI Shadow` most closely matches the bold, wide web title.

Example with `Big` font:
```
 _          _         _                  _    ___
| |    ___ | |__  ___| |_ ___ _ __      / \  |_ _|
| |   / _ \| '_ \/ __| __/ _ \ '__|    / _ \  | |
| |__| (_) | |_) \__ \ ||  __/ |      / ___ \ | |
|_____\___/|_.__/|___/\__\___|_|     /_/   \_\___|
```

### Color Mapping (Lip Gloss)

| Web token | Hex | Lip Gloss usage |
|-----------|-----|-----------------|
| Title text | `#f5f5f5` | `lipgloss.Color("#f5f5f5")` |
| Background | `#050505` | `lipgloss.Color("#050505")` |
| Accent | `#e5e5e5` | `lipgloss.Color("#e5e5e5")` |
| Muted | `#888888` | `lipgloss.Color("#888888")` |
| Nucleotide A | `#22c55e` | scramble frame color |
| Nucleotide T | `#ef4444` | scramble frame color |
| Nucleotide C | `#3b82f6` | scramble frame color |
| Nucleotide G | `#eab308` | scramble frame color |

---

## Bubble Tea Implementation Blueprint

### State Model

```go
type logoModel struct {
    text        string          // "Lobster AI"
    charStates  []charState     // per-character state
    phase       animPhase       // initialScramble | idle | sporadicFiring
    tick        int             // current frame within phase
    totalFrames int             // 24 for initial scramble
    sporadicIdx int             // which char is being scrambled (-1 = none)
    sporadicTick int            // 0..7
}

type charState struct {
    display     rune
    color       string  // "" = default foreground, else hex
    locked      bool    // final character reached
}

type animPhase int
const (
    phaseInitialScramble animPhase = iota
    phaseIdle
    phaseSporadicFiring
)
```

### Tick Timing

| Phase | Interval |
|-------|----------|
| Initial scramble | `50ms` (matches web `speed * 1000`) |
| Idle wait | `2000ms–4000ms` jitter (matches web `sporadicInterval + rand*sporadicInterval`) |
| Sporadic frame | `50ms` per frame, 8 frames = 400ms total |
| Entry fade | Optional: 3-step opacity ramp at 200ms each = 600ms total |

### Nucleotide Character Set

```go
var nucleotides = []rune{'A', 'T', 'C', 'G'}

var nucleotideColors = map[rune]string{
    'A': "#22c55e",
    'T': "#ef4444",
    'C': "#3b82f6",
    'G': "#eab308",
}
```

### Per-Character Rendering

For each ASCII cell in the figlet output, you can apply per-character coloring by iterating over the string and wrapping each rune in a Lip Gloss style. During scramble frames, replace the rune with a random nucleotide and apply its color; when locked, render with the default foreground style.

### Entry Animation (fade-in)

The web does `opacity: 0 → 1` over 600ms. In a TUI, simulate with:
- Frame 0 (0–200ms): render with `lipgloss.Color("#555555")` (subtle gray)
- Frame 1 (200–400ms): render with `lipgloss.Color("#888888")` (muted)
- Frame 2 (400–600ms): render with `lipgloss.Color("#f5f5f5")` (full foreground)
- Then transition to Phase 1 (initial scramble)

---

## Tagline (below title)

Web source:
```
The self-evolving agentic framework for bioinformatics
```
Style: `text-xl`, normal weight, centered, `text-foreground` (`#f5f5f5`)

Sub-tagline:
```
on-prem • python native • open-source
```
Style: `text-sm`, `font-mono`, `text-muted-foreground` (`#888888`)

For CLI: render tagline in `#f5f5f5`, sub-tagline in `#888888` with `|` or `•` separators. No animation on these.

---

## Summary: Key Numbers for CLI Port

| Parameter | Web value | CLI equivalent |
|-----------|-----------|----------------|
| Initial scramble duration | 1.2s | 24 ticks × 50ms |
| Scramble tick rate | 50ms | `time.NewTicker(50 * time.Millisecond)` |
| Sporadic interval | 2000–4000ms | `2000 + rand.Intn(2000)` ms |
| Sporadic fire chance | 70% | `rand.Float64() < 0.7` |
| Sporadic scramble frames | 8 | 8 ticks × 50ms = 400ms |
| Post-mount sporadic delay | 2200ms | wait `1.2s + 1.0s = 2200ms` |
| Entry fade steps | 3 | 3 × 200ms = 600ms |
| Character set | ATCG | `[]rune{'A','T','C','G'}` |
| Direction | left-to-right reveal | `progress = tick/totalFrames; lock if i < progress*len` |
