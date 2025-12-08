# Lobster Media Assets

This directory contains visual media (GIFs, images) used in documentation and README files.

## 📹 GIF Demos

### Core Workflow GIFs

| File | Description | Used In | Content |
|------|-------------|---------|---------|
| `lobster_installation.gif` | Installation demo | README.md Quick Start | Shows `uv tool install lobster-ai` execution |
| `lobster_init.gif` | Configuration wizard | README.md Quick Start | Interactive `lobster init` wizard walkthrough |
| `lobster_chat.gif` | Chat interface demo | README.md Features & Case Studies | Natural language analysis workflow |
| `lobster_gif_dashboard.gif` | Dashboard preview | README.md Dashboard section | Interactive dashboard interface (Alpha) |

## 🎬 Recording Guidelines

When creating new GIFs, follow these standards:

### Technical Specs
- **Max file size:** 5MB (GitHub renders smoothly)
- **Dimensions:** 1280×720 or 1920×1080
- **Frame rate:** 15-20 fps (smooth but compact)
- **Duration:** 10-30 seconds max
- **Colors:** 128-256 colors (optimize file size)

### Content Guidelines
- Show real workflows, not mock data
- Include terminal prompt for context
- Use clean, uncluttered terminal themes
- Demonstrate key features, not edge cases
- End on a clear success state

### Recommended Tools
- **macOS:** [ScreenToGif](https://www.screentogif.com/), [Kap](https://getkap.co/)
- **Linux:** [Peek](https://github.com/phw/peek), [gifcurry](https://github.com/lettier/gifcurry)
- **Windows:** [ScreenToGif](https://www.screentogif.com/)

### Optimization
```bash
# Reduce file size with gifsicle
gifsicle -O3 --colors 128 input.gif -o output.gif

# Or with ImageMagick
convert input.gif -fuzz 10% -layers Optimize output.gif
```

## 📝 Updating GIFs

When features change:
1. Re-record GIF following guidelines above
2. Optimize file size (target <5MB)
3. Replace old file (keep same filename)
4. Update this README if content changes significantly
5. Commit with message: `docs: update [feature] GIF demo`

## 🔗 External Links

For longer video content (>30s), prefer YouTube embeds:
- Create Omics-OS YouTube channel
- Upload tutorial/walkthrough videos
- Embed in README with thumbnail image

## 📊 Current Status

| GIF | File Size | Dimensions | Status |
|-----|-----------|------------|--------|
| lobster_installation.gif | TBD | TBD | ✅ Active |
| lobster_init.gif | TBD | TBD | ✅ Active |
| lobster_chat.gif | TBD | TBD | ✅ Active |
| lobster_gif_dashboard.gif | TBD | TBD | ⚠️ Alpha feature |

---

*Last updated: 2024-12-08*
