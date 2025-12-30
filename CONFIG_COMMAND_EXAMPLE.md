# /config Command - Per-Agent Model Display

## Overview

The `/config` command in interactive chat mode now displays which model each agent is using. This is especially useful when using Gemini or other providers to verify that all agents are configured consistently.

---

## Usage

```bash
# In interactive chat mode
lobster chat

# Type the command
/config
```

---

## Example Output

### Scenario: User initialized with Gemini in production mode

```
⚙️  Current Configuration
┌──────────┬────────────┬───────────────────┐
│ Setting  │ Value      │ Source            │
├──────────┼────────────┼───────────────────┤
│ Provider │ gemini     │ workspace config  │
│ Profile  │ production │ workspace config  │
└──────────┴────────────┴───────────────────┘

📁 Configuration Files
┌────────────────────┬────────────┬─────────────────────────────────────┐
│ Location           │ Status     │ Path                                │
├────────────────────┼────────────┼─────────────────────────────────────┤
│ Workspace Config   │ ✓ Exists   │ .lobster_workspace/provider_config  │
│ Global Config      │ ✗ Not found│ ~/.config/lobster/providers.json    │
└────────────────────┴────────────┴─────────────────────────────────────┘

🤖 Agent Models
┌────────────────────────────────────┬────────────────────┬────────────────────┐
│ Agent                              │ Model              │ Source             │
├────────────────────────────────────┼────────────────────┼────────────────────┤
│ Data Expert                        │ gemini-3-pro       │ workspace config   │
│ Research Agent                     │ gemini-3-pro       │ workspace config   │
│ Transcriptomics Expert             │ gemini-3-pro       │ workspace config   │
│ Annotation Expert                  │ gemini-3-pro       │ workspace config   │
│ DE Analysis Expert                 │ gemini-3-pro       │ workspace config   │
│ Visualization Expert               │ gemini-3-pro       │ workspace config   │
│ Protein Structure Visualization    │ gemini-3-pro       │ workspace config   │
└────────────────────────────────────┴────────────────────┴────────────────────┘

💡 Usage:
  • /config provider - List available providers
  • /config provider <name> - Switch provider (runtime only)
  • /config provider <name> --save - Switch and persist to workspace
  • /config model <name> - Set model for current provider
```

---

## Key Features

### 1. **Per-Agent Model Display**

Shows **exactly which model** each agent will use when invoked. This is critical for:
- Verifying consistent configuration (all agents using Gemini)
- Debugging mixed-provider setups
- Understanding cost implications (different models = different costs)

### 2. **Source Attribution**

Each agent's model displays its **configuration source**:

| Source | Meaning |
|--------|---------|
| `workspace config` | Set in `.lobster_workspace/provider_config.json` |
| `global config` | Set in `~/.config/lobster/providers.json` |
| `profile config` | From LOBSTER_PROFILE (development/production/ultra/godmode) |
| `runtime flag --model` | Temporary override via `--model` CLI flag |
| `provider default` | No explicit config, using provider's default |

### 3. **License Tier Filtering**

Only shows agents available for your current license tier:
- **Free tier**: Shows 7 core agents
- **Premium tier**: Shows all agents including metadata_assistant, proteomics_expert
- **Enterprise tier**: Shows all agents + custom agents

### 4. **Real-Time Configuration**

The display shows **actual runtime configuration**, not hardcoded profiles:
- Reflects workspace-specific settings
- Shows runtime overrides (if any)
- Matches what agents will actually use

---

## Common Scenarios

### Scenario 1: All Agents Using Same Model (Gemini)

**Setup:**
```bash
lobster init  # Selected Gemini provider
```

**Result:**
```
🤖 Agent Models
All agents → gemini-3-pro-preview (workspace config)
```

✅ **Expected behavior**: All agents use the same Gemini model because:
- Gemini provider is set globally
- Profile (production) doesn't override per-agent models
- No workspace-level per-agent overrides

---

### Scenario 2: Mixed Models (Development Profile)

**Setup:**
```bash
export LOBSTER_PROFILE=development
lobster chat
```

**Result:**
```
🤖 Agent Models
Supervisor              → claude-4-5-haiku      (profile config)
Research Agent          → claude-4-5-haiku      (profile config)
Transcriptomics Expert  → claude-4-5-haiku      (profile config)
Custom Feature Agent    → claude-4-5-sonnet     (profile config)
```

ℹ️ **Note**: Development profile uses lighter models to reduce cost during testing.

---

### Scenario 3: Per-Agent Override

**Setup:**
```yaml
# .lobster_workspace/provider_config.json
{
  "global_provider": "anthropic",
  "anthropic_model": "claude-sonnet-4-20250514",
  "per_agent_models": {
    "custom_feature_agent": "claude-opus-4-20250514"
  }
}
```

**Result:**
```
🤖 Agent Models
Data Expert            → claude-sonnet-4-20250514  (workspace config)
Research Agent         → claude-sonnet-4-20250514  (workspace config)
...
Custom Feature Agent   → claude-opus-4-20250514    (workspace config - per-agent)
```

✅ **Use case**: Use cheaper Sonnet for most agents, reserve Opus for code generation.

---

## Verification Steps

### 1. Check Your Configuration

```bash
lobster chat
/config
```

**What to verify:**
- ✅ Provider is correct (gemini, anthropic, bedrock, ollama)
- ✅ All agents show expected model
- ✅ Source attribution makes sense

### 2. Verify Model Consistency

**For single-provider setups (Gemini):**
- All agents should show **same model**
- Source should be consistent (all "workspace config" or all "profile config")

**For mixed setups:**
- Intentional overrides should be visible
- No unexpected variations

### 3. Cost Optimization Check

**Model pricing reference:**
```
Gemini 3 Pro:    $2.00 input / $12.00 output (per million tokens)
Gemini 3 Flash:  $0.50 input / $3.00 output (per million tokens)
Claude Haiku:    Lowest cost (development)
Claude Sonnet:   Balanced (production)
Claude Opus:     Highest cost (godmode)
```

Use `/config` to verify you're using cost-appropriate models for each agent.

---

## Troubleshooting

### Issue: Agents showing different models unexpectedly

**Possible causes:**
1. Mixed workspace + profile config
2. Environment variable overrides (`LOBSTER_<AGENT>_MODEL`)
3. Stale workspace config

**Solution:**
```bash
# Check actual config files
cat .lobster_workspace/provider_config.json
cat ~/.config/lobster/providers.json

# Reset to clean state
rm .lobster_workspace/provider_config.json
lobster init  # Reconfigure
```

---

### Issue: Model shows "runtime flag --model"

**Meaning:** Model was overridden via CLI flag (temporary).

**Check:**
```bash
# See if you started with --model flag
lobster chat --model custom-model-name  # This overrides workspace config
```

**Solution:**
- Remove `--model` flag to use workspace config
- Or use `/config model <name> --save` to persist the change

---

### Issue: Premium agents not showing

**Cause:** License tier restriction.

**Solution:**
```bash
# Check your tier
lobster status

# Upgrade to premium
lobster activate <premium-key>
```

---

## Implementation Details

### Code Location

**File:** `lobster/cli.py`
**Lines:** 7460-7503

**Key components:**
1. `ConfigResolver` - Resolves provider, profile, per-agent models
2. `Settings.get_agent_llm_params()` - Gets agent temperature/thinking config
3. `AGENT_REGISTRY` - List of all available agents
4. `is_agent_available()` - License tier filtering

### Model Resolution Priority

1. **Runtime flag** (`--model`)
2. **Workspace per-agent config** (`per_agent_models`)
3. **Workspace global model** (`<provider>_model`)
4. **Profile config** (from `agent_config.py`)
5. **Provider default**

---

## Related Commands

| Command | Purpose |
|---------|---------|
| `/config` | Show current configuration (this document) |
| `/config provider` | List/switch providers |
| `/config model` | List/switch models for current provider |
| `lobster config show-config` | CLI version with more details |
| `lobster status` | Show license tier |

---

## Summary

The `/config` command provides a **quick, at-a-glance view** of your configuration:
- ✅ Verify all agents use correct provider (e.g., Gemini)
- ✅ Check model consistency across agents
- ✅ Understand configuration sources
- ✅ Optimize costs by reviewing model assignments

**For Gemini users:** After running `lobster init` with Gemini, `/config` should show all agents using the same Gemini model with consistent source attribution.
