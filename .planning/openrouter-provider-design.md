# OpenRouter Provider Integration — Design Document

**Date:** 2026-03-03
**Status:** Approved
**Author:** ultrathink
**Motivation:** Model flexibility — access 600+ models via one API key without managing multiple provider credentials.

---

## Summary

Add OpenRouter as a first-class provider in Lobster AI's provider registry. OpenRouter speaks the OpenAI API protocol, so no new LangChain packages are needed — `ChatOpenAI` with `base_url="https://openrouter.ai/api/v1"` is the integration point.

---

## Architecture

### Approach: Standalone Provider + Lazy Live Catalog

Full independent `ILLMProvider` implementation with:
- Lazy live catalog fetched from `https://openrouter.ai/api/v1/models` on first `list_models()` call
- Class-level in-process cache (fetched once per process, never re-fetched)
- Curated fallback catalog of ~20 key models used when network is unavailable
- Hardcoded Lobster branding headers for OpenRouter leaderboard visibility

### Why not a subclass of OpenAIProvider

Subclassing would couple OpenRouter behavior to OpenAI internals. OpenRouter is semantically distinct: different API key env var, different model namespace (`provider/model-name`), different catalog strategy. The `ILLMProvider` interface is only 7 methods — standalone is clean.

---

## File Changes

### New File

| File | Purpose |
|------|---------|
| `lobster/config/providers/openrouter_provider.py` | Full `OpenRouterProvider(ILLMProvider)` implementation |

### Modified Files

| File | Change |
|------|--------|
| `lobster/config/constants.py` | Add `"openrouter"` to `VALID_PROVIDERS` + `PROVIDER_DISPLAY_NAMES` |
| `lobster/config/llm_factory.py` | Add `OPENROUTER = "openrouter"` to `LLMProvider` enum |
| `lobster/config/providers/registry.py` | Add `openrouter_provider` module to `_provider_specs` |
| `lobster/config/providers/__init__.py` | Export `OpenRouterProvider` |
| `lobster/config/provider_setup.py` | Add `create_openrouter_config(api_key)` |
| `lobster/config/global_config.py` | Add `openrouter_default_model: Optional[str]` field |
| `lobster/config/workspace_config.py` | Add `openrouter_model: Optional[str]` field |
| `lobster/cli.py` | Add option 7 to interactive init menu + OpenRouter setup flow |

**No new dependencies** — `langchain-openai` already in the dependency tree.

---

## `OpenRouterProvider` Design

```python
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://lobsterbio.com",
    "X-Title": "Lobster AI",
}
ENV_VAR = "OPENROUTER_API_KEY"
DEFAULT_MODEL = "anthropic/claude-sonnet-4-5"
```

### Curated Fallback Catalog (~20 models)

Representative models from major families, ordered by capability:

```
anthropic/claude-sonnet-4-5       (default)
anthropic/claude-3-5-haiku
openai/gpt-4o
openai/gpt-4o-mini
openai/o1
openai/o3-mini
meta-llama/llama-3.3-70b-instruct
meta-llama/llama-3.1-8b-instruct
deepseek/deepseek-r1
deepseek/deepseek-chat-v3-0324
google/gemini-2.0-flash-001
google/gemini-2.5-pro-preview
mistralai/mistral-large
mistralai/mistral-nemo
qwen/qwen-2.5-72b-instruct
x-ai/grok-2-1212
microsoft/phi-4
nvidia/llama-3.1-nemotron-70b-instruct
cohere/command-r-plus
amazon/nova-pro-v1
```

### Live Catalog Fetch

```python
_models_cache: ClassVar[Optional[List[ModelInfo]]] = None

def list_models(self) -> List[ModelInfo]:
    if OpenRouterProvider._models_cache is not None:
        return OpenRouterProvider._models_cache
    try:
        import httpx
        api_key = os.environ.get(self.ENV_VAR, "")
        headers = {"Authorization": f"Bearer {api_key}"}
        response = httpx.get(
            f"{self.OPENROUTER_BASE_URL}/models",
            headers=headers,
            timeout=5.0
        )
        response.raise_for_status()
        data = response.json().get("data", [])
        models = [self._parse_model(m) for m in data]
        OpenRouterProvider._models_cache = models
        logger.debug(f"Fetched {len(models)} models from OpenRouter API")
        return models
    except Exception as e:
        logger.debug(f"OpenRouter model fetch failed ({e}), using fallback catalog")
        return list(self.FALLBACK_MODELS)
```

### Model Validation

```python
def validate_model(self, model_id: str) -> bool:
    if not model_id:
        return False
    if OpenRouterProvider._models_cache:
        return model_id in [m.name for m in OpenRouterProvider._models_cache]
    # Cache not populated — accept any non-empty string (OpenRouter validates at inference)
    logger.debug(f"OpenRouter model '{model_id}' not validated (cache not loaded)")
    return True
```

### `create_chat_model`

```python
def create_chat_model(self, model_id, temperature=1.0, max_tokens=4096, **kwargs):
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError("langchain-openai not installed. Install with: uv pip install langchain-openai")

    api_key = kwargs.pop("api_key", None) or os.environ.get(self.ENV_VAR)
    if not api_key:
        raise ValueError(
            f"{self.ENV_VAR} not found. Set it with: export {self.ENV_VAR}=sk-or-..."
        )

    return ChatOpenAI(
        model=model_id,
        api_key=api_key,
        base_url=self.OPENROUTER_BASE_URL,
        default_headers=self.OPENROUTER_HEADERS,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
```

---

## Config Schema

### `GlobalProviderConfig` addition

```json
{
  "default_provider": "openrouter",
  "openrouter_default_model": "anthropic/claude-sonnet-4-5"
}
```

The field name `openrouter_default_model` follows the existing `{provider}_default_model` convention used by `base_config.get_model_for_provider()` — zero changes to `base_config.py` needed.

### `WorkspaceProviderConfig` addition

```json
{
  "global_provider": "openrouter",
  "openrouter_model": "openai/gpt-4o"
}
```

---

## CLI Init Flow

Menu option added to `lobster init` interactive flow:

```
  [cyan]7[/cyan] - OpenRouter - 600+ models via one API key (Claude, GPT-4o, Llama, DeepSeek, ...)
```

Setup flow:
1. Print: "Get your API key from: https://openrouter.ai/keys"
2. Prompt for `OPENROUTER_API_KEY` (password=True)
3. Prompt for default model (pre-filled: `anthropic/claude-sonnet-4-5`, accepts any string)
4. Save to `credentials.env` (mode 0o600) + `provider_config.json`

Non-interactive mode flag: `--openrouter-key=sk-or-...`

---

## Error Handling

| Scenario | Behavior |
|----------|----------|
| `OPENROUTER_API_KEY` not set | `is_configured()` → `False`; `create_chat_model()` raises `ValueError` |
| Live catalog fetch fails | Debug log, silently returns `FALLBACK_MODELS` |
| Invalid model at inference | OpenRouter API returns error; propagates as LangChain exception |
| `validate_model()` before cache populated | Returns `True` for any non-empty string |
| `httpx` not installed | `ImportError` caught in `list_models()`, falls back to curated list |

---

## Testing

New file: `tests/unit/config/test_openrouter_provider.py`

Tests:
- `test_is_configured_with_key` / `test_is_configured_without_key`
- `test_is_available_equals_is_configured`
- `test_create_chat_model_success` (mock `ChatOpenAI`, verify `base_url` and headers set)
- `test_create_chat_model_no_api_key` (raises `ValueError`)
- `test_list_models_live_catalog` (mock `httpx.get`, verify parsing + caching)
- `test_list_models_fallback_on_network_error` (mock `httpx.get` raises, verify fallback)
- `test_list_models_uses_cache` (call twice, verify only one fetch)
- `test_validate_model_with_cache` (True for known, False for empty)
- `test_validate_model_without_cache` (True for any non-empty)
- `test_get_default_model_is_in_fallback`

Extended existing test files:
- `tests/unit/config/test_constants.py` — add `"openrouter"` to `VALID_PROVIDERS` assertions
- `tests/unit/config/test_provider_setup.py` — add `test_create_openrouter_config_valid/invalid`
- `tests/unit/config/test_config_resolver.py` — include `"openrouter"` in provider resolution tests

---

## Provider Display Info

```python
# In constants.py
PROVIDER_DISPLAY_NAMES = {
    ...
    "openrouter": "OpenRouter (600+ models)",
}
```

---

## Implementation Order (Dependencies)

1. `constants.py` — gate for all validation
2. `openrouter_provider.py` — core implementation
3. `registry.py` — wire into ProviderRegistry
4. `llm_factory.py` — enum update (cosmetic)
5. `provider_setup.py` — config creation helper
6. `global_config.py` + `workspace_config.py` — config schema fields
7. `providers/__init__.py` — export
8. `cli.py` — user-facing init flow
9. Tests

---

## Non-Goals

- TTL-based cache invalidation (YAGNI — processes are short-lived)
- Custom `HTTP-Referer`/`X-Title` per user (hardcode Lobster branding)
- Reasoning model special-casing (OpenRouter handles this at the routing layer)
- OpenRouter provider-specific features (cost limits, fallback routing, transforms) — use kwargs passthrough for power users
