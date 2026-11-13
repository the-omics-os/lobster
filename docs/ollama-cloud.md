# Ollama Cloud

Use hosted Ollama models without installing a local Ollama daemon. Create an API key at https://ollama.com/settings/keys. Cloud inference consumes your Ollama usage credits.

Choose **Ollama Cloud** in `lobster init`, or configure non-interactively:

```bash
export OLLAMA_API_KEY="your-key"
lobster init --non-interactive --ollama-cloud-key "$OLLAMA_API_KEY" --model glm-5.2
lobster query --provider ollama-cloud --model glm-5.2 --session-id cloud-demo --json "Say hello"
```

The provider discovers hosted completion/tool models automatically and caches the catalog for the process lifetime. Restart the CLI to refresh it. If discovery fails, GLM-5.2 and GPT-OSS 120B/20B remain available as fallback choices. Explicit model IDs are also accepted. Missing prices are unknown, not free; known prices estimate standard token rates and do not account for cached-input or off-peak discounts.

The stable default is `glm-5.2`. Direct cloud IDs differ from local launcher names such as `glm-5.2:cloud`. Requests use https://ollama.com/v1 with OLLAMA_API_KEY; local Ollama settings do not configure this provider.
