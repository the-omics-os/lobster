# Test Migration Note

## test_llm_factory.py → test_llm_factory_OLD_DEPRECATED.py

**Date**: 2025-12-15
**Reason**: Provider system refactored in v0.4.0

The old `test_llm_factory.py` tested the pre-refactor architecture with methods like:
- `LLMFactory.detect_provider()` (removed - now in ConfigResolver)
- `LLMFactory._is_ollama_running()` (removed - now in OllamaProvider)
- `LLMFactory._translate_model_id()` (removed - handled by providers)

**New test file**: `test_provider_system.py` (30 tests)

Tests the new architecture:
- ProviderRegistry with ILLMProvider interface
- Individual provider implementations (Anthropic, Bedrock, Ollama)
- ConfigResolver 3-layer priority system
- LLMFactory integration with provider registry

**To clean up**: Delete `test_llm_factory_OLD_DEPRECATED.py` after verifying all functionality is covered by new tests.
