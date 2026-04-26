"""
OAuth authentication backends for LLM providers.

Provides PKCE-based OAuth 2.0 flows for providers that support it,
starting with Anthropic (Claude Pro/Max). The auth module is provider-agnostic
at the storage layer — any provider can store OAuth credentials via oauth_store.
"""
