# Lobster AI v1.0.0 Migration Guide

## Overview

Lobster AI v1.0.0 introduces a modular package architecture. This guide helps migrate existing `lobster-custom-*` packages.

## Key Changes

1. **Modular packages**: Agents are now separate PyPI packages
2. **Entry point discovery**: `AGENT_REGISTRY` removed, use entry points
3. **Version constraints**: Use `lobster-ai~=1.0.0` dependency

## Backward Compatibility

Your custom package continues working if:
- Entry points registered in pyproject.toml
- AGENT_CONFIG defined at module top
- Factory follows standard signature

## Quick Migration

1. Update dependency:
   ```toml
   dependencies = ["lobster-ai~=1.0.0"]
   ```

2. Verify entry points:
   ```toml
   [project.entry-points."lobster.agents"]
   my_agent = "lobster_custom_company.agents:AGENT_CONFIG"
   ```

3. Test discovery:
   ```bash
   lobster agents list
   ```

## Full Documentation

See https://docs.omics-os.com/docs/extending/migration-guide

## Support

Enterprise customers: support@omics-os.com
