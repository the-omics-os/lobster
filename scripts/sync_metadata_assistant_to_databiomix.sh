#!/bin/bash
#
# Sync DataBioMix metadata pipeline files from lobster to lobster-custom-databiomix
#

## Purpose

# Automatically sync metadata pipeline files (3 files) from the private `lobster` package
# to the custom `lobster-custom-databiomix` package with proper import path rewriting.
#
# Files synced:
# 1. metadata_assistant.py (agent)
# 2. metadata_filtering_service.py (service with disease fallback chain)
# 3. microbiome_filtering_service.py (service with metagenome HOST_ALIASES)

# ## Usage

# ```bash
# # Dry-run (preview changes)
# ./scripts/sync_metadata_assistant_to_databiomix.sh --dry-run

# # Apply changes
# ./scripts/sync_metadata_assistant_to_databiomix.sh
# ```

# ## What It Does

# 1. **Copies 3 files**:
#    - `lobster/agents/metadata_assistant.py` → `lobster_custom_databiomix/agents/metadata_assistant.py`
#    - `lobster/services/metadata/metadata_filtering_service.py` → `lobster_custom_databiomix/services/metadata/metadata_filtering_service.py`
#    - `lobster/services/metadata/microbiome_filtering_service.py` → `lobster_custom_databiomix/services/metadata/microbiome_filtering_service.py`
# 2. **Rewrites imports** for premium services:
#    - `lobster.services.metadata.*` → `lobster_custom_databiomix.services.metadata.*`
#    - `lobster.core.schemas.publication_queue` → `lobster_custom_databiomix.core.schemas.publication_queue`
#    - `factory_function="lobster.agents.*"` → `factory_function="lobster_custom_databiomix.agents.*"`
# 3. **Preserves fallbacks**: Keeps `except ImportError:` fallback structure intact
# 4. **Shows diff**: In dry-run mode, displays what would change for all 3 files

# ## When to Use

# Use this script when you've made changes to DataBioMix metadata pipeline files
# and need to propagate them to the custom package:

# - Fixed bugs in metadata_assistant or filtering services
# - Updated filtering logic (16S, host validation, disease extraction)
# - Added new tools or features to metadata processing
# - Changed HOST_ALIASES or filtering keywords
# - Updated documentation strings

# ## Example Workflow

# ```bash
# # 1. Make changes in lobster/ (any of the 3 files)
# vim lobster/agents/metadata_assistant.py
# vim lobster/services/metadata/metadata_filtering_service.py
# vim lobster/services/metadata/microbiome_filtering_service.py

# # 2. Test in base package
# pytest tests/unit/services/metadata/ -v

# # 3. Preview sync (dry-run) - shows diffs for all 3 files
# ./scripts/sync_metadata_assistant_to_databiomix.sh --dry-run

# # 4. Apply sync
# ./scripts/sync_metadata_assistant_to_databiomix.sh

# # 5. Test custom package imports
# cd ../lobster-custom-databiomix
# python -c "from lobster_custom_databiomix.agents.metadata_assistant import metadata_assistant"

# # 6. Commit both packages
# cd ../lobster && git add . && git commit -m "fix: DataBioMix metadata pipeline bug fixes"
# cd ../lobster-custom-databiomix && git add . && git commit -m "fix: sync metadata pipeline bug fixes from lobster"
# ```

# ## Import Rewrite Strategy

# ### What Gets Rewritten (Premium Services)

# These services exist in the custom package with enhancements:
# - `DiseaseStandardizationService` ✓ (not synced - assumed to exist)
# - `MicrobiomeFilteringService` ✓ (synced by this script)
# - `MetadataFilteringService` ✓ (synced by this script)
# - `extract_disease_with_fallback` ✓ (NEW function in metadata_filtering_service.py)
# - `PublicationQueue` schemas ✓ (not synced - assumed to exist)

# ### What Stays Unchanged (Public Lobster)

# These imports remain `from lobster.*` because they're in public PyPI package:
# - `CustomCodeExecutionService` (from lobster.services.execution)
# - `MetadataStandardizationService` (from lobster.services.metadata)
# - `SampleMappingService` (has try/except fallback)
# - All `core.*` modules (DataManagerV2, AnalysisStep, etc.)
# - All `config.*` modules (llm_factory, settings, etc.)

# ## Verification

# After running the script, verify:

# ```bash
# cd ../lobster-custom-databiomix

# # 1. Check syntax (all 3 files)
# python -m py_compile lobster_custom_databiomix/agents/metadata_assistant.py
# python -m py_compile lobster_custom_databiomix/services/metadata/metadata_filtering_service.py
# python -m py_compile lobster_custom_databiomix/services/metadata/microbiome_filtering_service.py

# # 2. Check imports
# python -c "from lobster_custom_databiomix.agents.metadata_assistant import metadata_assistant; print('OK')"
# python -c "from lobster_custom_databiomix.services.metadata.metadata_filtering_service import extract_disease_with_fallback; print('OK')"
# python -c "from lobster_custom_databiomix.services.metadata.microbiome_filtering_service import HOST_ALIASES; print('OK')"

# # 3. Review diffs (should show bug fixes)
# git diff lobster_custom_databiomix/agents/metadata_assistant.py
# git diff lobster_custom_databiomix/services/metadata/
# ```

# ## Troubleshooting

# **Error: "Target directory not found"**
# - Ensure `lobster-custom-databiomix` repo exists at `../lobster-custom-databiomix/`
# - Run from lobster root directory

# **Error: "Import fallback structure may be affected"**
# - Review the diff carefully before applying
# - The script preserves `except ImportError:` blocks by design

# **Import errors after sync**
# - Check that all premium services exist in custom package:
#   - `lobster_custom_databiomix/services/metadata/disease_standardization_service.py` (not synced - assumed to exist)
#   - `lobster_custom_databiomix/services/metadata/microbiome_filtering_service.py` (synced by this script)
#   - `lobster_custom_databiomix/services/metadata/metadata_filtering_service.py` (synced by this script)
#   - `lobster_custom_databiomix/core/schemas/publication_queue.py` (not synced - assumed to exist)


set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOBSTER_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATABIOMIX_ROOT="$(cd "$LOBSTER_ROOT/../lobster-custom-databiomix" && pwd)"

# File paths for syncing (expanded to include all DataBioMix bug fixes)
SOURCE_AGENT="$LOBSTER_ROOT/lobster/agents/metadata_assistant.py"
TARGET_AGENT="$DATABIOMIX_ROOT/lobster_custom_databiomix/agents/metadata_assistant.py"

SOURCE_METADATA_FILTERING="$LOBSTER_ROOT/lobster/services/metadata/metadata_filtering_service.py"
TARGET_METADATA_FILTERING="$DATABIOMIX_ROOT/lobster_custom_databiomix/services/metadata/metadata_filtering_service.py"

SOURCE_MICROBIOME_FILTERING="$LOBSTER_ROOT/lobster/services/metadata/microbiome_filtering_service.py"
TARGET_MICROBIOME_FILTERING="$DATABIOMIX_ROOT/lobster_custom_databiomix/services/metadata/microbiome_filtering_service.py"

# Temp files
TEMP_AGENT="/tmp/metadata_assistant_sync_$$.py"
TEMP_METADATA_FILTERING="/tmp/metadata_filtering_sync_$$.py"
TEMP_MICROBIOME_FILTERING="/tmp/microbiome_filtering_sync_$$.py"

# Check if dry-run mode
DRY_RUN=false
if [ "$1" = "--dry-run" ]; then
    DRY_RUN=true
    echo -e "${YELLOW}DRY RUN MODE${NC}"
fi

# Verify all source files exist
echo "Verifying source files..."
for src in "$SOURCE_AGENT" "$SOURCE_METADATA_FILTERING" "$SOURCE_MICROBIOME_FILTERING"; do
    if [ ! -f "$src" ]; then
        echo -e "${RED}✗ Source file not found: $src${NC}"
        exit 1
    fi
done
echo -e "${GREEN}✓${NC} All source files found"
echo ""

# Verify target directories exist
echo "Verifying target directories..."
for tgt in "$TARGET_AGENT" "$TARGET_METADATA_FILTERING" "$TARGET_MICROBIOME_FILTERING"; do
    TARGET_DIR="$(dirname "$tgt")"
    if [ ! -d "$TARGET_DIR" ]; then
        echo -e "${RED}✗ Target directory not found: $TARGET_DIR${NC}"
        exit 1
    fi
done
echo -e "${GREEN}✓${NC} All target directories found"
echo ""

echo "Syncing DataBioMix bug fixes (3 files)..."
echo "  1. metadata_assistant.py"
echo "  2. metadata_filtering_service.py"
echo "  3. microbiome_filtering_service.py"
echo ""

# ============================================================================
# Helper function: Rewrite imports for a single file
# ============================================================================
rewrite_imports() {
    local temp_file=$1
    local file_type=$2  # "agent" or "service"

    echo "  Rewriting import paths..."

    # Premium services that exist in custom package
    sed -i '' 's/from lobster\.services\.metadata\.disease_standardization_service/from lobster_custom_databiomix.services.metadata.disease_standardization_service/g' "$temp_file"
    sed -i '' 's/from lobster\.services\.metadata\.microbiome_filtering_service/from lobster_custom_databiomix.services.metadata.microbiome_filtering_service/g' "$temp_file"
    sed -i '' 's/from lobster\.services\.metadata\.metadata_filtering_service/from lobster_custom_databiomix.services.metadata.metadata_filtering_service/g' "$temp_file"

    # Publication queue schemas (premium)
    sed -i '' 's/from lobster\.core\.schemas\.publication_queue/from lobster_custom_databiomix.core.schemas.publication_queue/g' "$temp_file"

    # Agent-specific: factory function
    if [ "$file_type" = "agent" ]; then
        sed -i '' 's/factory_function="lobster\.agents\.metadata_assistant/factory_function="lobster_custom_databiomix.agents.metadata_assistant/g' "$temp_file"
        echo -e "    ${GREEN}✓${NC} Rewrote factory_function"
    fi

    echo -e "    ${GREEN}✓${NC} Rewrote premium service imports"
}

# ============================================================================
# Step 1: Sync metadata_assistant.py (Agent)
# ============================================================================
echo -e "${YELLOW}[1/3] Syncing metadata_assistant.py${NC}"

if [ ! -f "$SOURCE_AGENT" ]; then
    echo -e "${RED}✗ Source not found: $SOURCE_AGENT${NC}"
    exit 1
fi

cp "$SOURCE_AGENT" "$TEMP_AGENT"
rewrite_imports "$TEMP_AGENT" "agent"

# Verify fallback structure
if grep -q "try:" "$TEMP_AGENT" && grep -q "except ImportError:" "$TEMP_AGENT"; then
    echo -e "  ${GREEN}✓${NC} Import fallback structure preserved"
else
    echo -e "  ${YELLOW}⚠${NC} Warning: Import fallback structure may be affected"
fi
echo ""

# ============================================================================
# Step 2: Sync metadata_filtering_service.py (Service)
# ============================================================================
echo -e "${YELLOW}[2/3] Syncing metadata_filtering_service.py${NC}"

if [ ! -f "$SOURCE_METADATA_FILTERING" ]; then
    echo -e "${RED}✗ Source not found: $SOURCE_METADATA_FILTERING${NC}"
    exit 1
fi

cp "$SOURCE_METADATA_FILTERING" "$TEMP_METADATA_FILTERING"
rewrite_imports "$TEMP_METADATA_FILTERING" "service"
echo ""

# ============================================================================
# Step 3: Sync microbiome_filtering_service.py (Service)
# ============================================================================
echo -e "${YELLOW}[3/3] Syncing microbiome_filtering_service.py${NC}"

if [ ! -f "$SOURCE_MICROBIOME_FILTERING" ]; then
    echo -e "${RED}✗ Source not found: $SOURCE_MICROBIOME_FILTERING${NC}"
    exit 1
fi

# Note: microbiome_filtering_service.py only imports from lobster.core (public)
# No import rewriting needed, but copy for consistency
cp "$SOURCE_MICROBIOME_FILTERING" "$TEMP_MICROBIOME_FILTERING"
echo -e "  ${GREEN}✓${NC} No import rewrites needed (uses public lobster.core only)"
echo ""

# ============================================================================
# Dry-run mode: Show diffs
# ============================================================================
if [ "$DRY_RUN" = true ]; then
    echo ""
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Changes that would be applied:${NC}"
    echo -e "${YELLOW}========================================${NC}"
    echo ""

    echo -e "${YELLOW}[1/3] metadata_assistant.py:${NC}"
    diff -u "$TARGET_AGENT" "$TEMP_AGENT" || true
    echo ""

    echo -e "${YELLOW}[2/3] metadata_filtering_service.py:${NC}"
    diff -u "$TARGET_METADATA_FILTERING" "$TEMP_METADATA_FILTERING" || true
    echo ""

    echo -e "${YELLOW}[3/3] microbiome_filtering_service.py:${NC}"
    diff -u "$TARGET_MICROBIOME_FILTERING" "$TEMP_MICROBIOME_FILTERING" || true
    echo ""

    # Cleanup temp files
    rm "$TEMP_AGENT" "$TEMP_METADATA_FILTERING" "$TEMP_MICROBIOME_FILTERING"
    exit 0
fi

# ============================================================================
# Apply changes
# ============================================================================
echo -e "${YELLOW}Applying changes...${NC}"
cp "$TEMP_AGENT" "$TARGET_AGENT"
echo -e "${GREEN}✓${NC} Applied metadata_assistant.py"

cp "$TEMP_METADATA_FILTERING" "$TARGET_METADATA_FILTERING"
echo -e "${GREEN}✓${NC} Applied metadata_filtering_service.py"

cp "$TEMP_MICROBIOME_FILTERING" "$TARGET_MICROBIOME_FILTERING"
echo -e "${GREEN}✓${NC} Applied microbiome_filtering_service.py"

# Cleanup temp files
rm "$TEMP_AGENT" "$TEMP_METADATA_FILTERING" "$TEMP_MICROBIOME_FILTERING"

echo ""
echo -e "${GREEN}✓ Sync complete!${NC}"
echo ""

# ============================================================================
# Summary
# ============================================================================
echo "Summary:"
echo "  - Synced 3 files with DataBioMix bug fixes:"
echo "    • metadata_assistant.py (agent)"
echo "    • metadata_filtering_service.py (disease fallback chain)"
echo "    • microbiome_filtering_service.py (metagenome HOST_ALIASES)"
echo "  - Rewrote import paths for premium services"
echo "  - Preserved public lobster imports (fallback structure intact)"
echo ""
echo "Next steps:"
echo "  1. Review changes: cd $DATABIOMIX_ROOT && git diff"
echo "  2. Test imports:   python -c 'from lobster_custom_databiomix.agents.metadata_assistant import metadata_assistant'"
echo "  3. Run tests:      cd $DATABIOMIX_ROOT && pytest tests/unit/ -v"
echo "  4. Commit:         cd $DATABIOMIX_ROOT && git add . && git commit -m 'fix: sync DataBioMix metadata pipeline bug fixes'"
