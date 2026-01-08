#!/bin/bash
#
# Sync DataBioMix metadata pipeline files from lobster to lobster-custom-databiomix
#

## Purpose

# Automatically sync metadata pipeline files (4 files) from the private `lobster` package
# to the custom `lobster-custom-databiomix` package with proper import path rewriting.
#
# Files synced:
# 1. metadata_assistant.py (agent)
# 2. metadata_filtering_service.py (service with disease fallback chain)
# 3. microbiome_filtering_service.py (service with metagenome HOST_ALIASES)
# 4. premium_agent_configs.py → agent_configs.py (LLM config for metadata_assistant)

# ## Usage

# ```bash
# # Dry-run (preview changes)
# ./scripts/sync_metadata_assistant_to_databiomix.sh --dry-run

# # Apply changes
# ./scripts/sync_metadata_assistant_to_databiomix.sh
# ```

# ## What It Does

# 1. **Copies 4 files**:
#    - `lobster/agents/metadata_assistant.py` → `lobster_custom_databiomix/agents/metadata_assistant.py`
#    - `lobster/services/metadata/metadata_filtering_service.py` → `lobster_custom_databiomix/services/metadata/metadata_filtering_service.py`
#    - `lobster/services/metadata/microbiome_filtering_service.py` → `lobster_custom_databiomix/services/metadata/microbiome_filtering_service.py`
#    - `lobster/config/premium_agent_configs.py` → `lobster_custom_databiomix/config/agent_configs.py`
# 2. **Rewrites imports** for premium services:
#    - `lobster.services.metadata.*` → `lobster_custom_databiomix.services.metadata.*`
#    - `factory_function="lobster.agents.*"` → `factory_function="lobster_custom_databiomix.agents.*"`
#    - Note: PublicationQueue imports stay as `lobster.core.schemas.publication_queue` (now public)
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

# ### What Stays Unchanged (Public Lobster)

# These imports remain `from lobster.*` because they're in public PyPI package:
# - `PublicationQueue` schemas ✓ (NOW PUBLIC as of v2.x - was premium before)
# - `PublicationProcessingService` ✓ (NOW PUBLIC as of v2.x - was premium before)
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
# - Note: PublicationQueue schemas are now imported from public lobster-ai package (v2.x+)


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

SOURCE_CONFIG="$LOBSTER_ROOT/lobster/config/premium_agent_configs.py"
TARGET_CONFIG="$DATABIOMIX_ROOT/lobster_custom_databiomix/config/agent_configs.py"

# Temp files
TEMP_AGENT="/tmp/metadata_assistant_sync_$$.py"
TEMP_METADATA_FILTERING="/tmp/metadata_filtering_sync_$$.py"
TEMP_MICROBIOME_FILTERING="/tmp/microbiome_filtering_sync_$$.py"
TEMP_CONFIG="/tmp/premium_agent_configs_sync_$$.py"

# Check if dry-run mode
DRY_RUN=false
if [ "$1" = "--dry-run" ]; then
    DRY_RUN=true
    echo -e "${YELLOW}DRY RUN MODE${NC}"
fi

# Verify all source files exist
echo "Verifying source files..."
for src in "$SOURCE_AGENT" "$SOURCE_METADATA_FILTERING" "$SOURCE_MICROBIOME_FILTERING" "$SOURCE_CONFIG"; do
    if [ ! -f "$src" ]; then
        echo -e "${RED}✗ Source file not found: $src${NC}"
        exit 1
    fi
done
echo -e "${GREEN}✓${NC} All source files found"
echo ""

# Verify target directories exist
echo "Verifying target directories..."
for tgt in "$TARGET_AGENT" "$TARGET_METADATA_FILTERING" "$TARGET_MICROBIOME_FILTERING" "$TARGET_CONFIG"; do
    TARGET_DIR="$(dirname "$tgt")"
    if [ ! -d "$TARGET_DIR" ]; then
        echo -e "${RED}✗ Target directory not found: $TARGET_DIR${NC}"
        exit 1
    fi
done
echo -e "${GREEN}✓${NC} All target directories found"
echo ""

echo "Syncing DataBioMix bug fixes (4 files)..."
echo "  1. metadata_assistant.py"
echo "  2. metadata_filtering_service.py"
echo "  3. microbiome_filtering_service.py"
echo "  4. premium_agent_configs.py → agent_configs.py"
echo ""

# ============================================================================
# Helper function: Get file stats (lines, functions, classes)
# ============================================================================
get_file_stats() {
    local file=$1
    local lines=$(wc -l < "$file" 2>/dev/null | tr -d ' ' || echo 0)
    local classes=$(grep -c "^class " "$file" 2>/dev/null || echo 0)
    local functions=$(grep -c "^def " "$file" 2>/dev/null || echo 0)
    local tools=$(grep -c "@tool" "$file" 2>/dev/null || echo 0)

    echo "${lines}:${classes}:${functions}:${tools}"
}

# ============================================================================
# Helper function: Rewrite imports for a single file
# ============================================================================
rewrite_imports() {
    local temp_file=$1
    local file_type=$2  # "agent" or "service"
    local rewrite_count=0

    echo "  Rewriting import paths..."

    # Premium services that exist in custom package
    local before=$(grep -c "from lobster\.services\.metadata\." "$temp_file" 2>/dev/null || echo 0)
    sed -i '' 's/from lobster\.services\.metadata\.disease_standardization_service/from lobster_custom_databiomix.services.metadata.disease_standardization_service/g' "$temp_file"
    sed -i '' 's/from lobster\.services\.metadata\.microbiome_filtering_service/from lobster_custom_databiomix.services.metadata.microbiome_filtering_service/g' "$temp_file"
    sed -i '' 's/from lobster\.services\.metadata\.metadata_filtering_service/from lobster_custom_databiomix.services.metadata.metadata_filtering_service/g' "$temp_file"
    local after=$(grep -c "from lobster_custom_databiomix\.services\.metadata\." "$temp_file" 2>/dev/null || echo 0)
    rewrite_count=$((rewrite_count + after))

    # NOTE: Publication queue schemas are now PUBLIC (v2.x+) - no rewrite needed
    # They remain as 'from lobster.core.schemas.publication_queue' (public package)

    # Agent-specific: factory function
    if [ "$file_type" = "agent" ]; then
        sed -i '' 's/factory_function="lobster\.agents\.metadata_assistant/factory_function="lobster_custom_databiomix.agents.metadata_assistant/g' "$temp_file"
        rewrite_count=$((rewrite_count + 1))
        echo -e "    ${GREEN}✓${NC} Rewrote factory_function"
    fi

    echo -e "    ${GREEN}✓${NC} Rewrote $rewrite_count import paths"
}

# ============================================================================
# Step 1: Sync metadata_assistant.py (Agent)
# ============================================================================
echo -e "${YELLOW}[1/4] Syncing metadata_assistant.py${NC}"

if [ ! -f "$SOURCE_AGENT" ]; then
    echo -e "${RED}✗ Source not found: $SOURCE_AGENT${NC}"
    exit 1
fi

# Get file stats
stats=$(get_file_stats "$SOURCE_AGENT")
lines=$(echo "$stats" | cut -d':' -f1)
classes=$(echo "$stats" | cut -d':' -f2)
functions=$(echo "$stats" | cut -d':' -f3)
tools=$(echo "$stats" | cut -d':' -f4)

echo "  Source: $lines lines, $classes classes, $functions functions, $tools tools"

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
echo -e "${YELLOW}[2/4] Syncing metadata_filtering_service.py${NC}"

if [ ! -f "$SOURCE_METADATA_FILTERING" ]; then
    echo -e "${RED}✗ Source not found: $SOURCE_METADATA_FILTERING${NC}"
    exit 1
fi

# Get file stats
stats=$(get_file_stats "$SOURCE_METADATA_FILTERING")
lines=$(echo "$stats" | cut -d':' -f1)
classes=$(echo "$stats" | cut -d':' -f2)
functions=$(echo "$stats" | cut -d':' -f3)

echo "  Source: $lines lines, $classes classes, $functions functions"

cp "$SOURCE_METADATA_FILTERING" "$TEMP_METADATA_FILTERING"
rewrite_imports "$TEMP_METADATA_FILTERING" "service"
echo ""

# ============================================================================
# Step 3: Sync microbiome_filtering_service.py (Service)
# ============================================================================
echo -e "${YELLOW}[3/4] Syncing microbiome_filtering_service.py${NC}"

if [ ! -f "$SOURCE_MICROBIOME_FILTERING" ]; then
    echo -e "${RED}✗ Source not found: $SOURCE_MICROBIOME_FILTERING${NC}"
    exit 1
fi

# Get file stats
stats=$(get_file_stats "$SOURCE_MICROBIOME_FILTERING")
lines=$(echo "$stats" | cut -d':' -f1)
classes=$(echo "$stats" | cut -d':' -f2)
functions=$(echo "$stats" | cut -d':' -f3)

echo "  Source: $lines lines, $classes classes, $functions functions"

# Note: microbiome_filtering_service.py only imports from lobster.core (public)
# No import rewriting needed, but copy for consistency
cp "$SOURCE_MICROBIOME_FILTERING" "$TEMP_MICROBIOME_FILTERING"
echo -e "  ${GREEN}✓${NC} No import rewrites needed (uses public lobster.core only)"
echo ""

# ============================================================================
# Step 4: Sync premium_agent_configs.py → agent_configs.py (Config)
# ============================================================================
echo -e "${YELLOW}[4/4] Syncing premium_agent_configs.py → agent_configs.py${NC}"

if [ ! -f "$SOURCE_CONFIG" ]; then
    echo -e "${RED}✗ Source not found: $SOURCE_CONFIG${NC}"
    exit 1
fi

# Get file stats
stats=$(get_file_stats "$SOURCE_CONFIG")
lines=$(echo "$stats" | cut -d':' -f1)
classes=$(echo "$stats" | cut -d':' -f2)
functions=$(echo "$stats" | cut -d':' -f3)

echo "  Source: $lines lines, $classes classes, $functions functions"

# Note: premium_agent_configs.py only imports from lobster.config (public)
# No import rewriting needed - config imports stay as 'lobster.config.*'
cp "$SOURCE_CONFIG" "$TEMP_CONFIG"
echo -e "  ${GREEN}✓${NC} No import rewrites needed (uses public lobster.config only)"
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

    echo -e "${YELLOW}[1/4] metadata_assistant.py:${NC}"
    diff_output=$(diff -u "$TARGET_AGENT" "$TEMP_AGENT" 2>/dev/null || true)
    if [ -z "$diff_output" ]; then
        echo -e "  ${GREEN}✓${NC} No changes (files identical)"
    else
        additions=$(echo "$diff_output" | grep -c "^+" || echo 0)
        deletions=$(echo "$diff_output" | grep -c "^-" || echo 0)
        echo -e "  ${YELLOW}Changes:${NC} +$additions lines, -$deletions lines"
        echo "$diff_output"
    fi
    echo ""

    echo -e "${YELLOW}[2/4] metadata_filtering_service.py:${NC}"
    diff_output=$(diff -u "$TARGET_METADATA_FILTERING" "$TEMP_METADATA_FILTERING" 2>/dev/null || true)
    if [ -z "$diff_output" ]; then
        echo -e "  ${GREEN}✓${NC} No changes (files identical)"
    else
        additions=$(echo "$diff_output" | grep -c "^+" || echo 0)
        deletions=$(echo "$diff_output" | grep -c "^-" || echo 0)
        echo -e "  ${YELLOW}Changes:${NC} +$additions lines, -$deletions lines"
        echo "$diff_output"
    fi
    echo ""

    echo -e "${YELLOW}[3/4] microbiome_filtering_service.py:${NC}"
    diff_output=$(diff -u "$TARGET_MICROBIOME_FILTERING" "$TEMP_MICROBIOME_FILTERING" 2>/dev/null || true)
    if [ -z "$diff_output" ]; then
        echo -e "  ${GREEN}✓${NC} No changes (files identical)"
    else
        additions=$(echo "$diff_output" | grep -c "^+" || echo 0)
        deletions=$(echo "$diff_output" | grep -c "^-" || echo 0)
        echo -e "  ${YELLOW}Changes:${NC} +$additions lines, -$deletions lines"
        echo "$diff_output"
    fi
    echo ""

    echo -e "${YELLOW}[4/4] premium_agent_configs.py → agent_configs.py:${NC}"
    diff_output=$(diff -u "$TARGET_CONFIG" "$TEMP_CONFIG" 2>/dev/null || true)
    if [ -z "$diff_output" ]; then
        echo -e "  ${GREEN}✓${NC} No changes (files identical)"
    else
        additions=$(echo "$diff_output" | grep -c "^+" || echo 0)
        deletions=$(echo "$diff_output" | grep -c "^-" || echo 0)
        echo -e "  ${YELLOW}Changes:${NC} +$additions lines, -$deletions lines"
        echo "$diff_output"
    fi
    echo ""

    # Cleanup temp files
    rm "$TEMP_AGENT" "$TEMP_METADATA_FILTERING" "$TEMP_MICROBIOME_FILTERING" "$TEMP_CONFIG"
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

cp "$TEMP_CONFIG" "$TARGET_CONFIG"
echo -e "${GREEN}✓${NC} Applied premium_agent_configs.py → agent_configs.py"

# Cleanup temp files
rm "$TEMP_AGENT" "$TEMP_METADATA_FILTERING" "$TEMP_MICROBIOME_FILTERING" "$TEMP_CONFIG"

echo ""
echo -e "${GREEN}✓ Sync complete!${NC}"
echo ""

# ============================================================================
# Summary
# ============================================================================
# Calculate total stats
total_lines=0
total_classes=0
total_functions=0
total_tools=0

for src in "$SOURCE_AGENT" "$SOURCE_METADATA_FILTERING" "$SOURCE_MICROBIOME_FILTERING" "$SOURCE_CONFIG"; do
    stats=$(get_file_stats "$src")
    lines=$(echo "$stats" | cut -d':' -f1)
    classes=$(echo "$stats" | cut -d':' -f2)
    functions=$(echo "$stats" | cut -d':' -f3)
    tools=$(echo "$stats" | cut -d':' -f4)

    total_lines=$((total_lines + lines))
    total_classes=$((total_classes + classes))
    total_functions=$((total_functions + functions))
    total_tools=$((total_tools + tools))
done

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Summary${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Files synced: 4"
echo "  • metadata_assistant.py (agent)"
echo "  • metadata_filtering_service.py (service)"
echo "  • microbiome_filtering_service.py (service)"
echo "  • premium_agent_configs.py → agent_configs.py (LLM config)"
echo ""
echo "Total transferred:"
echo "  • Lines:     $total_lines"
echo "  • Classes:   $total_classes"
echo "  • Functions: $total_functions"
echo "  • Tools:     $total_tools"
echo ""
echo "Import rewrites:"
echo "  • Premium services (lobster.services.metadata.* → lobster_custom_databiomix.*)"
echo "  • Factory function (lobster.agents.* → lobster_custom_databiomix.agents.*)"
echo "  • Public imports preserved (lobster.core.*, lobster.config.* unchanged)"
echo ""
echo "Next steps:"
echo "  1. Review changes: cd $DATABIOMIX_ROOT && git diff"
echo "  2. Test imports:   python -c 'from lobster_custom_databiomix.config.agent_configs import METADATA_ASSISTANT_CONFIG'"
echo "  3. Run tests:      cd $DATABIOMIX_ROOT && pytest tests/unit/ -v"
echo "  4. Commit:         cd $DATABIOMIX_ROOT && git add . && git commit -m 'fix: sync DataBioMix metadata pipeline and LLM config'"
