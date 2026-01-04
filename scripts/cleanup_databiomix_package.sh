#!/bin/bash
#
# Clean up lobster-custom-databiomix package by removing unused/redundant files
#
# PURPOSE: Remove files that are NOT imported by metadata_assistant and cause sync issues
#
# ANALYSIS: metadata_assistant.py imports from lobster.* (core package), NOT lobster_custom_databiomix.*
# Therefore, duplicated orchestration/core files are UNUSED and should be deleted.
#

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOBSTER_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATABIOMIX_ROOT="$(cd "$LOBSTER_ROOT/../lobster-custom-databiomix" && pwd)"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}DataBioMix Package Cleanup${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if dry-run mode
DRY_RUN=false
if [ "$1" = "--dry-run" ]; then
    DRY_RUN=true
    echo -e "${YELLOW}DRY RUN MODE - No files will be deleted${NC}"
    echo ""
fi

# Verify databiomix directory exists
if [ ! -d "$DATABIOMIX_ROOT" ]; then
    echo -e "${RED}✗ DataBioMix directory not found: $DATABIOMIX_ROOT${NC}"
    exit 1
fi

echo "DataBioMix package: $DATABIOMIX_ROOT"
echo ""

# =============================================================================
# FILES TO DELETE (Unused - metadata_assistant imports from core lobster)
# =============================================================================

FILES_TO_DELETE=(
    # Orchestration layer (unused, causes bugs)
    "lobster_custom_databiomix/services/orchestration/publication_processing_service.py"

    # Core layer (unused, causes bugs)
    "lobster_custom_databiomix/core/publication_queue.py"
    "lobster_custom_databiomix/core/ris_parser.py"
    "lobster_custom_databiomix/core/schemas/publication_queue.py"
)

echo -e "${YELLOW}Files to DELETE (unused by metadata_assistant):${NC}"
total_size=0
for file in "${FILES_TO_DELETE[@]}"; do
    full_path="$DATABIOMIX_ROOT/$file"
    if [ -f "$full_path" ]; then
        size_bytes=$(stat -f%z "$full_path" 2>/dev/null || echo "0")
        size_kb=$((size_bytes / 1024))
        total_size=$((total_size + size_kb))
        echo -e "  ❌ $file ${RED}(${size_kb}KB)${NC}"
    else
        echo -e "  ⚠️  $file (already deleted)"
    fi
done
echo -e "  ${RED}Total: ${total_size}KB of unused code${NC}"
echo ""

# =============================================================================
# FILES TO KEEP (Actually imported by metadata_assistant)
# =============================================================================

FILES_TO_KEEP=(
    # The agent itself
    "lobster_custom_databiomix/agents/metadata_assistant.py"

    # Services imported by metadata_assistant
    "lobster_custom_databiomix/services/metadata/metadata_filtering_service.py"
    "lobster_custom_databiomix/services/metadata/microbiome_filtering_service.py"
    "lobster_custom_databiomix/services/metadata/disease_standardization_service.py"
    "lobster_custom_databiomix/services/metadata/sample_mapping_service.py"
    "lobster_custom_databiomix/services/metadata/identifier_provenance_service.py"
)

# =============================================================================
# VERIFY KEPT FILES EXIST
# =============================================================================

echo -e "${GREEN}Files to KEEP (imported by metadata_assistant):${NC}"
missing_count=0
for file in "${FILES_TO_KEEP[@]}"; do
    full_path="$DATABIOMIX_ROOT/$file"
    if [ -f "$full_path" ]; then
        size_bytes=$(stat -f%z "$full_path" 2>/dev/null || echo "0")
        size_kb=$((size_bytes / 1024))
        echo -e "  ✅ $file ${GREEN}(${size_kb}KB)${NC}"
    else
        echo -e "  ${RED}⚠️  Missing: $file${NC}"
        missing_count=$((missing_count + 1))
    fi
done

if [ $missing_count -gt 0 ]; then
    echo ""
    echo -e "${RED}✗ $missing_count required files are missing!${NC}"
    echo -e "${RED}Cannot proceed with cleanup.${NC}"
    exit 1
fi
echo ""

# =============================================================================
# Execute cleanup
# =============================================================================

if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Dry run complete${NC}"
    echo -e "${YELLOW}========================================${NC}"
    echo ""
    echo "Would delete $total_size KB of unused code"
    echo "Run without --dry-run to apply changes"
    exit 0
fi

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Executing cleanup...${NC}"
echo -e "${YELLOW}========================================${NC}"
echo ""

# Delete unused files
echo -e "${YELLOW}Deleting unused files...${NC}"
deleted_count=0
for file in "${FILES_TO_DELETE[@]}"; do
    full_path="$DATABIOMIX_ROOT/$file"
    if [ -f "$full_path" ]; then
        rm "$full_path"
        echo -e "  ${GREEN}✓${NC} Deleted: $file"
        deleted_count=$((deleted_count + 1))
    else
        echo -e "  ⚠️  Already gone: $file"
    fi
done

# Clean up empty directories
echo ""
echo -e "${YELLOW}Cleaning up empty directories...${NC}"
find "$DATABIOMIX_ROOT/lobster_custom_databiomix" -type d -empty -delete 2>/dev/null || true
echo -e "  ${GREEN}✓${NC} Removed empty directories"

# Verify imports still work
echo ""
echo -e "${YELLOW}Verifying imports...${NC}"
cd "$DATABIOMIX_ROOT"
if python3 -c "from lobster_custom_databiomix.agents.metadata_assistant import metadata_assistant; print('OK')" 2>/dev/null; then
    echo -e "  ${GREEN}✓${NC} metadata_assistant imports successfully"
else
    echo -e "  ${RED}✗${NC} Import failed! Cleanup may have broken dependencies"
    exit 1
fi

# Clean bytecode
echo ""
echo -e "${YELLOW}Clearing bytecode cache...${NC}"
find "$DATABIOMIX_ROOT" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$DATABIOMIX_ROOT" -name "*.pyc" -delete 2>/dev/null || true
echo -e "  ${GREEN}✓${NC} Bytecode cleared"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✅ Cleanup complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Summary:"
echo "  - Deleted: $deleted_count unused files (${total_size}KB freed)"
echo "  - Kept: ${#FILES_TO_KEEP[@]} essential files"
echo "  - Imports: ✓ Verified working"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "  1. Reinstall package: cd $DATABIOMIX_ROOT && pip uninstall lobster-custom-databiomix -y && pip install -e ."
echo "  2. Test bug fix: lobster query --debug 'process first 3 pending entries with 2 workers'"
echo "  3. Check for TRACE logs: Should see '🔍 TRACE' and '🔍 VERIFY' in output"
echo "  4. Commit changes: cd $DATABIOMIX_ROOT && git add -A && git commit -m 'cleanup: remove unused orchestration/core files (shadowed core lobster)'"
echo ""
echo -e "${YELLOW}IMPORTANT: After testing, remember to remove debug logging from core files${NC}"
echo ""
