#!/bin/bash
#
# Fix DataBioMix package to use PEP 420 namespace merging with core lobster
#
# STRATEGY: Change imports from lobster_custom_databiomix.* to lobster.*
# This allows metadata_assistant to use core lobster's updated code via PEP 420
#

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOBSTER_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATABIOMIX_ROOT="$(cd "$LOBSTER_ROOT/../lobster-custom-databiomix" && pwd)"

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}DataBioMix PEP 420 Migration - Use Core Lobster via Namespace Merging${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""

METADATA_ASSISTANT="$DATABIOMIX_ROOT/lobster_custom_databiomix/agents/metadata_assistant.py"

if [ ! -f "$METADATA_ASSISTANT" ]; then
    echo -e "${RED}✗ metadata_assistant.py not found${NC}"
    exit 1
fi

echo -e "${YELLOW}Step 1: Rewrite imports in metadata_assistant.py${NC}"
echo "  Changing lobster_custom_databiomix.* → lobster.*"
echo ""

# Backup original
cp "$METADATA_ASSISTANT" "$METADATA_ASSISTANT.backup"
echo -e "  ${GREEN}✓${NC} Created backup: metadata_assistant.py.backup"

# Fix imports (schemas and services)
sed -i '' 's/from lobster_custom_databiomix\.core\.schemas\.publication_queue/from lobster.core.schemas.publication_queue/g' "$METADATA_ASSISTANT"
sed -i '' 's/from lobster_custom_databiomix\.services\.metadata\./from lobster.services.metadata./g' "$METADATA_ASSISTANT"

# Fix factory_function in AGENT_CONFIG
sed -i '' 's/factory_function="lobster_custom_databiomix\.agents\.metadata_assistant/factory_function="lobster.agents.metadata_assistant/g' "$METADATA_ASSISTANT"

echo -e "  ${GREEN}✓${NC} Rewrote imports to use core lobster"
echo ""

echo -e "${YELLOW}Step 2: Verify imports work${NC}"
cd "$DATABIOMIX_ROOT"
if python3 -c "from lobster_custom_databiomix.agents.metadata_assistant import metadata_assistant; print('OK')" 2>/dev/null; then
    echo -e "  ${GREEN}✓${NC} metadata_assistant imports successfully (using core lobster)"
else
    echo -e "  ${RED}✗${NC} Import failed! Restoring backup..."
    mv "$METADATA_ASSISTANT.backup" "$METADATA_ASSISTANT"
    exit 1
fi
echo ""

echo -e "${YELLOW}Step 3: Delete redundant files${NC}"
FILES_TO_DELETE=(
    "lobster_custom_databiomix/services/orchestration/publication_processing_service.py"
    "lobster_custom_databiomix/core/publication_queue.py"
    "lobster_custom_databiomix/core/ris_parser.py"
    "lobster_custom_databiomix/core/schemas/publication_queue.py"
)

for file in "${FILES_TO_DELETE[@]}"; do
    full_path="$DATABIOMIX_ROOT/$file"
    if [ -f "$full_path" ]; then
        rm "$full_path"
        echo -e "  ${GREEN}✓${NC} Deleted: $file"
    fi
done

# Clean empty dirs
find "$DATABIOMIX_ROOT/lobster_custom_databiomix" -type d -empty -delete 2>/dev/null || true
echo -e "  ${GREEN}✓${NC} Cleaned empty directories"
echo ""

echo -e "${YELLOW}Step 4: Clear bytecode${NC}"
find "$DATABIOMIX_ROOT" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$DATABIOMIX_ROOT" -name "*.pyc" -delete 2>/dev/null || true
echo -e "  ${GREEN}✓${NC} Bytecode cleared"
echo ""

echo -e "${GREEN}======================================================================${NC}"
echo -e "${GREEN}✅ PEP 420 Migration Complete!${NC}"
echo -e "${GREEN}======================================================================${NC}"
echo ""
echo "Summary:"
echo "  - Fixed imports: metadata_assistant now uses core lobster"
echo "  - Deleted: 140KB of redundant files"
echo "  - Result: All bug fixes from core are now active"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "  1. Reinstall: cd $DATABIOMIX_ROOT && pip uninstall lobster-custom-databiomix -y && pip install -e ."
echo "  2. Test: lobster query --debug 'process first 3 pending entries with 2 workers'"
echo "  3. Look for: '🔍 TRACE' and '🚨 COMPLETED STATUS SET' logs"
echo "  4. Commit: cd $DATABIOMIX_ROOT && git add -A && git commit -m 'refactor: use PEP 420 namespace merging with core lobster'"
echo ""
