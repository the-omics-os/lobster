#!/usr/bin/env python3
"""
Quick test script to verify SRA provider keyword search fix.
Tests the original user query that triggered the bug.
"""

import sys
import os

# Add lobster to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.providers.sra_provider import SRAProvider, SRAProviderConfig
from lobster.utils.logger import get_logger

logger = get_logger(__name__)

def test_original_query():
    """Test the original user query that failed."""

    # Initialize DataManager and SRA provider
    data_manager = DataManagerV2(workspace_path="/tmp/test_sra", console=None)
    config = SRAProviderConfig(max_results=5)
    provider = SRAProvider(data_manager=data_manager, config=config)

    # Original query from user
    query = "IBS irritable bowel syndrome human microbiome 16S rRNA"
    filters = {"organism": "Homo sapiens", "strategy": "AMPLICON"}  # Using scientific name

    logger.info("=" * 70)
    logger.info(f"Testing SRA keyword search with original query:")
    logger.info(f"Query: {query}")
    logger.info(f"Filters: {filters}")
    logger.info("=" * 70)

    try:
        result = provider.search_publications(
            query=query,
            max_results=5,
            filters=filters
        )

        logger.info("\n" + "=" * 70)
        logger.info("RESULT:")
        logger.info("=" * 70)
        print(result)
        logger.info("=" * 70)

        # Verify it's not returning guidance message
        if "Keyword Search Limitation" in result:
            logger.error("❌ FAILED: Still returning guidance message!")
            return False

        if "SRA Database Search Results" in result or "No SRA Results Found" in result:
            logger.info("✅ SUCCESS: Keyword search working correctly!")
            return True
        else:
            logger.warning("⚠️ UNEXPECTED: Result format doesn't match expected patterns")
            return False

    except Exception as e:
        logger.error(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_original_query()
    sys.exit(0 if success else 1)
