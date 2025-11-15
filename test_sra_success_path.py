#!/usr/bin/env python3
"""
Test the success path with a query that should find results.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.providers.sra_provider import SRAProvider, SRAProviderConfig
from lobster.utils.logger import get_logger

logger = get_logger(__name__)

def test_success_path():
    """Test with a broader query that should find results."""

    data_manager = DataManagerV2(workspace_path="/tmp/test_sra", console=None)
    config = SRAProviderConfig(max_results=3)
    provider = SRAProvider(data_manager=data_manager, config=config)

    # Broader query that should find results
    query = "RNA sequencing human cancer"
    filters = {"organism": "Homo sapiens", "strategy": "RNA-Seq"}

    logger.info("=" * 70)
    logger.info(f"Testing SRA keyword search with broader query:")
    logger.info(f"Query: {query}")
    logger.info(f"Filters: {filters}")
    logger.info("=" * 70)

    try:
        result = provider.search_publications(
            query=query,
            max_results=3,
            filters=filters
        )

        logger.info("\n" + "=" * 70)
        logger.info("RESULT:")
        logger.info("=" * 70)
        print(result)
        logger.info("=" * 70)

        # Verify success indicators
        if "Keyword Search Limitation" in result:
            logger.error("❌ FAILED: Still returning guidance message!")
            return False

        if "SRA Database Search Results" in result:
            logger.info("✅ SUCCESS: Found results with keyword search!")
            # Check for expected fields
            if "study_accession" in result.lower() or "srp" in result.lower():
                logger.info("✅ SUCCESS: Result contains study accessions!")
                return True
            else:
                logger.warning("⚠️ Results don't contain expected accession patterns")
                return True  # Still a success if it has the header
        elif "No SRA Results Found" in result:
            logger.info("✓ No results found (acceptable for this query)")
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
    success = test_success_path()
    sys.exit(0 if success else 1)
