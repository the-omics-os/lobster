#!/usr/bin/env python3
"""
Comprehensive test of SRA provider functionality.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.providers.sra_provider import SRAProvider, SRAProviderConfig
from lobster.utils.logger import get_logger

logger = get_logger(__name__)

def test_accession_based_search():
    """Test that accession-based search still works (regression test)."""
    data_manager = DataManagerV2(workspace_path="/tmp/test_sra", console=None)
    config = SRAProviderConfig(max_results=3)
    provider = SRAProvider(data_manager=data_manager, config=config)

    logger.info("\n" + "=" * 70)
    logger.info("TEST 1: Accession-Based Search (Regression Test)")
    logger.info("=" * 70)

    # Use known GEUVADIS study from the integration tests
    accession = "SRP033351"
    logger.info(f"Query: {accession}")

    try:
        result = provider.search_publications(accession)

        print(result)
        logger.info("=" * 70)

        if "SRA Database Search Results" in result and accession in result:
            logger.info("✅ PASS: Accession-based search works")
            return True
        else:
            logger.error("❌ FAIL: Accession-based search broken")
            return False

    except Exception as e:
        logger.error(f"❌ ERROR: {e}")
        return False

def test_keyword_search_no_filters():
    """Test keyword search without filters."""
    data_manager = DataManagerV2(workspace_path="/tmp/test_sra", console=None)
    config = SRAProviderConfig(max_results=3)
    provider = SRAProvider(data_manager=data_manager, config=config)

    logger.info("\n" + "=" * 70)
    logger.info("TEST 2: Keyword Search Without Filters")
    logger.info("=" * 70)

    query = "cancer"
    logger.info(f"Query: {query}")

    try:
        result = provider.search_publications(query, max_results=3)

        print(result)
        logger.info("=" * 70)

        # Should not return guidance message
        if "Keyword Search Limitation" in result:
            logger.error("❌ FAIL: Returned guidance message")
            return False

        # Should perform actual search
        if "SRA Database Search Results" in result or "No SRA Results Found" in result:
            logger.info("✅ PASS: Keyword search executed (no guidance message)")
            return True
        else:
            logger.error("❌ FAIL: Unexpected response format")
            return False

    except Exception as e:
        logger.error(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_keyword_search_with_filters():
    """Test keyword search with filters."""
    data_manager = DataManagerV2(workspace_path="/tmp/test_sra", console=None)
    config = SRAProviderConfig(max_results=3)
    provider = SRAProvider(data_manager=data_manager, config=config)

    logger.info("\n" + "=" * 70)
    logger.info("TEST 3: Keyword Search With Filters")
    logger.info("=" * 70)

    query = "microbiome"
    filters = {"strategy": "AMPLICON"}
    logger.info(f"Query: {query}")
    logger.info(f"Filters: {filters}")

    try:
        result = provider.search_publications(query, max_results=3, filters=filters)

        print(result)
        logger.info("=" * 70)

        # Should not return guidance message
        if "Keyword Search Limitation" in result:
            logger.error("❌ FAIL: Returned guidance message")
            return False

        # Should perform actual search
        if "SRA Database Search Results" in result or "No SRA Results Found" in result:
            logger.info("✅ PASS: Keyword search with filters executed")
            return True
        else:
            logger.error("❌ FAIL: Unexpected response format")
            return False

    except Exception as e:
        logger.error(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    results = []

    results.append(("Accession-based search", test_accession_based_search()))
    results.append(("Keyword search (no filters)", test_keyword_search_no_filters()))
    results.append(("Keyword search (with filters)", test_keyword_search_with_filters()))

    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status}: {test_name}")

    all_passed = all(result[1] for result in results)
    logger.info("=" * 70)

    if all_passed:
        logger.info("🎉 ALL TESTS PASSED!")
    else:
        logger.error("❌ SOME TESTS FAILED")

    sys.exit(0 if all_passed else 1)
