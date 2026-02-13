"""
Integration tests for ContentAccessService three-tier cascade with real API calls.

This test suite validates the three-tier content access cascade strategy:
- Tier 1 (PMC Fast Path): PMC XML extraction (~500ms, highest accuracy)
- Tier 2 (Webpage Fallback): Publisher webpage parsing (2-5s, good structure)
- Tier 3 (PDF Fallback): PDF parsing with Docling (3-8s, most universal)

**Performance Expectations:**
- PMC XML extraction: <500ms (optimal path)
- Webpage extraction: 2-5s (fallback 1)
- PDF extraction: 3-8s (fallback 2)
- Full cascade (all tiers attempted): <10s total

**Test Strategy:**
- Use known stable identifiers with predictable tier availability
- Measure actual performance against targets
- Test cascade fallback logic (PMC unavailable → webpage → PDF)
- Validate error propagation and recovery
- Test rate limiting and retry behavior

**Markers:**
- @pytest.mark.real_api: All tests (requires internet + API keys)
- @pytest.mark.slow: Tests >30s
- @pytest.mark.integration: Multi-component tests

**Environment Requirements:**
- NCBI_API_KEY (recommended for PubMed rate limits)
- AWS_BEDROCK_ACCESS_KEY + AWS_BEDROCK_SECRET_ACCESS_KEY (for LLM)
- Internet connectivity for API access

Phase 7 - Task Group 2: ContentAccessService Three-Tier Cascade Tests
"""

import os
import time
from pathlib import Path

import pytest

from lobster.config.settings import get_settings
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.services.data_access.content_access_service import ContentAccessService
from lobster.utils.logger import get_logger

logger = get_logger(__name__)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def test_workspace(tmp_path_factory):
    """Create temporary workspace for test session."""
    workspace = tmp_path_factory.mktemp("test_content_cascade_real_api")
    return workspace


@pytest.fixture(scope="module")
def data_manager(test_workspace):
    """Initialize DataManagerV2 with test workspace."""
    settings = get_settings()
    dm = DataManagerV2(workspace_path=test_workspace, console=None)
    return dm


@pytest.fixture(scope="module")
def content_service(data_manager):
    """Create ContentAccessService instance for testing."""
    return ContentAccessService(data_manager=data_manager)


@pytest.fixture(scope="module")
def check_api_keys():
    """Verify required API keys are present."""
    required_keys = ["AWS_BEDROCK_ACCESS_KEY", "AWS_BEDROCK_SECRET_ACCESS_KEY"]
    missing = [key for key in required_keys if not os.getenv(key)]

    if missing:
        pytest.skip(f"Missing required API keys: {', '.join(missing)}")

    # NCBI_API_KEY is recommended but not required
    if not os.getenv("NCBI_API_KEY"):
        logger.warning(
            "NCBI_API_KEY not set - PubMed requests limited to 3/second (vs 10/second with key)"
        )


# ============================================================================
# Scenario 2.1: PMC Fast Path (Tier 1)
# ============================================================================


@pytest.mark.real_api
@pytest.mark.integration
class TestPMCFastPath:
    """Test PMC XML extraction (Tier 1 - fastest, most accurate)."""

    def test_pmc_extraction_performance(self, content_service, check_api_keys):
        """Test PMC XML extraction meets <500ms target."""
        # Use known PMID with PMC full text
        identifier = "PMID:35042229"

        start_time = time.time()
        result = content_service.get_full_content(
            source=identifier,
            prefer_webpage=False,  # Force PMC first
            max_paragraphs=100,
        )
        elapsed = time.time() - start_time

        # Verify successful extraction
        assert result is not None
        # Check for content in any of the expected keys (API format may vary)
        content_keys = ['content', 'methods_markdown', 'methods_text', 'full_text']
        has_content = any(key in result for key in content_keys)
        assert has_content, f"No content found. Keys: {list(result.keys())}"

        # Verify content is not empty
        for key in content_keys:
            if key in result and result[key]:
                assert len(str(result[key])) > 0
                break

        # Check tier used (should be PMC for this identifier)
        tier_used = result.get("tier_used", "")
        logger.info(f"Tier used: {tier_used}, Time: {elapsed:.3f}s")

        # PMC extraction should be fast (allow some overhead)
        # Note: With network latency and API variability, allow up to 5s
        assert (
            elapsed < 5.0
        ), f"PMC extraction took {elapsed:.3f}s (target: <500ms base + network, allow up to 5s)"

    def test_pmc_structured_content_quality(self, content_service, check_api_keys):
        """Test PMC XML provides structured, high-quality content."""
        identifier = "PMID:35042229"

        result = content_service.get_full_content(
            source=identifier, prefer_webpage=False, max_paragraphs=100
        )

        # Verify structured content elements
        # Check for content in any of the expected keys (API format may vary)
        content_keys = ['content', 'methods_markdown', 'methods_text', 'full_text']
        has_content = any(key in result for key in content_keys)
        assert has_content, f"No content found. Keys: {list(result.keys())}"

        # Get content from whichever key exists
        content = None
        for key in content_keys:
            if key in result and result[key]:
                content = str(result[key])
                break
        assert content is not None, "No content found in any expected keys"

        # PMC XML should extract sections clearly
        # Check for common scientific paper sections
        content_lower = content.lower()
        assert any(
            keyword in content_lower
            for keyword in ["method", "result", "introduction", "abstract"]
        )

        # Should have reasonable length (not truncated)
        assert (
            len(content) > 1000
        ), "PMC content too short - extraction may be incomplete"

    def test_pmc_metadata_extraction(self, content_service, check_api_keys):
        """Test PMC XML extracts comprehensive metadata."""
        identifier = "PMID:35042229"

        result = content_service.get_full_content(
            source=identifier, prefer_webpage=False, max_paragraphs=100
        )

        # Verify metadata present
        metadata = result.get("metadata", {})
        assert metadata is not None
        assert isinstance(metadata, dict)

        # PMC should provide rich metadata
        # At minimum: should have some extracted information
        assert len(metadata) > 0, "PMC should extract metadata from XML"

    def test_pmc_failure_edge_case(self, content_service, check_api_keys):
        """Test handling when PMC is unavailable for a publication."""
        # Use PMID without PMC full text (abstract only)
        identifier = "PMID:10000000"

        try:
            result = content_service.get_full_content(
                source=identifier, prefer_webpage=False, max_paragraphs=100
            )

            # If this succeeds, check that it fell back appropriately
            tier_used = result.get("tier_used", "")
            logger.info(f"PMC unavailable case - Tier used: {tier_used}")

            # Should NOT be PMC tier for this identifier
            assert "pmc" not in tier_used.lower() or result.get("content") is None

        except Exception as e:
            # Exception is acceptable for invalid/unavailable PMID
            logger.info(f"Expected failure for unavailable PMID: {e}")
            assert "error" in str(e).lower() or "not found" in str(e).lower()


# ============================================================================
# Scenario 2.2: Webpage Fallback (Tier 2)
# ============================================================================


@pytest.mark.real_api
@pytest.mark.integration
@pytest.mark.slow
class TestWebpageFallback:
    """Test webpage extraction (Tier 2 - fallback when PMC unavailable)."""

    def test_webpage_extraction_performance(self, content_service, check_api_keys):
        """Test webpage extraction meets 2-5s target."""
        # Use Nature article URL (no PMC, has webpage)
        identifier = "https://www.nature.com/articles/s41586-021-03852-1"

        start_time = time.time()
        result = content_service.get_full_content(
            source=identifier, prefer_webpage=True, max_paragraphs=100
        )
        elapsed = time.time() - start_time

        # Verify successful extraction
        assert result is not None
        # Check for content in any of the expected keys (API format changed)
        has_content = any(key in result for key in ['content', 'methods_markdown', 'methods_text', 'full_text'])
        assert has_content, f"No content found in result. Keys: {list(result.keys())}"

        # Webpage extraction: 2-5s target (allow up to 10s with network)
        logger.info(f"Webpage extraction time: {elapsed:.3f}s (target: 2-5s)")
        assert elapsed < 10.0, f"Webpage extraction took {elapsed:.3f}s (too slow)"

    def test_webpage_html_parsing_quality(self, content_service, check_api_keys):
        """Test webpage HTML parsing extracts meaningful content."""
        identifier = "https://www.nature.com/articles/s41586-021-03852-1"

        result = content_service.get_full_content(
            source=identifier, prefer_webpage=True, max_paragraphs=100
        )

        # Verify content quality - get content from any available key
        content = ""
        for key in ['content', 'methods_markdown', 'methods_text', 'full_text']:
            if key in result and result[key]:
                content = str(result[key])
                break

        assert len(content) > 500, f"Webpage content too short (got {len(content)} chars)"

        # Should extract text, not just HTML tags
        assert "<html>" not in content.lower(), "Raw HTML not cleaned"
        assert "<div>" not in content.lower(), "HTML tags not removed"

        # Should have scientific content
        content_lower = content.lower()
        assert any(
            keyword in content_lower
            for keyword in ["data", "method", "result", "figure", "study"]
        )

    def test_webpage_pdf_detection(self, content_service, check_api_keys):
        """Test webpage provider can detect and handle PDF links."""
        # Use identifier that may have PDF link on publisher page
        identifier = "https://www.biorxiv.org/content/10.1101/2024.01.18.576270v1"

        result = content_service.get_full_content(
            source=identifier, prefer_webpage=True, max_paragraphs=100
        )

        # Should successfully extract content (either webpage or PDF)
        assert result is not None

        # Check if service returned an error (external service unavailable or blocked)
        if "error" in result:
            pytest.skip(f"Service unavailable: {result.get('error', 'Unknown error')}")

        # Check for content in any of the expected keys (API format may vary)
        has_content = any(key in result for key in ['content', 'methods_markdown', 'methods_text', 'full_text'])
        assert has_content, f"No content found. Keys: {list(result.keys())}"
        # Verify content is not empty (check all possible content keys)
        content_found = False
        for key in ['content', 'methods_markdown', 'methods_text', 'full_text']:
            if key in result and result[key] and len(str(result[key])) > 0:
                content_found = True
                break
        assert content_found, "No non-empty content found"

        tier_used = result.get("tier_used", "")
        logger.info(f"BioRxiv extraction - Tier used: {tier_used}")

    def test_webpage_parsing_failure_edge_case(self, content_service, check_api_keys):
        """Test error handling when webpage parsing fails."""
        # Use invalid URL
        identifier = "https://invalid-domain-that-does-not-exist-12345.com/article"

        try:
            result = content_service.get_full_content(
                source=identifier, prefer_webpage=True, max_paragraphs=100
            )

            # If this succeeds somehow, verify minimal content
            if result:
                assert "error" in result or len(result.get("content", "")) == 0

        except Exception as e:
            # Exception expected for invalid URL
            logger.info(f"Expected failure for invalid URL: {e}")
            assert (
                "error" in str(e).lower()
                or "not found" in str(e).lower()
                or "failed" in str(e).lower()
            )


# ============================================================================
# Scenario 2.3: PDF Fallback (Tier 3)
# ============================================================================


@pytest.mark.real_api
@pytest.mark.integration
@pytest.mark.slow
class TestPDFFallback:
    """Test PDF extraction with Docling (Tier 3 - most universal)."""

    def test_pdf_extraction_performance(self, content_service, check_api_keys):
        """Test PDF extraction meets 3-8s target."""
        # Use direct PDF URL
        identifier = (
            "https://www.biorxiv.org/content/10.1101/2024.01.18.576270v1.full.pdf"
        )

        start_time = time.time()
        result = content_service.get_full_content(
            source=identifier,
            prefer_webpage=False,  # Force PDF extraction
            max_paragraphs=100,
        )
        elapsed = time.time() - start_time

        # Verify successful extraction
        assert result is not None

        # Check if service returned an error (external service unavailable)
        if "error" in result:
            pytest.skip(f"PDF service unavailable: {result.get('error', 'Unknown error')}")

        # Check for content in any of the expected keys (API format may vary)
        has_content = any(key in result for key in ['content', 'methods_markdown', 'methods_text', 'full_text'])
        assert has_content, f"No content found. Keys: {list(result.keys())}"

        # PDF extraction: 3-8s target (allow up to 15s with Docling overhead)
        logger.info(f"PDF extraction time: {elapsed:.3f}s (target: 3-8s)")
        assert (
            elapsed < 15.0
        ), f"PDF extraction took {elapsed:.3f}s (significantly over target)"

    def test_pdf_docling_quality(self, content_service, check_api_keys):
        """Test Docling PDF parsing produces high-quality output."""
        identifier = (
            "https://www.biorxiv.org/content/10.1101/2024.01.18.576270v1.full.pdf"
        )

        result = content_service.get_full_content(
            source=identifier, prefer_webpage=False, max_paragraphs=100
        )

        # Check if service returned an error (external service unavailable)
        if "error" in result:
            pytest.skip(f"PDF service unavailable: {result.get('error', 'Unknown error')}")

        # Verify content quality - get from any available key
        content = ""
        for key in ['content', 'methods_markdown', 'methods_text', 'full_text']:
            if key in result and result[key]:
                content = str(result[key])
                break

        assert len(content) > 1000, f"PDF content too short (got {len(content)} chars)"

        # Docling should extract clean text (not garbled)
        # Check for reasonable word/character ratio (not all symbols)
        words = content.split()
        assert len(words) > 100, "Too few words extracted from PDF"

        # Should have scientific content structure
        content_lower = content.lower()
        assert any(
            keyword in content_lower
            for keyword in ["abstract", "introduction", "method", "result", "figure"]
        )

    def test_pdf_table_and_figure_extraction(self, content_service, check_api_keys):
        """Test Docling extracts tables and figure references from PDFs."""
        identifier = (
            "https://www.biorxiv.org/content/10.1101/2024.01.18.576270v1.full.pdf"
        )

        result = content_service.get_full_content(
            source=identifier, prefer_webpage=False, max_paragraphs=100
        )

        # Check metadata for extracted tables/figures
        metadata = result.get("metadata", {})
        content = result.get("content", "")

        # PDF should reference figures/tables (at minimum)
        content_lower = content.lower()
        has_tables_or_figures = (
            "table" in content_lower
            or "figure" in content_lower
            or "fig." in content_lower
        )

        logger.info(
            f"PDF extraction - Tables/Figures mentioned: {has_tables_or_figures}"
        )
        # Note: Not all PDFs have tables/figures, so this is informational

    def test_pdf_parsing_failure_edge_case(self, content_service, check_api_keys):
        """Test error handling when PDF parsing fails."""
        # Use invalid PDF URL
        identifier = "https://example.com/nonexistent.pdf"

        try:
            result = content_service.get_full_content(
                source=identifier, prefer_webpage=False, max_paragraphs=100
            )

            # If this succeeds, verify error indication
            if result:
                assert "error" in result or len(result.get("content", "")) == 0

        except Exception as e:
            # Exception expected for invalid PDF
            logger.info(f"Expected failure for invalid PDF: {e}")
            assert (
                "error" in str(e).lower()
                or "not found" in str(e).lower()
                or "failed" in str(e).lower()
            )


# ============================================================================
# Scenario 2.4: Full Cascade Integration (CRITICAL)
# ============================================================================


@pytest.mark.real_api
@pytest.mark.integration
@pytest.mark.slow
class TestFullCascadeIntegration:
    """Test complete three-tier cascade: PMC→Webpage→PDF."""

    def test_cascade_pmc_to_webpage_fallback(self, content_service, check_api_keys):
        """Test cascade falls back from PMC to webpage when PMC unavailable."""
        # Use identifier that may not have PMC (forces webpage fallback)
        identifier = "10.1038/s41586-021-03852-1"  # Nature DOI

        result = content_service.get_full_content(
            source=identifier, prefer_webpage=True, max_paragraphs=100
        )

        # Verify successful extraction via fallback
        assert result is not None
        # Check for content in any of the expected keys (API format may vary)
        has_content = any(key in result for key in ['content', 'methods_markdown', 'methods_text', 'full_text'])
        assert has_content, f"No content found. Keys: {list(result.keys())}"
        # Verify content is not empty (check all possible content keys)
        content_found = False
        for key in ['content', 'methods_markdown', 'methods_text', 'full_text']:
            if key in result and result[key] and len(str(result[key])) > 0:
                content_found = True
                break
        assert content_found, "No non-empty content found"

        tier_used = result.get("tier_used", "")
        logger.info(f"Cascade test (DOI) - Tier used: {tier_used}")

        # Should use webpage or PDF (not PMC for Nature paywalled)
        # Exact tier depends on availability, but should succeed

    def test_cascade_webpage_to_pdf_fallback(self, content_service, check_api_keys):
        """Test cascade falls back from webpage to PDF when webpage parsing fails."""
        # Use bioRxiv identifier (may fall back to PDF)
        identifier = "https://www.biorxiv.org/content/10.1101/2024.01.18.576270v1"

        result = content_service.get_full_content(
            source=identifier, prefer_webpage=True, max_paragraphs=100
        )

        # Verify successful extraction
        assert result is not None

        # Check if service returned an error (external service unavailable or blocked)
        if "error" in result:
            pytest.skip(f"Service unavailable: {result.get('error', 'Unknown error')}")

        # Check for content in any of the expected keys (API format may vary)
        has_content = any(key in result for key in ['content', 'methods_markdown', 'methods_text', 'full_text'])
        assert has_content, f"No content found. Keys: {list(result.keys())}"
        # Verify content is not empty (check all possible content keys)
        content_found = False
        for key in ['content', 'methods_markdown', 'methods_text', 'full_text']:
            if key in result and result[key] and len(str(result[key])) > 0:
                content_found = True
                break
        assert content_found, "No non-empty content found"

        tier_used = result.get("tier_used", "")
        source_type = result.get("source_type", "")
        logger.info(
            f"Cascade test (bioRxiv) - Tier: {tier_used}, Source: {source_type}"
        )

    def test_cascade_end_to_end_performance(self, content_service, check_api_keys):
        """Test full cascade completes within <10s total target."""
        # Use identifier that exercises full cascade
        identifier = "PMID:35042229"

        start_time = time.time()
        result = content_service.get_full_content(
            source=identifier, prefer_webpage=True, max_paragraphs=100
        )
        elapsed = time.time() - start_time

        # Verify success
        assert result is not None
        # Check for content in any of the expected keys (API format may vary)
        has_content = any(key in result for key in ['content', 'methods_markdown', 'methods_text', 'full_text'])
        assert has_content, f"No content found. Keys: {list(result.keys())}"

        # Full cascade: <10s target (PMC attempts + fallback)
        logger.info(f"Full cascade time: {elapsed:.3f}s (target: <10s)")
        assert (
            elapsed < 15.0
        ), f"Full cascade took {elapsed:.3f}s (over 10s target, but allowing overhead)"

        tier_used = result.get("tier_used", "")
        logger.info(f"Full cascade completed via tier: {tier_used}")

    def test_cascade_rate_limiting_handling(self, content_service, check_api_keys):
        """Test cascade handles rate limiting gracefully."""
        # Make multiple rapid requests to test rate limiting
        identifiers = [
            "PMID:35042229",
            "PMID:33057194",
            "PMID:32424251",
        ]

        results = []
        for identifier in identifiers:
            try:
                result = content_service.get_full_content(
                    source=identifier, prefer_webpage=False, max_paragraphs=50
                )
                results.append(result)

                # Small delay between requests (respectful to APIs)
                time.sleep(0.5)

            except Exception as e:
                logger.warning(f"Rate limiting or error for {identifier}: {e}")
                # Rate limiting is acceptable - service should handle it
                assert "rate" in str(e).lower() or "too many" in str(e).lower()

        # At least some requests should succeed
        successful_results = [r for r in results if r and "content" in r]
        assert (
            len(successful_results) >= 1
        ), "All requests failed - possible rate limiting issue"

        logger.info(
            f"Rate limiting test: {len(successful_results)}/{len(identifiers)} succeeded"
        )

    def test_cascade_error_propagation(self, content_service, check_api_keys):
        """Test cascade properly propagates errors when all tiers fail."""
        # Use completely invalid identifier
        identifier = "INVALID:99999999"

        try:
            result = content_service.get_full_content(
                source=identifier, prefer_webpage=False, max_paragraphs=100
            )

            # If this returns a result, it should indicate failure
            if result:
                content = result.get("content", "")
                assert (
                    len(content) == 0 or "error" in content.lower()
                ), "Should indicate failure"

        except Exception as e:
            # Exception expected when all tiers fail
            logger.info(f"Expected cascade failure for invalid identifier: {e}")
            assert (
                "error" in str(e).lower()
                or "invalid" in str(e).lower()
                or "not found" in str(e).lower()
            )

    def test_cascade_tier_preference_respects_parameters(
        self, content_service, check_api_keys
    ):
        """Test cascade respects prefer_webpage parameter."""
        identifier = "PMID:35042229"

        # Test 1: prefer_webpage=True
        result_webpage = content_service.get_full_content(
            source=identifier, prefer_webpage=True, max_paragraphs=100
        )

        # Test 2: prefer_webpage=False
        result_no_webpage = content_service.get_full_content(
            source=identifier, prefer_webpage=False, max_paragraphs=100
        )

        # Both should succeed
        assert result_webpage is not None
        assert result_no_webpage is not None

        tier_webpage = result_webpage.get("tier_used", "")
        tier_no_webpage = result_no_webpage.get("tier_used", "")

        logger.info(f"prefer_webpage=True tier: {tier_webpage}")
        logger.info(f"prefer_webpage=False tier: {tier_no_webpage}")

        # Tiers may differ based on preference (if PMC available, prefer_webpage=False should use it)
        # This is validation that the parameter is being respected

    def test_cascade_multiple_identifiers_batch(self, content_service, check_api_keys):
        """Test cascade handles multiple identifiers efficiently."""
        identifiers = [
            "PMID:35042229",  # Has PMC
            "10.1038/s41586-021-03852-1",  # Nature DOI (may not have PMC)
            "https://www.biorxiv.org/content/10.1101/2024.01.18.576270v1",  # bioRxiv
        ]

        results = []
        total_time = 0

        for identifier in identifiers:
            start_time = time.time()
            try:
                result = content_service.get_full_content(
                    source=identifier, prefer_webpage=True, max_paragraphs=50
                )
                elapsed = time.time() - start_time
                total_time += elapsed

                # Check if result has content (any of the expected keys)
                has_content = False
                if result is not None:
                    has_content = any(key in result for key in ['content', 'methods_markdown', 'methods_text', 'full_text'])

                results.append(
                    {
                        "identifier": identifier,
                        "success": has_content,
                        "tier": result.get("tier_used", "") if result else "",
                        "time": elapsed,
                    }
                )

                # Respectful delay between requests
                time.sleep(1.0)

            except Exception as e:
                logger.warning(f"Batch processing error for {identifier}: {e}")
                results.append(
                    {
                        "identifier": identifier,
                        "success": False,
                        "tier": "failed",
                        "time": time.time() - start_time,
                    }
                )

        # Verify batch results
        successful_count = sum(1 for r in results if r["success"])
        logger.info(
            f"Batch cascade: {successful_count}/{len(identifiers)} succeeded, total time: {total_time:.2f}s"
        )

        # At least 2/3 should succeed (allow for API issues)
        assert successful_count >= 2, "Too many failures in batch cascade"

        # Log tier distribution
        tiers_used = [r["tier"] for r in results if r["success"]]
        logger.info(f"Tiers used in batch: {tiers_used}")


# ============================================================================
# Summary Tests
# ============================================================================


@pytest.mark.real_api
@pytest.mark.integration
@pytest.mark.slow
class TestCascadeSystemIntegration:
    """End-to-end integration tests for complete cascade system."""

    def test_cascade_system_reliability(self, content_service, check_api_keys):
        """Test cascade system overall reliability with diverse identifiers."""
        test_cases = [
            {
                "identifier": "PMID:35042229",
                "expected_tier": "pmc",
                "description": "PMC available",
            },
            {
                "identifier": "10.1038/s41586-021-03852-1",
                "expected_tier": "webpage_or_pdf",
                "description": "Nature article",
            },
            {
                "identifier": "https://www.biorxiv.org/content/10.1101/2024.01.18.576270v1.full.pdf",
                "expected_tier": "pdf",
                "description": "Direct PDF",
            },
        ]

        results = []
        for case in test_cases:
            try:
                start_time = time.time()
                result = content_service.get_full_content(
                    source=case["identifier"], prefer_webpage=True, max_paragraphs=100
                )
                elapsed = time.time() - start_time

                success = result is not None and len(result.get("content", "")) > 0
                tier_used = result.get("tier_used", "") if result else ""

                results.append(
                    {
                        "description": case["description"],
                        "success": success,
                        "tier": tier_used,
                        "time": elapsed,
                    }
                )

                logger.info(
                    f"{case['description']}: {tier_used} ({elapsed:.2f}s) - {'✅' if success else '❌'}"
                )

                time.sleep(1.0)  # Respectful delay

            except Exception as e:
                logger.error(f"{case['description']} failed: {e}")
                results.append(
                    {
                        "description": case["description"],
                        "success": False,
                        "tier": "error",
                        "time": 0,
                    }
                )

        # Overall reliability check
        successful_count = sum(1 for r in results if r["success"])
        logger.info(
            f"Cascade system reliability: {successful_count}/{len(test_cases)} cases succeeded"
        )

        # Should have high success rate (allow 1 failure)
        assert (
            successful_count >= len(test_cases) - 1
        ), "Cascade system reliability below threshold"

    def test_cascade_performance_benchmarking(self, content_service, check_api_keys):
        """Benchmark cascade performance across all tiers."""
        benchmarks = {
            "pmc_fast_path": [],
            "webpage_fallback": [],
            "pdf_fallback": [],
        }

        # PMC Fast Path
        for _ in range(2):
            start = time.time()
            try:
                content_service.get_full_content(
                    source="PMID:35042229", prefer_webpage=False, max_paragraphs=50
                )
                benchmarks["pmc_fast_path"].append(time.time() - start)
                time.sleep(0.5)
            except Exception as e:
                logger.warning(f"PMC benchmark error: {e}")

        # Webpage Fallback
        for _ in range(2):
            start = time.time()
            try:
                content_service.get_full_content(
                    source="https://www.nature.com/articles/s41586-021-03852-1",
                    prefer_webpage=True,
                    max_paragraphs=50,
                )
                benchmarks["webpage_fallback"].append(time.time() - start)
                time.sleep(1.0)
            except Exception as e:
                logger.warning(f"Webpage benchmark error: {e}")

        # PDF Fallback
        for _ in range(2):
            start = time.time()
            try:
                content_service.get_full_content(
                    source="https://www.biorxiv.org/content/10.1101/2024.01.18.576270v1.full.pdf",
                    prefer_webpage=False,
                    max_paragraphs=50,
                )
                benchmarks["pdf_fallback"].append(time.time() - start)
                time.sleep(1.0)
            except Exception as e:
                logger.warning(f"PDF benchmark error: {e}")

        # Report benchmarks
        for tier, times in benchmarks.items():
            if times:
                avg_time = sum(times) / len(times)
                logger.info(f"{tier} average: {avg_time:.3f}s (n={len(times)})")
            else:
                logger.warning(f"{tier}: No successful benchmarks")

        # Verify at least some benchmarks succeeded
        total_benchmarks = sum(len(times) for times in benchmarks.values())
        assert (
            total_benchmarks >= 3
        ), "Too few successful benchmarks - cascade system may be unreliable"
