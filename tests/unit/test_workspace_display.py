"""
Unit tests for workspace display functionality.

Tests the intelligent truncation and display logic introduced in Phase 1
of the workspace display enhancement project.
"""

import pytest


def truncate_middle(text: str, max_length: int = 60) -> str:
    """
    Truncate text in the middle with ellipsis, preserving start and end.

    This is a copy of the function from cli.py for testing purposes.
    """
    if len(text) <= max_length:
        return text

    # Calculate how much to keep on each side
    # Reserve 3 characters for "..."
    available_chars = max_length - 3
    start_length = (available_chars + 1) // 2  # Slightly prefer start
    end_length = available_chars // 2

    return f"{text[:start_length]}...{text[-end_length:]}"


class TestTruncateMiddle:
    """Test suite for intelligent middle truncation function."""

    def test_no_truncation_short_string(self):
        """Test that short strings are not truncated."""
        text = "geo_gse12345"
        result = truncate_middle(text, max_length=60)
        assert result == text
        assert len(result) == len(text)

    def test_no_truncation_exact_length(self):
        """Test that strings exactly at max_length are not truncated."""
        text = "a" * 60
        result = truncate_middle(text, max_length=60)
        assert result == text
        assert len(result) == 60

    def test_truncation_long_string(self):
        """Test that long strings are truncated correctly."""
        text = "geo_gse12345_quality_assessed_filtered_normalized_doublets_detected_clustered_markers_annotated_pseudobulk"
        result = truncate_middle(text, max_length=60)

        # Should be exactly 60 characters
        assert len(result) == 60

        # Should contain ellipsis
        assert "..." in result

        # Should preserve start and end
        assert result.startswith("geo_gse12345")
        assert result.endswith("pseudobulk")

    def test_truncation_preserves_key_information(self):
        """Test that truncation preserves dataset origin and processing stage."""
        text = "geo_gse12345_quality_assessed_filtered_normalized_doublets_detected_clustered_markers_annotated_pseudobulk"
        result = truncate_middle(text, max_length=60)

        # Should preserve dataset origin (start)
        assert "geo_gse12345" in result

        # Should preserve processing stage (end)
        assert "pseudobulk" in result

    def test_truncation_character_distribution(self):
        """Test that start and end portions have correct character counts."""
        text = "a" * 100  # Long uniform string
        result = truncate_middle(text, max_length=60)

        # Should be 60 chars total
        assert len(result) == 60

        # Calculate expected distribution:
        # available_chars = 60 - 3 = 57
        # start_length = (57 + 1) // 2 = 58 // 2 = 29
        # end_length = 57 // 2 = 28
        # Total: 29 + 3 + 28 = 60
        parts = result.split("...")
        assert len(parts) == 2
        assert len(parts[0]) == 29
        assert len(parts[1]) == 28

    def test_custom_max_length(self):
        """Test truncation with custom max_length parameter."""
        text = "geo_gse12345_quality_assessed_filtered_normalized"
        result = truncate_middle(text, max_length=40)

        assert len(result) == 40
        assert "..." in result
        assert result.startswith("geo_gse12345")
        assert result.endswith("normalized")

    def test_empty_string(self):
        """Test handling of empty string."""
        result = truncate_middle("", max_length=60)
        assert result == ""

    def test_very_short_max_length(self):
        """Test truncation with very short max_length."""
        text = "geo_gse12345_filtered"
        result = truncate_middle(text, max_length=10)

        # 10 - 3 = 7 available
        # start: (7+1)//2 = 4
        # end: 7//2 = 3
        # Total: 4 + 3 + 3 = 10
        assert len(result) == 10
        assert "..." in result

    def test_unicode_handling(self):
        """Test truncation with unicode characters."""
        text = "data_α_β_γ_" + "x" * 100
        result = truncate_middle(text, max_length=60)

        assert len(result) == 60
        assert "..." in result
        # Should preserve unicode characters if they're in preserved portions
        assert "data_α_β_γ_" in result or result.endswith("x" * 28)

    def test_real_world_dataset_names(self):
        """Test with realistic bioinformatics dataset names."""
        test_cases = [
            # Short name - no truncation
            ("geo_gse12345", 60, False),
            # Medium name - no truncation
            ("geo_gse12345_quality_assessed_filtered", 60, False),
            # Long name - truncated
            (
                "geo_gse12345_quality_assessed_filtered_normalized_doublets_detected_clustered",
                60,
                True,
            ),
            # Very long name - truncated
            (
                "geo_gse12345_quality_assessed_filtered_normalized_doublets_detected_clustered_markers_annotated_pseudobulk",
                60,
                True,
            ),
            # Bulk RNA-seq name - truncated
            (
                "bulk_rna_seq_project_alpha_experimental_condition_treatment_vs_control_filtered_normalized_de_analyzed",
                60,
                True,
            ),
        ]

        for name, max_len, should_truncate in test_cases:
            result = truncate_middle(name, max_length=max_len)

            # Check length constraint
            assert len(result) <= max_len

            # Check truncation expectation
            if should_truncate:
                assert "..." in result
                assert len(result) == max_len
            else:
                assert "..." not in result
                assert result == name

    def test_underscores_preserved(self):
        """Test that underscore separators are handled correctly."""
        text = "geo_gse12345_quality_assessed_filtered_normalized_doublets_detected_clustered_markers"
        result = truncate_middle(text, max_length=60)

        # Should still contain underscores
        assert "_" in result

        # Splitting by ellipsis should give valid parts
        parts = result.split("...")
        assert len(parts) == 2
        assert "_" in parts[0]
        assert "_" in parts[1]


class TestWorkspaceDisplayIntegration:
    """Integration tests for workspace display functionality."""

    def test_truncated_names_sortable(self):
        """Test that truncated names maintain sortability."""
        names = [
            "geo_gse12345_quality_assessed",
            "geo_gse12345_quality_assessed_filtered",
            "geo_gse12345_quality_assessed_filtered_normalized",
        ]

        truncated = [truncate_middle(name, 40) for name in names]

        # All should start with "geo_gse12345"
        for t in truncated:
            assert t.startswith("geo_gse12345")

    def test_truncation_consistency(self):
        """Test that truncation is consistent for same input."""
        text = "geo_gse12345_quality_assessed_filtered_normalized_doublets_detected_clustered_markers_annotated"

        result1 = truncate_middle(text, max_length=60)
        result2 = truncate_middle(text, max_length=60)

        assert result1 == result2

    def test_different_lengths_different_results(self):
        """Test that different max_length values produce different results."""
        text = "geo_gse12345_quality_assessed_filtered_normalized_doublets_detected_clustered_markers"

        result40 = truncate_middle(text, max_length=40)
        result60 = truncate_middle(text, max_length=60)

        assert len(result40) == 40
        assert len(result60) == 60
        assert result40 != result60

        # Shorter truncation should show less information
        assert len(result40) < len(result60)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
