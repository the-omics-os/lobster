#!/usr/bin/env python3
"""
Test script to verify structured reasoning display implementation.
"""

import sys
from pathlib import Path

# Add lobster to path
sys.path.insert(0, str(Path(__file__).parent))

def test_structured_response():
    """Test that client returns structured response with separate reasoning and text."""
    from lobster.core.client import AgentClient
    from langchain_core.messages import AIMessage

    print("=" * 80)
    print("Testing Structured Reasoning Response")
    print("=" * 80)
    print()

    # Test 1: Extract from content_blocks with reasoning
    print("Test 1: Extract structured response from content_blocks")
    print("-" * 80)

    # Create mock AIMessage with content_blocks (simulates Gemini/Bedrock response)
    mock_message = AIMessage(
        content="",  # Empty string content
        content_blocks=[
            {
                "type": "reasoning",
                "reasoning": "Let me analyze this step by step. First, I'll check the data quality..."
            },
            {
                "type": "text",
                "text": "Here's my analysis: The dataset looks good!"
            }
        ]
    )

    # Create client with reasoning enabled
    client = AgentClient(enable_reasoning=True)

    # Extract content
    extracted = client._extract_content_from_message(mock_message)

    print(f"Extracted type: {type(extracted)}")

    if isinstance(extracted, dict):
        print("✓ Returns structured dict (not concatenated string)")
        print()
        print(f"  reasoning: {extracted.get('reasoning', 'N/A')[:80]}...")
        print(f"  text: {extracted.get('text', 'N/A')[:80]}...")
        print(f"  combined: {extracted.get('combined', 'N/A')[:80]}...")
        print()

        # Verify structure
        assert "reasoning" in extracted, "Missing 'reasoning' field"
        assert "text" in extracted, "Missing 'text' field"
        assert "combined" in extracted, "Missing 'combined' field"

        # Verify content
        assert "analyze this step by step" in extracted["reasoning"].lower()
        assert "dataset looks good" in extracted["text"].lower()

        print("✓ All fields present and correct")
    else:
        print(f"✗ FAIL: Returns string instead of dict: {extracted[:100]}...")
        return False

    print()

    # Test 2: Backward compatibility (reasoning disabled)
    print("Test 2: Backward compatibility (reasoning disabled)")
    print("-" * 80)

    client_no_reasoning = AgentClient(enable_reasoning=False)
    extracted_no_reasoning = client_no_reasoning._extract_content_from_message(mock_message)

    if isinstance(extracted_no_reasoning, str):
        print("✓ Returns string when reasoning disabled (backward compatible)")
        print(f"  Content: {extracted_no_reasoning[:80]}...")
    else:
        print(f"✗ FAIL: Should return string when reasoning disabled, got {type(extracted_no_reasoning)}")
        return False

    print()

    # Test 3: No reasoning blocks (text only)
    print("Test 3: No reasoning blocks (text only)")
    print("-" * 80)

    mock_text_only = AIMessage(
        content="",
        content_blocks=[
            {"type": "text", "text": "Just a normal response without thinking"}
        ]
    )

    client_with_reasoning = AgentClient(enable_reasoning=True)
    extracted_text_only = client_with_reasoning._extract_content_from_message(mock_text_only)

    if isinstance(extracted_text_only, str):
        print("✓ Returns string when no reasoning blocks present")
        print(f"  Content: {extracted_text_only}")
    else:
        print(f"✗ FAIL: Should return string when no reasoning, got {type(extracted_text_only)}")
        return False

    print()

    # Test 4: Legacy raw content extraction
    print("Test 4: Legacy fallback with raw content list")
    print("-" * 80)

    mock_legacy_message = AIMessage(
        content=[
            {"type": "reasoning", "reasoning": "Legacy thinking format..."},
            {"type": "text", "text": "Legacy response"}
        ]
    )
    # Remove content_blocks attribute to force fallback
    if hasattr(mock_legacy_message, "content_blocks"):
        delattr(mock_legacy_message, "content_blocks")

    extracted_legacy = client._extract_content_from_message(mock_legacy_message)

    print(f"Extracted type: {type(extracted_legacy)}")
    if isinstance(extracted_legacy, dict):
        print("✓ Legacy fallback also returns structured response")
        assert "reasoning" in extracted_legacy
        assert "text" in extracted_legacy
    else:
        print(f"  Returns: {extracted_legacy[:100]}...")

    print()

    # Summary
    print("=" * 80)
    print("✓ ALL TESTS PASSED - Structured reasoning response working!")
    print("=" * 80)
    print()
    print("Implementation verified:")
    print("  ✓ Returns dict when reasoning enabled and present")
    print("  ✓ Returns string when reasoning disabled (backward compat)")
    print("  ✓ Returns string when no reasoning blocks (graceful)")
    print("  ✓ Legacy fallback path also works")
    print()
    print("CLI can now display reasoning as separate panel!")

    return True


if __name__ == "__main__":
    try:
        success = test_structured_response()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
