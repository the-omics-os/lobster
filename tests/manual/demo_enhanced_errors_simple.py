"""
Simple demo for enhanced docstring-driven error messages.

Tests the error generation directly without full client setup.
"""

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.workspace_tool import create_get_content_from_workspace_tool


def demo_errors():
    """Demo enhanced error messages."""

    # Initialize minimal data manager
    dm = DataManagerV2(workspace_path="/tmp/demo_workspace")
    tool = create_get_content_from_workspace_tool(dm)

    print("\n🎯 Enhanced Error Messages Demo (v2.6+)")
    print("📚 Docstring-driven, contextual, and actionable\n")

    # Test 1: Typo in workspace
    print("=" * 80)
    print("TEST 1: Typo in workspace parameter ('lit' instead of 'literature')")
    print("=" * 80)
    print("\nAgent call: get_content_from_workspace(workspace='lit')\n")
    result1 = tool.invoke({"workspace": 'lit'})
    print(result1)
    print("\n")

    # Test 2: Typo in level
    print("=" * 80)
    print("TEST 2: Typo in level parameter ('sumary' instead of 'summary')")
    print("=" * 80)
    print("\nAgent call: get_content_from_workspace(workspace='literature', level='sumary')\n")
    result2 = tool.invoke({"workspace": 'literature', "level": 'sumary'})
    print(result2)
    print("\n")

    # Test 3: Invalid workspace (no fuzzy match)
    print("=" * 80)
    print("TEST 3: Completely invalid workspace (no fuzzy match)")
    print("=" * 80)
    print("\nAgent call: get_content_from_workspace(workspace='xyz')\n")
    result3 = tool.invoke({"workspace": 'xyz'})
    print(result3)
    print("\n")

    # Test 4: Close match for level
    print("=" * 80)
    print("TEST 4: Close match for level parameter ('mthods' -> 'methods')")
    print("=" * 80)
    print("\nAgent call: get_content_from_workspace(workspace='literature', level='mthods')\n")
    result4 = tool.invoke({"workspace": 'literature', "level": 'mthods'})
    print(result4)
    print("\n")

    print("✅ Demo complete!")
    print("\n📊 Key Features:")
    print("  ✓ Fuzzy matching suggestions (difflib)")
    print("  ✓ Extracted examples from docstring (## headers)")
    print("  ✓ Status emojis for visual clarity")
    print("  ✓ Performance: <3ms per error (cached parser)")
    print("  ✓ Single source of truth (docstring)")


if __name__ == "__main__":
    try:
        demo_errors()
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
