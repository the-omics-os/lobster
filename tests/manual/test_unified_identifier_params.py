#!/usr/bin/env python3
"""
Manual integration test for unified identifier parameter architecture.

Tests that the renamed parameters (identifier) work correctly in the tools.
Run this after implementing the unified_id_plan.md changes.

Usage:
    python tests/manual/test_unified_identifier_params.py
"""

import sys
from pathlib import Path

# Add lobster to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.agents.research_agent import research_agent


def test_validate_dataset_metadata_signature():
    """Test that validate_dataset_metadata accepts 'identifier' parameter."""
    print("\n" + "=" * 80)
    print("TEST 1: validate_dataset_metadata parameter signature")
    print("=" * 80)

    # Create minimal data manager
    workspace = Path("/tmp/test_unified_id")
    workspace.mkdir(exist_ok=True)
    dm = DataManagerV2(workspace)

    # Create agent
    agent = research_agent(dm)

    # Get the validate_dataset_metadata tool
    tools = agent.tools
    validate_tool = None
    for tool in tools:
        if tool.name == "validate_dataset_metadata":
            validate_tool = tool
            break

    if not validate_tool:
        print("✗ FAILED: validate_dataset_metadata tool not found")
        return False

    # Check parameter signature using tool's args_schema
    args_schema = validate_tool.args_schema
    if args_schema:
        fields = args_schema.model_fields if hasattr(args_schema, 'model_fields') else {}

        print(f"\nTool name: {validate_tool.name}")
        print(f"Parameters: {list(fields.keys())}")

        if 'identifier' in fields:
            print("✓ PASSED: 'identifier' parameter found")
            has_identifier = True
        else:
            print("✗ FAILED: 'identifier' parameter NOT found")
            has_identifier = False

        if 'accession' in fields:
            print("✗ FAILED: Old 'accession' parameter still exists!")
            has_old_param = True
        else:
            print("✓ PASSED: Old 'accession' parameter removed")
            has_old_param = False

        return has_identifier and not has_old_param
    else:
        print("✗ WARNING: Could not inspect args_schema")
        return False


def test_extract_methods_signature():
    """Test that extract_methods accepts 'identifier' parameter."""
    print("\n" + "=" * 80)
    print("TEST 2: extract_methods parameter signature")
    print("=" * 80)

    # Create minimal data manager
    workspace = Path("/tmp/test_unified_id")
    workspace.mkdir(exist_ok=True)
    dm = DataManagerV2(workspace)

    # Create agent
    agent = research_agent(dm)

    # Get the extract_methods tool
    tools = agent.tools
    extract_tool = None
    for tool in tools:
        if tool.name == "extract_methods":
            extract_tool = tool
            break

    if not extract_tool:
        print("✗ FAILED: extract_methods tool not found")
        return False

    # Check parameter signature
    args_schema = extract_tool.args_schema
    if args_schema:
        fields = args_schema.model_fields if hasattr(args_schema, 'model_fields') else {}

        print(f"\nTool name: {extract_tool.name}")
        print(f"Parameters: {list(fields.keys())}")

        if 'identifier' in fields:
            print("✓ PASSED: 'identifier' parameter found")
            has_identifier = True
        else:
            print("✗ FAILED: 'identifier' parameter NOT found")
            has_identifier = False

        if 'url_or_pmid' in fields:
            print("✗ FAILED: Old 'url_or_pmid' parameter still exists!")
            has_old_param = True
        else:
            print("✓ PASSED: Old 'url_or_pmid' parameter removed")
            has_old_param = False

        return has_identifier and not has_old_param
    else:
        print("✗ WARNING: Could not inspect args_schema")
        return False


def test_consistent_naming_across_tools():
    """Test that all external-identifier tools use 'identifier' parameter."""
    print("\n" + "=" * 80)
    print("TEST 3: Consistent naming across all external-identifier tools")
    print("=" * 80)

    # Create minimal data manager
    workspace = Path("/tmp/test_unified_id")
    workspace.mkdir(exist_ok=True)
    dm = DataManagerV2(workspace)

    # Create agent
    agent = research_agent(dm)
    tools = agent.tools

    # Expected tools with 'identifier' parameter
    expected_identifier_tools = [
        "find_related_entries",
        "get_dataset_metadata",
        "fast_abstract_search",
        "read_full_publication",
        "validate_dataset_metadata",
        "extract_methods",
    ]

    # Expected tools with 'entry_id' parameter (internal queue IDs)
    expected_entry_id_tools = [
        "process_publication_entry",
    ]

    print(f"\nChecking {len(expected_identifier_tools)} external-identifier tools...")

    all_correct = True
    for tool_name in expected_identifier_tools:
        tool = None
        for t in tools:
            if t.name == tool_name:
                tool = t
                break

        if not tool:
            print(f"  ✗ {tool_name}: NOT FOUND")
            all_correct = False
            continue

        args_schema = tool.args_schema
        if args_schema:
            fields = args_schema.model_fields if hasattr(args_schema, 'model_fields') else {}

            if 'identifier' in fields:
                print(f"  ✓ {tool_name}: uses 'identifier'")
            else:
                print(f"  ✗ {tool_name}: does NOT use 'identifier' (params: {list(fields.keys())})")
                all_correct = False
        else:
            print(f"  ⚠ {tool_name}: Cannot inspect schema")

    print(f"\nChecking {len(expected_entry_id_tools)} internal-queue-ID tools...")
    for tool_name in expected_entry_id_tools:
        tool = None
        for t in tools:
            if t.name == tool_name:
                tool = t
                break

        if not tool:
            print(f"  ✗ {tool_name}: NOT FOUND")
            all_correct = False
            continue

        args_schema = tool.args_schema
        if args_schema:
            fields = args_schema.model_fields if hasattr(args_schema, 'model_fields') else {}

            if 'entry_id' in fields:
                print(f"  ✓ {tool_name}: uses 'entry_id'")
            else:
                print(f"  ✗ {tool_name}: does NOT use 'entry_id' (params: {list(fields.keys())})")
                all_correct = False

    return all_correct


def main():
    """Run all manual tests."""
    print("=" * 80)
    print("UNIFIED IDENTIFIER PARAMETER - INTEGRATION TESTS")
    print("=" * 80)
    print("\nThis test verifies that parameter renames from unified_id_plan.md")
    print("have been correctly applied to research_agent tools.")
    print()

    results = []

    # Test 1
    try:
        result = test_validate_dataset_metadata_signature()
        results.append(("validate_dataset_metadata signature", result))
    except Exception as e:
        print(f"✗ Test 1 EXCEPTION: {e}")
        results.append(("validate_dataset_metadata signature", False))

    # Test 2
    try:
        result = test_extract_methods_signature()
        results.append(("extract_methods signature", result))
    except Exception as e:
        print(f"✗ Test 2 EXCEPTION: {e}")
        results.append(("extract_methods signature", False))

    # Test 3
    try:
        result = test_consistent_naming_across_tools()
        results.append(("Consistent naming across tools", result))
    except Exception as e:
        print(f"✗ Test 3 EXCEPTION: {e}")
        results.append(("Consistent naming across tools", False))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed ({100 * passed / total:.1f}%)")

    if passed == total:
        print("\n✅ ALL TESTS PASSED - Implementation verified!")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED - Review implementation")
        return 1


if __name__ == "__main__":
    sys.exit(main())
