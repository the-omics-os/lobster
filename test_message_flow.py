"""
Test to verify what messages are returned by create_react_agent.

This test will:
1. Create a simple agent with a tool
2. Invoke it with a task that requires tool calling
3. Examine the full result object to see if it contains all messages or just the final one
"""

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_aws import ChatBedrock
import json
import os


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    print(f"[TOOL CALL] multiply({a}, {b}) = {a * b}")
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    print(f"[TOOL CALL] add({a}, {b}) = {a + b}")
    return a + b


def main():
    # Create a simple agent with tools (using Bedrock)
    model = ChatBedrock(
        model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
        region_name="us-east-1",
    )
    tools = [multiply, add]

    agent = create_react_agent(model, tools)

    # Invoke with a task that requires multiple tool calls
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Calculate (5 * 3) + (2 * 4)"}]}
    )

    print("=" * 80)
    print("FULL RESULT STRUCTURE:")
    print("=" * 80)
    print(f"Result keys: {result.keys()}")
    print(f"\nNumber of messages: {len(result.get('messages', []))}")
    print("\n" + "=" * 80)
    print("MESSAGE-BY-MESSAGE BREAKDOWN:")
    print("=" * 80)

    for i, msg in enumerate(result.get("messages", [])):
        print(f"\n--- Message {i} ---")
        print(f"Type: {type(msg).__name__}")
        if hasattr(msg, "type"):
            print(f"msg.type: {msg.type}")

        # Check for tool calls
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print(f"Tool calls: {len(msg.tool_calls)}")
            for tc in msg.tool_calls:
                print(f"  - {tc.get('name', 'unknown')}: {tc.get('args', {})}")

        # Check content
        if hasattr(msg, "content"):
            content_preview = str(msg.content)[:200]
            print(f"Content: {content_preview}")
            if len(str(msg.content)) > 200:
                print(f"... (truncated, total length: {len(str(msg.content))})")

        # Check for tool response
        if hasattr(msg, "name"):
            print(f"Tool name: {msg.name}")
        if hasattr(msg, "artifact"):
            print(f"Artifact: {msg.artifact}")

    print("\n" + "=" * 80)
    print("CRITICAL FINDING:")
    print("=" * 80)

    # Check what the supervisor sees
    final_msg = result.get("messages", [])[-1] if result.get("messages") else None
    if final_msg:
        content = final_msg.content if hasattr(final_msg, "content") else str(final_msg)
        print(f"What supervisor sees (final_msg.content):")
        print(f"  Length: {len(content)} characters")
        print(f"  Content: {content}")

    print("\n" + "=" * 80)
    print("HYPOTHESIS TEST:")
    print("=" * 80)

    # Calculate total message content if ALL messages were passed
    total_chars = 0
    for msg in result.get("messages", []):
        if hasattr(msg, "content"):
            total_chars += len(str(msg.content))
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            total_chars += len(str(msg.tool_calls))

    final_msg_chars = len(content) if final_msg else 0

    print(f"Final message only: {final_msg_chars} characters")
    print(f"All messages combined: {total_chars} characters")
    print(f"Multiplier if full history passed: {total_chars / final_msg_chars if final_msg_chars > 0 else 0:.1f}x")

    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print("=" * 80)

    if len(result.get("messages", [])) > 2:
        print("✓ create_react_agent DOES return full message history")
        print("✓ Current code extracts only [-1], so supervisor sees ONLY final message")
        print("✓ No context overflow from intermediate tool calls")
        print("\nHypothesis: REJECTED")
        print("The gibberish bug is NOT caused by intermediate tool call leakage.")
    else:
        print("✗ Only 2 messages in result (unexpected)")


if __name__ == "__main__":
    main()
