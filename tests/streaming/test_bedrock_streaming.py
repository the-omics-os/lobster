"""
Bedrock Streaming Evaluation Script

Tests whether langchain-aws ChatBedrockConverse actually streams tokens
from AWS Bedrock Claude models (Sonnet 4.5, Sonnet 4, Haiku 4.5).

Usage:
    python tests/streaming/test_bedrock_streaming.py
    python tests/streaming/test_bedrock_streaming.py --model sonnet4
    python tests/streaming/test_bedrock_streaming.py --all-models
    python tests/streaming/test_bedrock_streaming.py --async

Requires:
    AWS_BEDROCK_ACCESS_KEY and AWS_BEDROCK_SECRET_ACCESS_KEY env vars
"""

import argparse
import asyncio
import os
import sys
import time
from typing import Any

# ---------------------------------------------------------------------------
# Model catalog
# ---------------------------------------------------------------------------
MODELS = {
    "sonnet4.5": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "sonnet4": "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "opus4.5": "global.anthropic.claude-opus-4-5-20251101-v1:0",
}

DEFAULT_MODEL = "sonnet4"

PROMPT = "Explain in 3 sentences what DNA polymerase does during replication."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _check_credentials() -> bool:
    key = os.environ.get("AWS_BEDROCK_ACCESS_KEY")
    secret = os.environ.get("AWS_BEDROCK_SECRET_ACCESS_KEY")
    if not key or not secret:
        print("ERROR: Missing AWS_BEDROCK_ACCESS_KEY or AWS_BEDROCK_SECRET_ACCESS_KEY")
        print("Set these environment variables before running.")
        return False
    return True


def _create_model(model_id: str, **kwargs: Any):
    from langchain_aws import ChatBedrockConverse

    region = os.environ.get("AWS_REGION", "us-east-1")
    return ChatBedrockConverse(
        model_id=model_id,
        region_name=region,
        aws_access_key_id=os.environ["AWS_BEDROCK_ACCESS_KEY"],
        aws_secret_access_key=os.environ["AWS_BEDROCK_SECRET_ACCESS_KEY"],
        temperature=0.3,
        **kwargs,
    )


def _separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Test 1: Sync .stream() — token-by-token
# ---------------------------------------------------------------------------
def test_sync_stream(model_id: str):
    _separator(f"SYNC .stream()  |  {model_id}")

    llm = _create_model(model_id)

    chunk_count = 0
    total_text = ""
    first_token_time = None
    t0 = time.perf_counter()

    try:
        for chunk in llm.stream(PROMPT):
            now = time.perf_counter()
            if first_token_time is None:
                first_token_time = now - t0

            content = chunk.content
            if isinstance(content, str) and content:
                chunk_count += 1
                total_text += content
                # Print each chunk inline to visualize streaming
                sys.stdout.write(content)
                sys.stdout.flush()
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        if text:
                            chunk_count += 1
                            total_text += text
                            sys.stdout.write(text)
                            sys.stdout.flush()

    except Exception as e:
        print(f"\nERROR during stream: {e}")
        return False

    elapsed = time.perf_counter() - t0

    print(f"\n\n--- Results ---")
    print(f"  Chunks received : {chunk_count}")
    print(f"  Total chars     : {len(total_text)}")
    print(f"  Time to 1st tok : {first_token_time:.3f}s" if first_token_time else "  Time to 1st tok : N/A")
    print(f"  Total time      : {elapsed:.3f}s")
    print(f"  Streaming?      : {'YES' if chunk_count > 1 else 'NO (single chunk = buffered)'}")

    return chunk_count > 1


# ---------------------------------------------------------------------------
# Test 2: Sync .invoke() — baseline (non-streaming)
# ---------------------------------------------------------------------------
def test_sync_invoke(model_id: str):
    _separator(f"SYNC .invoke()  |  {model_id}")

    llm = _create_model(model_id)

    t0 = time.perf_counter()
    try:
        response = llm.invoke(PROMPT)
    except Exception as e:
        print(f"ERROR during invoke: {e}")
        return False

    elapsed = time.perf_counter() - t0

    content = response.content
    text = content if isinstance(content, str) else str(content)
    print(text)
    print(f"\n--- Results ---")
    print(f"  Total chars     : {len(text)}")
    print(f"  Total time      : {elapsed:.3f}s")
    print(f"  (Baseline — no streaming)")

    return True


# ---------------------------------------------------------------------------
# Test 3: Async .astream() — token-by-token
# ---------------------------------------------------------------------------
async def test_async_stream(model_id: str):
    _separator(f"ASYNC .astream()  |  {model_id}")

    llm = _create_model(model_id)

    chunk_count = 0
    total_text = ""
    first_token_time = None
    t0 = time.perf_counter()

    try:
        async for chunk in llm.astream(PROMPT):
            now = time.perf_counter()
            if first_token_time is None:
                first_token_time = now - t0

            content = chunk.content
            if isinstance(content, str) and content:
                chunk_count += 1
                total_text += content
                sys.stdout.write(content)
                sys.stdout.flush()
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        if text:
                            chunk_count += 1
                            total_text += text
                            sys.stdout.write(text)
                            sys.stdout.flush()

    except Exception as e:
        print(f"\nERROR during astream: {e}")
        return False

    elapsed = time.perf_counter() - t0

    print(f"\n\n--- Results ---")
    print(f"  Chunks received : {chunk_count}")
    print(f"  Total chars     : {len(total_text)}")
    print(f"  Time to 1st tok : {first_token_time:.3f}s" if first_token_time else "  Time to 1st tok : N/A")
    print(f"  Total time      : {elapsed:.3f}s")
    print(f"  Streaming?      : {'YES' if chunk_count > 1 else 'NO (single chunk = buffered)'}")

    return chunk_count > 1


# ---------------------------------------------------------------------------
# Test 4: Introspection — _should_stream check
# ---------------------------------------------------------------------------
def test_should_stream_flag(model_id: str):
    _separator(f"_should_stream introspection  |  {model_id}")

    llm = _create_model(model_id)

    # Check if _stream is overridden (not the base class default)
    from langchain_core.language_models.chat_models import BaseChatModel

    stream_overridden = type(llm)._stream is not BaseChatModel._stream
    astream_overridden = type(llm)._astream is not BaseChatModel._astream

    print(f"  _stream overridden  : {stream_overridden}")
    print(f"  _astream overridden : {astream_overridden}")
    print(f"  disable_streaming   : {getattr(llm, 'disable_streaming', 'N/A')}")

    # Try calling _should_stream directly
    try:
        result = llm._should_stream(async_api=False, run_manager=None)
        print(f"  _should_stream(sync): {result}")
    except Exception as e:
        print(f"  _should_stream(sync): ERROR - {e}")

    try:
        result = llm._should_stream(async_api=True, run_manager=None)
        print(f"  _should_stream(async): {result}")
    except Exception as e:
        print(f"  _should_stream(async): ERROR - {e}")

    return stream_overridden


# ---------------------------------------------------------------------------
# Test 5: LangGraph-style streaming (stream_mode="messages")
# ---------------------------------------------------------------------------
def test_langgraph_style_stream(model_id: str):
    """Simulates how Lobster's client.py uses LangGraph streaming."""
    _separator(f"LangGraph-style stream  |  {model_id}")

    try:
        from langchain_core.messages import HumanMessage, AIMessageChunk
        from langgraph.graph import StateGraph, MessagesState, START, END
    except ImportError:
        print("  SKIP: langgraph not installed")
        return None

    llm = _create_model(model_id)

    # Build minimal graph: single LLM node
    def call_model(state: MessagesState):
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node("model", call_model)
    graph_builder.add_edge(START, "model")
    graph_builder.add_edge("model", END)
    graph = graph_builder.compile()

    chunk_count = 0
    total_text = ""
    first_token_time = None
    t0 = time.perf_counter()

    try:
        for event in graph.stream(
            {"messages": [HumanMessage(content=PROMPT)]},
            stream_mode="messages",
        ):
            now = time.perf_counter()

            # stream_mode="messages" yields (message_chunk, metadata) tuples
            if isinstance(event, tuple) and len(event) == 2:
                msg_chunk, metadata = event
                if isinstance(msg_chunk, AIMessageChunk):
                    content = msg_chunk.content
                    if isinstance(content, str) and content:
                        if first_token_time is None:
                            first_token_time = now - t0
                        chunk_count += 1
                        total_text += content
                        sys.stdout.write(content)
                        sys.stdout.flush()
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                text = block.get("text", "")
                                if text:
                                    if first_token_time is None:
                                        first_token_time = now - t0
                                    chunk_count += 1
                                    total_text += text
                                    sys.stdout.write(text)
                                    sys.stdout.flush()

    except Exception as e:
        print(f"\nERROR during LangGraph stream: {e}")
        return False

    elapsed = time.perf_counter() - t0

    print(f"\n\n--- Results ---")
    print(f"  Chunks received : {chunk_count}")
    print(f"  Total chars     : {len(total_text)}")
    print(f"  Time to 1st tok : {first_token_time:.3f}s" if first_token_time else "  Time to 1st tok : N/A")
    print(f"  Total time      : {elapsed:.3f}s")
    print(f"  Streaming?      : {'YES' if chunk_count > 1 else 'NO (single chunk = buffered)'}")

    return chunk_count > 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Test Bedrock streaming with langchain-aws")
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        default=DEFAULT_MODEL,
        help=f"Model to test (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("--all-models", action="store_true", help="Test all models")
    parser.add_argument("--async", dest="run_async", action="store_true", help="Include async tests")
    parser.add_argument("--langgraph", action="store_true", help="Include LangGraph-style streaming test")
    parser.add_argument("--invoke-baseline", action="store_true", help="Include non-streaming invoke baseline")
    args = parser.parse_args()

    if not _check_credentials():
        sys.exit(1)

    models_to_test = list(MODELS.keys()) if args.all_models else [args.model]

    results = {}
    for model_key in models_to_test:
        model_id = MODELS[model_key]
        print(f"\n{'#'*60}")
        print(f"  MODEL: {model_key} ({model_id})")
        print(f"{'#'*60}")

        # Introspection
        test_should_stream_flag(model_id)

        # Sync stream (always run)
        stream_ok = test_sync_stream(model_id)
        results[f"{model_key}/sync_stream"] = stream_ok

        # Invoke baseline (optional)
        if args.invoke_baseline:
            test_sync_invoke(model_id)

        # Async stream (optional)
        if args.run_async:
            async_ok = asyncio.run(test_async_stream(model_id))
            results[f"{model_key}/async_stream"] = async_ok

        # LangGraph-style (optional)
        if args.langgraph:
            lg_ok = test_langgraph_style_stream(model_id)
            results[f"{model_key}/langgraph"] = lg_ok

    # Summary
    _separator("SUMMARY")
    all_pass = True
    for test_name, passed in results.items():
        status = "PASS" if passed else ("SKIP" if passed is None else "FAIL")
        if not passed and passed is not None:
            all_pass = False
        print(f"  [{status}] {test_name}")

    print()
    if all_pass:
        print("  All streaming tests passed.")
    else:
        print("  Some tests FAILED — streaming may not be working as expected.")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
