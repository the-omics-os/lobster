#!/bin/bash
# Test the actual CLI flow timing

echo "🧪 Testing Real CLI Flow"
echo "============================================================"
echo

# Test 1: lobster --help (no client init)
echo "📦 Test 1: lobster --help (lightweight command)"
echo "------------------------------------------------------------"
time lobster --help > /dev/null
echo

# Test 2: lobster chat with immediate exit (full flow)
echo "⚙️  Test 2: lobster chat startup (full initialization)"
echo "------------------------------------------------------------"
echo "(Ctrl+C after animation completes)"
echo
time (echo "exit" | lobster chat 2>&1 | head -50)
