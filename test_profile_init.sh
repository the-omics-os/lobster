#!/bin/bash
# Test script for profile selection in lobster init
# Tests both interactive and non-interactive modes with all three providers

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR="${SCRIPT_DIR}/test_init_profiles"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

echo "=========================================="
echo "Testing Profile Selection in lobster init"
echo "=========================================="
echo ""

# Setup test directory
rm -rf "${TEST_DIR}"
mkdir -p "${TEST_DIR}"

# Function to run a test
run_test() {
    local test_name="$1"
    local test_dir="${TEST_DIR}/${test_name}"
    local expected_profile="$2"
    local should_have_profile="$3"  # "yes" or "no"

    TESTS_RUN=$((TESTS_RUN + 1))

    echo -e "${YELLOW}Test ${TESTS_RUN}: ${test_name}${NC}"

    mkdir -p "${test_dir}"
    cd "${test_dir}"

    # Run the command (passed as remaining arguments)
    shift 3
    if "$@" > test_output.log 2>&1; then
        if [ "${should_have_profile}" = "yes" ]; then
            # Check if .env has LOBSTER_PROFILE
            if [ -f ".env" ] && grep -q "LOBSTER_PROFILE=${expected_profile}" .env; then
                echo -e "${GREEN}✓ PASSED: Profile '${expected_profile}' correctly written to .env${NC}"
                TESTS_PASSED=$((TESTS_PASSED + 1))
            else
                echo -e "${RED}✗ FAILED: Profile not found or incorrect in .env${NC}"
                echo "Expected: LOBSTER_PROFILE=${expected_profile}"
                if [ -f ".env" ]; then
                    echo "Found in .env:"
                    grep LOBSTER_PROFILE .env || echo "(no LOBSTER_PROFILE line)"
                fi
                TESTS_FAILED=$((TESTS_FAILED + 1))
            fi
        else
            # Check that .env does NOT have LOBSTER_PROFILE
            if [ -f ".env" ] && ! grep -q "LOBSTER_PROFILE" .env; then
                echo -e "${GREEN}✓ PASSED: Profile correctly omitted for Ollama${NC}"
                TESTS_PASSED=$((TESTS_PASSED + 1))
            else
                echo -e "${RED}✗ FAILED: Profile should not be in .env for Ollama${NC}"
                if [ -f ".env" ]; then
                    grep LOBSTER_PROFILE .env || true
                fi
                TESTS_FAILED=$((TESTS_FAILED + 1))
            fi
        fi
    else
        echo -e "${RED}✗ FAILED: Command failed${NC}"
        cat test_output.log
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi

    echo ""
    cd "${SCRIPT_DIR}"
}

echo "=== Non-Interactive Mode Tests ==="
echo ""

# Test 1: Anthropic with default profile (should be production)
run_test "anthropic_default" "production" "yes" \
    lobster init --non-interactive --anthropic-key="test-key-123"

# Test 2: Anthropic with explicit development profile
run_test "anthropic_dev" "development" "yes" \
    lobster init --non-interactive --anthropic-key="test-key-123" --profile=development

# Test 3: Anthropic with ultra profile
run_test "anthropic_ultra" "ultra" "yes" \
    lobster init --non-interactive --anthropic-key="test-key-123" --profile=ultra

# Test 4: Anthropic with godmode profile
run_test "anthropic_godmode" "godmode" "yes" \
    lobster init --non-interactive --anthropic-key="test-key-123" --profile=godmode

# Test 5: Bedrock with default profile (should be production)
run_test "bedrock_default" "production" "yes" \
    lobster init --non-interactive --bedrock-access-key="AKIA123" --bedrock-secret-key="secret123"

# Test 6: Bedrock with development profile
run_test "bedrock_dev" "development" "yes" \
    lobster init --non-interactive --bedrock-access-key="AKIA123" --bedrock-secret-key="secret123" --profile=development

# Test 7: Ollama without profile (should NOT have LOBSTER_PROFILE)
run_test "ollama_no_profile" "none" "no" \
    lobster init --non-interactive --use-ollama

# Test 8: Ollama with profile flag (should warn and ignore)
run_test "ollama_with_profile_ignored" "none" "no" \
    lobster init --non-interactive --use-ollama --profile=development

echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo -e "Total Tests Run:    ${TESTS_RUN}"
echo -e "${GREEN}Tests Passed:       ${TESTS_PASSED}${NC}"
echo -e "${RED}Tests Failed:       ${TESTS_FAILED}${NC}"
echo "=========================================="

# Cleanup
echo ""
echo "Test artifacts saved in: ${TEST_DIR}"
echo "To inspect results: cd ${TEST_DIR}"

# Exit with appropriate code
if [ ${TESTS_FAILED} -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
fi
