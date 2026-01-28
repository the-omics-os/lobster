#!/bin/bash
#
# Genomics Agent Systematic Test Runner
# =====================================
# Executes tests from SYSTEMATIC_TESTING_PLAN.md using lobster query
#
# Usage:
#   ./run_systematic_tests.sh              # Run all phases
#   ./run_systematic_tests.sh smoke        # Run smoke tests only
#   ./run_systematic_tests.sh phase1       # Run Category 1 (Data Loading)
#   ./run_systematic_tests.sh phase2       # Run Category 2 (QC Workflows)
#   ./run_systematic_tests.sh phase3       # Run Categories 3-4 (GWAS + PCA)
#   ./run_systematic_tests.sh phase4       # Run Category 5 (Workflows)
#   ./run_systematic_tests.sh errors       # Run Category 6 (Error Handling)
#   ./run_systematic_tests.sh integration  # Run Category 7 (Multi-Agent)
#
# Output: Test results saved to test_results/genomics_test_TIMESTAMP.log
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOBSTER_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TEST_DATA_DIR="$SCRIPT_DIR"
RESULTS_DIR="$SCRIPT_DIR/test_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$RESULTS_DIR/genomics_test_$TIMESTAMP.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create results directory
mkdir -p "$RESULTS_DIR"

# Helper functions
log() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

log_header() {
    log ""
    log "${BLUE}================================================================${NC}"
    log "${BLUE}$1${NC}"
    log "${BLUE}================================================================${NC}"
    log ""
}

log_test() {
    local test_id=$1
    local test_name=$2
    log "${YELLOW}>>> Test $test_id: $test_name${NC}"
}

log_pass() {
    log "${GREEN}[PASS]${NC} $1"
}

log_fail() {
    log "${RED}[FAIL]${NC} $1"
}

log_skip() {
    log "${YELLOW}[SKIP]${NC} $1"
}

run_test() {
    local test_id=$1
    local session_id=$2
    local query=$3
    local timeout=${4:-120}  # Default 2 minute timeout

    log_test "$test_id" "$session_id"
    log "Query: $query"
    log ""

    # Run lobster query with timeout
    set +e
    output=$(cd "$LOBSTER_ROOT" && timeout "$timeout" lobster query --session-id "$session_id" "$query" 2>&1)
    exit_code=$?
    set -e

    log "Output:"
    log "$output"
    log ""

    if [ $exit_code -eq 0 ]; then
        log_pass "Test completed successfully"
    elif [ $exit_code -eq 124 ]; then
        log_fail "Test timed out after ${timeout}s"
    else
        log_fail "Test failed with exit code $exit_code"
    fi

    log "---"
    log ""

    return $exit_code
}

# ============================================================================
# SMOKE TESTS (Phase 0)
# ============================================================================
run_smoke_tests() {
    log_header "SMOKE TESTS (Basic Functionality)"

    local passed=0
    local failed=0

    # Smoke Test 1: VCF Loading
    if run_test "SMOKE-1" "smoke_vcf_load" \
        "Load the VCF file at test_data/genomics/chr22.vcf.gz as 'smoke_vcf'. Report the number of samples and variants loaded."; then
        ((passed++))
    else
        ((failed++))
    fi

    # Smoke Test 2: PLINK Loading
    if run_test "SMOKE-2" "smoke_plink_load" \
        "Load the PLINK file at test_data/genomics/plink_test/test_chr22.bed as 'smoke_plink'. Report the number of individuals and SNPs."; then
        ((passed++))
    else
        ((failed++))
    fi

    # Smoke Test 3: QC Assessment
    if run_test "SMOKE-3" "smoke_qc" \
        "Load test_data/genomics/chr22.vcf.gz as 'smoke_qc_data'. Run quality assessment with default thresholds. Report the number of variants passing QC."; then
        ((passed++))
    else
        ((failed++))
    fi

    # Smoke Test 4: PCA
    if run_test "SMOKE-4" "smoke_pca" \
        "Load test_data/genomics/chr22.vcf.gz as 'smoke_pca_data'. Run quality assessment and filter variants with MAF >= 0.01. Calculate 5 principal components. Report PC1 variance explained." 180; then
        ((passed++))
    else
        ((failed++))
    fi

    log_header "SMOKE TEST SUMMARY"
    log "Passed: $passed / $((passed + failed))"

    if [ $failed -gt 0 ]; then
        log_fail "$failed smoke test(s) failed. Fix before proceeding."
        return 1
    else
        log_pass "All smoke tests passed!"
        return 0
    fi
}

# ============================================================================
# CATEGORY 1: DATA LOADING (Phase 1)
# ============================================================================
run_phase1() {
    log_header "CATEGORY 1: DATA LOADING TESTS"

    local passed=0
    local failed=0

    # Test 1.1: VCF Loading - Basic
    if run_test "1.1" "genomics_test_1_1" \
        "Load the VCF file at test_data/genomics/chr22.vcf.gz as 'test_vcf_basic'. Report the exact number of samples (should be 2504 for 1000 Genomes) and variants."; then
        ((passed++))
    else
        ((failed++))
    fi

    # Test 1.2: VCF Loading - Region Filter
    # Note: Uses Ensembl format "22" not UCSC "chr22"
    if run_test "1.2" "genomics_test_1_2" \
        "Load the VCF file test_data/genomics/chr22.vcf.gz with region filter '22:16000000-17000000' as 'test_vcf_region'. How many variants are in this 1Mb region compared to the full file?"; then
        ((passed++))
    else
        ((failed++))
    fi

    # Test 1.3: List and Inspect
    if run_test "1.3" "genomics_test_1_3" \
        "Load test_data/genomics/chr22.vcf.gz as 'inspect_test'. Then list all loaded modalities and get detailed information for 'inspect_test'."; then
        ((passed++))
    else
        ((failed++))
    fi

    # Test 1.4: PLINK Loading - Basic
    if run_test "1.4" "genomics_test_1_4" \
        "Load the PLINK file at test_data/genomics/plink_test/test_chr22.bed as 'test_plink_basic'. Report individual and SNP counts. Verify the FAM and BIM metadata columns are present."; then
        ((passed++))
    else
        ((failed++))
    fi

    # Test 1.5: PLINK Loading - MAF Filter
    if run_test "1.5" "genomics_test_1_5" \
        "Load the PLINK file at test_data/genomics/plink_test/test_chr22.bed with MAF filter >= 0.05 as 'test_plink_maf'. How many SNPs pass the MAF filter compared to loading without filter?"; then
        ((passed++))
    else
        ((failed++))
    fi

    log_header "CATEGORY 1 SUMMARY"
    log "Passed: $passed / $((passed + failed))"
    return 0
}

# ============================================================================
# CATEGORY 2: QC WORKFLOWS (Phase 2)
# ============================================================================
run_phase2() {
    log_header "CATEGORY 2: QC WORKFLOW TESTS"

    local passed=0
    local failed=0

    # Test 2.1: Basic QC Assessment
    if run_test "2.1" "genomics_test_2_1" \
        "Load test_data/genomics/chr22.vcf.gz as 'qc_basic'. Run quality assessment with default thresholds (call rate >= 0.95, MAF >= 0.01, HWE p >= 1e-10). Report: mean sample call rate, mean variant call rate, mean MAF, and number of variants passing QC."; then
        ((passed++))
    else
        ((failed++))
    fi

    # Test 2.2: Strict QC
    if run_test "2.2" "genomics_test_2_2" \
        "Load test_data/genomics/chr22.vcf.gz as 'qc_strict'. Run quality assessment with STRICT thresholds: call rate >= 0.99, MAF >= 0.05, HWE p >= 1e-6. How does the variant pass rate compare to default thresholds (MAF >= 0.01)?"; then
        ((passed++))
    else
        ((failed++))
    fi

    # Test 2.3: Sample Filtering
    if run_test "2.3" "genomics_test_2_3" \
        "Load test_data/genomics/chr22.vcf.gz as 'sample_qc'. Run quality assessment. Then filter samples with call rate >= 0.95 and heterozygosity z-score within 3 SD. How many samples are retained? Are any removed for low call rate or extreme heterozygosity?"; then
        ((passed++))
    else
        ((failed++))
    fi

    # Test 2.4: Variant Filtering
    if run_test "2.4" "genomics_test_2_4" \
        "Load test_data/genomics/chr22.vcf.gz as 'variant_qc'. Run quality assessment. Then filter variants with call rate >= 0.99, MAF >= 0.01, HWE p >= 1e-10. What percentage of variants are retained? What is the primary reason for variant removal?"; then
        ((passed++))
    else
        ((failed++))
    fi

    # Test 2.5: Complete QC Pipeline
    if run_test "2.5" "genomics_test_2_5" \
        "Execute complete QC pipeline on test_data/genomics/chr22.vcf.gz as 'full_qc':
         1. Load data
         2. Assess quality (default thresholds)
         3. Filter samples (call rate >= 0.95, het z <= 3)
         4. Filter variants (call rate >= 0.99, MAF >= 0.01, HWE p >= 1e-10)
         Report starting and final sample/variant counts." 180; then
        ((passed++))
    else
        ((failed++))
    fi

    log_header "CATEGORY 2 SUMMARY"
    log "Passed: $passed / $((passed + failed))"
    return 0
}

# ============================================================================
# CATEGORIES 3-4: GWAS & PCA (Phase 3)
# ============================================================================
run_phase3() {
    log_header "CATEGORIES 3-4: GWAS & PCA TESTS"

    local passed=0
    local failed=0

    # Test 3.1: Lambda GC Understanding
    if run_test "3.1" "genomics_test_3_1" \
        "Explain what Lambda GC (genomic inflation factor) means in GWAS. Why would it be elevated for the 1000 Genomes dataset which has 26 populations? What is the acceptable range for Lambda GC?"; then
        ((passed++))
    else
        ((failed++))
    fi

    # Test 3.2: GWAS Workflow Description
    if run_test "3.2" "genomics_test_3_2" \
        "Describe the complete GWAS workflow steps for analyzing a case-control disease association study. Include: 1) Data loading, 2) QC steps, 3) Population stratification assessment, 4) Covariate selection, 5) Statistical model choice, 6) Results interpretation."; then
        ((passed++))
    else
        ((failed++))
    fi

    # Test 4.1: Basic PCA
    if run_test "4.1" "genomics_test_4_1" \
        "Load test_data/genomics/chr22.vcf.gz as 'pca_test'. Run QC and filter variants (MAF >= 0.01). Calculate 10 principal components. Report: PC1 variance explained, cumulative variance for top 5 PCs, and whether there is evidence of population stratification (PC1 > 5%)."; then
        ((passed++))
    else
        ((failed++))
    fi

    # Test 4.2: PCA Interpretation
    if run_test "4.2" "genomics_test_4_2" \
        "For the 1000 Genomes dataset with 5 continental ancestry groups (AFR, AMR, EAS, EUR, SAS), explain: 1) What would PC1 likely represent? 2) Why is PC1 variance > 5% significant? 3) How many PCs should typically be included as GWAS covariates?"; then
        ((passed++))
    else
        ((failed++))
    fi

    # Test 4.3: PCA-GWAS Integration
    if run_test "4.3" "genomics_test_4_3" \
        "Load test_data/genomics/chr22.vcf.gz, run QC, and calculate PCA. Based on the PC1 variance, determine if population stratification correction is needed. If PC1 > 5%, explain how PCs would be used as GWAS covariates to reduce Lambda GC from ~1.6 to ~1.0." 180; then
        ((passed++))
    else
        ((failed++))
    fi

    log_header "CATEGORIES 3-4 SUMMARY"
    log "Passed: $passed / $((passed + failed))"
    return 0
}

# ============================================================================
# CATEGORY 5: MULTI-STEP WORKFLOWS (Phase 4)
# ============================================================================
run_phase4() {
    log_header "CATEGORY 5: MULTI-STEP WORKFLOW TESTS"

    local passed=0
    local failed=0

    # Test 5.1: Complete GWAS Pipeline
    if run_test "5.1" "genomics_test_5_1" \
        "Execute a complete GWAS analysis pipeline on 1000 Genomes chr22 (test_data/genomics/chr22.vcf.gz):
         1. Load VCF data
         2. Run quality assessment (call rate >= 0.95, MAF >= 0.01, HWE p >= 1e-10)
         3. Filter samples (call rate >= 0.95, het z-score <= 3)
         4. Filter variants (call rate >= 0.99, MAF >= 0.01)
         5. Calculate PCA (10 components)
         6. Report: final sample/variant counts, PC1 variance, and recommendation for GWAS covariates" 240; then
        ((passed++))
    else
        ((failed++))
    fi

    # Test 5.2: Multi-Ancestry Workflow Description
    if run_test "5.2" "genomics_test_5_2" \
        "The 1000 Genomes dataset has 26 populations across 5 continental groups. Describe the best practices for conducting GWAS in such a multi-ancestry cohort. Include: 1) Meta-analysis approaches, 2) Trans-ethnic fine-mapping, 3) Ancestry-specific replication strategies."; then
        ((passed++))
    else
        ((failed++))
    fi

    # Test 5.3: UK Biobank Standards
    if run_test "5.3" "genomics_test_5_3" \
        "Describe the UK Biobank quality control standards for GWAS. What are the typical thresholds for: sample call rate, variant call rate, MAF, HWE p-value, and heterozygosity? How do these compare to the thresholds used in our QC pipeline?"; then
        ((passed++))
    else
        ((failed++))
    fi

    log_header "CATEGORY 5 SUMMARY"
    log "Passed: $passed / $((passed + failed))"
    return 0
}

# ============================================================================
# CATEGORY 6: ERROR HANDLING (errors)
# ============================================================================
run_errors() {
    log_header "CATEGORY 6: ERROR HANDLING TESTS"

    local passed=0
    local failed=0

    # Test 6.1: Missing File
    if run_test "6.1" "genomics_test_6_1" \
        "Try to load a VCF file that does not exist: /nonexistent/path/fake.vcf.gz as 'error_test'. What error message do you receive? Is it user-friendly?"; then
        ((passed++))
    else
        ((failed++))
    fi

    # Test 6.2: Invalid Modality
    if run_test "6.2" "genomics_test_6_2" \
        "Try to run quality assessment on a modality called 'nonexistent_modality'. What error occurs? Does it list available modalities?"; then
        ((passed++))
    else
        ((failed++))
    fi

    # Test 6.3: Small Sample Limitations
    if run_test "6.3" "genomics_test_6_3" \
        "If I have genomics data with only 10 samples, what limitations would I encounter in: 1) Heterozygosity z-score calculation, 2) HWE testing, 3) GWAS statistical power, 4) PCA (number of components)? What are the minimum recommended sample sizes?"; then
        ((passed++))
    else
        ((failed++))
    fi

    # Test 6.4: High Missingness
    if run_test "6.4" "genomics_test_6_4" \
        "Explain how the QC pipeline handles data with high missingness (some samples/variants with < 50% call rate). What is the recommended iterative QC approach?"; then
        ((passed++))
    else
        ((failed++))
    fi

    log_header "CATEGORY 6 SUMMARY"
    log "Passed: $passed / $((passed + failed))"
    return 0
}

# ============================================================================
# CATEGORY 7: MULTI-AGENT INTEGRATION (integration)
# ============================================================================
run_integration() {
    log_header "CATEGORY 7: MULTI-AGENT INTEGRATION TESTS"

    local passed=0
    local failed=0

    # Test 7.1: Natural Routing
    if run_test "7.1" "genomics_test_7_1" \
        "I have a VCF file with whole genome sequencing data from a GWAS study. Can you help me load it at test_data/genomics/chr22.vcf.gz and assess the data quality?"; then
        ((passed++))
    else
        ((failed++))
    fi

    # Test 7.2: Explicit Handoff
    if run_test "7.2" "genomics_test_7_2" \
        "ADMIN SUPERUSER: Route to genomics_expert only. Load the VCF file at test_data/genomics/chr22.vcf.gz as 'handoff_test'. Run quality assessment and report variant pass rate."; then
        ((passed++))
    else
        ((failed++))
    fi

    log_header "CATEGORY 7 SUMMARY"
    log "Passed: $passed / $((passed + failed))"
    return 0
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

main() {
    log_header "GENOMICS AGENT SYSTEMATIC TEST SUITE"
    log "Timestamp: $TIMESTAMP"
    log "Log file: $LOG_FILE"
    log "Test data directory: $TEST_DATA_DIR"
    log ""

    # Verify test data exists
    if [ ! -f "$TEST_DATA_DIR/chr22.vcf.gz" ]; then
        log_fail "Test data not found: $TEST_DATA_DIR/chr22.vcf.gz"
        log "Please download 1000 Genomes chr22 data first."
        exit 1
    fi

    if [ ! -f "$TEST_DATA_DIR/plink_test/test_chr22.bed" ]; then
        log_fail "PLINK test data not found: $TEST_DATA_DIR/plink_test/test_chr22.bed"
        log "Please run: python $TEST_DATA_DIR/generate_plink_test_data.py"
        exit 1
    fi

    log_pass "Test data verified"
    log ""

    # Parse command line argument
    case "${1:-all}" in
        smoke)
            run_smoke_tests
            ;;
        phase1)
            run_phase1
            ;;
        phase2)
            run_phase2
            ;;
        phase3)
            run_phase3
            ;;
        phase4)
            run_phase4
            ;;
        errors)
            run_errors
            ;;
        integration)
            run_integration
            ;;
        all)
            log "Running all test phases..."
            log ""

            # Run smoke tests first - if they fail, stop
            if ! run_smoke_tests; then
                log_fail "Smoke tests failed. Fix before running comprehensive tests."
                exit 1
            fi

            # Run all phases
            run_phase1
            run_phase2
            run_phase3
            run_phase4
            run_errors
            run_integration
            ;;
        *)
            echo "Usage: $0 {smoke|phase1|phase2|phase3|phase4|errors|integration|all}"
            echo ""
            echo "Phases:"
            echo "  smoke       - Quick validation tests"
            echo "  phase1      - Category 1: Data Loading"
            echo "  phase2      - Category 2: QC Workflows"
            echo "  phase3      - Categories 3-4: GWAS & PCA"
            echo "  phase4      - Category 5: Multi-Step Workflows"
            echo "  errors      - Category 6: Error Handling"
            echo "  integration - Category 7: Multi-Agent Integration"
            echo "  all         - Run all phases"
            exit 1
            ;;
    esac

    log_header "TEST SUITE COMPLETE"
    log "Results saved to: $LOG_FILE"
}

# Run main function
main "$@"
