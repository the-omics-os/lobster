# Troubleshooting Guide

This comprehensive troubleshooting guide provides solutions to common issues encountered while using Lobster AI. Each problem includes symptoms, causes, and step-by-step solutions.

## Table of Contents

1. [Installation & Setup Issues](#installation-setup-issues)
   - API Keys Not Working
   - CLI Interface Not Working
   - **Rate Limit Errors (429)** ⚠️
   - **Authentication Errors (401)**
   - **Network Errors**
   - **Quota Exceeded Errors**
2. [Data Loading Problems](#data-loading-problems)
3. [Publication Intelligence & Docling Issues](#publication-intelligence-docling-issues) 🆕
   - Docling Not Installed
   - MemoryError During PDF Parsing
   - Methods Section Not Found
   - Page Dimensions RuntimeError
   - Cache Issues
4. [Analysis Failures](#analysis-failures)
5. [Performance Issues](#performance-issues)
6. [Visualization Problems](#visualization-problems)
7. [Cloud Integration Issues](#cloud-integration-issues)
8. [Agent & Tool Errors](#agent-tool-errors)
9. [Memory & Resource Problems](#memory-resource-problems)
10. [Output & Export Issues](#output-export-issues)
11. [Advanced Troubleshooting](#advanced-troubleshooting)

---

## Installation & Setup Issues

### Issue: Cannot Install Lobster AI

**Symptoms:**
- `pip install` fails with dependency errors
- Python version compatibility issues
- Missing system dependencies

**Solutions:**

#### Check Python Version
```bash
# Verify Python version (requires 3.12+)
python --version

# If using wrong version, create conda environment
conda create -n lobster python=3.12
conda activate lobster
```

#### Clean Installation
```bash
# Remove existing installation
pip uninstall lobster-ai

# Clean install with development dependencies
git clone https://github.com/the-omics-os/lobster.git
cd lobster
make clean-install
```

#### Resolve Dependency Conflicts
```bash
# Install with verbose output to see exact error
pip install -e . -v

# If conflicts occur, try constraint file
pip install -e . --constraint constraints.txt

# For conda users
conda env create -f environment.yml
```

### Issue: API Keys Not Working

**Symptoms:**
- "API key not found" errors
- Authentication failures
- Cannot access LLM models

**Solutions:**

#### Check Environment Variables
```bash
# Verify API keys are set
echo $AWS_BEDROCK_ACCESS_KEY
echo $AWS_BEDROCK_SECRET_ACCESS_KEY

# Check .env file exists and is correctly formatted
cat .env
```

#### Fix .env File
```bash
# Create or update .env file
cat > .env << EOF
AWS_BEDROCK_ACCESS_KEY=your-aws-access-key
AWS_BEDROCK_SECRET_ACCESS_KEY=your-aws-secret-key
NCBI_API_KEY=your-ncbi-api-key-optional
EOF

# Ensure no extra spaces or quotes
```

#### Test API Connection
```bash
# Test with minimal example
python -c "
from lobster.config.settings import get_settings
settings = get_settings()
print('Settings loaded successfully')
"
```

### Issue: CLI Interface Not Working

**Symptoms:**
- Plain text interface instead of Rich CLI
- Missing orange branding
- Arrow keys not working

**Solutions:**

#### Install Enhanced CLI Dependencies
```bash
# Install optional CLI enhancements
pip install prompt-toolkit

# Verify installation
python -c "import prompt_toolkit; print('Enhanced CLI available')"
```

#### Check Terminal Compatibility
```bash
# Test terminal capabilities
echo $TERM
python -c "
import sys
print(f'Terminal: {sys.stdout.isatty()}')
print(f'Colors: {hasattr(sys.stdout, \"isatty\")}')
"
```

#### Force Rich CLI Mode
```bash
# Start with explicit Rich mode
FORCE_RICH=1 lobster chat

# Or disable if causing issues
DISABLE_RICH=1 lobster chat
```

### Issue: Rate Limit Errors (429)

**Symptoms:**
```
⚠️  Rate Limit Exceeded
Error code: 429 - {'type': 'error', 'error': {'type': 'rate_limit_error', 'message': 'This request would exceed your organization's maximum usage increase rate...'}}
```

**Causes:**
- Anthropic's conservative rate limits for new accounts
- Exceeded requests per minute/hour quota
- Burst usage patterns triggering throttling
- Organization-level limits reached

**Solutions:**

#### Immediate Actions (Quick Fix)
```bash
# Wait 60 seconds and retry
# Rate limits typically reset within 1-2 minutes

# Check current rate limit status
🦞 You: "What are my current API rate limits?"
```

#### Short-term Solutions
```bash
# 1. Request rate limit increase from Anthropic
# Visit: https://docs.anthropic.com/en/api/rate-limits
# Fill out their rate increase request form

# 2. Reduce concurrent requests
# Run analysis tasks sequentially instead of parallel

# 3. Use retry logic with exponential backoff
# Lobster AI will automatically retry with delays
```

#### Long-term Solutions (Recommended)
```bash
# Switch to AWS Bedrock (enterprise-grade limits)
# 1. Set up AWS Bedrock credentials
cat > .env << EOF
AWS_BEDROCK_ACCESS_KEY=your-aws-access-key
AWS_BEDROCK_SECRET_ACCESS_KEY=your-aws-secret-key
EOF

# 2. Restart Lobster
lobster chat

# 3. Verify Bedrock connection
🦞 You: "/session"  # Check which provider is active
```

#### Contact Support
```bash
# For urgent rate limit increases or assistance:
# Email: info@omics-os.com
# Include:
# - Your organization ID (from error message)
# - Use case description
# - Expected usage volume
```

**Prevention:**
- Use AWS Bedrock for production deployments
- Request rate increases proactively before large analyses
- Monitor usage patterns to stay within limits
- Consider batch processing for large datasets

### Issue: Authentication Errors (401)

**Symptoms:**
```
🔑 Authentication Failed
Error code: 401 - {'type': 'error', 'error': {'type': 'invalid_api_key'}}
```

**Causes:**
- Invalid or expired API key
- API key not configured in environment
- Incorrect key format
- Missing required permissions (AWS Bedrock)

**Solutions:**

#### Verify API Key Configuration
```bash
# Check environment variables
echo $ANTHROPIC_API_KEY    # Should show: sk-ant-api03-...
echo $AWS_BEDROCK_ACCESS_KEY  # For AWS users

# Check .env file
cat .env

# Ensure proper format (no quotes, spaces, or line breaks)
ANTHROPIC_API_KEY=sk-ant-api03-your-actual-key-here
```

#### Fix Common Issues
```bash
# 1. Key in .env but not loaded
source .env && lobster chat

# 2. Key has extra whitespace
# Edit .env and remove any spaces:
# ✓ ANTHROPIC_API_KEY=sk-ant-...
# ✗ ANTHROPIC_API_KEY = sk-ant-...
# ✗ ANTHROPIC_API_KEY="sk-ant-..."

# 3. Generate new key
# Visit https://console.anthropic.com/settings/keys
# Create new key and update .env
```

#### Test API Connection
```bash
# Test authentication
python -c "
import os
from anthropic import Anthropic
client = Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))
print('✓ Authentication successful')
"
```

#### AWS Bedrock Permissions
```bash
# If using AWS Bedrock, verify IAM permissions include:
# - bedrock:InvokeModel
# - bedrock:InvokeModelWithResponseStream

# Test AWS credentials
aws bedrock list-foundation-models --region us-east-1
```

### Issue: Network Errors

**Symptoms:**
```
🌐 Network Error
Connection timeout / Connection refused / DNS resolution failed
```

**Causes:**
- No internet connectivity
- Firewall blocking HTTPS connections
- DNS resolution issues
- API service temporary outage
- Proxy misconfiguration

**Solutions:**

#### Check Basic Connectivity
```bash
# Test internet connection
ping -c 3 anthropic.com
ping -c 3 api.anthropic.com

# Test HTTPS access
curl -I https://api.anthropic.com/v1/messages

# Check DNS resolution
nslookup api.anthropic.com
```

#### Firewall Configuration
```bash
# Ensure firewall allows HTTPS (port 443)
# For corporate networks, contact IT to whitelist:
# - api.anthropic.com
# - bedrock-runtime.*.amazonaws.com (for AWS)

# Test with firewall temporarily disabled
sudo ufw disable  # Linux
# Try connection, then re-enable
sudo ufw enable
```

#### Proxy Configuration
```bash
# If behind a proxy, set environment variables
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080
export NO_PROXY=localhost,127.0.0.1

# Test with proxy
lobster chat
```

#### Check API Status
```bash
# Check Anthropic service status
# Visit: https://status.anthropic.com

# For AWS Bedrock
# Visit: https://status.aws.amazon.com
```

### Issue: Quota Exceeded Errors

**Symptoms:**
```
💳 Usage Quota Exceeded
Error code: 402 - insufficient_quota
```

**Causes:**
- Monthly spending limit reached
- Usage quota exhausted
- Payment method issues
- Billing not set up

**Solutions:**

#### Check Billing Status
```bash
# 1. Visit billing dashboard
# Anthropic: https://console.anthropic.com/settings/billing
# Check current usage and limits

# 2. Review usage metrics
# - Current month usage
# - Remaining quota
# - Next reset date
```

#### Increase Quota
```bash
# 1. Upgrade plan or add credits
# 2. Set up automatic billing
# 3. Contact billing support for enterprise quotas

# For AWS Bedrock users:
# Contact AWS support for quota increases
# Visit: https://console.aws.amazon.com/support
```

#### Alternative: Switch Providers
```bash
# Switch to AWS Bedrock for higher quotas
# See installation guide for AWS setup
# wiki/02-installation.md#aws-bedrock-access
```

---

## Data Loading Problems

### Issue: FTP Download Failures or Corruption (v0.2+ FIXED with Fix #7)

**Symptoms (Should be RARE after Fix #7):**
```
⚠️  FTP download failed after 3 retries
Corrupted gzip file detected
File size mismatches (140-285% larger than expected)
Gzip errors: "not in gzip format", "invalid header", "CRC check failed"
```

**Automatic Recovery (v0.2+ Fix #7 - HTTPS Pre-Download):**

Lobster AI v0.2+ includes **Fix #7**, which eliminates FTP corruption entirely:

✅ **HTTPS Pre-Download**: SOFT files are pre-downloaded using HTTPS before calling GEOparse
✅ **TLS Integrity Checking**: Automatic corruption detection via cryptographic MACs
✅ **91% → <5% Corruption Rate**: Expected 20x reduction in download failures
✅ **Fail-Fast Behavior**: SSL/HTTP errors instead of silent corruption
✅ **Graceful Fallback**: Falls back to FTP only if HTTPS fails

**How It Works:**
```
1. HTTPS pre-download of SOFT file (99% of cases)
   ↓ (if HTTPS fails)
2. GEOparse FTP fallback (rare)
   ↓ (if FTP fails)
3. Next pipeline step (multiple strategies)
```

**Manual Intervention (If Download Still Fails):**
```bash
# Clear cache and force fresh download
rm -rf ~/.lobster_workspace/cache/geo/GSE12345*
🦞 You: "Download GSE12345 with fresh cache"

# Check internet connectivity (HTTPS, not FTP)
ping ftp.ncbi.nlm.nih.gov

# Verify GEO accession exists
🦞 You: "Search for GSE12345 in GEO database"
```

**SSL Certificate Issues:**

If you see SSL certificate verification errors:

```bash
# macOS - Install Python certificates
cd "/Applications/Python 3.12/"
./Install Certificates.command

# Linux (Ubuntu/Debian) - Update CA certificates
sudo apt-get install ca-certificates
sudo update-ca-certificates

# Linux (Fedora/RHEL)
sudo dnf install ca-certificates
sudo update-ca-trust
```

**Technical Details:**

For comprehensive technical documentation including:
- Complete implementation details (9 locations across 3 files)
- Root cause analysis of FTP corruption
- Before/After log evidence
- Related bug fixes (H5AD serialization, metadata storage)
- Troubleshooting guide for SSL issues

See: [**Fix #7: HTTPS Pre-Download Technical Documentation**](47-fix7-https-geo-download.md)

---

### Issue: VDJ Data "Duplicate Barcode" Errors (v0.2+)

**Symptoms:**
```
⚠️  Duplicate cell barcodes detected: 48%
Dataset GSE248556 rejected due to data quality issues
Validation failed: non-unique cell barcodes
```

**Cause (FIXED in v0.2+):**
VDJ/TCR/BCR sequencing data legitimately has duplicate cell barcodes because each cell can express multiple receptor chains (heavy + light chain, alpha + beta chain). The system now automatically detects VDJ data types and accepts duplicates.

**Expected Behavior (v0.2+):**
- **VDJ/TCR/BCR data**: Duplicate barcodes **accepted** (biologically valid)
- **RNA/Protein data**: Duplicate barcodes **rejected** (indicates corruption)
- System uses sample metadata keywords: "VDJ", "TCR", "BCR", "immunology", "receptor"

**Auto-Detection:**
```bash
# System automatically detects VDJ datasets
🦞 You: "Download GSE248556"
# Output: "Detected VDJ/TCR sequencing data, accepting duplicate barcodes (48%)"
```

**Manual Override (If Misclassified):**
```bash
🦞 You: "Load GSE248556 treating samples as VDJ data"
🦞 You: "Override duplicate barcode validation for immunology dataset"
```

---

### Issue: H5AD Export Failures with GEO Metadata (v0.2+ FIXED with Bug Fix #3)

**Symptoms (Should NOT occur after Bug Fix #3):**
```
TypeError: Can't implicitly convert non-string objects to strings
TypeError: Cannot serialize mixed types to H5AD
ValueError: Boolean values not supported in AnnData metadata
KeyError: Metadata column contains None values
```

**Root Cause (FIXED in v0.2+):**
HDF5 (the underlying format for H5AD) cannot serialize scalar integers, floats, booleans, or Python lists in nested dictionaries. GEO datasets commonly have metadata structures like:

```python
# Problematic metadata structure:
{
    'contact_zip/postal_code': 12345,  # ❌ int cannot be serialized
    'sample_count': 13,  # ❌ int cannot be serialized
    'is_processed': True,  # ❌ bool cannot be serialized
    'platforms': ['GPL20795', 'GPL24676'],  # ❌ list cannot be serialized
    'submission_date': None,  # ❌ None cannot be serialized
}
```

**Automatic Resolution (v0.2+ Bug Fix #3):**

Lobster AI now performs **aggressive stringification** during H5AD export:
- `int/float → string` (e.g., `42 → '42'`)
- `bool → string` (e.g., `True → 'True'`)
- `None → ""` (empty string)
- `list → numpy string array` (e.g., `[1, 2] → array(['1', '2'])`)
- `list-of-dict → stringified representation`
- Keys with `/` → replaced with `__` (e.g., `'a/b' → 'a__b'`)
- **No user action required** - Metadata cleaned transparently

**Before/After Comparison:**

| Data Type | Before (Failed) | After (Fixed) |
|-----------|-----------------|---------------|
| `int` | `{'count': 42}` | `{'count': '42'}` ✅ |
| `float` | `{'score': 3.14}` | `{'score': '3.14'}` ✅ |
| `bool` | `{'flag': True}` | `{'flag': 'True'}` ✅ |
| `list` | `{'items': [1, 2]}` | `{'items': array(['1', '2'])}` ✅ |

**Impact:**
- **100% → 0%** H5AD serialization failure rate for GEO datasets
- Fixes GSE267814 (was 13/13 failures → now 0/13 failures)
- All biological/scientific data (`.X`, `.obs`, `.var`) preserved perfectly
- Metadata types converted to strings (acceptable for GEO metadata use case)

**When It Happens:**
GEO datasets often have poor metadata quality with:
- Boolean flags as actual bool type (not H5AD-compatible)
- Missing values as None (not serializable)
- Mixed integer/string columns
- Complex nested structures (lists-of-dicts in provenance metadata)

**Manual Verification:**
```bash
# Check metadata before export
🦞 You: "Show metadata summary for current dataset"

# Force H5AD export with sanitization (automatic in v0.2+)
🦞 You: "Export to H5AD"
```

**If Serialization Still Fails:**

This should NOT happen in v0.2+, but if you encounter new edge cases:

1. **Check sanitization logs**:
   ```bash
   grep "Sanitized column" lobster.log
   # Should show: [DEBUG] Sanitized column 'mt' - converted bool to string
   ```

2. **Inspect problematic metadata structure**:
   ```python
   import anndata as ad
   adata = ad.read_h5ad("problem_file.h5ad")
   print(adata.obs.dtypes)
   print(adata.uns)
   ```

3. **Report new edge case**:
   - File GitHub issue with dataset accession
   - Include metadata structure that failed
   - Helps improve sanitization logic

**Technical Details:**

For comprehensive documentation including:
- Complete root cause analysis
- Sanitization algorithm details
- Testing validation (5/5 test cases pass)
- Edge case handling

See: [**Fix #7: HTTPS Pre-Download Technical Documentation**](47-fix7-https-geo-download.md) (Bug Fix #3 section)

---

### Issue: Bulk RNA-seq "Inverted Dimensions" Warning (v0.2+)

**Symptoms:**
```
⚠️  Matrix dimensions may be inverted: 187,697 features × 4 observations
Expected: samples × genes for bulk RNA-seq
Applying automatic transpose...
```

**Automatic Resolution (v0.2+):**
Lobster AI applies biology-aware transpose logic:
- **Checks**: Gene count ranges (10K-60K for human/mouse)
- **Checks**: Sample count ranges (2-200 typical for bulk RNA-seq)
- **Checks**: >100x imbalance (conservative fallback for edge cases)
- Matrix automatically transposed to correct orientation
- **No user action required** - Biology-aware validation handles this

**Why It Happens:**
Some bulk RNA-seq datasets (e.g., GSE130036) have few samples:
- 4 samples × 187,697 genes → Looks inverted to naive algorithms
- System uses biological knowledge to correctly orient the matrix

**Manual Override (Rare):**
```bash
# If auto-transpose is incorrect (very rare)
🦞 You: "Load GSE12345 without auto-transpose"
🦞 You: "Keep original matrix orientation for GSE12345"
```

---

### Issue: Malformed GEO Accessions (v0.2+)

**Symptoms:**
```
❌ Invalid accession format: GDS200157007
Expected format: GSE/GSM/GPL/GDS + digits
Accession has 9 digits, expected 4-7
```

**Resolution (FIXED in v0.2+):**
- Case sensitivity bug fixed (lowercase "accession" field)
- Database migrated from "gds" (deprecated, ~5K datasets) to "geo" (active, 200K+ datasets)
- Correct accessions now retrieved: `GSE157007` (not `GDS200157007`)
- Dataset coverage increased 40x with active database

**Manual Verification:**
```bash
# Verify accession format
🦞 You: "Search for GSE157007 in GEO database and verify accession format"

# System now returns correct format automatically
# GSE prefix: Series (multiple samples)
# GSM prefix: Sample (single sample)
# GPL prefix: Platform (array/sequencing tech)
# GDS prefix: Curated dataset (deprecated but still supported)
```

---

### Issue: Cannot Load Dataset from GEO (General)

**Symptoms:**
- "Dataset not found" errors
- Download timeouts
- Network errors

**Solutions:**

#### Verify GEO Accession
```bash
🦞 You: "Search for GSE12345 in GEO database and verify it exists"
🦞 You: "Download GSE12345 with verbose output to see detailed progress"
```

#### Handle Network Issues
```bash
# Check internet connectivity
ping ncbi.nlm.nih.gov

# Use alternative download method
🦞 You: "Download GSE12345 using alternative mirror or cached version"

# Manual download and load
🦞 You: "Load local file that I downloaded manually from GEO"
```

#### Clear Cache and Retry
```bash
# Clear GEO cache
rm -rf ~/.lobster_workspace/geo_cache/

# Retry download
🦞 You: "Download GSE12345 with fresh cache"
```

**Note**: Most GEO issues are now handled automatically in v0.2+ with robust error handling, retry logic, and intelligent validation.

### Issue: File Format Not Recognized

**Symptoms:**
- "Unknown file format" errors
- Cannot parse file headers
- Encoding issues

**Solutions:**

#### Specify File Format Explicitly
```bash
🦞 You: "Load CSV file with genes as rows and samples as columns"
🦞 You: "Load TSV file with tab separators and first row as header"
🦞 You: "Load Excel file from sheet named 'RNAseq_data'"
```

#### Check File Encoding
```bash
# Check file encoding
file -i your_data.csv

# Convert if needed
iconv -f iso-8859-1 -t utf-8 your_data.csv > your_data_utf8.csv
```

#### Provide File Structure Information
```bash
🦞 You: "This is a count matrix with gene symbols in first column, sample IDs in header row"
🦞 You: "The file has metadata in the first 3 rows, data starts from row 4"
🦞 You: "File uses semicolon separators instead of commas"
```

### Issue: Large Files Won't Load

**Symptoms:**
- Memory errors during loading
- Loading process hangs
- "File too large" messages

**Solutions:**

#### Use Chunked Loading
```bash
🦞 You: "Load large file in chunks of 10000 rows to save memory"
🦞 You: "Subsample 50% of the data for initial exploration"
🦞 You: "Use sparse matrix format to reduce memory usage"
```

#### Optimize File Format
```bash
# Convert to more efficient format
🦞 You: "Convert CSV to H5AD format for faster loading"
🦞 You: "Compress data using sparse matrix representation"
```

#### Increase Available Memory
```bash
# Monitor memory usage
🦞 You: "/dashboard"  # Check system resources

# Use cloud processing for large files
export LOBSTER_CLOUD_KEY="your-api-key"
🦞 You: "Process this large dataset using cloud resources"
```

---

## Publication Intelligence & Docling Issues

### Issue: Docling Not Installed

**Symptoms:**
- `ImportError: No module named 'docling'`
- "Docling parser unavailable, falling back to PyPDF2"
- Warning messages about missing Docling dependencies

**Causes:**
- Docling package not installed
- Version mismatch with required dependencies
- Optional dependencies missing (OCR, table extraction)

**Solutions:**

#### Install Docling Package
```bash
# Install Docling with all dependencies
pip install docling

# Verify installation
python -c "from docling.document_converter import DocumentConverter; print('✓ Docling installed')"
```

#### Install Optional Features
```bash
# For enhanced table extraction
pip install "docling[table]"

# For OCR support (PDFs with scanned images)
pip install "docling[ocr]"

# Full installation with all features
pip install "docling[all]"
```

#### Verify Docling Functionality
```bash
🦞 You: "Test Docling installation by extracting methods from a sample paper"
🦞 You: "Extract methods from PMID:38448586 using Docling"
```

**Fallback Behavior:**
- System automatically falls back to PyPDF2 if Docling unavailable
- Extraction still works but with lower Methods section detection rate (~30% vs >90%)
- Tables and formulas won't be extracted with PyPDF2

### Issue: MemoryError During PDF Parsing

**Symptoms:**
```
MemoryError: Unable to allocate memory for document parsing
RuntimeError: PDF parsing failed after 2 retries
```

**Causes:**
- Large PDF documents (>100 pages)
- Complex layouts with many images
- Insufficient system memory (<4GB available)
- Multiple concurrent parsing operations

**Solutions:**

#### Immediate Actions
```bash
# Clear memory before parsing
🦞 You: "Clear workspace cache to free memory"
python -c "import gc; gc.collect()"

# Parse one document at a time
🦞 You: "Extract methods from PMID:12345678"  # Sequential processing
# Wait for completion before starting next extraction
```

#### Optimize Memory Usage
```bash
# Docling automatically retries with garbage collection
# The retry logic handles MemoryError automatically
# Just wait for the automatic retry to complete

# For very large PDFs, use PyPDF2 fallback explicitly
🦞 You: "Extract methods using PyPDF2 fallback for memory efficiency"
```

#### Monitor Memory
```bash
# Check available memory
free -h  # Linux
vm_stat  # macOS

# Monitor during extraction
🦞 You: "/dashboard"  # Check memory usage in real-time
```

#### Batch Processing Best Practices
```bash
# Process papers sequentially (not in parallel)
🦞 You: "Extract methods from these papers one at a time: PMID:123, PMID:456, PMID:789"

# Clear cache between large documents
rm -rf ~/.lobster_workspace/literature_cache/parsed_docs/
```

**Prevention:**
- Parse papers sequentially rather than in parallel
- Docling's retry logic includes explicit `gc.collect()` between attempts
- Cache prevents re-parsing (30-50x faster on subsequent access)
- Consider increasing system RAM for large-scale analysis

### Issue: DOI/PMID Not Resolving to Accessible URLs (v0.2+ Fix)

**Symptoms:**
```
⚠️  Could not resolve DOI to accessible URL
Failed to extract content from identifier: 10.1038/...
PaywalledError: Paper 10.18632/aging.204666 is paywalled
```

**Causes:**
- Paywalled article with no open access version available
- Invalid or malformed DOI/PMID
- Publisher website temporarily unavailable
- DOI not yet indexed in resolution databases
- Network connectivity issues

**Solutions:**

#### Verify DOI/PMID Format
```bash
# Test if identifier is detected correctly
🦞 You: "Check if DOI:10.1038/s41586-025-09686-5 is accessible"

# System will show resolution attempt and results:
# "✓ Detected identifier (DOI): 10.1038/..., resolving to URL..."
# "✓ Resolved to: https://www.nature.com/articles/..."
# OR
# "⚠️ Paper is not accessible: paywalled"
```

#### Try Alternative Identifiers
```bash
# If DOI doesn't resolve, try the PMID
🦞 You: "Extract methods from PMID:38448586"

# Or search for preprint version
🦞 You: "Find bioRxiv preprint for cellular senescence human fibroblasts"
```

#### Manual URL Provision
```bash
# If you have institutional access, provide the article page URL directly
🦞 You: "Extract methods from https://www.nature.com/articles/s41586-025-09686-5"

# For PMC papers, try the main article page (not /pdf/ directory)
🦞 You: "Extract methods from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12496192/"
```

#### Check Resolution Logs
The v0.2+ system provides detailed logging of resolution attempts:

```bash
# Successful resolution shows:
INFO Detected identifier (DOI): 10.1101/..., resolving to URL...
INFO Resolved via preprint server: https://www.biorxiv.org/content/...
INFO Content extraction successful (pdf auto-detected) in 2.3s

# Failed resolution shows:
WARNING Paper 10.18632/aging.204666 is not accessible: paywalled
INFO Alternative suggestions: [institutional access, preprints, author contact]
```

**Expected Behavior (v0.2+):**
- ✅ System automatically detects DOI/PMID format
- ✅ Tries multiple resolution strategies (PMC → bioRxiv/medRxiv → publisher)
- ✅ Format auto-detection (HTML vs PDF) handled by Docling
- ✅ If resolution fails, provides helpful alternative access suggestions
- ✅ No more crashes with FileNotFoundError for valid DOIs

### Issue: Methods Section Not Found

**Symptoms:**
```
⚠️  Methods section not found in document
Extracted 0 paragraphs from Methods section
```

**Causes:**
- Non-standard section naming (e.g., "Materials and Methods", "Experimental Procedures")
- Methods split across multiple sections
- PDF parsing failed to detect document structure
- Incompatible PDF format (page-dimensions error)

**Solutions:**

#### Verify Document Structure
```bash
🦞 You: "Show me the document structure and available sections"
🦞 You: "List all section headings found in the paper"
```

#### Try Alternative Keywords
```bash
# Docling searches for these keywords by default:
# "method", "material", "experimental", "procedure", "analysis"
#
# If paper uses non-standard terms, Docling may miss the section

# Check if paper is accessible
🦞 You: "Check if PMID:12345678 is accessible"

# Try extraction with PyPDF2 fallback
# (captures more text but less structured)
```

#### Manual Verification
```bash
# View full PDF text to check section names
🦞 You: "Extract full text from the paper to identify section structure"

# Check if paper has Methods at all
# Some papers (reviews, perspectives) may not have Methods sections
```

#### Check for Incompatible PDFs
```bash
# If you see "page-dimensions" RuntimeError:
# This indicates an incompatible PDF format
# System will automatically fall back to PyPDF2

# Verify fallback behavior
🦞 You: "Extract methods from PMID:12345678"
# Check provenance metadata: {"parser": "pypdf2", "fallback": true}
```

**Quality Metrics:**
- Docling achieves >90% Methods section detection on scientific papers
- PyPDF2 fallback achieves ~30% detection (first 10K chars naive truncation)
- Some papers legitimately don't have Methods sections (reviews, opinions)

### Issue: Page Dimensions RuntimeError

**Symptoms:**
```
RuntimeError: PDF contains page-dimensions errors
Falling back to PyPDF2 after detecting incompatible PDF format
```

**Causes:**
- PDF with malformed page dimension metadata
- Scanned PDFs with inconsistent page sizes
- PDFs created with non-standard tools
- Corrupted PDF files

**Solutions:**

#### Automatic Fallback (No Action Needed)
```bash
# Docling automatically detects this error and falls back to PyPDF2
# Extraction continues with reduced functionality:
# - Methods section still extracted (lower hit rate)
# - Tables won't be extracted
# - Formulas won't be detected
# - Provenance will show: {"parser": "pypdf2", "fallback": true}
```

#### Verify Fallback Success
```bash
🦞 You: "Extract methods from PMID:12345678"
# Check response for "Extraction completed using PyPDF2 fallback"

# Verify provenance metadata
🦞 You: "Show extraction provenance for the last paper"
# Should show: {"parser": "pypdf2", "fallback": true, "fallback_reason": "page-dimensions"}
```

#### PDF Repair (Advanced)
```bash
# Attempt to repair PDF with external tools
# Only if PyPDF2 fallback also fails

# Option 1: Ghostscript repair
gs -o repaired.pdf -sDEVICE=pdfwrite -dPDFSETTINGS=/prepress original.pdf

# Option 2: qpdf repair
qpdf --linearize original.pdf repaired.pdf

# Then try extraction again
🦞 You: "Extract methods from repaired.pdf"
```

**Expected Behavior:**
- System tries Docling first (max_retries=2 with memory management)
- If RuntimeError with "page-dimensions", immediately falls back to PyPDF2
- PyPDF2 extraction succeeds for most papers (~95% success rate)
- Fallback is logged in provenance for transparency

### Issue: Cache Issues

**Symptoms:**
- Unexpected cache hits for different papers
- Stale cache returning outdated extractions
- Cache consuming excessive disk space
- "Cache read failed" warnings

**Causes:**
- MD5 hash collisions (extremely rare)
- Manual cache modifications
- Corrupted cache files
- Cache directory permissions

**Solutions:**

#### Clear Cache
```bash
# Remove all cached documents
rm -rf ~/.lobster_workspace/literature_cache/parsed_docs/

# Clear specific paper cache
# Cache files named by MD5 hash of source URL
# Example: parsed_docs/abc123def456.json
```

#### Verify Cache Location
```bash
# Check cache directory exists and is writable
ls -la ~/.lobster_workspace/literature_cache/parsed_docs/

# Check cache file sizes
du -sh ~/.lobster_workspace/literature_cache/
# Typical: 500KB-2MB per paper
```

#### Monitor Cache Performance
```bash
# Cache hit: <100ms
# Cache miss (first parse): 2-5 seconds

# You'll see timing in responses:
# "Extraction completed in 0.08s (cached)" - Cache hit
# "Extraction completed in 3.2s" - Fresh parse
```

#### Cache Management Best Practices
```bash
# Cache is persistent across sessions (good for reproducibility)
# Automatic cache invalidation not implemented
# Manual cleanup recommended if:
# - Papers are updated/corrected by publishers
# - Testing different extraction parameters
# - Cache directory exceeds 1GB

# Selective cache cleanup
cd ~/.lobster_workspace/literature_cache/parsed_docs/
# Delete specific paper cache by finding its MD5 hash
```

**Cache Behavior:**
- Cache key: MD5 hash of source URL
- Storage format: JSON (Pydantic serialization)
- Non-fatal failures: Extraction continues if cache read/write fails
- Performance: 30-50x faster on cache hit

### Performance Optimization

#### Batch Processing
```bash
# Process 2-5 papers at a time (not more)
🦞 You: "Extract methods from PMID:123, PMID:456, PMID:789"

# System processes sequentially to avoid memory issues
# Wait for batch completion before starting next batch
```

#### Memory Management
```bash
# Docling's built-in retry logic:
# 1. First attempt: Parse with Docling
# 2. MemoryError → gc.collect() → Retry
# 3. Second MemoryError → Fall back to PyPDF2
# 4. RuntimeError (page-dimensions) → Immediate PyPDF2 fallback

# You don't need to manage retries manually
```

#### Troubleshooting Checklist

When extraction fails, check:
1. ✅ Docling installed: `pip list | grep docling`
2. ✅ Available memory: `free -h` (need >2GB free)
3. ✅ Paper accessibility: `🦞 "Check if PMID:12345 is accessible"`
4. ✅ Cache corruption: Clear cache and retry
5. ✅ Provenance metadata: Check for fallback indicators

### See Also

For detailed technical information about Docling integration:
- **[Publication Intelligence Deep Dive](37-publication-intelligence-deep-dive.md)** - Comprehensive technical guide
- **[Research Agent API](15-agents-api.md)** - Research Agent documentation
- **[Services API](16-services-api.md)** - ContentAccessService reference
- **[Literature Integration Workflow](06-data-analysis-workflows.md)** - Usage examples

---

## Analysis Failures

### Issue: No Cells Pass Quality Control

**Symptoms:**
- "0 cells remaining after filtering"
- All cells filtered out
- QC thresholds too strict

**Solutions:**

#### Review QC Thresholds
```bash
🦞 You: "Show QC metric distributions before applying any filters"
🦞 You: "What are the recommended QC thresholds for this data type?"
```

#### Adjust Filtering Parameters
```bash
🦞 You: "Use more lenient QC thresholds: >500 genes per cell and <30% mitochondrial"
🦞 You: "Filter based on median absolute deviation instead of fixed thresholds"
🦞 You: "Show me the effect of different threshold combinations"
```

#### Check Data Quality
```bash
🦞 You: "Is this data extremely low quality or are the thresholds inappropriate?"
🦞 You: "Generate comprehensive QC report with recommendations"
🦞 You: "Compare QC metrics to typical ranges for this experiment type"
```

### Issue: Clustering Produces Poor Results

**Symptoms:**
- All cells in one cluster
- Too many small clusters
- Clusters don't make biological sense

**Solutions:**

#### Optimize Clustering Parameters
```bash
🦞 You: "Test clustering resolutions from 0.1 to 2.0 and show silhouette scores"
🦞 You: "Try different clustering algorithms: Leiden, Louvain, hierarchical"
🦞 You: "Adjust number of neighbors from 5 to 50 and compare results"
```

#### Check Preprocessing
```bash
🦞 You: "Verify that data normalization was applied correctly"
🦞 You: "Check if highly variable genes were identified properly"
🦞 You: "Ensure PCA was computed with appropriate number of components"
```

#### Evaluate Data Quality
```bash
🦞 You: "Generate PCA plot to check for obvious batch effects"
🦞 You: "Show UMAP plot to assess overall data structure"
🦞 You: "Calculate and plot variance explained by each PC"
```

### Issue: No Significantly Differentially Expressed Genes

**Symptoms:**
- All p-values > 0.05
- No genes pass FDR threshold
- Effect sizes very small

**Solutions:**

#### Check Sample Sizes and Power
```bash
🦞 You: "How many samples per group do I have? Is this sufficient for DE analysis?"
🦞 You: "Calculate power analysis for detecting 2-fold changes"
🦞 You: "Show distribution of fold changes even if not significant"
```

#### Adjust Statistical Parameters
```bash
🦞 You: "Use less stringent FDR threshold (0.1 instead of 0.05)"
🦞 You: "Try different statistical methods: DESeq2, edgeR, limma"
🦞 You: "Test for fold change thresholds: |log2FC| > 0.5"
```

#### Investigate Experimental Design
```bash
🦞 You: "Check if experimental conditions are properly balanced"
🦞 You: "Look for confounding factors in sample metadata"
🦞 You: "Generate PCA plot colored by treatment to see separation"
```

---

## Performance Issues

### Issue: Analysis Takes Too Long

**Symptoms:**
- Processes hang for hours
- No progress updates
- System becomes unresponsive

**Solutions:**

#### Monitor Progress
```bash
🦞 You: "/progress"  # Check current operations
🦞 You: "/dashboard"  # Monitor system resources
```

#### Optimize Analysis Parameters
```bash
🦞 You: "Use faster approximate methods for initial exploration"
🦞 You: "Reduce number of genes/cells for testing parameters"
🦞 You: "Enable parallel processing using multiple CPU cores"
```

#### Use Cloud Resources
```bash
# Switch to cloud for intensive analyses
export LOBSTER_CLOUD_KEY="your-api-key"
🦞 You: "Move this analysis to cloud infrastructure for faster processing"
```

### Issue: Memory Errors

**Symptoms:**
- "Out of memory" errors
- System crashes
- Killed processes

**Solutions:**

#### Reduce Memory Usage
```bash
🦞 You: "Convert to sparse matrix format to save memory"
🦞 You: "Process data in smaller chunks"
🦞 You: "Remove unnecessary variables from workspace"
```

#### Optimize Data Types
```bash
🦞 You: "Use int32 instead of int64 for count data"
🦞 You: "Apply gene filtering to reduce matrix size"
🦞 You: "Subsample cells for parameter testing"
```

#### Monitor Memory Usage
```bash
🦞 You: "/dashboard"  # Check memory consumption
🦞 You: "Show memory usage of current datasets"
```

---

## Visualization Problems

### Issue: Plots Not Displaying

**Symptoms:**
- Empty plot windows
- "No plots generated" messages
- Visualization errors

**Solutions:**

#### Check Plot Generation
```bash
🦞 You: "/plots"  # List available plots
🦞 You: "Generate simple scatter plot to test visualization system"
```

#### Verify Data Requirements
```bash
🦞 You: "Do I have the required data for this plot type?"
🦞 You: "Show me the data structure needed for UMAP visualization"
```

#### Regenerate Plots
```bash
🦞 You: "Create UMAP plot with different parameters"
🦞 You: "Generate static plot instead of interactive version"
🦞 You: "Export plot data for external visualization"
```

### Issue: Poor Quality Visualizations

**Symptoms:**
- Overlapping labels
- Unclear color schemes
- Low resolution images

**Solutions:**

#### Improve Plot Parameters
```bash
🦞 You: "Create high-resolution plot (300 DPI) suitable for publication"
🦞 You: "Use distinct colors for better cluster separation"
🦞 You: "Adjust point sizes and transparency for better visibility"
```

#### Customize Appearance
```bash
🦞 You: "Generate plot with custom color palette"
🦞 You: "Create plot with larger fonts for better readability"
🦞 You: "Export plot with editable text for post-processing"
```

---

## Cloud Integration Issues

### Issue: Cloud API Not Working

**Symptoms:**
- Authentication failures
- "Cloud service unavailable"
- Timeout errors

**Solutions:**

#### Verify Cloud Setup
```bash
# Check API key is set
echo $LOBSTER_CLOUD_KEY

# Test cloud connectivity
🦞 You: "/session"  # Should show provider and session info
```

#### Switch to Local Mode
```bash
# Temporarily disable cloud
unset LOBSTER_CLOUD_KEY
🦞 You: "Continue analysis in local mode"
```

#### Retry Cloud Connection
```bash
# Re-export API key
export LOBSTER_CLOUD_KEY="your-api-key"
🦞 You: "Test cloud connection and retry analysis"
```

### Issue: Slow Cloud Processing

**Symptoms:**
- Long wait times
- Frequent timeouts
- Poor responsiveness

**Solutions:**

#### Optimize for Cloud
```bash
🦞 You: "Use cloud-optimized analysis parameters"
🦞 You: "Split large analyses into smaller chunks"
```

#### Check Network
```bash
# Test network speed
speedtest-cli

# Use local processing for small analyses
🦞 You: "Process this small dataset locally to save time"
```

---

## Agent & Tool Errors

### Issue: Agent Handoffs Fail

**Symptoms:**
- "Agent not available" errors
- Wrong agent selected
- Tool execution failures

**Solutions:**

#### Check Agent Status
```bash
🦞 You: "/status"  # Check available agents
🦞 You: "List all available agents and their capabilities"
```

#### Explicit Agent Selection
```bash
🦞 You: "Use the single-cell expert to analyze this scRNA-seq data"
🦞 You: "Hand this proteomics task to the MS proteomics expert"
```

#### Restart Session
```bash
# Exit and restart Lobster
🦞 You: "/exit"
lobster chat  # Fresh session
```

### Issue: Tool Execution Errors

**Symptoms:**
- "Tool failed" messages
- Incomplete analysis results
- Error tracebacks

**Solutions:**

#### Check Input Requirements
```bash
🦞 You: "What data is required for this analysis?"
🦞 You: "Verify that my data meets the requirements"
```

#### Use Alternative Tools
```bash
🦞 You: "Try alternative method for this analysis"
🦞 You: "Use simpler version of this analysis"
```

#### Report Detailed Errors
```bash
🦞 You: "Show detailed error message and suggest solutions"
🦞 You: "Generate debug information for this failed analysis"
```

---

## Memory & Resource Problems

### Issue: System Becomes Unresponsive

**Symptoms:**
- High CPU usage
- System freezing
- Slow response times

**Solutions:**

#### Monitor Resources
```bash
🦞 You: "/dashboard"  # Check system status
htop  # Monitor processes externally
```

#### Optimize Resource Usage
```bash
🦞 You: "Kill any running background processes"
🦞 You: "Reduce analysis complexity to save resources"
🦞 You: "Clear workspace cache to free memory"
```

#### Adjust Analysis Settings
```bash
🦞 You: "Use single-threaded processing to reduce CPU load"
🦞 You: "Process data in smaller batches"
```

---

## Output & Export Issues

### Issue: Cannot Export Results

**Symptoms:**
- "Export failed" errors
- Missing output files
- Permission denied errors

**Solutions:**

#### Check File Permissions
```bash
# Verify write permissions
ls -la ./
🦞 You: "Export to a different directory with write permissions"
```

#### Specify Export Format
```bash
🦞 You: "Export results as CSV files"
🦞 You: "Save plots in PNG format instead of SVG"
🦞 You: "Export data in H5AD format for preservation"
```

#### Use Alternative Export Methods
```bash
🦞 You: "/export results"  # Use CLI export command
🦞 You: "Show me the data so I can copy it manually"
```

### Issue: Missing Analysis Results

**Symptoms:**
- "No results found"
- Empty output directories
- Lost analysis history

**Solutions:**

#### Check Analysis Status
```bash
🦞 You: "Show me all completed analyses in this session"
🦞 You: "/data"  # Check loaded datasets
🦞 You: "/files"  # List all available files
```

#### Regenerate Missing Results
```bash
🦞 You: "Re-run the differential expression analysis"
🦞 You: "Recreate the clustering analysis from preprocessed data"
```

#### Access Analysis History
```bash
🦞 You: "Show analysis history and provenance tracking"
🦞 You: "Export session log with all commands and results"
```

---

## Advanced Troubleshooting

### Debug Mode and Logging

#### Enable Verbose Output
```bash
# Start with debug mode
LOBSTER_DEBUG=1 lobster chat

# Check log files
tail -f ~/.lobster/logs/lobster.log
```

#### Capture Error Details
```bash
🦞 You: "Enable detailed error reporting for troubleshooting"
🦞 You: "Show me the complete error traceback"
🦞 You: "Generate diagnostic report for this issue"
```

### Manual Intervention

#### Direct Data Access
```python
# Access data manager directly
from lobster.core.data_manager_v2 import DataManagerV2
from pathlib import Path

dm = DataManagerV2(workspace_path=Path(".lobster_workspace"))
print(dm.list_modalities())

# Inspect specific dataset
adata = dm.get_modality("your_dataset_name")
print(adata.obs.head())
```

#### Service-Level Debugging
```python
# Test individual services
from lobster.tools.preprocessing_service import PreprocessingService

service = PreprocessingService()
# Test service methods directly
```

### Recovery Procedures

#### Workspace Recovery
```bash
# Backup current workspace
cp -r .lobster_workspace .lobster_workspace_backup

# Clean and reinitialize
🦞 You: "Initialize fresh workspace and reload data"
```

#### Session Recovery
```bash
# Save current state
🦞 You: "/export session-state"

# Restart and restore
lobster chat
🦞 You: "/import session-state"
```

---

## ContentAccessService Issues (v0.2+)

### Issue: "ContentAccessService not available"

**Symptoms:**
```
ERROR: ContentAccessService not available or not initialized
ERROR: No providers registered for capability
```

**Causes:**
- Service not properly initialized in research_agent
- Provider registry configuration error
- Missing dependencies (docling, pypdf2, etc.)

**Solutions:**

#### Check Service Initialization
```bash
# Verify service is available
lobster chat
> "Query available capabilities"

# Should show:
# - AbstractProvider (fast abstracts)
# - PubMedProvider (literature search)
# - GEOProvider (dataset discovery)
# - PMCProvider (full-text, priority)
# - WebpageProvider (fallback, PDF support)
```

#### Verify Provider Registration
```bash
# Check which providers are active
> "What providers are available for literature access?"

# Expected output shows all 5 providers with priorities
```

#### Reinstall Dependencies
```bash
# Install Docling for PDF support
pip install lobster[docling]

# Verify installation
python -c "import docling; print('Docling OK')"
```

**Restart with Fresh Environment:**
```bash
# Clean workspace and restart
rm -rf ~/.lobster_workspace/
lobster chat --workspace ~/.lobster_new
```

### Issue: PDF Parsing Failures (Docling)

**Symptoms:**
```
ERROR: Failed to parse PDF content
WARNING: Docling service failed to extract content
MemoryError during PDF parsing
```

**Causes:**
- Corrupted or malformed PDF file
- Scanned PDFs without OCR text layer
- Large PDF files causing memory issues
- Docling dependencies not properly installed

**Solutions:**

#### Install Docling Dependencies
```bash
# Full Docling installation
pip install lobster[docling]

# Verify dependencies
python -c "import docling.document_converter; print('Docling installed')"
```

#### Handle Large PDFs
```bash
# For PDFs >50MB, increase memory limit
export LOBSTER_MAX_FILE_SIZE_MB=500

# Or use abstract-only for initial review
> "Get abstract for PMID:12345"  # Fast, always works
```

#### Try Alternative Methods
```bash
# If Docling fails, system automatically falls back to PyPDF2
# No action needed - fallback is automatic

# Manually request abstract instead of full-text
> "Extract abstract and keywords from PMID:12345"
```

#### Check PDF Format
```bash
# Test PDF integrity
pdfinfo your_file.pdf

# For scanned PDFs, use OCR first
# (Docling doesn't support image-only PDFs)
```

**Expected Behavior:**
- Docling tries first (max_retries=2)
- Automatic fallback to PyPDF2 on failure
- Provenance logs which parser was used

### Issue: Rate Limiting for Web Scraping

**Symptoms:**
```
ERROR: HTTP 429 Too Many Requests
WARNING: Rate limit exceeded for webpage extraction
ERROR: Publisher blocking automated access
```

**Causes:**
- Rapid sequential requests to same publisher
- Publisher anti-bot protection (Cloudflare)
- IP-based rate limiting

**Solutions:**

#### Use PMC Priority Path
```bash
# ContentAccessService tries PMC XML API first (fast, no rate limits)
> "Read full publication PMID:35042229"

# PMC covers 30-40% of biomedical literature
# 10x faster than webpage scraping
```

#### Let Service Handle Backoff
```bash
# Service implements exponential backoff automatically
# Just wait and retry after 60 seconds

# Check capabilities to see which providers are available
> "Query capabilities"
```

#### Use DOI URLs
```bash
# Direct DOI URLs often work better than publisher pages
> "Read content from https://doi.org/10.1038/s41586-021-12345-6"
```

#### Alternative: Preprints and Open Access
```bash
# Search for open access versions
> "Search bioRxiv for BRCA1 breast cancer"

# Filter by open access
> "Search literature cancer therapy filters:open_access=true"
```

### Issue: Authentication Issues for Paywalled Content

**Symptoms:**
```
ERROR: Content is behind paywall
INFO: PMC full-text not available for this publication
WARNING: Paper is not accessible: paywalled
```

**Causes:**
- Paper not in open access repositories
- Institution access required
- Not in PMC open access subset (70% of papers)

**Solutions:**

#### Three-Tier Cascade
```bash
# System automatically tries:
# 1. PMC XML API (30-40% coverage, fast)
# 2. Webpage/PDF extraction (60-70% coverage, slower)
# 3. Error with suggestions if paywalled

> "Read full publication PMID:12345"
# Automatic cascade - no manual intervention needed
```

#### Use Abstract + Methods
```bash
# For paywalled papers, get what you can
> "Get abstract for PMID:12345"
> "Extract methods from abstract"  # Limited but useful
```

#### Search for Preprints
```bash
> "Find bioRxiv preprint for [paper title]"
> "Search medRxiv for COVID-19 clinical trial"
```

#### Check Open Access Availability
```bash
> "Is PMID:12345 available in open access?"
> "Find open access version of DOI:10.1038/xxx"
```

**Alternative Strategies:**
- Request author preprints directly
- Check institutional library access
- Use Supplementary Materials (often freely available)

---

## WorkspaceContentService Issues (v0.2+)

### Issue: File Not Found in Workspace

**Symptoms:**
```
ERROR: Identifier 'publication_PMID12345' not found in workspace
FileNotFoundError: ~/.lobster_workspace/literature/pmid_12345.json
```

**Causes:**
- Content not cached yet
- Incorrect identifier format
- Wrong workspace directory

**Solutions:**

#### List Cached Content
```bash
# Check what's actually cached
> "What content do I have cached?"
> "Show me cached publications"
> "List all cached datasets"

# Use /workspace command
> /workspace
```

#### Verify Identifier Format
```bash
# Correct format: lowercase with underscores
# ✅ Correct: publication_PMID35042229
# ❌ Wrong: PMID:35042229 (has colon)
# ❌ Wrong: publication_pmid_35042229 (duplicate prefix)

# Check identifier in cache directory
ls ~/.lobster_workspace/literature/
```

#### Cache Content First
```bash
# Must cache before accessing
> "Read full publication PMID:35042229"
# This automatically caches to workspace

# Or explicitly cache
> "Cache PMID:35042229 in literature workspace"
```

#### Verify Workspace Path
```bash
# Check workspace exists
ls -la ~/.lobster_workspace/

# Should have subdirectories:
# - literature/
# - data/
# - metadata/

# Check in Lobster
> /workspace
```

### Issue: Workspace Path Resolution Issues

**Symptoms:**
```
ERROR: Permission denied: ~/.lobster_workspace/literature/
ERROR: Cannot create directory
OSError: [Errno 30] Read-only file system
```

**Causes:**
- Insufficient file permissions
- Workspace directory doesn't exist
- Disk full or read-only mount

**Solutions:**

#### Create Workspace Directories
```bash
# Create all required directories
mkdir -p ~/.lobster_workspace/{literature,data,metadata}
chmod 755 ~/.lobster_workspace/

# Verify creation
ls -la ~/.lobster_workspace/
```

#### Check Permissions
```bash
# Fix ownership
chown -R $USER:$USER ~/.lobster_workspace/

# Fix permissions
chmod -R u+rw ~/.lobster_workspace/
```

#### Check Disk Space
```bash
# Check available space
df -h ~

# If disk full, clean old caches
du -sh ~/.lobster_workspace/
find ~/.lobster_workspace/ -mtime +30 -delete  # Remove files >30 days old
```

#### Use Custom Workspace
```bash
# Specify different workspace path
export LOBSTER_WORKSPACE=/path/to/workspace
lobster chat

# Or at runtime
lobster chat --workspace /mnt/data/lobster_workspace
```

### Issue: Permission Errors Reading Workspace Files

**Symptoms:**
```
PermissionError: [Errno 13] Permission denied: '~/.lobster_workspace/literature/pmid_12345.json'
```

**Causes:**
- File created by different user
- Incorrect file permissions (chmod 000)
- SELinux or AppArmor restrictions (Linux)

**Solutions:**

#### Fix File Ownership
```bash
# Take ownership of all workspace files
chown -R $USER:$USER ~/.lobster_workspace/
```

#### Fix Permissions
```bash
# Make files readable/writable
chmod -R u+rw ~/.lobster_workspace/

# For directories, add execute permission
chmod -R u+rwx ~/.lobster_workspace/*/
```

#### Check SELinux (Linux Only)
```bash
# Check if SELinux is enforcing
getenforce

# If 'Enforcing', temporarily disable for testing
sudo setenforce 0

# Or configure SELinux policy properly
# (production systems should not disable SELinux)
```

#### Fresh Workspace
```bash
# Nuclear option: delete and recreate
rm -rf ~/.lobster_workspace/
lobster chat  # Will recreate with correct permissions
```

---

## Caching System Issues (v0.2+)

### Issue: Cache Hit/Miss Debugging

**Understanding Cache Behavior:**

Lobster v0.2+ has two-tier caching:
1. **Session cache** (in-memory, fast, temporary)
2. **Workspace cache** (filesystem, persistent)

**Debug Cache Status:**
```bash
# Check cache statistics
> /workspace
# Shows: cached publications, datasets, metadata

# List cached content by type
> "Show me all cached publications"
> "List cached datasets"

# Check cache directory directly
ls -lh ~/.lobster_workspace/literature/
ls -lh ~/.lobster_workspace/data/
```

**Force Cache Refresh:**
```bash
# Bypass cache and re-fetch
> "Read full publication PMID:12345 with force refresh"

# Delete specific cache file
rm ~/.lobster_workspace/literature/pmid_35042229.json
```

### Issue: Cache Invalidation Strategies

**When to Invalidate:**
- Dataset metadata updated on GEO/PRIDE
- Publication retracted or corrected
- Workspace migration to new system
- Cache corruption detected

**Manual Invalidation:**
```bash
# Delete specific cached item
rm ~/.lobster_workspace/literature/pmid_35042229.json

# Clear all cached publications
rm -rf ~/.lobster_workspace/literature/*.json

# Clear all cached datasets
rm -rf ~/.lobster_workspace/data/*.json

# Nuclear option: clear entire workspace
rm -rf ~/.lobster_workspace/
lobster chat  # Starts fresh
```

**Automatic Invalidation (v0.2+):**
```bash
# Cached content has timestamps
# Service checks age before using
# Default TTL:
# - Publications: 7 days
# - Datasets: 24 hours (metadata changes frequently)
# - Metadata: 24 hours

# No manual invalidation needed for most cases
```

### Issue: Disk Space Issues with Large Caches

**Symptoms:**
```
ERROR: No space left on device
WARNING: Workspace size exceeding 1GB
OSError: [Errno 28] No space left on device
```

**Check Disk Usage:**
```bash
# Check total workspace size
du -sh ~/.lobster_workspace/

# Break down by subdirectory
du -h ~/.lobster_workspace/ | sort -h

# Find largest cached items
find ~/.lobster_workspace/ -type f -size +10M -exec ls -lh {} \;

# Check available disk space
df -h ~
```

**Solutions:**

#### Clean Old Cache Files
```bash
# Remove files older than 30 days
find ~/.lobster_workspace/ -type f -mtime +30 -delete

# Remove files older than 7 days
find ~/.lobster_workspace/ -type f -mtime +7 -delete

# Verify cleanup
du -sh ~/.lobster_workspace/
```

#### Archive Old Workspace
```bash
# Backup to compressed archive
tar -czf lobster_workspace_backup_$(date +%Y%m%d).tar.gz ~/.lobster_workspace/

# Delete old workspace
rm -rf ~/.lobster_workspace/

# Restore if needed
tar -xzf lobster_workspace_backup_YYYYMMDD.tar.gz -C ~/
```

#### Use Workspace Size Limits
```bash
# Set maximum workspace size
export LOBSTER_MAX_WORKSPACE_SIZE_MB=500
lobster chat

# Service will warn when limit approached
```

#### Move to Larger Disk
```bash
# Move workspace to external/network drive
mv ~/.lobster_workspace /mnt/large_disk/lobster_workspace

# Create symbolic link
ln -s /mnt/large_disk/lobster_workspace ~/.lobster_workspace

# Verify
ls -la ~/.lobster_workspace
```

---

## Protein Structure Visualization Issues (v0.2+)

### Issue: PyMOL Installation Issues

**Symptoms:**
```
ERROR: PyMOL not found in PATH
WARNING: PyMOL visualization will not execute
INFO: Command script generated at: 1AKE_commands.pml
```

**Verification:**
```bash
# Check PyMOL installation
which pymol

# Test PyMOL (headless mode)
pymol -c -Q

# Check version
pymol -c -r "print(cmd.get_version())"
```

**Solutions by Platform:**

#### macOS - Automated
```bash
# Use Makefile (recommended)
cd lobster
make install-pymol

# Verify
pymol -c -Q
```

#### macOS - Manual
```bash
# Install via Homebrew
brew install brewsci/bio/pymol

# Verify installation
which pymol
pymol -c -Q
```

#### Linux (Ubuntu/Debian)
```bash
# Install from repositories
sudo apt-get update
sudo apt-get install pymol

# Verify
which pymol
```

#### Linux (Fedora/RHEL)
```bash
# Install via DNF
sudo dnf install pymol

# Verify
which pymol
```

#### Docker (Pre-installed)
```bash
# PyMOL is pre-installed in Docker image
docker run -it omicsos/lobster:latest pymol -c -Q

# No installation needed in Docker
```

#### Windows
```powershell
# Download from https://pymol.org/
# Install using GUI installer
# Add to PATH via System Environment Variables
```

**Fallback Without PyMOL:**
```bash
# Agent still generates command scripts
> "Visualize protein structure 1AKE"

# Manual execution later when PyMOL installed
pymol 1AKE_commands.pml  # Interactive mode
pymol -c 1AKE_commands.pml  # Batch mode (headless)
```

### Issue: PDB File Format Errors

**Symptoms:**
```
ERROR: Failed to parse PDB file: 1AKE.pdb
ERROR: Invalid PDB ID format
ValueError: PDB ID must be 4 characters
```

**Causes:**
- Invalid PDB ID format (must be exactly 4 alphanumeric characters)
- Corrupted download
- Wrong file format

**Solutions:**

#### Validate PDB ID
```bash
# ✅ Correct formats:
# - 1AKE (4 chars, alphanumeric)
# - 4HHB, 3A5D, 7BV2

# ❌ Wrong formats:
# - AKE (too short)
# - 1AKEE (too long)
# - 1-AKE (invalid character: hyphen)
# - 1ake (works but use uppercase for consistency)

# Use correct format
> "Fetch protein structure 1AKE"
```

#### Re-download Structure
```bash
# Use cached version
> "Fetch protein structure 1AKE"

# Force re-download
> "Fetch protein structure 1AKE with force refresh"

# Verify file integrity
ls -lh protein_structures/1AKE.*
# Should be >10KB for valid structure
```

#### Try Different Format
```bash
# mmCIF format (default, recommended)
> "Fetch protein structure 1AKE format=cif"

# Legacy PDB format
> "Fetch protein structure 1AKE format=pdb"
```

#### Verify Structure Exists
```bash
# Check on RCSB website
# https://www.rcsb.org/structure/1AKE

# Search for alternative structures
> "Find protein structures for gene BRCA1"
```

### Issue: Structure Rendering Failures

**Symptoms:**
```
ERROR: PyMOL execution timed out
ERROR: Failed to generate visualization
WARNING: PyMOL process exited with error code 1
```

**Causes:**
- Very large structure (>100K atoms)
- Insufficient memory
- Graphics driver issues (interactive mode)
- Corrupted structure file

**Solutions:**

#### Use Batch Mode
```bash
# Batch mode is faster, no GUI required
> "Visualize 1AKE with PyMOL mode=batch"

# Generates PNG image without opening GUI
```

#### Simplify Representation
```bash
# Cartoon is fastest (default)
> "Visualize 1AKE style=cartoon"

# Surface/spheres are slower
> "Visualize 1AKE style=surface"  # Slower, more memory
```

#### Check Structure Size
```bash
# Fetch structure first to see metadata
> "Fetch protein structure 1AKE"

# Look for: "Total atoms: X" in output
# If >100K atoms, expect longer rendering time
```

#### Increase Timeout
```bash
# For very large structures
export LOBSTER_PYMOL_TIMEOUT_SECONDS=300

# Restart Lobster
lobster chat
```

#### Use Headless Mode Manually
```bash
# Generate PNG without GUI
pymol -c 1AKE_commands.pml

# Faster than interactive mode
```

#### Check Memory
```bash
# Linux
free -h

# macOS
vm_stat

# Ensure >2GB free for large structures
```

### Issue: Interactive Mode Not Launching

**Symptoms:**
```
INFO: Launching PyMOL GUI...
WARNING: PyMOL GUI did not launch
ERROR: DISPLAY environment variable not set
```

**Causes:**
- No display environment (SSH session without X11)
- PyMOL not in PATH
- X11 forwarding disabled

**Solutions:**

#### Check Display Environment
```bash
# Should be set for GUI apps
echo $DISPLAY

# Expected values:
# - :0 (local display)
# - localhost:10.0 (X11 forwarding)
```

#### Enable X11 Forwarding (SSH)
```bash
# SSH with X11 forwarding
ssh -X user@host

# Or on macOS (requires XQuartz)
ssh -Y user@host
```

#### Test X11
```bash
# Simple X11 test
xeyes  # Should show GUI window

# If xeyes fails, X11 not configured
```

#### Use Batch Mode Instead
```bash
# Batch mode doesn't require display
> "Visualize 1AKE mode=batch style=cartoon"

# Generates PNG without GUI
```

#### Execute Script Manually Later
```bash
# Save command script now
> "Visualize 1AKE execute=false"

# Execute later when you have GUI access
pymol 1AKE_commands.pml
```

---

## S3 Backend Issues (v0.2+)

### Issue: AWS Credentials Configuration

**Symptoms:**
```
ERROR: Unable to locate credentials
ERROR: S3 backend connection failed
botocore.exceptions.NoCredentialsError
```

**Solutions:**

#### Configure AWS CLI
```bash
# Interactive configuration
aws configure

# Enter:
# AWS Access Key ID: AKIAIOSFODNN7EXAMPLE
# AWS Secret Access Key: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
# Default region: us-east-1
# Default output format: json
```

#### Set Environment Variables
```bash
# Export credentials
export AWS_ACCESS_KEY_ID=your_key_id
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1

# Verify
echo $AWS_ACCESS_KEY_ID
```

#### Use Credentials File
```bash
# Create credentials file
mkdir -p ~/.aws
cat > ~/.aws/credentials << EOF
[default]
aws_access_key_id = your_key_id
aws_secret_access_key = your_secret_key
EOF

# Set permissions
chmod 600 ~/.aws/credentials
```

#### Verify Credentials
```bash
# Test S3 access
aws s3 ls

# Should list your buckets
# If error, credentials are wrong
```

#### Test in Lobster
```bash
lobster chat
> "Use S3 backend for storage"
> /session  # Should show session info with loaded data
```

### Issue: S3 Bucket Permissions

**Symptoms:**
```
ERROR: Access Denied (403)
ERROR: Cannot write to S3 bucket: your-bucket-name
botocore.exceptions.ClientError: An error occurred (AccessDenied)
```

**Required IAM Permissions:**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:ListBucket",
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject"
      ],
      "Resource": [
        "arn:aws:s3:::your-bucket-name",
        "arn:aws:s3:::your-bucket-name/*"
      ]
    }
  ]
}
```

**Verify Permissions:**
```bash
# Test bucket listing
aws s3 ls s3://your-bucket-name/

# Test write permission
echo "test" | aws s3 cp - s3://your-bucket-name/test.txt

# Test read permission
aws s3 cp s3://your-bucket-name/test.txt -

# Test delete permission
aws s3 rm s3://your-bucket-name/test.txt
```

**Check IAM Policy:**
```bash
# Get user policies
aws iam list-user-policies --user-name your-username

# Get policy document
aws iam get-user-policy --user-name your-username --policy-name your-policy
```

**Solutions:**
```bash
# If permissions insufficient, contact AWS admin
# Or create new IAM user with correct permissions

# Alternative: Use local storage
> "Use local filesystem backend instead of S3"
```

### Issue: Network Connectivity Issues

**Symptoms:**
```
ERROR: Connection timeout to S3
ERROR: Unable to reach S3 endpoint
requests.exceptions.ConnectionError: Max retries exceeded
botocore.exceptions.EndpointConnectionError
```

**Causes:**
- Network firewall blocking AWS endpoints
- VPN issues
- DNS resolution failure
- Regional endpoint unavailable

**Solutions:**

#### Test S3 Connectivity
```bash
# Ping S3 endpoint
ping s3.amazonaws.com

# Test HTTPS connection
curl -I https://s3.amazonaws.com

# Should return HTTP 403 (forbidden but reachable)
```

#### Try Different Region
```bash
# Change default region
export AWS_DEFAULT_REGION=us-west-2
lobster chat

# Or specify in config
aws configure set default.region us-west-2
```

#### Check DNS Resolution
```bash
# Test DNS lookup
nslookup s3.amazonaws.com

# Should resolve to AWS IP addresses
```

#### Use VPC Endpoint (AWS Environment)
```bash
# If running in AWS EC2/ECS
export AWS_S3_ENDPOINT=https://vpce-xxxxx.s3.us-east-1.vpce.amazonaws.com

# VPC endpoints bypass internet gateway
```

#### Increase Timeout
```bash
# For slow connections
export LOBSTER_S3_TIMEOUT_SECONDS=60
lobster chat
```

#### Check Firewall Rules
```bash
# Ensure outbound HTTPS (443) allowed to:
# - s3.amazonaws.com
# - *.s3.amazonaws.com
# - s3.us-east-1.amazonaws.com (region-specific)
```

**Alternative: Use Local Storage**
```bash
# If S3 unavailable, switch to local
> "Use local filesystem backend"
> /session  # Verify session workspace
```

---

## Getting Help

### When to Seek Support

Contact support if you encounter:
- Persistent crashes or system instability
- Data corruption or loss
- Reproducible bugs in core functionality
- Performance issues that can't be resolved

### Information to Include

When reporting issues, provide:
1. **System Information**: OS, Python version, Lobster version
2. **Error Messages**: Complete error text and tracebacks
3. **Data Description**: Dataset type, size, source
4. **Reproduction Steps**: Exact commands that trigger the issue
5. **Expected vs Actual Results**: What you expected vs what happened

### Community Resources

- **GitHub Issues**: [Report bugs and feature requests](https://github.com/the-omics-os/lobster/issues)
- **Discord Community**: Real-time help and discussion
- **Documentation**: [Complete guide and tutorials](README.md)
- **Example Notebooks**: Working examples and best practices

### Quick Diagnostic Command

```bash
🦞 You: "Run system diagnostics and generate troubleshooting report"
```

This command generates a comprehensive report including:
- System specifications
- Installation status
- Current workspace state
- Recent error logs
- Performance metrics

---

This troubleshooting guide covers the most common issues encountered in Lobster AI. For additional help, consult the [FAQ](29-faq.md) or reach out to the community through the support channels listed above.
