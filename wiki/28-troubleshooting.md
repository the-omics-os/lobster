# Troubleshooting Guide

This comprehensive troubleshooting guide provides solutions to common issues encountered while using Lobster AI. Each problem includes symptoms, causes, and step-by-step solutions.

## Table of Contents

1. [Installation & Setup Issues](#installation--setup-issues)
   - API Keys Not Working
   - CLI Interface Not Working
   - **Rate Limit Errors (429)** ⚠️
   - **Authentication Errors (401)**
   - **Network Errors**
   - **Quota Exceeded Errors**
2. [Data Loading Problems](#data-loading-problems)
3. [Publication Intelligence & Docling Issues](#publication-intelligence--docling-issues) 🆕
   - Docling Not Installed
   - MemoryError During PDF Parsing
   - Methods Section Not Found
   - Page Dimensions RuntimeError
   - Cache Issues
4. [Analysis Failures](#analysis-failures)
5. [Performance Issues](#performance-issues)
6. [Visualization Problems](#visualization-problems)
7. [Cloud Integration Issues](#cloud-integration-issues)
8. [Agent & Tool Errors](#agent--tool-errors)
9. [Memory & Resource Problems](#memory--resource-problems)
10. [Output & Export Issues](#output--export-issues)
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
echo $OPENAI_API_KEY
echo $AWS_BEDROCK_ACCESS_KEY
echo $AWS_BEDROCK_SECRET_ACCESS_KEY

# Check .env file exists and is correctly formatted
cat .env
```

#### Fix .env File
```bash
# Create or update .env file
cat > .env << EOF
OPENAI_API_KEY=your-openai-api-key-here
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
🦞 You: "/status"  # Check which provider is active
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

### Issue: FTP Download Failures or Corruption (v2.3+)

**Symptoms:**
```
⚠️  FTP download failed after 3 retries
Corrupted gzip file detected
Large files (>50MB) timeout or fail
```

**Automatic Recovery (v2.3+):**
Lobster AI now includes robust error handling:
- System automatically retries with exponential backoff (2s, 4s, 8s delays)
- Chunked downloads (8KB blocks) prevent corruption for large files
- MD5 validation and gzip integrity checks detect corruption before caching
- **No user action required** - System handles retry logic automatically

**Manual Intervention (If Automatic Retry Fails):**
```bash
# Clear cache and force fresh download
rm -rf ~/.lobster_workspace/geo_cache/GSE12345*
🦞 You: "Download GSE12345 with fresh cache"

# Check internet connectivity
ping ftp.ncbi.nlm.nih.gov

# Verify GEO accession exists
🦞 You: "Search for GSE12345 in GEO database"
```

**Technical Details (v2.3+):**
- FTP retry logic with exponential backoff
- Chunked FTP downloads for files >70MB
- Gzip validation with 32KB chunked reading
- Automatic cache poisoning prevention

---

### Issue: VDJ Data "Duplicate Barcode" Errors (v2.3+)

**Symptoms:**
```
⚠️  Duplicate cell barcodes detected: 48%
Dataset GSE248556 rejected due to data quality issues
Validation failed: non-unique cell barcodes
```

**Cause (FIXED in v2.3+):**
VDJ/TCR/BCR sequencing data legitimately has duplicate cell barcodes because each cell can express multiple receptor chains (heavy + light chain, alpha + beta chain). The system now automatically detects VDJ data types and accepts duplicates.

**Expected Behavior (v2.3+):**
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

### Issue: H5AD Export Failures with GEO Metadata (v2.3+)

**Symptoms:**
```
TypeError: Cannot serialize mixed types to H5AD
ValueError: Boolean values not supported in AnnData metadata
KeyError: Metadata column contains None values
```

**Automatic Resolution (v2.3+):**
Lobster AI now sanitizes GEO metadata before H5AD export:
- `bool → string` ("True", "False")
- `None → ""` (empty string)
- Mixed types → string representation
- Empty columns dropped automatically
- **No user action required** - Metadata cleaned transparently

**When It Happens:**
GEO datasets often have poor metadata quality with:
- Boolean flags as actual bool type (not H5AD-compatible)
- Missing values as None (not serializable)
- Mixed integer/string columns

**Manual Verification:**
```bash
# Check metadata before export
🦞 You: "Show metadata summary for current dataset"

# Force H5AD export with sanitization
🦞 You: "Export to H5AD with metadata sanitization"
```

---

### Issue: Bulk RNA-seq "Inverted Dimensions" Warning (v2.3+)

**Symptoms:**
```
⚠️  Matrix dimensions may be inverted: 187,697 features × 4 observations
Expected: samples × genes for bulk RNA-seq
Applying automatic transpose...
```

**Automatic Resolution (v2.3+):**
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

### Issue: Malformed GEO Accessions (v2.3+)

**Symptoms:**
```
❌ Invalid accession format: GDS200157007
Expected format: GSE/GSM/GPL/GDS + digits
Accession has 9 digits, expected 4-7
```

**Resolution (FIXED in v2.3+):**
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

**Note**: Most GEO issues are now handled automatically in v2.3+ with robust error handling, retry logic, and intelligent validation.

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

### Issue: DOI/PMID Not Resolving to Accessible URLs (v2.3+ Fix)

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
The v2.3+ system provides detailed logging of resolution attempts:

```bash
# Successful resolution shows:
INFO Detected identifier (DOI): 10.1101/..., resolving to URL...
INFO Resolved via preprint server: https://www.biorxiv.org/content/...
INFO Content extraction successful (pdf auto-detected) in 2.3s

# Failed resolution shows:
WARNING Paper 10.18632/aging.204666 is not accessible: paywalled
INFO Alternative suggestions: [institutional access, preprints, author contact]
```

**Expected Behavior (v2.3+):**
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
- **[Services API](16-services-api.md)** - PublicationIntelligenceService reference
- **[Literature Integration Workflow](06-data-analysis-workflows.md#literature-integration-workflow)** - Usage examples

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
🦞 You: "/status"  # Should show cloud mode if working
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
- **Documentation**: [Complete guide and tutorials](../README.md)
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