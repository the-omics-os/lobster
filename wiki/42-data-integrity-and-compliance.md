# Data Integrity, Security, and Compliance

**Comprehensive Reference**: Security architecture, compliance features, and deployment guidance
**Available**: Lobster AI v0.3.4+
**Compliance**: 21 CFR Part 11, ALCOA+, GxP, HIPAA, GDPR, ISO/IEC 27001, SOC 2

---

## 1. Overview

### 1.1 Purpose and Audience

**What this document covers**: Lobster AI's comprehensive security architecture, data integrity features, compliance capabilities, and deployment guidance for regulated environments. This reference enables:

- **QA teams** to assess Lobster for regulated use (GxP, HIPAA, clinical trials)
- **DevOps teams** to deploy securely (local, cloud, validated environments)
- **Compliance officers** to map Lobster features to regulatory requirements (21 CFR Part 11, ALCOA+, ISO 27001)
- **Enterprise customers** to understand security posture and compliance readiness

**Target audiences**:

| Role | Primary Needs | Key Sections |
|------|--------------|--------------|
| **QA / Compliance** | Assess GxP readiness, audit trails | [3](#3-audit-trail--provenance), [10](#10-compliance-features-for-regulated-environments) |
| **DevOps / IT Security** | Deploy securely, monitor systems | [9](#9-deployment-security), [11](#11-security-best-practices) |
| **Analysts** | Understand data integrity, use securely | [2](#what-youll-see), [5](#5-secure-code-execution), [8](#8-validation--data-quality) |
| **Enterprise Buyers** | Evaluate security for procurement | [1.2](#12-why-security-matters-in-bioinformatics), [10.1](#101-gxp-ready-checklist) |

---

### 1.2 Why Security Matters in Bioinformatics

**Bioinformatics presents unique security challenges**:

1. **Sensitive data** - Patient genomic data (HIPAA), clinical trial results (GxP), proprietary research (trade secrets)
2. **Reproducibility crisis** - 70% of researchers unable to reproduce others' results (Nature survey)
3. **Data integrity** - Single base pair error can invalidate conclusions, impact patient care
4. **Regulatory complexity** - FDA, EMA, HIPAA, GDPR all impose different requirements
5. **Long-term value** - Analyses must remain valid for 7-10+ years (regulatory retention)

**Lobster's security philosophy**:

| Principle | Implementation | Benefit |
|-----------|----------------|---------|
| **Security by default** | W3C-PROV enabled, integrity manifests automatic | No opt-in required |
| **Audit everything** | Every operation logged with attribution | Complete audit trail |
| **Cryptographic proof** | SHA-256 hashes, RSA-2048 signatures | Tamper-evident records |
| **Principle of least privilege** | Workspace isolation, subscription tiers | Minimal attack surface |
| **Graceful degradation** | Local mode (max security) or cloud (scalability) | Flexible deployment |
| **Standards compliance** | W3C-PROV, NIST algorithms, ISO formats | Industry best practices |

---

### 1.3 Compliance Coverage Matrix

**What regulations does Lobster support?**

| Regulation | Current Status | Deployment Mode | Key Features |
|------------|---------------|-----------------|--------------|
| **21 CFR Part 11** | ✅ Ready | Local + Cloud | Audit trails, tamper-evidence, validation support |
| **ALCOA+** | ✅ Ready | Local + Cloud | All 9 principles implemented (see [10.1](#101-gxp-ready-checklist)) |
| **GxP (GAMP 5)** | ⚠️ Partial | Local (Cat 4 ready), Cloud (validation TBD) | IQ/OQ/PQ templates available |
| **HIPAA** | ⚠️ Conditional | Local (ready), Cloud (BAA required) | Encryption, audit logs, access control |
| **GDPR** | ⚠️ Conditional | Local (ready), Cloud (region + DPA) | Data residency, anonymization, retention |
| **ISO/IEC 27001** | ✅ Ready | Local + Cloud | Information security controls (A.8.1-A.8.24) |
| **SOC 2 Type II** | ⚠️ Partial | Cloud (AWS certified), Lobster (pending) | AWS inherits certification |

**Feature coverage by section**:

| Section | Regulation Support | Key Features |
|---------|-------------------|--------------|
| **[2] Data Integrity Manifest** | 21 CFR Part 11 § 11.10(a) | SHA-256 hashes, tamper-evidence |
| **[3] Audit Trail** | 21 CFR Part 11 § 11.10(d,e) | W3C-PROV, AnalysisStep IR, session tracking |
| **[4] Access Control** | HIPAA, GDPR, ISO 27001 | License management, tier enforcement, API keys |
| **[5] Secure Execution** | 21 CFR Part 11 § 11.10(k) | Subprocess isolation, forbidden modules |
| **[6] Data Protection** | HIPAA, GDPR | Workspace isolation, concurrent access protection |
| **[7] Network Security** | ISO 27001 A.13 | Rate limiting, timeout handling, HTTPS |
| **[8] Validation** | 21 CFR Part 11 § 11.10(k) | Schema validation, pre-download checks |
| **[9] Deployment** | SOC 2, HIPAA | Docker, S3 encryption, AWS security |
| **[10] Compliance** | GxP, 21 CFR Part 11 | ALCOA+ mapping, deployment patterns, SOPs |
| **[11] Best Practices** | All | Environment security, access control, monitoring |

---

### 1.4 Document Structure

**How to use this guide**:

**For quick assessment** (QA teams, 30 minutes):
1. Read [1.3 Compliance Coverage Matrix](#13-compliance-coverage-matrix) - Understand regulation support
2. Read [10.1 GxP-Ready Checklist](#101-gxp-ready-checklist) - See ALCOA+ and 21 CFR Part 11 mapping
3. Read [10.2 Deployment Patterns](#102-deployment-patterns-for-regulated-environments) - Choose deployment model
4. Review [10.3 SOPs](#103-standard-operating-procedures-sops) - Template procedures

**For deep technical review** (DevOps, 2-4 hours):
1. Read entire document (Sections 2-11)
2. Review linked detailed documentation (wiki pages)
3. Test features in staging environment
4. Validate with IQ/OQ/PQ scripts ([10.4](#104-validation-testing-for-gxp))

**For compliance audit** (inspectors, 1-2 hours):
1. Start with [3. Audit Trail](#3-audit-trail--provenance) - Verify W3C-PROV implementation
2. Review [2. Data Integrity Manifest](#what-youll-see) - Understand cryptographic controls
3. Check [10. Compliance Features](#10-compliance-features-for-regulated-environments) - Map to regulations
4. Request provenance export and verify hashes

**Navigation tips**:
- Each section starts with **"What it is"** (executive summary)
- **Tables** provide quick reference (capabilities, compliance benefits)
- **Code examples** show practical usage
- **"For complete details"** links to authoritative documentation (no duplication)

---

## 2. Data Integrity Manifest

### 2.1 What You'll See

When you export a notebook using `/pipeline export`, the second cell contains a data integrity manifest:

```markdown
## 🔒 Data Integrity Manifest

**Purpose**: Cryptographic verification of data integrity (ALCOA+ compliance)

{
  "data_integrity_manifest": {
    "generated_at": "2026-01-01T14:23:45.123456",
    "provenance": {
      "session_id": "session_20260101_142000",
      "sha256": "7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069",
      "activities": 15,
      "entities": 8
    },
    "input_files": {
      "geo_gse109564.h5ad": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    },
    "system": {
      "lobster_version": "0.3.4",
      "git_commit": "dd2c126f",
      "python_version": "3.13.9",
      "platform": "darwin"
    }
  }
}
```

---

## Understanding the Manifest

### Provenance Section
```json
"provenance": {
  "session_id": "session_20260101_142000",
  "sha256": "7f83b165...",
  "activities": 15,
  "entities": 8
}
```

**What this proves**:
- Links notebook to specific analysis session
- Cryptographic hash of the session's audit trail
- Documents scope: 15 analysis steps, 8 data entities

---

### Input Files Section
```json
"input_files": {
  "geo_gse109564.h5ad": "e3b0c442...",
  "geo_gse109564_filtered.h5ad": "5d41402a..."
}
```

**What this proves**:
- Exact data files used in analysis
- Each file has unique cryptographic fingerprint
- Any modification changes the hash

---

### System Section
```json
"system": {
  "lobster_version": "0.3.4",
  "git_commit": "dd2c126f",
  "python_version": "3.13.9",
  "platform": "darwin"
}
```

**What this proves**:
- Exact software version documented
- Enables long-term reproducibility
- Environment can be reconstructed

---

## How to Verify Data Integrity

### Basic Verification

**macOS/Linux**:
```bash
shasum -a 256 geo_gse109564.h5ad
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
```

**Windows PowerShell**:
```powershell
Get-FileHash -Algorithm SHA256 geo_gse109564.h5ad
```

**Python**:
```python
import hashlib

def verify_file_hash(filepath, expected_hash):
    """Verify file matches expected SHA-256 hash."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)

    actual_hash = sha256.hexdigest()
    if actual_hash == expected_hash:
        print("✅ VERIFIED: File hash matches manifest")
        return True
    else:
        print("❌ MISMATCH: File has been modified!")
        print(f"Expected: {expected_hash}")
        print(f"Actual:   {actual_hash}")
        return False

# Usage
verify_file_hash(
    "geo_gse109564.h5ad",
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
)
```

---

### Automated Verification Script

Create a verification script for your notebooks:

```python
#!/usr/bin/env python3
"""Verify data integrity for Lobster AI notebook."""

import json
import hashlib
import nbformat
from pathlib import Path

def verify_notebook_integrity(notebook_path, data_directory):
    """Verify all input files match manifest hashes."""
    # Read notebook
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    # Find manifest cell
    manifest = None
    for cell in nb.cells:
        if "data_integrity_manifest" in cell.source:
            # Extract JSON from cell
            lines = cell.source.split("\n")
            json_lines = []
            in_json = False
            for line in lines:
                if line.strip().startswith("{"):
                    in_json = True
                if in_json:
                    json_lines.append(line)
                if line.strip().endswith("}") and in_json:
                    break
            manifest = json.loads("\n".join(json_lines))
            break

    if not manifest:
        print("❌ No integrity manifest found in notebook")
        return False

    # Verify each input file
    input_files = manifest["data_integrity_manifest"]["input_files"]
    all_verified = True

    for filename, expected_hash in input_files.items():
        filepath = Path(data_directory) / filename

        if not filepath.exists():
            print(f"⚠️  {filename}: File not found")
            all_verified = False
            continue

        # Calculate hash
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        actual_hash = sha256.hexdigest()

        if actual_hash == expected_hash:
            print(f"✅ {filename}: Verified")
        else:
            print(f"❌ {filename}: HASH MISMATCH")
            print(f"   Expected: {expected_hash}")
            print(f"   Actual:   {actual_hash}")
            all_verified = False

    return all_verified

# Usage
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: verify_integrity.py <notebook.ipynb> <data_directory>")
        sys.exit(1)

    verified = verify_notebook_integrity(sys.argv[1], sys.argv[2])
    sys.exit(0 if verified else 1)
```

**Save as**: `verify_integrity.py`

**Usage**:
```bash
python verify_integrity.py my_analysis.ipynb ~/.lobster/
```

---

## Common Scenarios

### Scenario 1: Hashes Match ✅

```bash
$ shasum -a 256 geo_gse109564.h5ad
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
```

**Meaning**: File is authentic and unchanged
**Action**: Proceed with analysis review

---

### Scenario 2: Hash Mismatch ❌

```bash
$ shasum -a 256 geo_gse109564.h5ad
a1b2c3d4... (DIFFERENT HASH)
```

**Possible Causes**:
1. File was re-downloaded or updated (intentional)
2. File corruption (disk error, network issue)
3. File was modified (accidental or malicious)

**Action**:
1. Check file modification date
2. Verify with original data source
3. If intentional update: Re-run analysis to get new notebook with updated hashes
4. If unexpected: Investigate security incident

---

### Scenario 3: File Not Found

```bash
$ shasum -a 256 geo_gse109564.h5ad
shasum: geo_gse109564.h5ad: No such file or directory
```

**Meaning**: Data file has been moved or deleted

**Action**:
1. Check if file was archived
2. Restore from backup if needed
3. Cannot reproduce analysis without original file

---

## Why This Matters

### For Regulatory Compliance

| Principle | Requirement | How Manifest Helps |
|-----------|-------------|-------------------|
| **ALCOA+ "Original"** | Prove data is authentic | SHA-256 verifies file identity |
| **ALCOA+ "Accurate"** | Detect tampering | Hash mismatch reveals changes |
| **21 CFR Part 11** | Tamper-evident records | Cryptographic binding |
| **GxP Audit Trail** | Document system state | Version info captured |

---

### For Scientific Reproducibility

**Problem**: "Which version of the data did I use?"

**Solution**: The hash uniquely identifies the exact file version:
- Same data = Same hash (every time)
- Different data = Different hash (guaranteed)
- Cannot fake a hash (mathematically impossible)

---

## Best Practices

### 1. Verify Hashes Before Review

When reviewing a colleague's notebook:
```bash
# Extract hashes from manifest
# Verify each input file
# Only proceed if all hashes match
```

### 2. Archive Data with Notebooks

Store notebooks alongside their input data:
```
analysis_project/
├── my_analysis.ipynb          # Notebook with manifest
├── data/
│   ├── geo_gse109564.h5ad     # Input file
│   └── metadata.csv           # Metadata file
└── verify_integrity.py        # Verification script
```

### 3. Include Verification in SOPs

**Standard Operating Procedure Example**:
1. Analyst exports notebook
2. QA verifies hashes
3. If verified → Review code
4. If mismatch → Investigate before review

### 4. Document Hash Verification

Keep records of verification:
```
Analysis: GSE109564_clustering
Notebook: my_analysis.ipynb
Verified: 2026-01-01 by QA-UserName
Hash Status: ✅ All inputs verified
Reviewer: [Signature]
```

---

## FAQ

### Q: Is this automatic?

**A**: Yes. Every notebook export includes the manifest automatically. No extra steps required.

### Q: Does this slow down my analysis?

**A**: No. Hashing happens only during export (~0.5 seconds for typical files). Zero impact on analysis performance.

### Q: What if I need to update my data?

**A**: Re-run the analysis and export a new notebook. The new notebook will have new hashes reflecting the updated data. Both notebooks remain valid records of what data was used at each point in time.

### Q: Can I use this in non-regulated environments?

**A**: Absolutely! Even outside GxP environments, data integrity verification is a scientific best practice. It helps you:
- Track which version of data was used
- Prevent accidental use of wrong files
- Document your analysis provenance

### Q: What hash algorithm is used?

**A**: SHA-256 (Secure Hash Algorithm 256-bit). This is:
- NIST-approved standard
- Used by GitHub, Bitcoin, SSL certificates
- Collision-resistant (virtually impossible to find duplicates)
- Industry standard for data integrity

---

---

## 2.2 H5AD Validation and Compression (v3.4.2+)

**What it is**: Lobster includes utilities for validating H5AD file integrity and optimizing storage via compression. These features ensure data quality and efficient storage in production deployments.

**H5AD validation** (`core/utils/h5ad_utils.py`):

| Check | Purpose | Error Detection |
|-------|---------|----------------|
| **File format** | Verify valid HDF5 structure | Detects corrupted files |
| **Required keys** | Check for `.obs`, `.var`, `.X` | Detects incomplete files |
| **Shape consistency** | Verify `n_obs × n_vars` matches | Detects truncated data |
| **Compression valid** | Test gzip/lzf decompression | Detects compression errors |
| **Metadata present** | Check for `.uns` metadata | Detects missing annotations |

**Validation usage**:
```python
from lobster.core.utils.h5ad_utils import validate_h5ad

# Validate H5AD file before analysis
is_valid, error_msg = validate_h5ad("geo_gse109564.h5ad")

if is_valid:
    print("✅ H5AD file is valid")
    adata = sc.read_h5ad("geo_gse109564.h5ad")
else:
    print(f"❌ Validation failed: {error_msg}")
    # Handle error (re-download, investigate corruption)
```

**H5AD compression** (storage optimization):

| Compression | Method | Ratio | Speed | Use Case |
|-------------|--------|-------|-------|----------|
| **gzip (level 6)** | Deflate | 5-10x | Medium | Default, balanced |
| **gzip (level 9)** | Deflate | 8-15x | Slow | Long-term archival |
| **lzf** | LZF | 3-5x | Fast | Real-time processing |

**Compression usage**:
```python
from lobster.core.utils.h5ad_utils import compress_h5ad

# Compress H5AD for archival
original_size = Path("geo_gse109564.h5ad").stat().st_size
compress_h5ad("geo_gse109564.h5ad", compression="gzip", compression_opts=9)
compressed_size = Path("geo_gse109564.h5ad").stat().st_size

print(f"Original: {original_size / 1e9:.2f} GB")
print(f"Compressed: {compressed_size / 1e9:.2f} GB")
print(f"Ratio: {original_size / compressed_size:.1f}x")
# Output: Original: 2.4 GB, Compressed: 0.3 GB, Ratio: 8.0x
```

**Compliance benefits**:

- ✅ **Data integrity** - Pre-load validation catches corruption
- ✅ **Storage efficiency** - 5-10x compression reduces costs
- ✅ **Quality assurance** - Automated validation in CI/CD
- ✅ **Audit trail** - Validation results logged to provenance

**For complete implementation details**, see:
- [H5AD Utils](../lobster/core/utils/h5ad_utils.py) (validation and compression functions)
- [Data Management](20-data-management.md#h5ad-optimization) (usage patterns)

---

## 2.3 Atomic File Operations

**What it is**: Lobster uses atomic file operations (temp file + fsync + atomic replace) to ensure crash-safe writes for critical files (session metadata, queues, provenance logs). This prevents data corruption from crashes, power failures, or kill signals.

**Atomic write pattern**:
```python
def atomic_write_json(path: Path, data: dict):
    """Crash-safe JSON write."""
    temp_path = path.with_suffix('.tmp')

    # Step 1: Write to temp file
    with open(temp_path, 'w') as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())  # Force write to disk (bypass OS cache)

    # Step 2: Atomic replace (POSIX guarantee)
    os.replace(temp_path, path)  # Atomic on POSIX systems
```

**Why atomic writes matter**:

| Scenario | Without Atomic Writes | With Atomic Writes |
|----------|----------------------|-------------------|
| **Crash during write** | Partial data written, file corrupted | Temp file discarded, original intact |
| **Power failure** | File may contain garbage | Temp file or complete file, never partial |
| **Kill signal** | Incomplete JSON, parse error | Complete file or original preserved |
| **Concurrent access** | Race conditions, corruption | Combined with file locks, safe |

**Protected files** (using atomic writes):
- `.session.json` - Session metadata
- `provenance.json` - W3C-PROV audit trail
- `download_queue.jsonl` - Download queue entries
- `publication_queue.jsonl` - Publication queue entries
- `cache_metadata.json` - Cache tracking

**POSIX atomicity guarantee**:
```
os.replace(src, dst) on POSIX:
- Atomically replaces dst with src
- If dst exists: overwritten atomically
- If crash occurs: dst is either old OR new (never partial)
- Thread-safe + process-safe (when combined with locks)
```

**Compliance benefits**:

- ✅ **Data integrity** - Crash-safe writes prevent corruption
- ✅ **ALCOA+ "Accurate"** - File contents always valid
- ✅ **Audit trail integrity** - Provenance never corrupted
- ✅ **Reliability** - Production-ready for 24/7 operation

**For complete implementation details**, see:
- [Queue Storage](../lobster/core/queue_storage.py) (atomic write implementation)
- [Download Queue - Concurrency](35-download-queue-system.md#concurrency-infrastructure) (usage patterns)
- [CLAUDE.md - Concurrency Pattern](../CLAUDE.md#45-patterns--abstractions) (developer guide)

---

## 3. Audit Trail & Provenance

### 3.1 W3C-PROV Compliance

**What it is**: Lobster implements the World Wide Web Consortium (W3C) PROV standard for complete audit trails of all analysis operations. Every action is recorded as a directed acyclic graph (DAG) with three key components:

- **Activities**: What was done (e.g., clustering, quality control, differential expression)
- **Entities**: What data was used and generated (datasets, plots, metadata files)
- **Agents**: Who/what performed the action (singlecell_expert, data_expert, human users)

This creates an immutable, traceable record from raw data download through final publication-ready results.

**Why it matters for compliance**:

| Compliance Principle | Requirement | How W3C-PROV Helps |
|---------------------|-------------|-------------------|
| **ALCOA+ "Traceable"** | Complete operation history | DAG links all operations to source data |
| **ALCOA+ "Attributable"** | User/agent identification | Every activity attributed to specific agent |
| **21 CFR Part 11 § 11.10(e)** | Audit trail requirements | Timestamped, immutable activity log |
| **ISO/IEC 27001:2022** | Change logging | Complete provenance graph exportable as JSON |

**Key capabilities**:

| Feature | Description | Compliance Benefit |
|---------|-------------|--------------------|
| Activity tracking | All operations logged with parameters | Complete audit trail |
| Entity lineage | Data provenance from source to result | Traceability |
| Agent attribution | User/agent identification for each step | Accountability |
| Temporal ordering | Timestamp-based activity sequencing | Contemporaneous recording |
| Exportable format | W3C-PROV JSON standard | Portability & long-term archival |
| Query interface | Programmatic provenance queries | Audit support |

**Quick start**:
```bash
# View current session provenance
lobster status

# Export provenance for specific session (W3C-PROV JSON)
# Provenance automatically saved to: .lobster_workspace/provenance.json
```

**Example provenance graph** (simplified):
```
[PubMed Search] → [GEO GSE109564 Metadata] → [Download Dataset]
                                               ↓
                                          [geo_gse109564.h5ad]
                                               ↓
                                          [Quality Control]
                                               ↓
                                          [Filter Cells/Genes]
                                               ↓
                                          [Normalization]
                                               ↓
                                          [Clustering]
                                               ↓
                                          [Annotated Dataset]
```

**For complete implementation details**, see:
- [Data Management - Provenance System](20-data-management.md#provenance-system) (architecture, API reference)
- [CLAUDE.md Section 4.4](../CLAUDE.md#44-provenance--analysisstep-ir-w3cprov) (developer patterns)
- [Core API - ProvenanceTracker](14-core-api.md#provenance) (class documentation)

---

### 3.2 AnalysisStep Intermediate Representation (IR)

**What it is**: Every analysis operation returns an `AnalysisStep` object that captures the complete specification for reproducing that step. This IR (Intermediate Representation) enables:

1. **Notebook export** - Generates executable Python code from analysis history
2. **Parameter validation** - Ensures reproducibility through schema enforcement
3. **Audit trail integration** - Links provenance to executable protocols
4. **Method documentation** - Self-documenting analysis workflows

**3-Tuple Pattern** (all services follow this):
```python
def analyze(adata: AnnData, **params) -> Tuple[AnnData, Dict[str, Any], AnalysisStep]:
    # ... processing ...
    return processed_adata, stats, analysis_step_ir
```

**What gets recorded in AnalysisStep**:

| Field | Purpose | Example |
|-------|---------|---------|
| `operation` | Method name | `"scanpy.pp.filter_cells"` |
| `tool_name` | Service method | `"quality_service.assess_quality"` |
| `description` | Human explanation | `"Filter cells based on QC metrics"` |
| `library` | Software library | `"scanpy"`, `"pyDESeq2"` |
| `code_template` | Jinja2 template | `"sc.pp.filter_cells(adata, min_genes={{ min_genes }})"` |
| `imports` | Required imports | `["import scanpy as sc"]` |
| `parameters` | Actual values used | `{"min_genes": 200, "max_genes": 8000}` |
| `parameter_schema` | Validation rules | Types, defaults, constraints |
| `input_entities` | Data dependencies | `["geo_gse109564.h5ad"]` |
| `output_entities` | Generated data | `["geo_gse109564_filtered.h5ad"]` |
| `execution_context` | Runtime metadata | Timestamps, agent, session ID |

**Compliance benefits**:

- ✅ **Complete method documentation** - Every parameter recorded
- ✅ **Parameter validation** - Schema prevents invalid configurations
- ✅ **Reproducible protocols** - Code templates generate executable notebooks
- ✅ **Audit trail integration** - AnalysisStep embedded in W3C-PROV activities
- ✅ **ALCOA+ "Accurate"** - Parameter schema prevents data entry errors

**Example AnalysisStep** (clustering):
```python
AnalysisStep(
    operation="scanpy.tl.leiden",
    tool_name="clustering_service.perform_clustering",
    description="Leiden clustering with resolution 0.5",
    library="scanpy",
    code_template="sc.tl.leiden(adata, resolution={{ resolution }})",
    imports=["import scanpy as sc"],
    parameters={"resolution": 0.5, "random_state": 42},
    parameter_schema={
        "resolution": {"type": "float", "default": 1.0, "min": 0.0},
        "random_state": {"type": "int", "default": 0}
    },
    input_entities=["geo_gse109564_normalized.h5ad"],
    output_entities=["geo_gse109564_clustered.h5ad"]
)
```

**For complete implementation details**, see:
- [CLAUDE.md Section 4.4](../CLAUDE.md#44-provenance--analysisstep-ir-w3cprov) (developer guide, service pattern)
- [Data Management - AnalysisStep](20-data-management.md#analysisstep-intermediate-representation) (API reference)

---

### 3.3 Session and Tool Usage Tracking

**What it is**: Lobster maintains session-level metadata that tracks all tool invocations, agent handoffs, and data operations across multi-turn conversations. This enables:

- **Cross-session traceability** - Continue analysis from previous sessions
- **Usage auditing** - Track which agents/tools were used and when
- **Session restoration** - Recover from interruptions
- **Compliance reporting** - Generate audit reports per session

**Session metadata structure**:
```json
{
  "session_id": "session_20260101_142000",
  "created_at": "2026-01-01T14:20:00.123456Z",
  "last_updated": "2026-01-01T15:45:32.789012Z",
  "workspace_path": "/Users/analyst/.lobster_workspace",
  "modalities": {
    "geo_gse109564": {
      "created_at": "2026-01-01T14:23:15Z",
      "n_obs": 5000,
      "n_vars": 2000,
      "layers": ["counts", "normalized"],
      "last_modified": "2026-01-01T15:30:00Z"
    }
  },
  "tool_usage": [
    {
      "tool_name": "search_pubmed",
      "agent": "research_agent",
      "timestamp": "2026-01-01T14:20:30Z",
      "parameters": {"query": "single-cell CRISPR screening"},
      "result": "Found 15 publications"
    },
    {
      "tool_name": "download_geo_dataset",
      "agent": "data_expert",
      "timestamp": "2026-01-01T14:23:00Z",
      "parameters": {"accession": "GSE109564"},
      "result": "Downloaded 5000 cells × 2000 genes"
    }
  ],
  "agent_handoffs": [
    {
      "from": "supervisor",
      "to": "research_agent",
      "timestamp": "2026-01-01T14:20:15Z",
      "reason": "User requested literature search"
    },
    {
      "from": "research_agent",
      "to": "data_expert",
      "timestamp": "2026-01-01T14:22:45Z",
      "reason": "Download queue entry created for GSE109564"
    }
  ]
}
```

**Key capabilities**:

| Feature | Description | Compliance Benefit |
|---------|-------------|--------------------|
| Unique session IDs | Timestamp-based unique identifiers | Session-level traceability |
| UTC timestamps | All times in UTC (ISO 8601) | Contemporaneous recording |
| Agent attribution | Every action linked to agent | ALCOA+ "Attributable" |
| Tool usage log | Complete invocation history | Audit trail support |
| Cross-session continuity | `--session-id latest` continues work | Analysis reproducibility |
| Automatic backup | Session saved after each operation | Crash recovery |

**Session commands**:
```bash
# Start new session with custom ID
lobster query --session-id "project_gse109564" "Download GSE109564 and cluster"

# Continue previous session
lobster query --session-id latest "Add differential expression analysis"

# View current session status
lobster status

# Export session (includes provenance + metadata)
# Automatically saved to: .lobster_workspace/.session.json
```

**Compliance benefits**:

- ✅ **ALCOA+ "Contemporaneous"** - Timestamped in real-time
- ✅ **ALCOA+ "Attributable"** - User/agent identification
- ✅ **21 CFR Part 11 § 11.10(e)** - Session-level audit capability
- ✅ **ISO/IEC 27001** - Access logging requirements

**For complete implementation details**, see:
- [Data Management - Session Metadata](20-data-management.md#session-metadata) (session structure, API)
- [CLI Commands - Session Management](05-cli-commands.md#session-management) (usage examples)

---

### 3.4 Provenance Hash and Tamper-Evidence

**What it is**: Lobster creates a cryptographic hash (SHA-256) of the complete provenance graph (activities + entities + agents) and embeds it in the notebook's Data Integrity Manifest. This creates a tamper-evident link between the notebook and its audit trail.

**How it works**:

1. **Analysis phase**: Provenance graph built as operations execute
2. **Export phase**: Complete provenance serialized to JSON
3. **Hash calculation**: SHA-256 computed over canonical JSON representation
4. **Manifest embedding**: Hash included in notebook's second cell
5. **Verification**: Recompute hash from provenance.json and compare

**Verification guarantees**:

| Property | Guarantee | Attack Prevention |
|----------|-----------|-------------------|
| **Immutability** | Any provenance modification changes hash | Prevents retroactive edits |
| **Binding** | Hash proves notebook ↔ provenance linkage | Prevents data substitution |
| **Completeness** | Hash covers all activities/entities | Prevents omission of steps |
| **Non-repudiation** | Provenance includes agent attribution | Accountability enforcement |

**Example integrity manifest** (provenance section):
```json
{
  "data_integrity_manifest": {
    "generated_at": "2026-01-01T15:45:00Z",
    "provenance": {
      "session_id": "session_20260101_142000",
      "sha256": "7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069",
      "activities": 15,
      "entities": 8,
      "agents": 3,
      "time_span": {
        "start": "2026-01-01T14:20:00Z",
        "end": "2026-01-01T15:30:00Z"
      }
    },
    "input_files": { ... },
    "system": { ... }
  }
}
```

**Verification workflow**:
```python
import hashlib
import json

# Read provenance from workspace
with open(".lobster_workspace/provenance.json") as f:
    provenance = json.load(f)

# Compute hash (canonical JSON, sorted keys)
provenance_json = json.dumps(provenance, sort_keys=True)
computed_hash = hashlib.sha256(provenance_json.encode()).hexdigest()

# Compare to manifest hash
manifest_hash = "7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069"

if computed_hash == manifest_hash:
    print("✅ VERIFIED: Provenance matches notebook manifest")
else:
    print("❌ TAMPERED: Provenance has been modified!")
```

**Compliance benefits**:

- ✅ **21 CFR Part 11 § 11.10(a)** - Tamper-evident audit trails
- ✅ **ALCOA+ "Original"** - Proves audit trail authenticity
- ✅ **ISO/IEC 27001:2022** - Integrity monitoring
- ✅ **GxP** - Supports 21 CFR Part 11 requirements for electronic records

**For complete implementation details**, see:
- [Data Integrity Manifest - Provenance Hash](#understanding-the-manifest) (current page, Section 2)
- [Core API - ProvenanceTracker](14-core-api.md#provenance) (hash computation)
- [Notebook Exporter](../CLAUDE.md#where--code-layout) (`core/notebook_exporter.py`)

---

## 4. Access Control & Authentication

### 4.1 License Management System

**What it is**: Lobster uses a cryptographic license service (AWS-hosted) to validate entitlements and enforce subscription tiers. The system uses server-side RSA signing with client-side verification via JWKS (JSON Web Key Set), following industry-standard JWT/JWS patterns.

**Architecture** (AWS Serverless):
- **AWS Lambda** - License service endpoints (Python 3.12, ARM64)
- **API Gateway** - REST API (`https://x6gm9vfgl5.execute-api.us-east-1.amazonaws.com/v1`)
- **DynamoDB** - Entitlements, customers, audit logs
- **AWS KMS** - RSA-2048 signing key (HSM-backed, private key never leaves AWS)
- **S3 + CloudFront** - JWKS public endpoint for signature verification

**How it works** (5-step activation):

1. User activates: `lobster activate <cloud-key>`
2. CLI calls AWS license service: `POST /api/v1/activate`
3. Service validates key, signs entitlement via AWS KMS (server-side)
4. Entitlement saved to: `~/.lobster/license.json`
5. CLI verifies signature via JWKS on each run (client-side)

**Entitlement file structure**:
```json
{
  "cloud_key": "lbstr_abc123...",
  "customer_id": "cust_databiomix",
  "subscription_tier": "premium",
  "features": ["metadata_assistant", "proteomics_expert"],
  "issued_at": "2026-01-01T12:00:00Z",
  "expires_at": "2027-01-01T12:00:00Z",
  "signature": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
  "revocation_status": {
    "last_checked": "2026-01-01T18:00:00Z",
    "is_revoked": false
  }
}
```

**Security properties**:

| Property | Implementation | Benefit |
|----------|----------------|---------|
| **Server-side signing** | Private key in AWS KMS (never exposed) | Cannot be compromised by client |
| **Client-side verification** | JWKS public key from S3 | Standard JWT pattern, offline verification |
| **Tamper-evident** | RSA-2048 signature validation | Any modification invalidates entitlement |
| **Revocation checking** | Periodic status checks (24h interval) | Supports license revocation |
| **Audit logging** | DynamoDB AuditLogs table | Complete activation/refresh history |

**CLI commands**:
```bash
# Activate cloud license
lobster activate lbstr_abc123...

# Check license status and tier
lobster status

# Output shows:
# Subscription Tier: premium
# Features: metadata_assistant, proteomics_expert
# Expires: 2027-01-01
```

**Compliance benefits**:

- ✅ **Access control** - Tier-based feature restrictions
- ✅ **Audit trail** - All activations logged
- ✅ **Non-repudiation** - Cryptographic signatures prove entitlements
- ✅ **Revocation support** - Invalidate compromised keys

**For complete implementation details**, see:
- [Premium Licensing Technical Guide](../../docs/PREMIUM_LICENSING.md) (CTO/technical guide, 4 phases)
- [Premium Testing Checklist](../../docs/PREMIUM_TESTING_CHECKLIST.md) (QA/verification)
- [Commercial Licensing FAQ](../docs/commercial_licensing_faq.md) (customer-facing, AGPL-3.0 explanation)
- [CLAUDE.md - License Manager](../CLAUDE.md#411-feature-tiering--conditional-activation) (developer integration)

---

### 4.2 Subscription Tier Enforcement

**What it is**: Role-based access control (RBAC) implemented via three subscription tiers (FREE, PREMIUM, ENTERPRISE). Each tier unlocks specific agents and features, enforced at CLI startup and agent creation.

**Three-tier model**:

| Tier | Agents | Features | Use Case |
|------|--------|----------|----------|
| **FREE** | 7 agents | Core workflows | Open-source, academic |
| | • supervisor | Basic analysis | Individual researchers |
| | • research_agent | Literature search | Education |
| | • data_expert | Data loading | |
| | • transcriptomics_expert | RNA-seq | |
| | • visualization_expert | Plotting | |
| | • machine_learning_expert | ML models | |
| | • protein_structure_visualization_expert | Structural biology | |
| **PREMIUM** | +2 agents | Advanced workflows | Biotech, CRO |
| | • metadata_assistant | Publication processing | Small teams |
| | • proteomics_expert | Mass spec | Commercial use |
| **ENTERPRISE** | +custom packages | Customer-specific | Pharma, large orgs |
| | • Via `lobster-custom-*` | Proprietary agents | Validated environments |

**Enforcement mechanism** (4 layers):

1. **License Manager** (`core/license_manager.py`) - Validates tier at CLI startup
2. **Agent Registry** (`config/agent_registry.py`) - Checks tier before creating agents
3. **Handoff Restrictions** (`config/subscription_tiers.py`) - Prevents unauthorized delegation
4. **Component Registry** (`core/component_registry.py`) - Premium services check availability

**Example tier checking**:
```python
from lobster.config.subscription_tiers import is_agent_available

# Check if agent is available for current tier
if is_agent_available("metadata_assistant", current_tier="free"):
    # Agent available (False for FREE tier)
    create_agent()
else:
    # Show upgrade message
    print("⚠️ metadata_assistant requires PREMIUM tier")
```

**Handoff restrictions** (FREE tier example):
```python
# supervisor can handoff to research_agent (✅)
# supervisor can handoff to data_expert (✅)
# research_agent CANNOT handoff to metadata_assistant (❌ - PREMIUM only)
```

**Graceful degradation**:
```
User: "Process my publication queue and filter metadata"
Lobster (FREE): "⚠️ Publication queue processing requires PREMIUM tier
                 (metadata_assistant agent). Available with PREMIUM subscription.
                 Visit https://omics-os.com/pricing for upgrade options."
```

**Compliance benefits**:

- ✅ **Access control** - Feature restrictions without authentication overhead
- ✅ **Commercial licensing** - Supports AGPL-3.0 + commercial model
- ✅ **Audit trail** - Tier logged in session metadata
- ✅ **Customer segmentation** - Different capabilities per contract

**For complete implementation details**, see:
- [Architecture - Subscription Tiers](18-architecture-overview.md#subscription-tiers) (tier definitions, agent list)
- [CLAUDE.md - Feature Tiering](../CLAUDE.md#411-feature-tiering--conditional-activation) (development rules)
- [Commercial Licensing FAQ](../docs/commercial_licensing_faq.md) (customer guide)

---

### 4.3 API Key Security

**What it is**: Secure management of third-party API credentials (LLM providers, NCBI, cloud services) via environment variables, workspace-level configuration, and secret management integration.

**Supported API keys**:

| Key | Purpose | Required | Rate Limit Impact |
|-----|---------|----------|------------------|
| `ANTHROPIC_API_KEY` | Anthropic Direct LLM | Conditional* | N/A |
| `AWS_BEDROCK_ACCESS_KEY` | AWS Bedrock LLM | Conditional* | N/A |
| `AWS_BEDROCK_SECRET_ACCESS_KEY` | AWS Bedrock LLM | Conditional* | N/A |
| `GOOGLE_API_KEY` | Google Gemini LLM (v0.4.0+) | Conditional* | N/A |
| `NCBI_API_KEY` | NCBI E-utilities | Optional | 3 → 10 req/s |
| `LOBSTER_CLOUD_KEY` | Cloud service activation | Optional | N/A |

*At least one LLM provider required (Anthropic OR AWS Bedrock OR Google Gemini OR Ollama local)

**Configuration hierarchy** (priority order):
1. **Workspace-level** `.env` - Per-project keys (highest priority)
2. **Global** `~/.lobster/.env` - User-wide keys
3. **Environment variables** - System-level (e.g., CI/CD)

**Security best practices**:

| Practice | Implementation | Security Benefit |
|----------|----------------|------------------|
| **Never commit keys** | `.gitignore` includes `.env` files | Prevents credential leaks |
| **Use .env templates** | `.env.example` (no real keys) | Safe to commit, guides setup |
| **Rotate regularly** | Quarterly for NCBI, on team changes for LLMs | Limits exposure window |
| **Scope appropriately** | Workspace > Global > System | Principle of least privilege |
| **Validate on startup** | CLI checks key validity | Fail fast on invalid keys |

**Example secure setup**:
```bash
# ✅ GOOD: Use .env file (gitignored)
cat > .env << EOF
ANTHROPIC_API_KEY=sk-ant-api03-...
NCBI_API_KEY=abc123...
EOF

# ✅ GOOD: Workspace-specific keys
mkdir -p ~/project1/.lobster_workspace
echo "ANTHROPIC_API_KEY=sk-ant-project1..." > ~/project1/.env

# ❌ BAD: Hardcode in scripts (committed to git)
export ANTHROPIC_API_KEY="sk-ant-api03-..."  # Will be committed!

# ❌ BAD: Share keys across users
echo "ANTHROPIC_API_KEY=shared-key" > /etc/lobster/.env  # Security risk
```

**Enterprise secret management**:
```bash
# AWS Secrets Manager integration
export ANTHROPIC_API_KEY=$(aws secretsmanager get-secret-value \
  --secret-id lobster/anthropic-key \
  --query SecretString \
  --output text)

# HashiCorp Vault integration
export ANTHROPIC_API_KEY=$(vault kv get -field=api_key secret/lobster/anthropic)

# CI/CD environment injection (GitHub Actions example)
# Stored in repository secrets, injected at runtime
- name: Run Lobster analysis
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
  run: lobster query "Analyze GSE109564"
```

**Compliance benefits**:

- ✅ **Access control** - Workspace-level isolation
- ✅ **Audit trail** - Key usage logged (not key values)
- ✅ **Credential rotation** - Supports regular key updates
- ✅ **Principle of least privilege** - Scoped per project

**For complete implementation details**, see:
- [Configuration Guide - API Keys](03-configuration.md#api-key-setup) (setup instructions)
- [Configuration - Provider Setup](03-configuration.md#llm-provider-configuration) (LLM-specific)
- [CLAUDE.md - Environment Setup](../CLAUDE.md#52-environment-setup) (developer reference)

---

### 4.4 Cloud vs Local Security Models

**What it is**: Lobster supports two deployment modes with different security postures: **Local Mode** (default, maximum data control) and **Cloud Mode** (managed infrastructure, cloud key required).

**Local Mode** (Default):

| Property | Status | Details |
|----------|--------|---------|
| **Data location** | ✅ Local machine | Never leaves your environment |
| **Network egress** | ⚠️ Minimal | Only for data downloads (GEO, PubMed) |
| **Workspace isolation** | ✅ Full control | User manages permissions |
| **API key management** | ⚠️ User-managed | Store in `.env` files |
| **Suitable for** | ✅ Sensitive data | HIPAA, confidential, air-gapped |
| **Compliance** | ✅ GxP-ready | Full audit trail, local storage |

**Cloud Mode** (`LOBSTER_CLOUD_KEY` set):

| Property | Status | Details |
|----------|--------|---------|
| **Data location** | ⚠️ Cloud API | Data sent to cloud service |
| **Network dependency** | ⚠️ Required | Must have internet connection |
| **License validation** | ✅ Automatic | Cloud key verified via AWS |
| **Scalable compute** | ✅ Managed | Auto-scaling infrastructure |
| **API key management** | ✅ Server-side | No local credential storage |
| **Suitable for** | ⚠️ Non-sensitive | Validated, non-PHI data |
| **Compliance** | ⚠️ Check BAA | HIPAA requires Business Associate Agreement |

**Decision matrix** (choosing a mode):

| Requirement | Local | Cloud |
|-------------|-------|-------|
| Air-gapped environment | ✅ | ❌ |
| Sensitive patient data (HIPAA) | ✅ | ⚠️ (BAA required) |
| Large-scale processing (100s of datasets) | ⚠️ | ✅ |
| No infrastructure management | ❌ | ✅ |
| Maximum data control | ✅ | ⚠️ |
| Regulatory compliance (21 CFR Part 11) | ✅ | ⚠️ (validated cloud) |
| Multi-user collaboration | ⚠️ | ✅ |
| Cost predictability | ✅ (compute only) | ⚠️ (usage-based) |

**Hybrid deployment pattern** (recommended):

```
Development & Exploration → Local Mode
├─ Literature search
├─ Dataset discovery
├─ Small-scale testing
└─ Sensitive data analysis

Production & Scale → Cloud Mode
├─ Large batch processing
├─ Multi-user workflows
├─ Non-sensitive data
└─ Managed infrastructure
```

**Switching modes**:
```bash
# Local mode (default)
unset LOBSTER_CLOUD_KEY
lobster chat

# Cloud mode
export LOBSTER_CLOUD_KEY=lbstr_abc123...
lobster activate $LOBSTER_CLOUD_KEY
lobster chat  # Now uses cloud backend
```

**Security considerations** (cloud mode):

- **Data sovereignty**: Understand where data is processed (AWS region)
- **Compliance**: Verify BAA for HIPAA, DPA for GDPR
- **Network security**: Use VPN/private endpoints for enterprise
- **Access logs**: Cloud service logs all API calls
- **Data retention**: Understand cloud provider's retention policies

**Compliance benefits**:

- ✅ **Flexibility** - Choose security posture per use case
- ✅ **Data control** - Local mode for regulated data
- ✅ **Scalability** - Cloud mode for production
- ✅ **Audit trail** - Both modes support W3C-PROV

**For complete implementation details**, see:
- [Configuration - Cloud vs Local](03-configuration.md#cloud-vs-local-mode) (setup guide)
- [Cloud-Local Architecture](21-cloud-local-architecture.md) (architectural details)
- [CLAUDE.md - Client Layer](../CLAUDE.md#22-client-layer-core) (AgentClient vs CloudLobsterClient)

---

## 5. Secure Code Execution

### 5.1 Custom Code Execution Service

**What it is**: Lobster includes a `CustomCodeExecutionService` that allows analysts to execute arbitrary Python code for edge-case data manipulations (e.g., complex metadata transformations, custom filtering). This feature balances **flexibility** (handle unexpected scenarios) with **security** (protect the system).

**Use cases**:
- Complex metadata filtering not covered by standard tools
- Custom data transformations (e.g., multi-column merging)
- Edge-case QC operations
- Format conversions for specialized databases

**Security model** (Phase 1 - Current):

| Control | Implementation | Protection |
|---------|----------------|------------|
| **Subprocess isolation** | Runs in separate Python process | Crash isolation (subprocess failure doesn't crash main) |
| **Timeout enforcement** | 300-second default limit | Prevents infinite loops |
| **Workspace-only access** | Working directory set to workspace | Cannot access files outside workspace |
| **Forbidden module blocking** | AST analysis + import hooks | Blocks `subprocess`, `os.system`, `importlib`, `eval`, `exec` |
| **Output capture** | Stdout/stderr redirected | All output logged to provenance |
| **Error handling** | Exception isolation | User-friendly error messages |

**Known limitations** (acceptable for local CLI, NOT for cloud SaaS):

- ⚠️ **Network access not restricted** - Code can make HTTP requests
- ⚠️ **File permissions not sandboxed** - Uses user's OS permissions
- ⚠️ **Resource limits basic** - Only timeout, no CPU/memory quotas
- ❌ **NOT suitable for cloud SaaS** - Requires Docker isolation (Phase 2)

**Testing rigor**:
- **30+ security test files** in `tests/manual/custom_code_execution/`
- **201+ attack vectors** tested across 6 categories:
  - File system attacks (path traversal, permission bypass)
  - Network attacks (data exfiltration, SSRF)
  - Resource exhaustion (memory bombs, CPU thrashing)
  - Privilege escalation (setuid, sudo abuse)
  - Code injection (eval, exec, import manipulation)
  - Crash attacks (segfault, assertion failures)

**Example usage**:
```python
# Via agent tool
execute_custom_code("""
import pandas as pd

# Load metadata from workspace
metadata = pd.read_csv(WORKSPACE / 'metadata.csv')

# Complex filtering (example: multiple conditions)
filtered = metadata[
    (metadata['disease'] == 'cancer') &
    (metadata['tissue'].isin(['lung', 'breast'])) &
    (metadata['age'] > 50)
]

# Save to exports directory (user-facing files)
OUTPUT_DIR.mkdir(exist_ok=True)
filtered.to_csv(OUTPUT_DIR / 'filtered_metadata.csv', index=False)
""", persist=True)
```

**Compliance benefits**:

- ✅ **Audit trail** - All custom code logged to provenance
- ✅ **Reproducibility** - Code captured in AnalysisStep IR
- ✅ **Isolation** - Subprocess protects main process
- ✅ **Timeout enforcement** - Prevents runaway processes
- ⚠️ **Limited sandboxing** - Suitable for local, NOT cloud

**For complete implementation details**, see:
- [CLAUDE.md - Security Considerations](../CLAUDE.md#410-security-considerations) (detailed security analysis)
- [Custom Code Execution Service](../lobster/services/execution/custom_code_execution_service.py) (implementation)
- [Security Tests](../tests/manual/custom_code_execution/) (attack vector testing)

---

### 5.2 Security Controls

**What it is**: Multi-layered security controls to prevent malicious or accidental misuse of custom code execution.

**1. Forbidden Module Blocking** (AST-based static analysis):

```python
FORBIDDEN_MODULES = [
    'subprocess',       # Prevent shell command execution
    'os.system',        # Prevent system calls
    'os.popen',         # Prevent pipe-based execution
    'importlib',        # Prevent dynamic imports
    '__import__',       # Prevent import manipulation
    'eval',             # Prevent code evaluation
    'exec',             # Prevent code execution
    'compile',          # Prevent bytecode compilation
    'open',             # Restricted to workspace paths only
]
```

**Example blocked code**:
```python
# ❌ BLOCKED: Attempt to use subprocess
import subprocess
subprocess.run(['rm', '-rf', '/'])  # Detected and blocked at import time

# ❌ BLOCKED: Dynamic import
importlib.import_module('os').system('evil')  # Detected via AST analysis

# ❌ BLOCKED: Code evaluation
eval("__import__('os').system('evil')")  # Detected via AST analysis
```

**2. Timeout Enforcement**:

```python
# Subprocess killed after timeout
process = subprocess.run(
    [sys.executable, script_path],
    timeout=300,  # 5 minutes default
    capture_output=True,
    text=True,
    cwd=str(workspace_path)  # Workspace boundary
)
```

**Example timeout protection**:
```python
# ❌ BLOCKED: Infinite loop (killed after 300s)
while True:
    pass

# ❌ BLOCKED: Excessive computation
for i in range(10**15):
    x = i ** 2  # Killed after timeout
```

**3. Workspace Boundary Enforcement**:

```python
# Custom code executes in workspace directory
# All file paths relative to workspace
WORKSPACE = Path(workspace_path)  # e.g., /Users/analyst/.lobster_workspace
OUTPUT_DIR = WORKSPACE / "exports"  # User-facing files

# Accessing files outside workspace requires absolute paths
# (user's OS permissions apply)
```

**Example workspace access**:
```python
# ✅ ALLOWED: Read/write within workspace
data = pd.read_csv(WORKSPACE / 'metadata.csv')
data.to_csv(OUTPUT_DIR / 'result.csv')

# ⚠️ USER PERMISSION: Access outside workspace (if OS allows)
external_data = pd.read_csv('/external/path/data.csv')  # OS permissions apply
```

**4. Provenance Logging** (all custom code logged):

```python
# Every execution logged to provenance
log_tool_usage(
    tool_name="execute_custom_code",
    parameters={"code": code_snippet, "persist": True},
    stats={"execution_time_ms": 1234, "output_lines": 5},
    ir=AnalysisStep(...)  # Complete code captured for reproducibility
)
```

**Compliance benefits**:

- ✅ **Attack surface reduction** - Forbidden modules blocked
- ✅ **Resource protection** - Timeout prevents DoS
- ✅ **Workspace isolation** - Limited file access scope
- ✅ **Complete audit trail** - All code logged
- ✅ **Reproducibility** - Code captured in AnalysisStep

---

### 5.3 Deployment Recommendations

**What it is**: Guidance on when CustomCodeExecutionService is appropriate for different deployment environments.

**Deployment decision matrix**:

| Environment | Status | Rationale | Recommendation |
|-------------|--------|-----------|----------------|
| **Local CLI** | ✅ Production-ready | Trusted users, local data control, user's OS permissions | Deploy with current security model |
| **Enterprise (on-premise)** | ⚠️ Conditional | Assess risk tolerance, trusted users, air-gapped OK | Deploy with current model OR wait for Phase 2 |
| **Cloud SaaS** | ❌ Requires Phase 2 | Untrusted users, need full isolation, network restrictions | Wait for Docker sandboxing |
| **Academic/Research** | ✅ Production-ready | Trusted users, flexibility > security, local control | Deploy with current model |
| **Regulated (GxP)** | ⚠️ Conditional | Risk assessment required, consider code review workflow | Evaluate per use case |

**Phase 2 enhancements** (roadmap for cloud SaaS):

| Enhancement | Technology | Benefit |
|-------------|-----------|---------|
| **Docker sandboxing** | gVisor or Kata Containers | Full isolation (filesystem, network, process) |
| **Network isolation** | Docker bridge mode | No outbound connections allowed |
| **Resource quotas** | cgroups | CPU, memory, disk limits |
| **Read-only input mounts** | Docker volumes | Input data immutable |
| **Runtime security scanning** | Falco or Sysdig | Detect anomalous behavior |
| **Egress firewall** | iptables/nftables | Block all external connections |

**Phase 2 timeline**: Estimated 4-6 weeks implementation + testing

**Risk assessment questions** (for enterprise deployment):

1. **Who executes code?** Trusted employees? External analysts?
2. **What data sensitivity?** PHI/PII? Confidential? Public?
3. **Network environment?** Air-gapped? Internet-connected?
4. **Risk tolerance?** Low (wait for Phase 2)? Medium (conditional)? High (deploy now)?
5. **Code review workflow?** Manual review? Automated scanning? None?

**Recommendations by scenario**:

```
Academic Lab (trusted users, public data)
→ ✅ Deploy now with Phase 1 security

Biotech Startup (small team, confidential data, on-premise)
→ ✅ Deploy now + code review workflow

Pharma Enterprise (GxP, validated environment)
→ ⚠️ Conditional: Risk assessment + code review + SOP

Cloud SaaS Provider (untrusted users, multi-tenant)
→ ❌ Wait for Phase 2 (Docker isolation)
```

**Compliance considerations**:

- ✅ **Audit trail** - All code logged (GxP requirement)
- ✅ **Reproducibility** - Code captured in notebooks (21 CFR Part 11)
- ⚠️ **Validation** - Phase 2 required for IQ/OQ/PQ (validated environments)
- ⚠️ **Data integrity** - Phase 1 OK for local, Phase 2 for cloud

**For complete implementation details**, see:
- [CLAUDE.md - Security Considerations](../CLAUDE.md#410-security-considerations) (detailed analysis, testing)
- [Custom Code Execution Service](../lobster/services/execution/custom_code_execution_service.py) (implementation)

---

### 5.4 Best Practices for Custom Code

**What it is**: Operational guidance for analysts using custom code execution and administrators deploying the feature.

**For Analysts** (using custom code):

**✅ GOOD: Simple data manipulation**
```python
execute_custom_code("""
import pandas as pd

# Load data
metadata = pd.read_csv(WORKSPACE / 'metadata.csv')

# Filter by condition
cancer_samples = metadata[metadata['disease'] == 'cancer']

# Save to exports (user-facing files)
OUTPUT_DIR.mkdir(exist_ok=True)
cancer_samples.to_csv(OUTPUT_DIR / 'cancer_metadata.csv', index=False)
""", persist=True)
```

**✅ GOOD: Custom QC checks**
```python
execute_custom_code("""
import pandas as pd
import numpy as np

# Load metadata
meta = pd.read_csv(WORKSPACE / 'metadata.csv')

# Custom QC: Check for missing values
missing_counts = meta.isnull().sum()
qc_pass = missing_counts.max() < 10  # Threshold: < 10 missing per column

# Save QC report
report = pd.DataFrame({
    'column': missing_counts.index,
    'missing_count': missing_counts.values,
    'qc_status': ['PASS' if c < 10 else 'FAIL' for c in missing_counts]
})
report.to_csv(OUTPUT_DIR / 'qc_report.csv', index=False)
""", persist=True)
```

**❌ BAD: Attempting forbidden operations**
```python
execute_custom_code("""
import subprocess  # ❌ BLOCKED at import time
subprocess.run(['rm', '-rf', '/'])  # Won't execute

import os
os.system('evil command')  # ❌ BLOCKED (os.system forbidden)

eval("__import__('os').system('evil')")  # ❌ BLOCKED (eval forbidden)
""")
```

**❌ BAD: Inefficient operations (timeout risk)**
```python
execute_custom_code("""
# ❌ Will timeout after 300 seconds
while True:
    pass

# ❌ Excessive memory usage (may crash subprocess)
data = [0] * (10 ** 10)  # 10 billion elements
""")
```

**For Administrators** (deploying Lobster):

**1. Local Deployment Checklist**:
```bash
# ✅ Verify workspace permissions (user-only access)
chmod 700 ~/.lobster_workspace

# ✅ Set timeout (optional, default 300s)
export LOBSTER_CUSTOM_CODE_TIMEOUT=600  # 10 minutes

# ✅ Enable audit logging (automatically enabled)
lobster query "Your analysis request"

# ✅ Review provenance logs periodically
cat ~/.lobster_workspace/provenance.json | jq '.activities[] | select(.tool_name == "execute_custom_code")'
```

**2. Enterprise Deployment Checklist**:
```bash
# ⚠️ Risk assessment (required)
# - Document: Who executes code? What data sensitivity?
# - Review: Security controls sufficient for risk tolerance?

# ✅ Code review workflow (recommended)
# - Require peer review for custom code blocks
# - Document in SOP: "Custom code must be reviewed by lead analyst"

# ✅ Periodic audit (quarterly)
# - Review custom code usage via provenance logs
# - Identify patterns, create standard tools for common operations

# ✅ User training
# - Document allowed patterns (data filtering, QC checks)
# - Document forbidden patterns (network access, subprocess)
```

**3. Monitoring & Alerting** (enterprise):
```python
# Example: Monitor custom code usage
import json
from pathlib import Path

def audit_custom_code_usage(workspace_path):
    """Generate custom code usage report."""
    provenance_path = Path(workspace_path) / "provenance.json"
    with open(provenance_path) as f:
        prov = json.load(f)

    custom_code_activities = [
        act for act in prov.get('activities', [])
        if act.get('tool_name') == 'execute_custom_code'
    ]

    print(f"Total custom code executions: {len(custom_code_activities)}")
    for act in custom_code_activities:
        print(f"  - {act['timestamp']}: {act['agent']} (session: {act['session_id']})")

# Run monthly
audit_custom_code_usage("~/.lobster_workspace")
```

**4. Standard Operating Procedure** (SOP template):

```markdown
## SOP: Custom Code Execution in Lobster AI

**Purpose**: Define approved use of custom code execution feature

**Scope**: All analysts using Lobster AI for bioinformatics analysis

**Approved Use Cases**:
1. Complex metadata filtering (>3 conditions)
2. Custom QC checks not covered by standard tools
3. Format conversions for specialized databases

**Forbidden Operations**:
1. Network requests (data exfiltration risk)
2. File access outside workspace (data leakage risk)
3. Subprocess execution (security risk)

**Review Process**:
1. Analyst documents custom code in lab notebook
2. Lead analyst reviews code before execution
3. QA team audits custom code usage quarterly

**Audit Trail**:
- All custom code logged to provenance.json
- Captured in notebook exports (Data Integrity Manifest)
- Reviewable via `lobster status` command
```

**Compliance benefits**:

- ✅ **Documented workflows** - SOPs for custom code usage
- ✅ **Audit trail** - Complete logging via provenance
- ✅ **Training** - Clear guidance on allowed patterns
- ✅ **Periodic review** - Quarterly audits recommended
- ✅ **Risk mitigation** - Code review workflow for sensitive environments

**For complete implementation details**, see:
- [CLAUDE.md - Security Considerations](../CLAUDE.md#410-security-considerations) (comprehensive security analysis)
- [Custom Code Execution Service](../lobster/services/execution/custom_code_execution_service.py) (implementation)
- [Security Tests](../tests/manual/custom_code_execution/) (201+ attack vectors tested)

---

## 6. Data Protection & Isolation

### 6.1 Workspace Isolation

**What it is**: Lobster uses a workspace-based architecture where each analysis session operates in an isolated directory. This provides data isolation between projects, prevents cross-contamination, and enables clean archival of complete analyses.

**Workspace resolution order** (priority):
1. **`--workspace` CLI flag** - Explicit path (highest priority)
2. **`LOBSTER_WORKSPACE` environment variable** - Project/session-level configuration
3. **Current directory default** - `./.lobster_workspace` (automatic creation)

**Workspace structure**:
```
.lobster_workspace/
├── .session.json              # Current session metadata (multi-process safe)
├── provenance.json            # W3C-PROV audit trail
├── dataset_name.h5ad          # Modality data files
├── plots/                     # Visualizations (PNG, HTML)
│   ├── umap_plot.html
│   └── qc_metrics.png
├── exports/                   # User-facing files (v2.4+)
│   ├── metadata_filtered.csv  # Exported metadata
│   └── de_results.tsv         # Differential expression
├── literature/                # Cached papers (PDF, TXT)
│   └── PMID_12345678.txt
├── metadata/                  # Sample metadata (CSV, TSV)
│   └── GSE109564_metadata.csv
├── download_queue.jsonl       # Download coordination (multi-process safe)
└── publication_queue.jsonl    # Publication processing (multi-process safe)
```

**Security properties**:

| Property | Implementation | Security Benefit |
|----------|----------------|------------------|
| **Per-project isolation** | Separate workspace per project | No cross-project data leakage |
| **Clean archival** | Zip workspace = complete analysis | Easy backup, transfer, compliance |
| **Multi-user support** | Each user has own workspace | No data mixing in shared environments |
| **Permission inheritance** | OS-level permissions (chmod 700) | Access control via filesystem |
| **Centralized exports** | `exports/` for user files (v2.4+) | Clear distinction: internal vs user-facing |

**Best practices**:

```bash
# ✅ GOOD: Dedicated workspace per project
lobster chat --workspace ~/projects/gse109564_analysis/

# ✅ GOOD: Set global workspace for long session
export LOBSTER_WORKSPACE=~/current_project
lobster chat

# ✅ GOOD: Archive complete analysis
tar -czf gse109564_analysis.tar.gz ~/projects/gse109564_analysis/

# ❌ BAD: Mix multiple projects in same workspace
lobster query "Analyze GSE12345"
lobster query "Analyze GSE67890"  # Mixed in same workspace, confusing provenance
```

**Multi-user deployment** (shared server):
```bash
# Each user gets isolated workspace
export LOBSTER_WORKSPACE=/shared/workspaces/$USER
chmod 700 /shared/workspaces/$USER  # User-only access

# Or project-based workspaces
export LOBSTER_WORKSPACE=/shared/projects/project_123
# Set group permissions for team collaboration
chmod 770 /shared/projects/project_123
chgrp bioinfo_team /shared/projects/project_123
```

**Compliance benefits**:

- ✅ **Data isolation** - HIPAA/GDPR data separation
- ✅ **Complete audit trail** - All files in one location
- ✅ **Clean archival** - 21 CFR Part 11 electronic records
- ✅ **Access control** - OS-level permission enforcement
- ✅ **Multi-user support** - Enterprise deployment ready

**For complete implementation details**, see:
- [Data Management - Workspace Organization](20-data-management.md#workspace-organization) (detailed structure)
- [CLAUDE.md - Workspace Resolution](../CLAUDE.md#45-patterns--abstractions) (resolve_workspace pattern)
- [Core - Workspace Module](../lobster/core/workspace.py) (centralized resolution)

---

### 6.2 Concurrent Access Protection

**What it is**: Lobster implements multi-process safe file locking for shared state files (download queue, publication queue, session metadata). This prevents race conditions, data corruption, and lost updates in concurrent scenarios.

**Protected files** (multi-process safe):

| File | Protection | Use Case |
|------|-----------|----------|
| `download_queue.jsonl` | InterProcessFileLock | Multiple agents downloading datasets |
| `publication_queue.jsonl` | InterProcessFileLock | Batch publication processing |
| `.session.json` | InterProcessFileLock | Session state updates |
| `cache_metadata.json` | InterProcessFileLock | Cache tracking |

**Implementation** (`core/queue_storage.py`):

**1. Cross-platform file locking**:
```python
# POSIX (macOS, Linux): fcntl.flock
# Windows: msvcrt.locking

class InterProcessFileLock:
    """File-based lock for cross-process coordination."""
    def __enter__(self):
        if platform.system() == 'Windows':
            msvcrt.locking(self.fd, msvcrt.LK_LOCK, 1)
        else:
            fcntl.flock(self.fd, fcntl.LOCK_EX)  # Exclusive lock
```

**2. Thread + process locking**:
```python
from contextlib import contextmanager

@contextmanager
def queue_file_lock(thread_lock, lock_path):
    """Combines threading.Lock + file lock."""
    with thread_lock:  # Thread-safe within process
        with InterProcessFileLock(lock_path):  # Process-safe across processes
            yield
```

**3. Atomic writes** (crash-safe):
```python
def atomic_write_json(path: Path, data: dict):
    """Crash-safe JSON write (temp file + fsync + atomic replace)."""
    temp_path = path.with_suffix('.tmp')

    # Write to temp file
    with open(temp_path, 'w') as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())  # Force write to disk

    # Atomic replace (POSIX guarantees atomicity)
    os.replace(temp_path, path)
```

**Example usage** (download queue):
```python
# Thread A and Thread B both trying to update queue
with queue_file_lock(self._lock, self._lock_path):
    # Critical section - guaranteed exclusive access
    entries = self._read_all_entries()
    entries.append(new_entry)
    atomic_write_jsonl(self._queue_path, entries)
# Lock released - other threads/processes can proceed
```

**Concurrency scenarios protected**:

| Scenario | Without Locking | With Locking |
|----------|----------------|--------------|
| Concurrent writes | Lost updates, corruption | Sequential writes, no loss |
| Read during write | Partial data read | Read waits for write completion |
| Process crash during write | Corrupted file | Temp file discarded, original intact |
| Multi-user access | Race conditions | Coordinated access |

**Performance characteristics**:

- **Lock overhead**: ~1-5ms per acquisition (local filesystem)
- **Blocking**: Writers wait for exclusive access (FIFO order)
- **Scalability**: Suitable for 10-100 concurrent processes (local FS limitation)
- **Not recommended for**: NFS/SMB (file locking issues), high-frequency updates (>100 ops/sec)

**Compliance benefits**:

- ✅ **Data integrity** - Prevents corruption in concurrent scenarios
- ✅ **Crash safety** - Atomic writes prevent partial updates
- ✅ **Multi-user support** - Enterprise deployment ready
- ✅ **Audit trail integrity** - Session metadata protected
- ✅ **Queue reliability** - Download/publication queues consistent

**For complete implementation details**, see:
- [Download Queue - Concurrency Infrastructure](35-download-queue-system.md#concurrency-infrastructure) (detailed patterns)
- [Core - Queue Storage](../lobster/core/queue_storage.py) (implementation)
- [CLAUDE.md - Concurrency Pattern](../CLAUDE.md#45-patterns--abstractions) (developer guide)

---

### 6.3 Session Management and Data Restoration

**What it is**: Lobster maintains persistent session state that enables cross-session continuity, analysis restoration, and historical provenance queries. Sessions support multi-turn conversations with automatic state management.

**Session lifecycle**:

```
Session Creation → Multi-Turn Analysis → Session Snapshot → Restoration
     ↓                    ↓                    ↓                ↓
session_123.json    Updates per tool      .session.json     Resume analysis
  (metadata)         (provenance log)       (checkpoint)     (--session-id)
```

**Session metadata structure** (saved to `.session.json`):
```json
{
  "session_id": "session_20260101_142000",
  "created_at": "2026-01-01T14:20:00.123456Z",
  "last_updated": "2026-01-01T15:45:32.789012Z",
  "workspace_path": "/Users/analyst/.lobster_workspace",
  "subscription_tier": "premium",
  "modalities": {
    "geo_gse109564": {
      "created_at": "2026-01-01T14:23:15Z",
      "n_obs": 5000,
      "n_vars": 2000,
      "layers": ["counts", "normalized"],
      "last_modified": "2026-01-01T15:30:00Z",
      "file_path": "geo_gse109564.h5ad",
      "size_bytes": 12345678
    }
  },
  "tool_usage_count": 15,
  "agent_handoffs_count": 4
}
```

**Key capabilities**:

| Feature | Description | Use Case |
|---------|-------------|----------|
| **Session continuity** | `--session-id latest` resumes previous session | Multi-day analysis |
| **Named sessions** | `--session-id "project_gse109564"` | Long-term projects |
| **Automatic checkpointing** | Session saved after each operation | Crash recovery |
| **Cross-session restoration** | Load data from previous session | Reproduce results |
| **Historical queries** | Query provenance from any session | Audit support |

**Session commands**:
```bash
# Start new named session
lobster query --session-id "project_gse109564" "Download GSE109564 and cluster"

# Continue most recent session
lobster query --session-id latest "Add differential expression"

# Resume specific session
lobster query --session-id "session_20260101_142000" "Export results"

# View current session
lobster status
# Output:
# Session ID: session_20260101_142000
# Workspace: /Users/analyst/.lobster_workspace
# Modalities: 3 datasets loaded
# Tool usage: 15 operations
# Last updated: 2026-01-01 15:45:32 UTC
```

**Session restoration workflow**:
```python
# Automatically handled by DataManagerV2
class DataManagerV2:
    def restore_session(self, session_id: str):
        """Restore complete session state."""
        # 1. Load session metadata
        session_meta = self._load_session_file(session_id)

        # 2. Restore modalities (lazy loading)
        for modality_name, info in session_meta['modalities'].items():
            self.modalities[modality_name] = self._load_h5ad(info['file_path'])

        # 3. Restore provenance context
        self.provenance.load_session_activities(session_id)

        # 4. Resume analysis
        return session_meta
```

**Security considerations**:

| Aspect | Protection | Benefit |
|--------|-----------|---------|
| **Session files** | Workspace permissions (chmod 700) | User-only access |
| **Session IDs** | Timestamp-based (not guessable) | Prevents session hijacking |
| **Automatic backup** | Written after each operation | Crash recovery |
| **No sensitive data** | API keys NOT stored in session | Credential protection |

**Compliance benefits**:

- ✅ **ALCOA+ "Contemporaneous"** - Real-time session updates
- ✅ **ALCOA+ "Traceable"** - Complete session history
- ✅ **Crash recovery** - Automatic checkpointing
- ✅ **Historical audit** - Query any previous session
- ✅ **Multi-day analysis** - Session continuity support

**For complete implementation details**, see:
- [Data Management - Session Metadata](20-data-management.md#session-metadata) (detailed structure)
- [CLI Commands - Session Management](05-cli-commands.md#session-management) (usage examples)
- [CLAUDE.md - Session Continuity](../CLAUDE.md#54-running-the-app) (multi-turn conversations)

---

## 7. Network Security & Rate Limiting

### 7.1 Redis Rate Limiter Architecture

**What it is**: Lobster uses a Redis-backed token bucket rate limiter to prevent API rate limit violations when accessing external databases (NCBI, GEO, PubMed, PRIDE, MassIVE, etc.). This ensures **good API citizenship** and prevents 429 errors that interrupt analysis workflows.

**Architecture**:
- **Token bucket algorithm** - Tokens replenish over time, requests consume tokens
- **Redis connection pool** - Thread-safe, health-check enabled (30s interval)
- **Graceful degradation** - Fail-open if Redis unavailable (warning only)
- **Cross-process coordination** - Multiple lobster processes share rate limit

**Deployment modes**:

| Mode | Redis Required | Behavior | Use Case |
|------|---------------|----------|----------|
| **Development** | No | Warning logged, no blocking | Single-user, local analysis |
| **Production** | Yes | Rate limits enforced | Multi-user, shared server |
| **CI/CD** | Optional | Warning-only (no Redis) | Automated testing |

**Setup**:
```bash
# Development: No Redis needed (warning only)
lobster query "Search PubMed for CRISPR papers"
# Warning: Redis not available, rate limiting disabled

# Production: Redis for multi-process coordination
docker run -d -p 6379:6379 redis:alpine
export REDIS_URL=redis://localhost:6379
lobster query "Search PubMed for CRISPR papers"
# Rate limits enforced across all processes
```

**Key features**:

| Feature | Implementation | Benefit |
|---------|----------------|---------|
| **Connection pooling** | `redis.ConnectionPool` with health checks | Auto-recovery from stale connections |
| **Thread-safe** | Double-checked locking for lazy init | Safe for multi-threaded agents |
| **Process-safe** | Redis keys with TTL | Cross-process coordination |
| **Automatic retry** | Exponential backoff on 429 errors | Transparent error recovery |
| **Provider-specific** | Separate keys per domain | Precise rate limit compliance |

**Example usage** (automatic):
```python
from lobster.tools.rate_limiter import get_rate_limiter

# Decorator automatically applied to provider methods
@get_rate_limiter().with_rate_limit(domain="ncbi")
def search_pubmed(query: str):
    # Rate limit enforced before API call
    # If limit exceeded: waits for token availability
    return ncbi_api.search(query)
```

**Redis key structure**:
```
rate_limit:ncbi       → Token bucket for NCBI (10 req/s with API key)
rate_limit:pmc        → Token bucket for PMC (3 req/s)
rate_limit:geo        → Token bucket for GEO (10 req/s)
rate_limit:pride      → Token bucket for PRIDE (2 req/s)
```

**Compliance benefits**:

- ✅ **API compliance** - Respects provider rate limits (NCBI Terms of Service)
- ✅ **Reliability** - Prevents 429 errors that interrupt workflows
- ✅ **Good citizenship** - Prevents overloading public databases
- ✅ **Multi-user support** - Coordinates rate limits across users
- ✅ **Audit trail** - Rate limit violations logged to provenance

**For complete implementation details**, see:
- [Redis Rate Limiter Architecture](48-redis-rate-limiter-architecture.md) (comprehensive guide, connection pool patterns)
- [Rate Limiter](../lobster/tools/rate_limiter.py) (implementation)
- [CLAUDE.md - Rate Limiting Pattern](../CLAUDE.md#45-patterns--abstractions) (developer guide)

---

### 7.2 Multi-Domain Rate Limiting

**What it is**: Lobster integrates with **29+ external databases** (genomics, proteomics, metabolomics, literature), each with different rate limits. The system enforces provider-specific limits to ensure compliance and prevent access denial.

**Rate limits by provider**:

| Domain | Base Rate Limit | With API Key | Enforcement | Protocol |
|--------|----------------|--------------|-------------|----------|
| **NCBI E-utilities** | 3 req/s | 10 req/s | Redis + backoff | HTTPS |
| **PMC Open Access** | 3 req/s | N/A | Redis + backoff | HTTPS |
| **GEO** | 10 req/s | N/A | Redis + backoff | HTTPS/FTP |
| **SRA** | 3 req/s | 10 req/s (same NCBI key) | Redis + backoff | HTTPS |
| **PRIDE** | 2 req/s | N/A | Redis + backoff | HTTPS |
| **MassIVE** | 1 req/s | N/A | Redis + backoff | HTTPS |
| **MetaboLights** | 2 req/s | N/A | Redis + backoff | HTTPS |
| **Publisher APIs** | 0.3-2.0 req/s | Varies | Redis + backoff | HTTPS |

**NCBI API key benefits** (recommended):

| Metric | Without Key | With Key | Improvement |
|--------|------------|----------|-------------|
| Rate limit | 3 req/s | 10 req/s | 3.3x faster |
| Batch PubMed search | ~300 queries/min | ~1000 queries/min | 3.3x faster |
| Large dataset download | Rate-limited | Priority queue | Better service |

**Setup NCBI API key** (free, 2-minute registration):
```bash
# 1. Register at: https://www.ncbi.nlm.nih.gov/account/settings/
# 2. Generate API key
# 3. Add to .env file
echo "NCBI_API_KEY=your-key-here" >> .env

# 4. Verify (rate limit increases automatically)
lobster query "Search PubMed for 500 papers on CRISPR"
# Completes in ~30s instead of ~100s
```

**Exponential backoff** (automatic retry):

```python
# Automatic retry on 429 Too Many Requests
@rate_limiter.with_rate_limit(domain="ncbi")
def fetch_data(accession):
    """Automatically retries with exponential backoff."""
    # Retry schedule: 1s, 2s, 4s, 8s, 16s (max 5 attempts)
    # Total wait time: ~31s max
    return api.get(accession)
```

**Rate limit error handling**:

```python
# Example internal implementation
def _handle_rate_limit_error(response, attempt):
    """Handle 429 Too Many Requests."""
    if response.status_code == 429:
        retry_after = response.headers.get('Retry-After', 2 ** attempt)
        logger.warning(f"Rate limit exceeded, retrying in {retry_after}s")
        time.sleep(retry_after)
        return True  # Retry
    return False  # Don't retry
```

**Multi-domain coordination** (example workflow):

```
User: "Search PubMed for CRISPR papers, download top 10 datasets from GEO, fetch protein structures from PDB"
    ↓
research_agent → NCBI rate limiter (10 req/s with key)
    ↓
data_expert → GEO rate limiter (10 req/s)
    ↓
protein_structure_visualization_expert → PDB rate limiter (10 req/s)

Each domain has independent token bucket → No cross-domain interference
```

**Monitoring & alerting** (enterprise):

```python
# Example: Monitor rate limit usage
def check_rate_limit_health():
    """Check Redis connection and rate limit status."""
    limiter = get_rate_limiter()

    if not limiter.is_available():
        logger.error("⚠️ Redis unavailable - rate limiting disabled")
        return False

    # Check token availability for critical domains
    critical_domains = ["ncbi", "geo", "pride"]
    for domain in critical_domains:
        tokens = limiter.get_available_tokens(domain)
        if tokens < 10:  # Low token threshold
            logger.warning(f"⚠️ Low tokens for {domain}: {tokens}")

    return True

# Run periodically (e.g., every 5 minutes)
check_rate_limit_health()
```

**Compliance benefits**:

- ✅ **Terms of Service compliance** - Respects NCBI, GEO, PRIDE rate limits
- ✅ **Prevents access denial** - Avoids IP blocking from excessive requests
- ✅ **Good API citizenship** - Responsible use of public resources
- ✅ **Multi-domain coordination** - Independent limits per provider
- ✅ **Audit trail** - All API calls logged to provenance

**For complete implementation details**, see:
- [Architecture - Multi-Domain Rate Limiting](18-architecture-overview.md#multi-domain-rate-limiting) (provider list, rate limits)
- [Redis Rate Limiter Architecture](48-redis-rate-limiter-architecture.md) (token bucket algorithm)
- [Configuration - NCBI API Key](03-configuration.md#ncbi-api-key) (setup guide)

---

### 7.3 API Timeout and Error Handling

**What it is**: Lobster implements robust timeout handling and error recovery for all external API calls, ensuring graceful degradation when network issues occur or APIs are unavailable.

**Timeout configuration**:

| Timeout Type | Default | Configurable | Purpose |
|-------------|---------|--------------|---------|
| **Connection timeout** | 10s | Yes | Time to establish connection |
| **Read timeout** | 30s | Yes | Time to receive first byte |
| **Total timeout** | 5min | Yes | Maximum request duration |
| **Retry timeout** | 2min | Yes | Maximum retry duration |

**Error handling strategy**:

| Error Type | Retry? | Backoff | Max Attempts | Logged? |
|-----------|--------|---------|--------------|---------|
| **Connection errors** | ✅ Yes | Exponential | 5 | ✅ Yes |
| **429 Rate limit** | ✅ Yes | Exponential | 5 | ✅ Yes |
| **5xx Server errors** | ✅ Yes | Exponential | 3 | ✅ Yes |
| **4xx Client errors** | ❌ No | N/A | 1 | ✅ Yes |
| **Timeout** | ✅ Yes | Linear | 3 | ✅ Yes |

**Example timeout configuration**:
```python
# Provider-specific timeouts (in provider classes)
class PubMedProvider:
    DEFAULT_TIMEOUT = (10, 30)  # (connect, read) in seconds

    def search(self, query: str):
        try:
            response = requests.get(
                url,
                timeout=self.DEFAULT_TIMEOUT,
                headers={"User-Agent": "Lobster-AI/0.3.4"}
            )
            response.raise_for_status()
            return response.json()
        except requests.Timeout:
            logger.warning("PubMed search timed out, retrying...")
            # Automatic retry via decorator
        except requests.ConnectionError:
            logger.error("Connection to PubMed failed")
            raise ProviderError("Network error")
```

**Retry with exponential backoff**:
```python
# Automatic retry logic (internal)
def retry_with_backoff(func, max_attempts=5):
    """Exponential backoff: 1s, 2s, 4s, 8s, 16s."""
    for attempt in range(max_attempts):
        try:
            return func()
        except (requests.Timeout, requests.ConnectionError) as e:
            if attempt == max_attempts - 1:
                raise  # Final attempt failed
            wait_time = 2 ** attempt
            logger.warning(f"Retry {attempt + 1}/{max_attempts} in {wait_time}s")
            time.sleep(wait_time)
```

**Network security best practices**:

| Practice | Implementation | Security Benefit |
|----------|----------------|------------------|
| **HTTPS only** | All API calls use HTTPS | Encrypted communication |
| **SSL verification** | `verify=True` (default) | Prevents MITM attacks |
| **Timeouts enforced** | All requests have timeout | Prevents hanging |
| **User-Agent header** | `Lobster-AI/version` | Identifies client, enables rate limit cooperation |
| **Error logging** | All failures logged to provenance | Audit trail, debugging |

**Example error handling** (user-facing):

```
User: "Download GSE12345 from GEO"

# Scenario 1: Network timeout
research_agent: "⚠️ Network timeout while accessing GEO. Retrying... (attempt 1/5)"
# ... exponential backoff ...
research_agent: "✅ Successfully downloaded GSE12345 (retry 3 succeeded)"

# Scenario 2: Rate limit exceeded
research_agent: "⚠️ Rate limit exceeded (429). Waiting 8 seconds before retry..."
# ... automatic backoff ...
research_agent: "✅ Request succeeded after rate limit backoff"

# Scenario 3: Permanent failure
research_agent: "❌ Failed to download GSE12345 after 5 attempts. GEO may be unavailable. Please try again later."
```

**Monitoring network health** (enterprise):
```bash
# Check recent network errors
cat ~/.lobster_workspace/provenance.json | \
  jq '.activities[] | select(.status == "error") | .error_message'

# Count errors by type
cat ~/.lobster_workspace/provenance.json | \
  jq '.activities[] | select(.status == "error") | .error_type' | \
  sort | uniq -c
```

**Compliance benefits**:

- ✅ **Graceful degradation** - Analysis continues despite transient failures
- ✅ **Audit trail** - All network errors logged to provenance
- ✅ **Security** - HTTPS + SSL verification enforced
- ✅ **Reliability** - Automatic retry with backoff
- ✅ **User experience** - Clear error messages, transparent retry

**For complete implementation details**, see:
- [Redis Rate Limiter Architecture](48-redis-rate-limiter-architecture.md#error-handling) (retry logic)
- [Provider Base Class](../lobster/tools/providers/base_provider.py) (timeout patterns)
- [CLAUDE.md - Error Hierarchy](../CLAUDE.md#45-patterns--abstractions) (ProviderError, ServiceError)

---

## 8. Validation & Data Quality

### 8.1 Schema Validation

**What it is**: Lobster uses Pydantic-based schema validation for all modality data (transcriptomics, proteomics, metabolomics, metagenomics). This enforces **data integrity** and **standardization** at load time, preventing downstream analysis errors.

**Schema architecture**:
- **Per-modality schemas** - Domain-specific validation rules
- **Pydantic models** - Type checking, constraint validation, automatic coercion
- **Pre-load validation** - Errors caught before data enters workspace
- **Quality checks** - Missing value thresholds, column requirements

**Supported schemas** (`core/schemas/`):

| Schema | Purpose | Key Validations |
|--------|---------|----------------|
| **transcriptomics_schema.py** | RNA-seq QC metrics | Min cells/genes, count thresholds, QC column checks |
| **proteomics_schema.py** | Mass spec data | Missing value limits, intensity ranges, peptide columns |
| **metabolomics_schema.py** | Metabolite data | m/z ranges, retention times, peak intensity |
| **metagenomics_schema.py** | 16S/shotgun data | Taxonomy levels, abundance validation |
| **database_mappings.py** | Accession patterns | 29 database identifier formats |

**Example validation** (transcriptomics):

```python
from lobster.core.schemas.transcriptomics_schema import TranscriptomicsMetadata

# Validate H5AD metadata before loading
class TranscriptomicsMetadata(BaseModel):
    n_obs: int = Field(gt=0, description="Number of observations (cells)")
    n_vars: int = Field(gt=0, description="Number of variables (genes)")
    layers: List[str] = Field(..., description="Required layers")
    obs_columns: List[str] = Field(..., description="Observation annotations")

    @validator("layers")
    def check_required_layers(cls, v):
        required = ["counts"]
        if not any(layer in v for layer in required):
            raise ValueError(f"Missing required layer: {required}")
        return v

    @validator("n_obs")
    def check_minimum_cells(cls, v):
        if v < 10:
            raise ValueError(f"Too few cells: {v} (minimum: 10)")
        return v
```

**Validation workflow**:

```
User: "Load my single-cell data"
    ↓
data_expert → load_modality()
    ↓
ModalityAdapter.load() → H5AD file read
    ↓
TranscriptomicsMetadata.validate() → Schema checks
    ├─ ✅ PASS → Data loaded into workspace
    └─ ❌ FAIL → ValidationError with details

Example error:
"ValidationError: n_obs=5 (minimum: 10 cells required)"
"ValidationError: Missing required layer: 'counts'"
```

**Validation categories**:

| Category | Checks | Example |
|----------|--------|---------|
| **Structure** | Required columns, layers | `.obs['cell_type']`, `.layers['counts']` |
| **Thresholds** | Min/max values | `n_obs >= 10`, `n_vars >= 200` |
| **Data types** | Type enforcement | Integer counts, float normalized values |
| **Consistency** | Cross-field validation | `len(obs) == adata.shape[0]` |
| **Quality** | QC metric ranges | `pct_counts_mt < 20%` |

**Benefits for analysts**:

```
# ❌ WITHOUT VALIDATION: Silent failures, corrupted analysis
adata = load_data("bad_file.h5ad")  # Missing 'counts' layer
adata = sc.pp.normalize_total(adata)  # KeyError: 'counts'

# ✅ WITH VALIDATION: Early error detection
adata = load_data("bad_file.h5ad")
# ValidationError: Missing required layer: 'counts'
# Fix data before analysis, avoid wasted time
```

**Compliance benefits**:

- ✅ **ALCOA+ "Accurate"** - Data integrity enforced
- ✅ **ALCOA+ "Complete"** - Required fields validated
- ✅ **Quality assurance** - Pre-analysis QC
- ✅ **Audit trail** - Validation results logged to provenance
- ✅ **Reproducibility** - Schema version captured in metadata

**For complete implementation details**, see:
- [Data Management - Schema Validation](20-data-management.md#schema-validation) (validation workflow, Pydantic patterns)
- [Schemas](../lobster/core/schemas/) (schema implementations)
- [CLAUDE.md - Patterns](../CLAUDE.md#45-patterns--abstractions) (ValidationError hierarchy)

---

### 8.2 Accession Validation

**What it is**: Lobster uses a centralized `AccessionResolver` to validate and parse identifiers from **29 public databases** (GEO, SRA, PRIDE, MassIVE, MetaboLights, etc.). This prevents typos, detects invalid accessions, and enables database-specific download strategies.

**Supported databases** (29 patterns):

| Database | Pattern Examples | Category |
|----------|-----------------|----------|
| **GEO** | GSE109564, GSM*, GPL*, GDS* | Genomics |
| **SRA** | SRP*, SRX*, SRR*, SRS* | Sequencing |
| **ENA/DDBJ** | ERP*, ERX*, ERR*, DRP* | Sequencing |
| **BioProject/BioSample** | PRJNA*, SAMN* | Metadata |
| **PRIDE** | PXD012345 | Proteomics |
| **MassIVE** | MSV000082048 | Proteomics |
| **MetaboLights** | MTBLS123 | Metabolomics |
| **ArrayExpress** | E-MTAB-*, E-GEOD-* | Microarrays |
| **DOI** | 10.1234/example | Publications |

**Architecture**:
- **Thread-safe singleton** - Single instance via `get_accession_resolver()`
- **Pre-compiled regex** - 29 patterns compiled at import time
- **Case-insensitive** - `gse12345` = `GSE12345` (better UX)
- **URL generation** - Automatic URL construction per database
- **Centralized source** - `core/schemas/database_mappings.py` (single source of truth)

**Key methods**:

```python
from lobster.core.identifiers.accession_resolver import get_accession_resolver

resolver = get_accession_resolver()

# Detect database type
db = resolver.detect_database("GSE109564")
# Returns: "geo"

# Validate accession
is_valid = resolver.validate("GSE109564", "geo")
# Returns: True

# Extract accessions from text
accessions = resolver.extract_accessions_by_type("Check GSE109564 and SRP123456")
# Returns: {"geo": ["GSE109564"], "sra": ["SRP123456"]}

# Generate URL
url = resolver.get_url("GSE109564", "geo")
# Returns: "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE109564"
```

**Pre-download validation** (prevents failed downloads):

```
User: "Download INVALID12345"
    ↓
research_agent → validate accession
    ↓
AccessionResolver.detect_database("INVALID12345")
    ├─ ✅ Matched pattern → Proceed to download
    └─ ❌ No match → "❌ Invalid accession format: INVALID12345"

User: "Download GSE109564"
    ↓
AccessionResolver.detect_database("GSE109564")  → "geo"
    ↓
AccessionResolver.validate("GSE109564", "geo")  → True
    ↓
Create DownloadQueueEntry(accession="GSE109564", database="geo")
    ↓
data_expert → execute_download_from_queue()
```

**Helper methods** (commonly used):

```python
# Check if GEO identifier
if resolver.is_geo_identifier("GSE109564"):
    # Handle GEO-specific logic
    pass

# Check if SRA identifier
if resolver.is_sra_identifier("SRP123456"):
    # Handle SRA-specific logic
    pass

# Check if proteomics identifier
if resolver.is_proteomics_identifier("PXD012345"):
    # Handle PRIDE-specific logic
    pass
```

**Migration from hardcoded patterns** (✅ Done):

```python
# ❌ OLD: Hardcoded regex in every provider (duplicated, error-prone)
class GEOProvider:
    def _validate(self, accession):
        if not re.match(r"^GSE\d+$", accession):  # Duplicated across 10+ files
            raise ValueError("Invalid")

# ✅ NEW: Centralized validation (single source of truth)
class GEOProvider:
    def _validate(self, accession):
        if not get_accession_resolver().validate(accession, "geo"):
            raise ValueError("Invalid")
```

**Compliance benefits**:

- ✅ **Data integrity** - Invalid accessions rejected early
- ✅ **Audit trail** - All validation logged to provenance
- ✅ **Error prevention** - Typos caught before expensive downloads
- ✅ **Consistency** - Single source of truth for patterns
- ✅ **Extensibility** - Add new databases in one place

**For complete implementation details**, see:
- [Architecture - Accession Validation](18-architecture-overview.md#accession-resolver) (pattern list, database registry)
- [AccessionResolver](../lobster/core/identifiers/accession_resolver.py) (implementation)
- [Database Mappings](../lobster/core/schemas/database_mappings.py) (DATABASE_ACCESSION_REGISTRY)
- [CLAUDE.md - AccessionResolver Pattern](../CLAUDE.md#45-patterns--abstractions) (usage guide)

---

### 8.3 Pre-Download Validation

**What it is**: Lobster performs **multi-layer validation** before initiating dataset downloads. This prevents wasted time, bandwidth, and storage on invalid or problematic datasets.

**Validation layers** (executed sequentially):

| Layer | Checks | Example | Failure Action |
|-------|--------|---------|----------------|
| **1. Accession format** | Regex pattern match | `GSE109564` valid, `INVALID123` invalid | Reject immediately |
| **2. Database detection** | Identify data source | `GSE*` → GEO, `PXD*` → PRIDE | Route to correct service |
| **3. Metadata fetch** | API call for dataset info | Sample count, file sizes, organism | Log metadata to queue entry |
| **4. Availability check** | Verify dataset exists | HTTP HEAD request | Mark as FAILED if 404 |
| **5. Size estimation** | Check file sizes | Warn if >10 GB | User confirmation required |
| **6. Queue uniqueness** | Prevent duplicate downloads | Check existing queue entries | Skip if already queued |

**Example validation workflow**:

```
User: "Download GSE109564"
    ↓
Layer 1: Accession format validation
├─ AccessionResolver.validate("GSE109564", "geo")
└─ ✅ PASS: Valid GEO accession

Layer 2: Database detection
├─ AccessionResolver.detect_database("GSE109564")
└─ ✅ PASS: Detected as "geo"

Layer 3: Metadata fetch
├─ GEOProvider.get_metadata("GSE109564")
├─ Returns: {"n_samples": 5000, "organism": "Homo sapiens", "platform": "GPL24676"}
└─ ✅ PASS: Metadata retrieved

Layer 4: Availability check
├─ requests.head(f"https://ftp.ncbi.nlm.nih.gov/geo/series/GSE109nnn/GSE109564/")
└─ ✅ PASS: Status 200 (dataset exists)

Layer 5: Size estimation
├─ Estimated size: 1.2 GB
└─ ✅ PASS: Below 10 GB threshold (no confirmation needed)

Layer 6: Queue uniqueness
├─ DownloadQueue.check_existing("GSE109564")
└─ ✅ PASS: Not already in queue

Result: Create DownloadQueueEntry(status=PENDING)
```

**Early rejection examples** (prevents wasted resources):

```
# Invalid accession format
User: "Download BADFORMAT123"
→ ❌ Rejected at Layer 1: "Invalid accession format: BADFORMAT123"

# Dataset doesn't exist
User: "Download GSE999999999"
→ ❌ Rejected at Layer 4: "Dataset not found: GSE999999999 (404)"

# Already in queue
User: "Download GSE109564" (twice)
→ ⚠️ Skipped at Layer 6: "GSE109564 already in download queue (status: PENDING)"
```

**Size warnings** (large datasets):

```
User: "Download GSE150614"
    ↓
Layer 5: Size estimation
├─ Estimated size: 15 GB
└─ ⚠️ WARNING: Large dataset detected

research_agent: "⚠️ GSE150614 is large (~15 GB). Download may take 10-30 minutes depending on network speed. Proceed? (yes/no)"

User: "yes"
→ ✅ Proceed to download

User: "no"
→ ❌ Download cancelled
```

**Queue status tracking** (all stages logged):

```json
{
  "entry_id": "download_20260101_142000",
  "accession": "GSE109564",
  "database": "geo",
  "status": "PENDING",
  "validation_results": {
    "accession_format": "PASS",
    "database_detection": "PASS (geo)",
    "metadata_fetch": "PASS (5000 samples)",
    "availability_check": "PASS (200 OK)",
    "size_estimation": "PASS (1.2 GB)",
    "queue_uniqueness": "PASS"
  },
  "created_at": "2026-01-01T14:20:00Z"
}
```

**Benefits**:

| Benefit | Impact | Example |
|---------|--------|---------|
| **Time savings** | Reject invalid accessions in <1s vs 5-10min download attempt | Invalid format caught immediately |
| **Bandwidth savings** | Skip non-existent datasets | 404 check prevents failed downloads |
| **Storage savings** | Prevent duplicate downloads | Uniqueness check avoids re-downloading |
| **User experience** | Clear error messages | "Invalid format" vs generic failure |
| **Audit trail** | All validations logged | Compliance support |

**Compliance benefits**:

- ✅ **ALCOA+ "Accurate"** - Data integrity verified before download
- ✅ **ALCOA+ "Complete"** - Metadata completeness checked
- ✅ **Resource efficiency** - Prevents wasted bandwidth/storage
- ✅ **Audit trail** - Validation results logged to provenance
- ✅ **Quality assurance** - Pre-download QC

**For complete implementation details**, see:
- [Download Queue System](35-download-queue-system.md#validation-workflow) (validation layers, queue patterns)
- [Download Architecture](../CLAUDE.md#46-download-architecture-queue-based-pattern) (IDownloadService, orchestrator)
- [GEO Provider](../lobster/tools/providers/geo_provider.py) (GEO-specific validation)

---

## 9. Deployment Security

### 9.1 Docker Deployment

**What it is**: Lobster provides containerized deployment via Docker for consistent, reproducible environments across development, staging, and production. Two container types support different use cases: CLI (local analysis) and Server (cloud API).

**Container types**:

| Container | Image | Purpose | Published | Use Case |
|-----------|-------|---------|-----------|----------|
| **CLI** | `omicsos/lobster:latest` | Local analysis | ✅ Docker Hub | Individual users, CI/CD |
| **Server** | (private) | Cloud API service | ❌ Private only | Enterprise cloud deployment |

**CLI container architecture**:
```dockerfile
FROM python:3.11-slim

# Security: Non-root user
RUN useradd -m -u 1000 lobster
USER lobster

# Install lobster-ai package
RUN pip install --no-cache-dir lobster-ai

# Workspace mounted at runtime
WORKDIR /workspace

ENTRYPOINT ["lobster"]
```

**Running CLI container**:
```bash
# Basic usage
docker run -v $(pwd):/workspace omicsos/lobster:latest query "Analyze GSE109564"

# With API keys
docker run \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  -e NCBI_API_KEY=$NCBI_API_KEY \
  -v $(pwd):/workspace \
  omicsos/lobster:latest chat

# With persistent workspace
docker run \
  -v $(pwd)/.lobster_workspace:/workspace \
  omicsos/lobster:latest query "Download GSE109564"

# With Redis for rate limiting
docker network create lobster-net
docker run -d --name redis --network lobster-net redis:alpine
docker run \
  --network lobster-net \
  -e REDIS_URL=redis://redis:6379 \
  -v $(pwd):/workspace \
  omicsos/lobster:latest query "Search PubMed"
```

**Security properties**:

| Property | Implementation | Security Benefit |
|----------|----------------|------------------|
| **Non-root user** | UID 1000 (lobster) | Prevents privilege escalation |
| **Minimal base image** | python:3.11-slim | Reduced attack surface |
| **No secrets in image** | API keys via env vars | Prevents credential leaks |
| **Read-only filesystem** | Optional `--read-only` flag | Immutable container |
| **Resource limits** | `--memory`, `--cpus` flags | DoS prevention |

**Multi-service deployment** (docker-compose):
```yaml
version: '3.8'

services:
  redis:
    image: redis:alpine
    restart: unless-stopped
    volumes:
      - redis-data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s

  lobster-cli:
    image: omicsos/lobster:latest
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./workspace:/workspace
    depends_on:
      - redis

volumes:
  redis-data:
```

**Compliance benefits**:

- ✅ **Reproducibility** - Fixed environment, version pinning
- ✅ **Isolation** - Container-level separation
- ✅ **Audit trail** - Image tags document versions
- ✅ **Portability** - Run anywhere (local, cloud, HPC)
- ✅ **Security** - Non-root user, minimal attack surface

**For complete implementation details**, see:
- [Docker Deployment Guide](43-docker-deployment-guide.md) (comprehensive deployment patterns)
- [Dockerfile](../Dockerfile) (CLI container specification)
- [CLAUDE.md - Deployment](../CLAUDE.md#34-deployment--infrastructure) (build strategy)

---

### 9.2 S3 Backend Security

**What it is**: Lobster supports Amazon S3 as a storage backend for workspaces, enabling cloud-native deployments with centralized data management. The S3 backend implements AWS security best practices for data protection.

**Architecture**:
- **S3DataBackend** - Implements `IDataBackend` interface
- **boto3 integration** - AWS SDK for Python
- **Workspace prefix** - Per-workspace isolation in bucket
- **Server-side encryption** - AES-256 or KMS encryption
- **Access control** - IAM policies + bucket policies

**S3 workspace structure**:
```
s3://lobster-workspaces/
├── user1_project_gse109564/           # Workspace prefix
│   ├── .session.json                   # Session metadata
│   ├── provenance.json                 # W3C-PROV audit trail
│   ├── geo_gse109564.h5ad              # Modality data
│   ├── plots/
│   │   └── umap_plot.html
│   └── exports/
│       └── metadata_filtered.csv
├── user2_project_xyz/
│   └── ...
```

**Security configuration**:

| Security Control | Implementation | Compliance Benefit |
|-----------------|----------------|-------------------|
| **Encryption at rest** | S3 SSE-S3 (AES-256) or SSE-KMS | HIPAA, GDPR compliance |
| **Encryption in transit** | HTTPS (TLS 1.2+) | Data protection |
| **Access control** | IAM roles + bucket policies | Principle of least privilege |
| **Versioning** | S3 versioning enabled | Data recovery, audit trail |
| **Logging** | S3 access logs + CloudTrail | Audit support |
| **MFA delete** | Required for object deletion | Accidental deletion prevention |

**IAM policy example** (least privilege):
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::lobster-workspaces/user1_*/*",
        "arn:aws:s3:::lobster-workspaces"
      ],
      "Condition": {
        "StringLike": {
          "s3:prefix": ["user1_*"]
        }
      }
    }
  ]
}
```

**Bucket policy example** (enforce encryption):
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "DenyUnencryptedObjectUploads",
      "Effect": "Deny",
      "Principal": "*",
      "Action": "s3:PutObject",
      "Resource": "arn:aws:s3:::lobster-workspaces/*",
      "Condition": {
        "StringNotEquals": {
          "s3:x-amz-server-side-encryption": "AES256"
        }
      }
    }
  ]
}
```

**Usage** (automatic backend selection):
```python
# Configure S3 backend (environment variables)
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key
export LOBSTER_S3_BUCKET=lobster-workspaces
export LOBSTER_S3_WORKSPACE_PREFIX=user1_project_gse109564

# Lobster automatically uses S3 backend
lobster query "Analyze GSE109564"
# Data stored to: s3://lobster-workspaces/user1_project_gse109564/
```

**Data lifecycle policies** (cost optimization + compliance):
```json
{
  "Rules": [
    {
      "Id": "TransitionToIA",
      "Status": "Enabled",
      "Transitions": [
        {
          "Days": 30,
          "StorageClass": "STANDARD_IA"
        }
      ],
      "NoncurrentVersionTransitions": [
        {
          "NoncurrentDays": 7,
          "StorageClass": "GLACIER"
        }
      ]
    },
    {
      "Id": "ExpireOldVersions",
      "Status": "Enabled",
      "NoncurrentVersionExpiration": {
        "NoncurrentDays": 90
      }
    }
  ]
}
```

**Compliance benefits**:

- ✅ **HIPAA compliance** - Encryption at rest + in transit, access logs
- ✅ **GDPR compliance** - Data residency (region selection), encryption
- ✅ **21 CFR Part 11** - Audit trails (CloudTrail), versioning
- ✅ **SOC 2** - AWS SOC 2 certification + Lobster audit trail
- ✅ **Cost optimization** - Lifecycle policies for long-term storage

**For complete implementation details**, see:
- [S3 Backend Guide](43-s3-backend-guide.md) (comprehensive setup, IAM policies)
- [S3 Backend](../lobster/core/backends/s3_backend.py) (implementation)
- [CLAUDE.md - Data Backends](../CLAUDE.md#45-patterns--abstractions) (IDataBackend interface)

---

### 9.3 AWS License Service Deployment

**What it is**: Lobster's license service is deployed as an AWS serverless application using Lambda, API Gateway, DynamoDB, and KMS. This section covers the **security architecture** and **deployment best practices** for the license service.

**Architecture overview**:

```
API Gateway (REST) → Lambda (Python 3.12) → DynamoDB + KMS
       ↓                    ↓                       ↓
  HTTPS only          ARM64 runtime         RSA-2048 signing
  Rate limiting       256 MB memory         Private key in HSM
  IAM auth            10s timeout           Automatic rotation
```

**Security layers**:

| Layer | Control | Implementation |
|-------|---------|----------------|
| **API Gateway** | Rate limiting | 1000 req/sec burst, 5000 req/sec steady |
| **API Gateway** | HTTPS enforcement | TLS 1.2+ required |
| **Lambda** | IAM execution role | Least privilege (DynamoDB, KMS only) |
| **Lambda** | VPC isolation | Optional (private subnet) |
| **DynamoDB** | Encryption at rest | AWS-managed keys (KMS) |
| **DynamoDB** | Point-in-time recovery | Automatic backups |
| **KMS** | HSM-backed keys | FIPS 140-2 Level 2 validated |
| **KMS** | Key rotation | Automatic annual rotation |
| **CloudWatch** | Audit logging | All API calls logged |

**Lambda execution role** (least privilege):
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "dynamodb:GetItem",
        "dynamodb:PutItem",
        "dynamodb:UpdateItem",
        "dynamodb:Query"
      ],
      "Resource": [
        "arn:aws:dynamodb:us-east-1:*:table/LobsterEntitlements",
        "arn:aws:dynamodb:us-east-1:*:table/LobsterCustomers",
        "arn:aws:dynamodb:us-east-1:*:table/LobsterAuditLogs"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "kms:Sign",
        "kms:GetPublicKey"
      ],
      "Resource": "arn:aws:kms:us-east-1:*:key/license-signing-key"
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:us-east-1:*:log-group:/aws/lambda/LobsterLicenseService:*"
    }
  ]
}
```

**DynamoDB table security**:

| Table | Encryption | Backup | TTL |
|-------|-----------|--------|-----|
| **Entitlements** | KMS | PITR | No |
| **Customers** | KMS | PITR | No |
| **AuditLogs** | KMS | PITR | 90 days |

**Deployment** (AWS CDK):
```bash
# Install dependencies
cd lobster-cloud
source .venv/bin/activate
pip install -r requirements.txt

# Deploy license service
export AWS_REGION=us-east-1
export JSII_SILENCE_WARNING_UNTESTED_NODE_VERSION=1

cdk deploy LobsterLicenseService \
  --context signing_key_arn=arn:aws:kms:us-east-1:*:key/your-key-id \
  --context environment=production

# Output:
# API Gateway URL: https://x6gm9vfgl5.execute-api.us-east-1.amazonaws.com/v1
# JWKS URL: https://d123abc456.cloudfront.net/.well-known/jwks.json
```

**Monitoring & alerting**:
```bash
# CloudWatch alarms (CDK automatically creates)
- High error rate (>5% errors)
- High latency (>1s P99)
- DynamoDB throttling
- KMS rate limiting
- Lambda concurrency limit

# CloudWatch Insights queries
fields @timestamp, @message
| filter @message like /ERROR/
| stats count() by bin(5m)
```

**Security best practices**:

1. **API Gateway**: Enable AWS WAF for DDoS protection
2. **Lambda**: Use VPC endpoints for DynamoDB/KMS (no internet)
3. **KMS**: Enable automatic key rotation (annual)
4. **DynamoDB**: Enable point-in-time recovery (PITR)
5. **CloudWatch**: Set up alarms for security events
6. **IAM**: Regularly audit IAM roles and policies

**Compliance benefits**:

- ✅ **SOC 2** - AWS SOC 2 certified services
- ✅ **HIPAA** - DynamoDB encryption, audit logging
- ✅ **GDPR** - Data residency (region selection)
- ✅ **21 CFR Part 11** - Audit trails (CloudTrail, CloudWatch)
- ✅ **FedRAMP** - AWS GovCloud deployment option

**For complete implementation details**, see:
- [Premium Licensing Technical Guide](../../docs/PREMIUM_LICENSING.md) (comprehensive architecture, 4 phases)
- [License Service Stack](../lobster-cloud/infrastructure/license_service_stack.py) (CDK infrastructure)
- [CLAUDE.md - License Service](../CLAUDE.md#license-service-architecture-aws-serverless) (overview)

---

## 10. Compliance Features for Regulated Environments

### 10.1 GxP-Ready Checklist

**What it is**: Lobster implements multiple features that align with **Good Practice (GxP)** requirements for regulated pharmaceutical and clinical research. This checklist maps Lobster features to GxP principles and regulatory requirements.

**ALCOA+ Principles** (FDA Data Integrity Guidance):

| Principle | Requirement | Lobster Implementation | Section Reference |
|-----------|-------------|------------------------|-------------------|
| **Attributable** | Who performed the action? | Agent attribution in W3C-PROV, session metadata tracks users | [3.1](#31-w3c-prov-compliance) |
| **Legible** | Can data be read/understood? | Human-readable JSON, Plotly visualizations, markdown reports | [2.0](#understanding-the-manifest) |
| **Contemporaneous** | Recorded in real-time? | UTC timestamps on all operations, immediate session updates | [3.3](#33-session-and-tool-usage-tracking) |
| **Original** | First recording preserved? | W3C-PROV provenance, immutable activity log | [3.1](#31-w3c-prov-compliance) |
| **Accurate** | Free from errors? | Schema validation, pre-download validation, QC checks | [8.1](#81-schema-validation) |
| **+ Complete** | All data present? | Required field validation, session metadata complete | [8.1](#81-schema-validation) |
| **+ Consistent** | Data relationships valid? | Cross-field validation, modality compatibility checks | [8.3](#83-pre-download-validation) |
| **+ Enduring** | Long-term preservation? | H5AD/MuData formats, S3 archival, notebook exports | [9.2](#92-s3-backend-security) |
| **+ Available** | Accessible when needed? | Session restoration, workspace archival, S3 backend | [6.3](#63-session-management-and-data-restoration) |

**21 CFR Part 11 Requirements** (Electronic Records):

| Regulation | Requirement | Lobster Implementation | Status |
|------------|-------------|------------------------|--------|
| **§ 11.10(a)** | System validation | Testing framework, CI/CD, reproducible builds | ✅ Ready |
| **§ 11.10(b)** | Ability to generate accurate copies | Notebook export, workspace archival, provenance export | ✅ Ready |
| **§ 11.10(c)** | Protection against unauthorized access | Workspace permissions, subscription tiers, license validation | ✅ Ready |
| **§ 11.10(d)** | Secure timestamped audit trails | W3C-PROV with UTC timestamps, session tracking | ✅ Ready |
| **§ 11.10(e)** | Audit trail review | Provenance queries, session status, activity logs | ✅ Ready |
| **§ 11.10(k)(1)** | System checks (validation) | Schema validation, accession validation, QC metrics | ✅ Ready |
| **§ 11.10(k)(2)** | Authority checks | Subscription tier enforcement, license manager | ✅ Ready |

**ISO/IEC 27001:2022** (Information Security):

| Control | Category | Lobster Implementation | Section |
|---------|----------|------------------------|---------|
| **A.8.1** | Asset management | Modality tracking, workspace inventory | [6.1](#61-workspace-isolation) |
| **A.8.10** | Information deletion | Modality removal with audit trail | [6.3](#63-session-management-and-data-restoration) |
| **A.8.15** | Access logging | Session metadata, tool usage tracking | [3.3](#33-session-and-tool-usage-tracking) |
| **A.8.16** | Audit logs | W3C-PROV provenance, DynamoDB audit logs | [3.1](#31-w3c-prov-compliance) |
| **A.8.24** | Cryptographic controls | SHA-256 hashing, RSA-2048 signatures, KMS | [2.0](#understanding-the-manifest), [4.1](#41-license-management-system) |

**Compliance readiness matrix**:

| Regulation | Current Status | Deployment Mode | Notes |
|------------|---------------|-----------------|-------|
| **21 CFR Part 11** | ✅ Ready | Local + Cloud | Full audit trail, validation, access control |
| **HIPAA** | ⚠️ Conditional | Local (ready), Cloud (BAA required) | Encryption, access logs available |
| **GDPR** | ⚠️ Conditional | Local (ready), Cloud (region + DPA) | Data residency configurable |
| **GxP (GAMP 5)** | ⚠️ Partial | Local (ready for Cat 4), Cloud (validation TBD) | IQ/OQ/PQ documentation needed |
| **ISO/IEC 27001** | ✅ Ready | Local + Cloud | Information security controls implemented |
| **SOC 2 Type II** | ⚠️ Partial | Cloud (AWS certified), Lobster (audit pending) | AWS inherits certification |

**Quick deployment checklist** (regulated environments):

```bash
# 1. ✅ Enable all audit features
export LOBSTER_ENABLE_PROVENANCE=true  # Default: enabled
export LOBSTER_ENABLE_INTEGRITY_MANIFEST=true  # Default: enabled

# 2. ✅ Configure secure workspace
mkdir -p /validated/workspaces/project_gse109564
chmod 700 /validated/workspaces/project_gse109564
export LOBSTER_WORKSPACE=/validated/workspaces/project_gse109564

# 3. ✅ Use PREMIUM tier (metadata_assistant for publication processing)
lobster activate lbstr_premium_key_abc123

# 4. ✅ Enable Redis for rate limiting (multi-user)
docker run -d -p 6379:6379 redis:alpine
export REDIS_URL=redis://localhost:6379

# 5. ✅ Set up API keys securely
cat > .env << EOF
ANTHROPIC_API_KEY=sk-ant-api03-...
NCBI_API_KEY=abc123...
EOF
chmod 600 .env

# 6. ✅ Verify compliance features
lobster status
# Check: Provenance enabled, Subscription tier, Workspace path
```

**Compliance benefits**:

- ✅ **Complete ALCOA+ coverage** - All 9 principles supported
- ✅ **21 CFR Part 11 ready** - Electronic records & signatures
- ✅ **Multi-regulation support** - HIPAA, GDPR, GxP, ISO 27001
- ✅ **Audit-ready** - Comprehensive provenance + integrity manifests
- ✅ **Validation-friendly** - Reproducible notebooks, fixed environments

**For complete implementation details**, see:
- [Regulatory Compliance Roadmap](../../docs/REGULATORY_COMPLIANCE_ROADMAP.md) (comprehensive GxP guidance)
- [Premium Licensing - Compliance](../../docs/PREMIUM_LICENSING.md#compliance-considerations) (technical compliance)

---

### 10.2 Deployment Patterns for Regulated Environments

**What it is**: Recommended deployment architectures for different regulatory compliance levels (GxP, HIPAA, GDPR, SOC 2).

**Pattern 1: Academic/Research** (minimal compliance):

```
Deployment: Local CLI
Security: Basic
Compliance: Internal only

Components:
├── Local machine (macOS/Linux/Windows)
├── Lobster CLI (pip install lobster-ai)
├── Local workspace (~/.lobster_workspace)
└── API keys in .env file

Suitable for:
- Academic research (public data)
- Exploratory analysis
- Individual researchers
```

**Pattern 2: Biotech Startup** (moderate compliance):

```
Deployment: Local CLI + shared workspaces
Security: Enhanced
Compliance: GLP, internal QA

Components:
├── Shared Linux server
├── Lobster CLI (Docker container)
├── Redis (rate limiting coordination)
├── Shared workspaces (/shared/projects/*)
├── API keys in HashiCorp Vault
└── Weekly provenance audits

Suitable for:
- Small biotech companies (5-20 users)
- Confidential but non-GxP data
- Internal QA requirements
```

**Pattern 3: Pharma Enterprise** (full compliance):

```
Deployment: Validated environment
Security: Maximum
Compliance: GxP (GAMP Cat 4), 21 CFR Part 11

Components:
├── Validated Linux environment (air-gapped)
├── Lobster CLI (Docker, validated image)
├── Redis (high availability cluster)
├── S3 backend (encrypted, versioned, WORM)
├── API keys in AWS Secrets Manager
├── Automated IQ/OQ/PQ testing
├── Change control process
└── Annual validation review

Suitable for:
- Pharmaceutical companies (GxP data)
- Clinical trials (patient data)
- Regulatory submissions (FDA, EMA)
```

**Pattern 4: Cloud SaaS** (multi-tenant):

```
Deployment: AWS serverless
Security: Maximum + isolation
Compliance: HIPAA (BAA), SOC 2, GDPR

Components:
├── AWS Lambda (auto-scaling)
├── API Gateway (rate limiting)
├── DynamoDB (encrypted)
├── S3 (per-tenant workspaces)
├── Redis ElastiCache (rate limiting)
├── KMS (encryption keys)
├── CloudWatch (audit logs)
└── AWS WAF (DDoS protection)

Suitable for:
- Multi-tenant SaaS
- Managed service offering
- Large-scale processing (100s of users)
```

**Comparison matrix**:

| Requirement | Academic | Biotech | Pharma | Cloud SaaS |
|-------------|----------|---------|--------|------------|
| **Setup time** | 10 min | 2 hours | 2-4 weeks | 1-2 weeks |
| **Compliance** | None | Internal QA | GxP validated | HIPAA/SOC 2 |
| **Cost** | $0 | $100-500/mo | $5K-20K one-time | $10K-50K setup |
| **Security** | Basic | Enhanced | Maximum | Maximum |
| **User capacity** | 1-5 | 5-20 | 20-100 | 100-1000s |

**Deployment decision tree**:

```
Q: Do you handle patient data (PHI/PII)?
├─ YES: Pattern 3 (Pharma) or Pattern 4 (Cloud with BAA)
└─ NO: Continue

Q: Do you need GxP validation?
├─ YES: Pattern 3 (Pharma)
└─ NO: Continue

Q: Do you have >20 users?
├─ YES: Pattern 4 (Cloud SaaS)
└─ NO: Continue

Q: Do you need multi-user coordination?
├─ YES: Pattern 2 (Biotech)
└─ NO: Pattern 1 (Academic)
```

**Compliance benefits**:

- ✅ **Flexible deployment** - Choose pattern per compliance needs
- ✅ **Scalability** - Start simple, upgrade as regulations require
- ✅ **Cost-effective** - Pay only for compliance level needed
- ✅ **Audit-ready** - All patterns support provenance tracking

**For complete implementation details**, see:
- [Docker Deployment Guide](43-docker-deployment-guide.md) (Pattern 1, 2, 3)
- [Cloud-Local Architecture](21-cloud-local-architecture.md) (Pattern 4)
- [S3 Backend Guide](43-s3-backend-guide.md) (Pattern 3, 4)

---

### 10.3 Standard Operating Procedures (SOPs)

**What it is**: Template SOPs for integrating Lobster AI into regulated workflows. These templates can be customized for specific organizational requirements.

**SOP 1: Data Analysis with Lobster AI** (template):

```markdown
## SOP-LOBSTER-001: Bioinformatics Data Analysis

**Purpose**: Standardize use of Lobster AI for bioinformatics analysis in GxP environment

**Scope**: All analysts performing bioinformatics analysis on GxP data

**Responsibilities**:
- Analyst: Execute analysis, document decisions
- Lead Analyst: Review analysis, approve results
- QA: Verify data integrity, audit provenance

**Procedure**:

1. **Session Initialization**
   - Create dedicated workspace: `lobster chat --workspace /validated/project_name/`
   - Verify subscription tier: `lobster status` (PREMIUM required for GxP)
   - Document session ID in lab notebook

2. **Data Loading**
   - Use validated data sources only (GEO, internal repositories)
   - Verify accession format before download
   - Check Data Integrity Manifest after download
   - Document: Dataset ID, download date, file hash

3. **Analysis Execution**
   - Follow validated workflows (clustering, DE, etc.)
   - Document all custom code with justification
   - Review QC metrics at each step
   - Save all plots to workspace/plots/

4. **Notebook Export**
   - Export pipeline: `/pipeline export`
   - Verify Data Integrity Manifest present
   - Verify provenance hash included
   - Archive notebook + data + provenance.json

5. **Review and Approval**
   - Lead analyst reviews notebook
   - QA verifies file hashes match manifest
   - Approve for downstream use (submission, publication)
   - Document approval in QMS (Quality Management System)

**Audit Trail**:
- All operations logged to provenance.json
- Session metadata saved to .session.json
- Notebook includes Data Integrity Manifest
- Hashes verify data authenticity

**Revision History**:
- Version 1.0: Initial SOP (2026-01-01)
```

**SOP 2: Hash Verification for Data Integrity** (template):

```markdown
## SOP-LOBSTER-002: Data Integrity Verification

**Purpose**: Verify cryptographic hashes in Lobster AI notebooks

**Scope**: All notebooks used for regulatory submissions or GxP decisions

**Procedure**:

1. **Notebook Receipt**
   - Receive notebook file (.ipynb) from analyst
   - Receive data files referenced in notebook
   - Receive provenance.json from workspace

2. **Hash Extraction**
   - Open notebook in Jupyter or text editor
   - Locate "🔒 Data Integrity Manifest" cell (cell 2)
   - Extract input_files section with SHA-256 hashes

3. **Hash Verification**
   ```bash
   # For each file in input_files
   shasum -a 256 filename.h5ad
   # Compare output to manifest hash
   ```

4. **Provenance Verification**
   ```python
   # Verify provenance hash
   python verify_provenance_hash.py provenance.json manifest_hash
   ```

5. **Documentation**
   - Record verification results in QA log
   - If PASS: Approve notebook for review
   - If FAIL: Return to analyst for investigation

**Acceptance Criteria**:
- ✅ All file hashes match manifest
- ✅ Provenance hash matches manifest
- ✅ System info documented (lobster version, git commit)

**Revision History**:
- Version 1.0: Initial SOP (2026-01-01)
```

**SOP 3: Custom Code Review** (template):

```markdown
## SOP-LOBSTER-003: Custom Code Review

**Purpose**: Review and approve custom code blocks in Lobster analyses

**Scope**: All analyses using execute_custom_code tool

**Procedure**:

1. **Pre-Execution Review** (required for GxP):
   - Analyst documents custom code justification
   - Lead analyst reviews code for:
     - Security risks (subprocess, network access)
     - Data integrity risks (file manipulation)
     - Scientific validity (correct operations)
   - Approval documented in lab notebook

2. **Post-Execution Audit** (quarterly):
   - QA extracts custom code from provenance.json
   - Review for patterns (can standardize?)
   - Check for forbidden operations
   - Document findings in QA report

**Forbidden Patterns** (auto-blocked):
- ❌ subprocess, os.system, importlib
- ❌ eval, exec, compile
- ❌ File access outside workspace

**Allowed Patterns**:
- ✅ Pandas filtering (complex conditions)
- ✅ Custom QC checks
- ✅ Format conversions

**Revision History**:
- Version 1.0: Initial SOP (2026-01-01)
```

**Compliance benefits**:

- ✅ **Standardized procedures** - Consistent workflows across teams
- ✅ **Documentation** - SOPs required for GxP validation
- ✅ **Training** - Clear guidance for analysts and QA
- ✅ **Audit support** - Procedures documented for inspections

---

### 10.4 Validation Testing for GxP

**What it is**: Guidance for performing Installation Qualification (IQ), Operational Qualification (OQ), and Performance Qualification (PQ) testing for Lobster AI in validated environments.

**GAMP 5 Category**: Category 4 (Configurable software)
**Validation approach**: Risk-based, leveraging vendor testing

**IQ (Installation Qualification)** - Verify correct installation:

| Test | Procedure | Acceptance Criteria |
|------|-----------|---------------------|
| **Version verification** | `lobster --version` | Matches validated version |
| **Dependency check** | `pip list | grep -E "(scanpy|anndata|pydeseq2)"` | All dependencies present |
| **Workspace creation** | `lobster query "test" --workspace /validated/test/` | Workspace created successfully |
| **Provenance enabled** | Check `.lobster_workspace/provenance.json` exists | File created |
| **License validation** | `lobster status` | PREMIUM tier active |

**IQ checklist**:
```bash
# 1. Version check
lobster --version
# Expected: Lobster AI CLI v0.3.4 (or specified version)

# 2. Dependency verification
pip list | grep -E "(scanpy|anndata|pydeseq2|plotly)"

# 3. Create test workspace
lobster query "test installation" --workspace /validated/iq_test/
ls /validated/iq_test/.lobster_workspace/

# 4. Verify provenance file
cat /validated/iq_test/.lobster_workspace/provenance.json

# 5. Check license
lobster status
# Expected: Subscription Tier: premium

# 6. Document results in IQ report
```

**OQ (Operational Qualification)** - Verify features work correctly:

| Test | Procedure | Acceptance Criteria |
|------|-----------|---------------------|
| **Data download** | Download GSE109564 | Data loaded, hash in manifest |
| **Quality control** | Run QC on test dataset | QC metrics calculated |
| **Clustering** | Perform Leiden clustering | Clusters assigned |
| **Visualization** | Generate UMAP plot | Plot saved to plots/ |
| **Notebook export** | `/pipeline export` | Notebook with integrity manifest |
| **Hash verification** | Verify SHA-256 hashes | All hashes match |
| **Provenance query** | Query session activities | Complete audit trail |

**OQ test script** (automated):
```python
#!/usr/bin/env python3
"""OQ test script for Lobster AI validation."""

import subprocess
import json
import hashlib
from pathlib import Path

WORKSPACE = Path("/validated/oq_test/.lobster_workspace")

def run_oq_test():
    """Execute operational qualification tests."""
    tests = [
        ("Download dataset", "Download GSE109564 and assess quality"),
        ("Clustering", "Cluster the data with resolution 0.5"),
        ("Export notebook", "Export the analysis pipeline"),
    ]

    for test_name, command in tests:
        print(f"\n{'='*60}")
        print(f"OQ Test: {test_name}")
        print(f"{'='*60}")

        result = subprocess.run(
            ["lobster", "query", command, "--workspace", str(WORKSPACE.parent)],
            capture_output=True,
            text=True,
            timeout=600
        )

        if result.returncode == 0:
            print(f"✅ PASS: {test_name}")
        else:
            print(f"❌ FAIL: {test_name}")
            print(f"Error: {result.stderr}")
            return False

    # Verify provenance file
    prov_path = WORKSPACE / "provenance.json"
    if prov_path.exists():
        with open(prov_path) as f:
            prov = json.load(f)
        print(f"\n✅ Provenance file exists: {len(prov.get('activities', []))} activities")
    else:
        print("\n❌ Provenance file missing")
        return False

    return True

if __name__ == "__main__":
    success = run_oq_test()
    exit(0 if success else 1)
```

**PQ (Performance Qualification)** - Verify performance with real data:

| Test | Procedure | Acceptance Criteria |
|------|-----------|---------------------|
| **Large dataset** | Download GSE150614 (15 GB) | Completes in <30 min |
| **Complex analysis** | Full workflow (QC → cluster → DE) | Completes in <2 hours |
| **Batch processing** | Process 10 datasets | All complete successfully |
| **Concurrent users** | 5 users simultaneous | No conflicts, no errors |
| **Data integrity** | Verify hashes for all outputs | All hashes match |

**PQ acceptance criteria**:
- ✅ Performance meets specifications (time limits)
- ✅ Results scientifically valid (QC metrics within range)
- ✅ Data integrity maintained (all hashes verify)
- ✅ Audit trail complete (all operations logged)
- ✅ Reproducibility confirmed (re-run generates same results)

**Validation documentation structure**:

```
validation_package/
├── VP_001_Validation_Plan.pdf
├── IQ_001_Installation_Qualification.pdf
│   ├── Test cases 1-6
│   ├── Screenshots
│   └── Signatures (analyst, QA, manager)
├── OQ_001_Operational_Qualification.pdf
│   ├── Test cases 1-7
│   ├── Test data
│   └── Signatures
├── PQ_001_Performance_Qualification.pdf
│   ├── Test cases 1-5
│   ├── Performance data
│   └── Signatures
└── Summary_Report.pdf
    ├── Validation summary
    ├── Deviations (if any)
    └── Final approval signatures
```

**Compliance benefits**:

- ✅ **GxP validation** - IQ/OQ/PQ documented
- ✅ **Risk-based** - GAMP 5 Category 4 approach
- ✅ **Audit-ready** - Complete validation package
- ✅ **Reproducible** - Automated test scripts
- ✅ **Change control** - Re-validation on version updates

**For complete implementation details**, see:
- [Regulatory Compliance Roadmap](../../docs/REGULATORY_COMPLIANCE_ROADMAP.md) (IQ/OQ/PQ templates)
- [Testing Guide](12-testing-guide.md) (test framework)

---

## 11. Security Best Practices

### 11.1 Environment Configuration Security

**What it is**: Best practices for securely configuring Lobster AI environments (development, staging, production) to prevent credential leaks, unauthorized access, and configuration drift.

**Configuration hierarchy** (secure defaults):

| Level | File Location | Priority | Use Case | Security |
|-------|--------------|----------|----------|----------|
| **Workspace** | `./project/.env` | Highest | Project-specific keys | Project isolation |
| **Global** | `~/.lobster/.env` | Medium | User-wide keys | User isolation |
| **System** | `/etc/lobster/.env` | Low | System-wide (enterprise) | Shared configs only |
| **Environment** | `export VAR=value` | Lowest | CI/CD, containers | Ephemeral |

**Secure .env file setup**:

```bash
# ✅ GOOD: Create workspace-specific .env (highest priority)
cat > ~/project1/.env << EOF
# LLM Provider (required)
ANTHROPIC_API_KEY=sk-ant-api03-...

# Optional: NCBI API key (3x rate limit increase)
NCBI_API_KEY=abc123...

# Optional: Cloud key (for cloud mode)
LOBSTER_CLOUD_KEY=lbstr_premium_...
EOF

# Secure permissions (user-only)
chmod 600 ~/project1/.env

# ✅ GOOD: Add to .gitignore (prevent commits)
echo ".env" >> .gitignore
echo "*.env" >> .gitignore
git add .gitignore
```

**Common security mistakes**:

```bash
# ❌ BAD: Commit secrets to git
git add .env
git commit -m "Add config"  # Credentials leaked to history!

# ❌ BAD: World-readable permissions
chmod 644 .env  # Anyone can read API keys

# ❌ BAD: Hardcode in scripts
export ANTHROPIC_API_KEY="sk-ant-..."  >> setup.sh
git add setup.sh  # Key committed to git

# ❌ BAD: Share keys across users
echo "ANTHROPIC_API_KEY=shared" > /etc/lobster/.env  # Security risk

# ❌ BAD: Store keys in plaintext in cloud
aws s3 cp .env s3://public-bucket/.env  # Publicly accessible!
```

**Environment-specific best practices**:

**Development**:
```bash
# Use personal API keys (not shared)
cat > .env << EOF
ANTHROPIC_API_KEY=$PERSONAL_ANTHROPIC_KEY
NCBI_API_KEY=$PERSONAL_NCBI_KEY
EOF
chmod 600 .env

# Use .env.example for team (no real keys)
cat > .env.example << EOF
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
NCBI_API_KEY=your-ncbi-key-here
EOF
git add .env.example
```

**Staging**:
```bash
# Use staging-specific keys (rotated monthly)
export ANTHROPIC_API_KEY=$(aws secretsmanager get-secret-value \
  --secret-id staging/lobster/anthropic \
  --query SecretString --output text)

# Separate workspace
export LOBSTER_WORKSPACE=/staging/workspaces/
```

**Production**:
```bash
# Use production keys (rotated quarterly)
export ANTHROPIC_API_KEY=$(vault kv get -field=api_key secret/prod/lobster/anthropic)

# Read-only data sources
export LOBSTER_WORKSPACE=/production/workspaces/
chmod 500 /production/data/  # Read + execute only

# Enable all audit features
export LOBSTER_ENABLE_PROVENANCE=true
export LOBSTER_ENABLE_INTEGRITY_MANIFEST=true
```

**Secret rotation schedule**:

| Secret Type | Rotation Frequency | Trigger | Process |
|-------------|-------------------|---------|---------|
| **NCBI API keys** | Quarterly | Calendar | Generate new key, update .env, test |
| **Anthropic keys** | Quarterly | Calendar | Rotate via Anthropic Console |
| **AWS credentials** | On departure | Team member leaves | Revoke IAM access, issue new keys |
| **Lobster cloud keys** | Annual | Subscription renewal | License service auto-renews |
| **SSH keys** | Bi-annually | Calendar | Generate new keypair |

**Compliance benefits**:

- ✅ **Credential protection** - Secure storage, rotation policies
- ✅ **Access control** - File permissions enforce isolation
- ✅ **Audit trail** - Configuration changes logged
- ✅ **Incident response** - Rapid key rotation on compromise
- ✅ **Principle of least privilege** - Workspace > global > system

**For complete implementation details**, see:
- [Configuration Guide](03-configuration.md) (comprehensive configuration)
- [API Key Security](#43-api-key-security) (current page, Section 4.3)

---

### 11.2 Access Control Best Practices

**What it is**: Operational best practices for managing user access, workspace permissions, and data isolation in multi-user deployments.

**User access model** (recommended):

| User Type | Access Level | Workspace | Subscription Tier | Use Case |
|-----------|-------------|-----------|-------------------|----------|
| **Analyst** | Read/Write own workspace | `~/workspaces/$USERNAME/` | FREE/PREMIUM | Individual analysis |
| **Lead Analyst** | Read all team workspaces | `/team/workspaces/*/` | PREMIUM | Team oversight |
| **QA** | Read-only all workspaces | `/validated/workspaces/*/` | PREMIUM | Audit and review |
| **Admin** | Full system access | All paths | ENTERPRISE | System maintenance |

**Filesystem permissions** (Linux/macOS):

```bash
# Individual analyst workspace (user-only)
mkdir -p ~/workspaces/analyst1
chmod 700 ~/workspaces/analyst1  # rwx------
chown analyst1:analyst1 ~/workspaces/analyst1

# Team workspace (group access)
mkdir -p /team/project_gse109564
chmod 770 /team/project_gse109564  # rwxrwx---
chown analyst1:bioinfo_team /team/project_gse109564

# QA workspace (read-only for QA team)
mkdir -p /validated/project_gse109564
chmod 750 /validated/project_gse109564  # rwxr-x---
chown analyst1:qa_team /validated/project_gse109564

# Archive workspace (read-only for everyone)
mkdir -p /archives/completed_analyses
chmod 555 /archives/completed_analyses  # r-xr-xr-x
```

**Docker multi-user deployment**:

```yaml
# docker-compose.yml (multi-user)
version: '3.8'

services:
  lobster-analyst1:
    image: omicsos/lobster:latest
    user: "1000:1000"  # UID:GID for analyst1
    environment:
      - ANTHROPIC_API_KEY=${ANALYST1_API_KEY}
      - REDIS_URL=redis://redis:6379
    volumes:
      - /workspaces/analyst1:/workspace:rw  # Read-write own workspace
      - /team/shared:/team:rw                # Read-write team workspace
      - /validated:/validated:ro             # Read-only validated data

  lobster-qa:
    image: omicsos/lobster:latest
    user: "2000:2000"  # UID:GID for QA user
    environment:
      - ANTHROPIC_API_KEY=${QA_API_KEY}
    volumes:
      - /validated:/workspace:ro             # Read-only access
      - /archives:/archives:ro               # Read-only archives
```

**Access logging** (audit trail):

```bash
# Monitor workspace access (Linux)
auditctl -w /validated/workspaces/ -p rwa -k lobster_access

# View access logs
ausearch -k lobster_access

# Or use inotify for real-time monitoring
inotifywait -m -r -e access,modify,create,delete /validated/workspaces/
```

**Access control checklist** (enterprise):

```bash
# 1. ✅ Verify user isolation
ls -la ~/workspaces/
# Each user should only see their own directory

# 2. ✅ Test read-only restrictions (QA user)
su - qa_user
lobster query "Load data" --workspace /validated/project/
# Should succeed (read-only)

echo "test" > /validated/project/.lobster_workspace/unauthorized.txt
# Should fail (permission denied)

# 3. ✅ Verify subscription tier enforcement
su - analyst_free
lobster status
# Should show: Subscription Tier: free

lobster query "Process publication queue"
# Should fail: metadata_assistant requires PREMIUM

# 4. ✅ Audit trail verification
cat /validated/project/.lobster_workspace/.session.json | jq '.tool_usage'
# Should show all operations with timestamps + agent attribution
```

**Compliance benefits**:

- ✅ **Access control** - OS-level + subscription tier enforcement
- ✅ **Data isolation** - Per-user/per-team workspaces
- ✅ **Audit trail** - All access logged
- ✅ **Principle of least privilege** - Role-based permissions
- ✅ **Multi-user support** - Enterprise deployment ready

---

### 11.3 Data Handling Best Practices

**What it is**: Operational guidance for securely handling sensitive data (PHI, PII, confidential) with Lobster AI.

**Data classification** (example):

| Classification | Examples | Lobster Deployment | Compliance |
|----------------|----------|-------------------|------------|
| **Public** | GEO datasets, published papers | Local or Cloud | None |
| **Internal** | Unpublished experiments | Local (recommended) | Internal QA |
| **Confidential** | Proprietary assays, IP | Local only | NDA, trade secret |
| **Sensitive** | Patient data (PHI/PII) | Local (validated) | HIPAA, GDPR |

**Handling sensitive data**:

```bash
# ✅ GOOD: Local mode for PHI/PII
unset LOBSTER_CLOUD_KEY  # Ensure local mode
export LOBSTER_WORKSPACE=/encrypted/phi_data/project_xyz
lobster chat --workspace /encrypted/phi_data/project_xyz

# ✅ GOOD: Encrypted filesystem (Linux)
cryptsetup luksFormat /dev/sdb1
cryptsetup luksOpen /dev/sdb1 encrypted_data
mkfs.ext4 /dev/mapper/encrypted_data
mount /dev/mapper/encrypted_data /encrypted/phi_data

# ✅ GOOD: Automatic workspace cleanup (after archival)
tar -czf project_archive.tar.gz /encrypted/phi_data/project_xyz
shasum -a 256 project_archive.tar.gz >> archive_manifest.txt
rm -rf /encrypted/phi_data/project_xyz  # After archival only

# ❌ BAD: Cloud mode with PHI (without BAA)
export LOBSTER_CLOUD_KEY=lbstr_...
lobster query "Analyze patient data"  # PHI sent to cloud (HIPAA violation!)
```

**Data retention policies** (example):

| Data Type | Retention | Storage | Deletion |
|-----------|-----------|---------|----------|
| **Raw data** | 7 years | S3 Glacier | Automated (lifecycle) |
| **Analysis results** | 3 years | S3 Standard | Manual review |
| **Provenance logs** | 7 years | S3 Glacier | Automated |
| **Notebooks** | Permanent | S3 Standard-IA | Never (archival) |
| **Temporary files** | 30 days | Local disk | Automated cleanup |

**Data anonymization** (for PHI):

```python
# ✅ GOOD: Anonymize metadata before analysis
execute_custom_code("""
import pandas as pd

# Load metadata with PHI
metadata = pd.read_csv(WORKSPACE / 'metadata_with_phi.csv')

# Remove PHI columns
phi_columns = ['patient_id', 'patient_name', 'date_of_birth', 'ssn']
metadata_anon = metadata.drop(columns=phi_columns)

# Generate anonymous IDs
metadata_anon['sample_id'] = [f"SAMPLE_{i:06d}" for i in range(len(metadata_anon))]

# Save anonymized version
metadata_anon.to_csv(OUTPUT_DIR / 'metadata_anonymized.csv', index=False)

# Delete original (after verification)
# os.remove(WORKSPACE / 'metadata_with_phi.csv')  # Manual step
""")
```

**Data transfer security**:

```bash
# ✅ GOOD: Encrypted transfer (SCP)
scp -i ~/.ssh/id_rsa -C \
  analyst1@server:/workspaces/project/analysis.tar.gz \
  ~/local_copy/

# ✅ GOOD: Verify integrity after transfer
shasum -a 256 ~/local_copy/analysis.tar.gz
# Compare to hash from manifest

# ✅ GOOD: Use SFTP for large files
sftp analyst1@server
sftp> get /workspaces/project/large_dataset.h5ad

# ❌ BAD: Unencrypted transfer
ftp analyst1@server  # Plain FTP (no encryption)
scp -o "StrictHostKeyChecking=no" ...  # Disables host verification (MITM risk)
```

**Compliance benefits**:

- ✅ **Data protection** - Classification-based handling
- ✅ **HIPAA compliance** - PHI isolation, local mode
- ✅ **GDPR compliance** - Data retention, anonymization
- ✅ **Audit trail** - All data operations logged
- ✅ **Incident response** - Clear procedures for data breaches

---

### 11.4 Monitoring and Incident Response

**What it is**: Proactive monitoring, alerting, and incident response procedures for Lobster AI deployments.

**Monitoring stack** (recommended):

| Component | Tool | Purpose | Alert Threshold |
|-----------|------|---------|----------------|
| **System health** | `lobster status` | Check CLI functionality | Errors in output |
| **Disk usage** | `df -h` | Monitor workspace size | >80% full |
| **Provenance logs** | `jq` queries | Audit trail analysis | Error rate >5% |
| **Redis health** | `redis-cli ping` | Rate limiter availability | Connection failures |
| **API errors** | Log aggregation | Network failure tracking | >10 failures/hour |

**Health check script** (automated monitoring):

```python
#!/usr/bin/env python3
"""Lobster AI health check for monitoring systems."""

import subprocess
import json
from pathlib import Path

def check_lobster_health():
    """Run health checks and return status."""
    checks = {
        "cli_available": False,
        "workspace_writable": False,
        "provenance_enabled": False,
        "redis_available": False,
        "disk_space_ok": False
    }

    # 1. CLI availability
    result = subprocess.run(["lobster", "--version"], capture_output=True)
    checks["cli_available"] = result.returncode == 0

    # 2. Workspace writable
    workspace = Path.home() / ".lobster_workspace"
    try:
        test_file = workspace / ".health_check"
        test_file.touch()
        test_file.unlink()
        checks["workspace_writable"] = True
    except:
        pass

    # 3. Provenance enabled
    result = subprocess.run(
        ["lobster", "query", "test", "--workspace", str(workspace.parent)],
        capture_output=True,
        timeout=30
    )
    prov_file = workspace / "provenance.json"
    checks["provenance_enabled"] = prov_file.exists()

    # 4. Redis availability (if configured)
    result = subprocess.run(["redis-cli", "ping"], capture_output=True)
    checks["redis_available"] = b"PONG" in result.stdout

    # 5. Disk space
    result = subprocess.run(["df", "-h", str(workspace)], capture_output=True, text=True)
    # Parse disk usage (simplified)
    checks["disk_space_ok"] = "100%" not in result.stdout

    # Report
    all_ok = all(checks.values())
    if all_ok:
        print("✅ ALL CHECKS PASSED")
        return 0
    else:
        print("❌ HEALTH CHECK FAILURES:")
        for check, status in checks.items():
            if not status:
                print(f"  - {check}: FAIL")
        return 1

if __name__ == "__main__":
    exit(check_lobster_health())
```

**Alerting rules** (example for Prometheus/Grafana):

```yaml
# Lobster health monitoring
groups:
  - name: lobster_alerts
    interval: 5m
    rules:
      - alert: LobsterHighErrorRate
        expr: rate(lobster_errors_total[5m]) > 0.05
        annotations:
          summary: "Lobster error rate >5% in last 5 minutes"

      - alert: LobsterWorkspaceFull
        expr: node_filesystem_avail_bytes{mountpoint="/workspaces"} / node_filesystem_size_bytes < 0.2
        annotations:
          summary: "Workspace disk <20% free"

      - alert: RedisDown
        expr: redis_up == 0
        annotations:
          summary: "Redis unavailable - rate limiting disabled"

      - alert: HighAPILatency
        expr: histogram_quantile(0.95, rate(lobster_api_duration_seconds[5m])) > 5
        annotations:
          summary: "95th percentile API latency >5 seconds"
```

**Incident response procedures**:

**Incident 1: Suspected data corruption**

```markdown
1. **Immediate Actions** (within 1 hour):
   - Isolate affected workspace (chmod 000)
   - Notify QA team and data owner
   - Preserve logs (copy provenance.json, .session.json)

2. **Investigation** (within 4 hours):
   - Verify file hashes against manifest
   - Check provenance for unexpected operations
   - Review access logs (who accessed workspace?)
   - Identify root cause (corruption, tampering, bug)

3. **Recovery** (within 24 hours):
   - Restore from backup (if corruption)
   - Re-run analysis from validated data (if tampering)
   - Document incident in QA log

4. **Prevention** (within 1 week):
   - Fix root cause (bug fix, permission change)
   - Update SOP if procedural issue
   - Re-train users if human error
```

**Incident 2: API key compromise**

```markdown
1. **Immediate Actions** (within 30 minutes):
   - Revoke compromised key (Anthropic Console / AWS IAM)
   - Generate new key
   - Update .env files on all systems
   - Notify security team

2. **Investigation** (within 2 hours):
   - Review API usage logs (unusual activity?)
   - Check git history (was key committed?)
   - Identify exposure vector (how was key leaked?)

3. **Remediation** (within 4 hours):
   - Rotate all related keys (defense in depth)
   - Update .gitignore (prevent future commits)
   - Scan git history for other secrets

4. **Prevention** (within 1 week):
   - Implement pre-commit hooks (detect secrets)
   - User training (secure credential handling)
   - Consider secret management system (Vault, Secrets Manager)
```

**Monitoring dashboard** (example metrics):

```
Lobster AI - Production Dashboard

System Health:
├── CLI Status: ✅ Healthy
├── Workspace Disk: 65% used (warning at 80%)
├── Redis: ✅ Connected (15ms latency)
└── API Keys: ✅ Valid (expires: 2027-01-01)

Usage Metrics (last 24h):
├── Queries: 1,245
├── Downloads: 87
├── Errors: 12 (0.96% error rate)
└── Avg query time: 2.3 minutes

Rate Limiting:
├── NCBI: 8,432 requests (no throttling)
├── GEO: 1,234 requests (no throttling)
└── PRIDE: 45 requests (no throttling)

Security Events:
├── Failed license validations: 0
├── Workspace permission errors: 2 (investigate)
└── API key rotation: Due in 15 days
```

**Compliance benefits**:

- ✅ **Proactive monitoring** - Issues detected early
- ✅ **Incident response** - Clear procedures documented
- ✅ **Audit trail** - All incidents logged
- ✅ **Continuous improvement** - Metrics drive optimization
- ✅ **Regulatory readiness** - Demonstrates control

**For complete implementation details**, see:
- [Troubleshooting Guide](28-troubleshooting.md) (common issues, debugging)
- [Architecture - Monitoring](18-architecture-overview.md) (system observability)

---

## 12. Future Enhancements

### 12.1 Security Roadmap (Phases 2-4)

**What's next**: Lobster AI's security roadmap includes four phases. **Phase 1 (current)** provides production-ready security for local CLI deployments. Future phases target cloud SaaS, full GxP validation, and advanced compliance automation.

**Phase 2: Enhanced Sandboxing** (Target: Q2 2026)

| Enhancement | Technology | Benefit | Effort |
|-------------|-----------|---------|--------|
| **Docker sandboxing for custom code** | gVisor or Kata Containers | Full isolation (filesystem, network, process) | 4-6 weeks |
| **Network isolation** | Docker bridge mode + iptables | No outbound connections | 2 weeks |
| **Resource quotas** | cgroups (CPU, memory, disk) | DoS prevention, fair resource allocation | 2 weeks |
| **Read-only input mounts** | Docker volumes | Input data immutable | 1 week |
| **Runtime security scanning** | Falco or Sysdig | Detect anomalous behavior in real-time | 3 weeks |

**Phase 2 deliverables**:
- ✅ Cloud SaaS ready (multi-tenant isolation)
- ✅ Custom code sandboxing (untrusted users)
- ✅ Network egress firewall (prevent data exfiltration)
- ✅ Resource limits (prevent resource exhaustion)

---

**Phase 3: HIPAA & SOC 2 Certification** (Target: Q3 2026)

| Enhancement | Technology | Benefit | Effort |
|-------------|-----------|---------|--------|
| **Business Associate Agreement** | Legal + technical controls | HIPAA-compliant cloud | 6-8 weeks |
| **SOC 2 Type II audit** | Independent auditor | Third-party validation | 12-16 weeks |
| **PHI de-identification** | Automated PII scrubbing | Safe use of patient data | 4 weeks |
| **Breach notification** | Automated alerting | HIPAA § 164.404 compliance | 2 weeks |
| **Access logs (HIPAA)** | Enhanced logging + retention | HIPAA § 164.312(b) compliance | 3 weeks |

**Phase 3 deliverables**:
- ✅ HIPAA-compliant cloud service (with BAA)
- ✅ SOC 2 Type II certified
- ✅ PHI de-identification workflows
- ✅ HIPAA audit trail enhancements

---

**Phase 4: Full GxP Validation** (Target: Q4 2026)

| Enhancement | Technology | Benefit | Effort |
|-------------|-----------|---------|--------|
| **IQ/OQ/PQ automation** | Automated validation framework | Reduces validation time from 2-4 weeks to 2-3 days | 6 weeks |
| **Electronic signatures** | 21 CFR Part 11 § 11.50/11.70 | Secure, auditable approvals | 4 weeks |
| **Change control integration** | Git-based workflow + approvals | GAMP 5 change control | 4 weeks |
| **CAPA tracking** | Corrective/Preventive Action log | Quality management | 3 weeks |
| **Validation package generator** | Auto-generate IQ/OQ/PQ docs | 90% reduction in validation effort | 8 weeks |

**Phase 4 deliverables**:
- ✅ Full GxP validation support (GAMP 5 Cat 4)
- ✅ Electronic signatures (21 CFR Part 11 compliant)
- ✅ Automated validation package generation
- ✅ Change control + CAPA tracking

---

### 12.2 Feature Roadmap

**Data Integrity** (near-term):

| Feature | Description | Timeline | Compliance Impact |
|---------|-------------|----------|-------------------|
| **Runtime hash verification** | Auto-verify hashes when notebook re-runs | Q1 2026 | ALCOA+ "Accurate" |
| **Visual hash indicators** | Green ✅ / Red ❌ in notebook cells | Q1 2026 | User experience |
| **Hash history tracking** | Track hash changes for evolving datasets | Q2 2026 | Data lineage |
| **Batch verification CLI** | `lobster verify --workspace /path/` | Q1 2026 | QA automation |

**Access Control** (mid-term):

| Feature | Description | Timeline | Compliance Impact |
|---------|-------------|----------|-------------------|
| **LDAP/Active Directory** | Enterprise authentication integration | Q2 2026 | ISO 27001 A.9.2 |
| **Role-based permissions** | Fine-grained workspace access control | Q2 2026 | Principle of least privilege |
| **Audit user actions** | User-level attribution (not just agent) | Q2 2026 | ALCOA+ "Attributable" |
| **MFA enforcement** | Two-factor authentication | Q3 2026 | Enhanced security |

**Compliance Automation** (long-term):

| Feature | Description | Timeline | Compliance Impact |
|---------|-------------|----------|-------------------|
| **Auto-generate compliance reports** | 21 CFR Part 11 compliance report from provenance | Q3 2026 | Reduces audit burden |
| **ALCOA+ validator** | Automatic checks for ALCOA+ compliance | Q3 2026 | Quality assurance |
| **Regulatory submission package** | FDA/EMA submission-ready exports | Q4 2026 | Streamlines submissions |
| **GxP dashboard** | Real-time compliance metrics | Q4 2026 | Continuous monitoring |

---

### 12.3 Community Feedback

**What to expect**: Lobster AI's security roadmap is influenced by customer feedback, regulatory changes, and industry best practices. Contributions welcome!

**Request a feature**:
- GitHub Issues: [Report feature requests](https://github.com/the-omics-os/lobster/issues)
- Enterprise customers: Contact via customer success team
- Community discussion: GitHub Discussions

**Upcoming based on customer requests**:
1. **GDPR right-to-erasure** - Automated data deletion workflows (Q2 2026)
2. **Data residency controls** - Region-specific S3 backends (Q2 2026)
3. **Audit report templates** - Pre-built compliance reports (Q3 2026)
4. **Validation test library** - IQ/OQ/PQ test templates (Q3 2026)

---

## 13. Related Documentation

### 13.1 Security & Compliance Documentation

**This wiki page (42)**: Security architecture, compliance features, deployment guidance (executive summaries + deep links)

**Detailed technical documentation**:
- [Premium Licensing Technical Guide](../../docs/PREMIUM_LICENSING.md) - CTO/technical guide (4 phases, 1,500+ lines)
- [Premium Testing Checklist](../../docs/PREMIUM_TESTING_CHECKLIST.md) - QA verification procedures
- [Regulatory Compliance Roadmap](../../docs/REGULATORY_COMPLIANCE_ROADMAP.md) - GxP roadmap, IQ/OQ/PQ templates
- [Custom Packages Activation Flow](../../docs/CUSTOM_PACKAGES_ACTIVATION_FLOW.md) - Enterprise custom packages

**Customer-facing documentation**:
- [Commercial Licensing FAQ](../docs/commercial_licensing_faq.md) - AGPL-3.0 explanation, use cases

---

### 13.2 Architecture & Implementation Documentation

**Core architecture**:
- [18 Architecture Overview](18-architecture-overview.md) - Complete system architecture (1,900+ lines)
- [19 Agent System](19-agent-system.md) - Multi-agent coordination, LangGraph
- [20 Data Management](20-data-management.md) - DataManagerV2, provenance (W3C-PROV details)
- [21 Cloud-Local Architecture](21-cloud-local-architecture.md) - Deployment modes

**Key subsystems**:
- [35 Download Queue System](35-download-queue-system.md) - Queue-based downloads, concurrency
- [48 Redis Rate Limiter Architecture](48-redis-rate-limiter-architecture.md) - Token bucket algorithm, connection pools
- [38 Workspace Content Service](38-workspace-content-service.md) - Unified workspace tools

**Developer documentation**:
- [CLAUDE.md](../CLAUDE.md) - Developer guide (2,000+ lines)
- [08 Developer Overview](08-developer-overview.md) - Contributing guidelines
- [09 Creating Agents](09-creating-agents.md) - Agent development patterns
- [12 Testing Guide](12-testing-guide.md) - Testing framework, pytest patterns

---

### 13.3 Configuration & Deployment Documentation

**Configuration guides**:
- [03 Configuration](03-configuration.md) - LLM providers, API keys, workspace setup
- [43 Docker Deployment Guide](43-docker-deployment-guide.md) - Container deployments
- [43 S3 Backend Guide](43-s3-backend-guide.md) - Cloud storage configuration
- [44 Custom Providers Guide](44-custom-providers-guide.md) - Extend with new data sources

**User guides**:
- [01 Getting Started](01-getting-started.md) - Installation, first analysis
- [04 User Guide Overview](04-user-guide-overview.md) - Feature overview
- [05 CLI Commands](05-cli-commands.md) - Command reference
- [06 Data Analysis Workflows](06-data-analysis-workflows.md) - Analysis patterns

---

### 13.4 API & Developer Reference

**API documentation**:
- [13 API Overview](13-api-overview.md) - API structure, conventions
- [14 Core API](14-core-api.md) - Client, DataManagerV2, ProvenanceTracker
- [15 Agents API](15-agents-api.md) - Agent interfaces, tools
- [16 Services API](16-services-api.md) - Analysis services, patterns
- [17 Interfaces API](17-interfaces-api.md) - IDataBackend, IModalityAdapter

**Tutorials**:
- [23 Tutorial: Single-Cell](23-tutorial-single-cell.md) - scRNA-seq workflow
- [24 Tutorial: Bulk RNA-seq](24-tutorial-bulk-rnaseq.md) - Differential expression
- [25 Tutorial: Proteomics](25-tutorial-proteomics.md) - Mass spectrometry
- [26 Tutorial: Custom Agent](26-tutorial-custom-agent.md) - Extend Lobster

---

### 13.5 Troubleshooting & Support

**Troubleshooting**:
- [28 Troubleshooting](28-troubleshooting.md) - Common issues, debugging
- [29 FAQ](29-faq.md) - Frequently asked questions
- [30 Glossary](30-glossary.md) - Terminology reference

**Support channels**:
- **GitHub Issues**: [Report bugs or request features](https://github.com/the-omics-os/lobster/issues)
- **GitHub Discussions**: Community support and questions
- **Enterprise support**: Contact customer success team (PREMIUM/ENTERPRISE only)
- **Security issues**: Email security@omics-os.com (responsible disclosure)

---

## Technical Implementation (Advanced)

### Architecture

The manifest is generated by `NotebookExporter` class during the `export()` method:

1. **Hash Calculation** - SHA-256 of each input file (chunked for memory efficiency)
2. **Provenance Hash** - Fingerprint of the session's audit trail
3. **System Info** - Captures Lobster version, Git commit, Python version
4. **Manifest Cell** - Inserted as second cell in notebook (after header)

### Code Location

- **Implementation**: `lobster/core/notebook_exporter.py`
- **Methods**:
  - `_create_integrity_manifest_cell()` - Creates manifest
  - `_get_input_file_hashes()` - Hashes data files
  - `_get_provenance_hash()` - Hashes session
  - `_calculate_file_hash()` - SHA-256 computation

---

## Future Enhancements

Coming in future versions:

1. **Runtime Verification** - Auto-verify hashes when notebook is re-run
2. **Visual Indicators** - Green ✅ / Red ❌ status in notebook
3. **Hash History** - Track hash changes over time for evolving datasets
4. **Verification Tools** - Built-in CLI command for batch verification

---

## Support

For questions about data integrity features:
- GitHub Issues: [Report issues](https://github.com/the-omics-os/lobster/issues)
- Documentation: See this guide
- Compliance questions: Contact your organization's QA/compliance team

---

---

*Last Updated: 2026-01-01*
*Document Version: 2.0*
*Sections: 13 (Overview, Data Integrity, Audit Trail, Access Control, Secure Execution, Data Protection, Network Security, Validation, Deployment, Compliance, Best Practices, Future Enhancements, Related Documentation)*
*Compliance Coverage: 21 CFR Part 11, ALCOA+, GxP, HIPAA, GDPR, ISO/IEC 27001, SOC 2*
