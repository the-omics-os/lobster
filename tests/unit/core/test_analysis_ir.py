"""
Unit tests for AnalysisStep Intermediate Representation (IR).

Tests cover serialization, validation, template rendering,
and utility functions for the Service-Emitted IR architecture.
"""

import pytest
from lobster.core.analysis_ir import (
    AnalysisStep,
    ParameterSpec,
    create_minimal_ir,
    extract_unique_imports,
    validate_ir_list,
)


class TestParameterSpec:
    """Test ParameterSpec dataclass."""

    def test_parameter_spec_creation(self):
        """Test creating a ParameterSpec."""
        spec = ParameterSpec(
            param_type="int",
            papermill_injectable=True,
            default_value=200,
            required=False,
            validation_rule="min_genes > 0",
            description="Minimum genes per cell",
        )

        assert spec.param_type == "int"
        assert spec.papermill_injectable is True
        assert spec.default_value == 200
        assert spec.required is False
        assert spec.validation_rule == "min_genes > 0"
        assert spec.description == "Minimum genes per cell"

    def test_parameter_spec_serialization(self):
        """Test ParameterSpec to_dict/from_dict round-trip."""
        spec = ParameterSpec(
            param_type="float",
            papermill_injectable=False,
            default_value=0.05,
            required=True,
            description="P-value threshold",
        )

        # Serialize
        spec_dict = spec.to_dict()

        assert isinstance(spec_dict, dict)
        assert spec_dict["param_type"] == "float"
        assert spec_dict["papermill_injectable"] is False
        assert spec_dict["default_value"] == 0.05

        # Deserialize
        spec_restored = ParameterSpec.from_dict(spec_dict)

        assert spec_restored.param_type == spec.param_type
        assert spec_restored.papermill_injectable == spec.papermill_injectable
        assert spec_restored.default_value == spec.default_value
        assert spec_restored.required == spec.required

    def test_parameter_spec_minimal(self):
        """Test ParameterSpec with minimal fields."""
        spec = ParameterSpec(
            param_type="str",
            papermill_injectable=True,
            default_value="test",
            required=True,
        )

        assert spec.validation_rule is None
        assert spec.description == ""

    def test_parameter_spec_non_json_serializable_default(self):
        """Test that non-JSON-serializable default values raise TypeError."""
        from pathlib import Path

        # Create spec with non-serializable default (Path object)
        spec = ParameterSpec(
            param_type="Path",
            papermill_injectable=True,
            default_value=Path("/tmp/test.h5ad"),  # Path objects are not JSON-serializable
            required=True,
            description="Test path parameter",
        )

        # Should raise TypeError when attempting to serialize
        with pytest.raises(TypeError, match="not JSON-serializable"):
            spec.to_dict()

    def test_parameter_spec_json_serializable_default(self):
        """Test that JSON-serializable defaults pass validation."""
        # Test various JSON-serializable types
        test_cases = [
            ("str", "test_string"),
            ("int", 42),
            ("float", 3.14),
            ("bool", True),
            ("list", [1, 2, 3]),
            ("dict", {"key": "value"}),
            ("None", None),
        ]

        for param_type, default_value in test_cases:
            spec = ParameterSpec(
                param_type=param_type,
                papermill_injectable=True,
                default_value=default_value,
                required=False,
            )

            # Should not raise error
            spec_dict = spec.to_dict()
            assert spec_dict["default_value"] == default_value


class TestAnalysisStep:
    """Test AnalysisStep dataclass."""

    @pytest.fixture
    def sample_parameter_schema(self):
        """Sample parameter schema for testing."""
        return {
            "target_sum": ParameterSpec(
                param_type="float",
                papermill_injectable=True,
                default_value=1e4,
                required=False,
                description="Target sum for normalization",
            ),
            "qc_vars": ParameterSpec(
                param_type="List[str]",
                papermill_injectable=False,
                default_value=["mt", "ribo"],
                required=True,
                description="QC variables to calculate",
            ),
        }

    @pytest.fixture
    def sample_ir(self, sample_parameter_schema):
        """Sample AnalysisStep for testing."""
        return AnalysisStep(
            operation="scanpy.pp.normalize_total",
            tool_name="normalize",
            description="Total-count normalize expression data",
            library="scanpy",
            code_template="sc.pp.normalize_total(adata, target_sum={{ target_sum }})",
            imports=["import scanpy as sc", "import numpy as np"],
            parameters={"target_sum": 1e4},
            parameter_schema=sample_parameter_schema,
            input_entities=["adata"],
            output_entities=["adata"],
            execution_context={"scanpy_version": "1.9.3"},
        )

    def test_analysis_step_creation(self, sample_ir):
        """Test creating an AnalysisStep."""
        assert sample_ir.operation == "scanpy.pp.normalize_total"
        assert sample_ir.tool_name == "normalize"
        assert sample_ir.library == "scanpy"
        assert len(sample_ir.imports) == 2
        assert "target_sum" in sample_ir.parameters
        assert "target_sum" in sample_ir.parameter_schema
        assert sample_ir.validates_on_export is True
        assert sample_ir.requires_validation is False

    def test_analysis_step_serialization(self, sample_ir):
        """Test AnalysisStep to_dict/from_dict round-trip."""
        # Serialize
        ir_dict = sample_ir.to_dict()

        assert isinstance(ir_dict, dict)
        assert ir_dict["operation"] == "scanpy.pp.normalize_total"
        assert ir_dict["tool_name"] == "normalize"

        # ParameterSpec should be converted to dict
        assert isinstance(ir_dict["parameter_schema"]["target_sum"], dict)
        assert ir_dict["parameter_schema"]["target_sum"]["param_type"] == "float"

        # Deserialize
        ir_restored = AnalysisStep.from_dict(ir_dict)

        assert ir_restored.operation == sample_ir.operation
        assert ir_restored.tool_name == sample_ir.tool_name
        assert ir_restored.library == sample_ir.library
        assert ir_restored.code_template == sample_ir.code_template
        assert len(ir_restored.imports) == len(sample_ir.imports)
        assert ir_restored.parameters == sample_ir.parameters

        # ParameterSpec should be reconstructed
        assert isinstance(
            ir_restored.parameter_schema["target_sum"], ParameterSpec
        )
        assert (
            ir_restored.parameter_schema["target_sum"].param_type
            == sample_ir.parameter_schema["target_sum"].param_type
        )

    def test_analysis_step_json_compatibility(self, sample_ir):
        """Test that serialized IR is JSON-compatible."""
        import json

        ir_dict = sample_ir.to_dict()

        # Should be able to serialize to JSON
        json_str = json.dumps(ir_dict)
        assert isinstance(json_str, str)

        # Should be able to deserialize from JSON
        ir_dict_restored = json.loads(json_str)
        ir_restored = AnalysisStep.from_dict(ir_dict_restored)

        assert ir_restored.operation == sample_ir.operation
        assert ir_restored.tool_name == sample_ir.tool_name

    def test_analysis_step_minimal_fields(self):
        """Test AnalysisStep with minimal required fields."""
        ir = AnalysisStep(
            operation="test_operation",
            tool_name="test_tool",
            description="Test description",
            library="test_library",
            code_template="# Test code",
            imports=["import test"],
            parameters={},
            parameter_schema={},
        )

        assert ir.input_entities == []
        assert ir.output_entities == []
        assert ir.execution_context == {}
        assert ir.validates_on_export is True
        assert ir.requires_validation is False

    def test_analysis_step_missing_required_fields(self):
        """Test that missing required fields raise ValueError."""
        incomplete_dict = {
            "operation": "test",
            "tool_name": "test",
            # Missing other required fields
        }

        with pytest.raises(ValueError, match="Missing required fields"):
            AnalysisStep.from_dict(incomplete_dict)

    def test_template_validation_valid(self, sample_ir):
        """Test template validation with valid Jinja2 template."""
        assert sample_ir.validate_template() is True

    def test_template_validation_invalid(self):
        """Test template validation with invalid Jinja2 template."""
        ir = AnalysisStep(
            operation="test",
            tool_name="test",
            description="test",
            library="test",
            code_template="{{ unclosed",  # Invalid Jinja2
            imports=[],
            parameters={},
            parameter_schema={},
        )

        with pytest.raises(ValueError, match="Invalid Jinja2 template"):
            ir.validate_template()

    def test_render_with_parameters(self, sample_ir):
        """Test rendering code template with parameters."""
        code = sample_ir.render()

        assert "sc.pp.normalize_total" in code
        assert "10000.0" in code  # target_sum=1e4
        assert "{{" not in code  # No unrendered placeholders

    def test_render_with_override_parameters(self, sample_ir):
        """Test rendering with parameter overrides."""
        code = sample_ir.render(target_sum=5000)

        assert "5000" in code
        assert "10000" not in code

    def test_render_invalid_template(self):
        """Test that invalid template raises error during render."""
        ir = AnalysisStep(
            operation="test",
            tool_name="test",
            description="test",
            library="test",
            code_template="{{ undefined_var | unknown_filter }}",
            imports=[],
            parameters={},
            parameter_schema={},
        )

        with pytest.raises(ValueError, match="Template rendering failed"):
            ir.render()

    def test_validate_rendered_code_valid(self, sample_ir):
        """Test validation of rendered code with valid Python syntax."""
        assert sample_ir.validate_rendered_code() is True

    def test_validate_rendered_code_invalid(self):
        """Test validation catches syntax errors in rendered code."""
        ir = AnalysisStep(
            operation="test",
            tool_name="test",
            description="test",
            library="test",
            code_template="if True\n    pass",  # Missing colon
            imports=[],
            parameters={},
            parameter_schema={},
        )

        with pytest.raises(SyntaxError):
            ir.validate_rendered_code()

    def test_get_papermill_parameters(self, sample_ir):
        """Test extracting Papermill-injectable parameters."""
        papermill_params = sample_ir.get_papermill_parameters()

        # Should include target_sum (injectable=True)
        assert "target_sum" in papermill_params
        assert papermill_params["target_sum"] == 1e4

        # Should NOT include qc_vars (injectable=False)
        assert "qc_vars" not in papermill_params

    def test_get_papermill_parameters_empty(self):
        """Test getting Papermill parameters when none are injectable."""
        ir = AnalysisStep(
            operation="test",
            tool_name="test",
            description="test",
            library="test",
            code_template="pass",
            imports=[],
            parameters={"param1": "value1"},
            parameter_schema={
                "param1": ParameterSpec(
                    param_type="str",
                    papermill_injectable=False,
                    default_value="value1",
                    required=True,
                )
            },
        )

        papermill_params = ir.get_papermill_parameters()
        assert papermill_params == {}

    def test_repr(self, sample_ir):
        """Test string representation."""
        repr_str = repr(sample_ir)

        assert "AnalysisStep" in repr_str
        assert "scanpy.pp.normalize_total" in repr_str
        assert "normalize" in repr_str


class TestUtilityFunctions:
    """Test utility functions for IR manipulation."""

    def test_validate_ir_list_valid(self):
        """Test validation of valid IR list."""
        irs = [
            {
                "operation": "op1",
                "tool_name": "tool1",
                "description": "desc1",
                "library": "lib1",
                "code_template": "pass",
                "imports": ["import lib1"],
                "parameters": {},
                "parameter_schema": {},
            },
            {
                "operation": "op2",
                "tool_name": "tool2",
                "description": "desc2",
                "library": "lib2",
                "code_template": "x = 1",
                "imports": ["import lib2"],
                "parameters": {},
                "parameter_schema": {},
            },
        ]

        errors = validate_ir_list(irs)
        assert errors == []

    def test_validate_ir_list_invalid_template(self):
        """Test validation catches invalid templates."""
        irs = [
            {
                "operation": "op1",
                "tool_name": "tool1",
                "description": "desc1",
                "library": "lib1",
                "code_template": "{{ unclosed",  # Invalid Jinja2
                "imports": ["import lib1"],
                "parameters": {},
                "parameter_schema": {},
            }
        ]

        errors = validate_ir_list(irs)
        assert len(errors) > 0
        assert "Invalid template" in errors[0]

    def test_validate_ir_list_missing_fields(self):
        """Test validation catches missing required fields."""
        irs = [
            {
                "operation": "op1",
                # Missing required fields
            }
        ]

        errors = validate_ir_list(irs)
        assert len(errors) > 0
        assert "Deserialization failed" in errors[0]

    def test_validate_ir_list_empty_required_fields(self):
        """Test validation catches empty required fields."""
        irs = [
            {
                "operation": "",  # Empty operation
                "tool_name": "tool1",
                "description": "desc1",
                "library": "lib1",
                "code_template": "",  # Empty template
                "imports": [],  # No imports
                "parameters": {},
                "parameter_schema": {},
            }
        ]

        errors = validate_ir_list(irs)
        assert len(errors) >= 3  # Three empty field errors

    def test_extract_unique_imports_deduplication(self):
        """Test import deduplication."""
        ir1 = AnalysisStep(
            operation="op1",
            tool_name="tool1",
            description="desc1",
            library="lib1",
            code_template="pass",
            imports=["import scanpy as sc", "import numpy as np"],
            parameters={},
            parameter_schema={},
        )

        ir2 = AnalysisStep(
            operation="op2",
            tool_name="tool2",
            description="desc2",
            library="lib2",
            code_template="pass",
            imports=["import scanpy as sc", "import pandas as pd"],  # Duplicate scanpy
            parameters={},
            parameter_schema={},
        )

        unique_imports = extract_unique_imports([ir1, ir2])

        # Should have 3 unique imports (scanpy appears only once)
        assert len(unique_imports) == 3
        assert "import scanpy as sc" in unique_imports
        assert "import numpy as np" in unique_imports
        assert "import pandas as pd" in unique_imports

    def test_extract_unique_imports_ordering(self):
        """Test import ordering: stdlib → third-party → local."""
        ir = AnalysisStep(
            operation="test",
            tool_name="test",
            description="test",
            library="test",
            code_template="pass",
            imports=[
                "from lobster.utils import helper",  # Local (should be last)
                "import pandas as pd",  # Third-party (should be middle)
                "import os",  # Stdlib (should be first)
                "import scanpy as sc",  # Third-party
                "from pathlib import Path",  # Stdlib
            ],
            parameters={},
            parameter_schema={},
        )

        ordered_imports = extract_unique_imports([ir])

        # Find indices
        os_idx = ordered_imports.index("import os")
        pathlib_idx = ordered_imports.index("from pathlib import Path")
        pandas_idx = ordered_imports.index("import pandas as pd")
        scanpy_idx = ordered_imports.index("import scanpy as sc")
        lobster_idx = ordered_imports.index("from lobster.utils import helper")

        # Stdlib should come first
        assert os_idx < pandas_idx
        assert pathlib_idx < pandas_idx

        # Third-party should come before local
        assert pandas_idx < lobster_idx
        assert scanpy_idx < lobster_idx

    def test_extract_unique_imports_empty(self):
        """Test with no IRs."""
        unique_imports = extract_unique_imports([])
        assert unique_imports == []

    def test_create_minimal_ir(self):
        """Test creating minimal IR fallback."""
        ir = create_minimal_ir(
            operation="unmapped_operation",
            tool_name="unmapped_tool",
            code="# Placeholder code",
            library="unknown",
        )

        assert ir.operation == "unmapped_operation"
        assert ir.tool_name == "unmapped_tool"
        assert "TODO" in ir.description
        assert "TODO" in ir.code_template
        assert "# Placeholder code" in ir.code_template
        assert ir.library == "unknown"
        assert ir.imports == []
        assert ir.parameters == {}
        assert ir.parameter_schema == {}
        assert ir.validates_on_export is False
        assert ir.requires_validation is True

    def test_create_minimal_ir_defaults(self):
        """Test minimal IR with default library."""
        ir = create_minimal_ir(
            operation="test_op", tool_name="test_tool", code="pass"
        )

        assert ir.library == "unknown"
        assert ir.validates_on_export is False
        assert ir.requires_validation is True


class TestComplexScenarios:
    """Test complex real-world scenarios."""

    def test_complete_workflow_serialization(self):
        """Test serialization of multiple IRs simulating a workflow."""
        # Create workflow steps
        qc_ir = AnalysisStep(
            operation="scanpy.pp.calculate_qc_metrics",
            tool_name="assess_quality",
            description="Calculate QC metrics",
            library="scanpy",
            code_template="sc.pp.calculate_qc_metrics(adata, qc_vars={{ qc_vars }})",
            imports=["import scanpy as sc"],
            parameters={"qc_vars": ["mt", "ribo"]},
            parameter_schema={
                "qc_vars": ParameterSpec(
                    param_type="List[str]",
                    papermill_injectable=False,
                    default_value=["mt", "ribo"],
                    required=True,
                )
            },
        )

        filter_ir = AnalysisStep(
            operation="scanpy.pp.filter_cells",
            tool_name="filter_cells",
            description="Filter cells by quality",
            library="scanpy",
            code_template="sc.pp.filter_cells(adata, min_genes={{ min_genes }})",
            imports=["import scanpy as sc"],
            parameters={"min_genes": 200},
            parameter_schema={
                "min_genes": ParameterSpec(
                    param_type="int",
                    papermill_injectable=True,
                    default_value=200,
                    required=False,
                )
            },
        )

        # Serialize workflow
        workflow = [qc_ir.to_dict(), filter_ir.to_dict()]

        # Should be JSON-compatible
        import json

        json_str = json.dumps(workflow)
        assert isinstance(json_str, str)

        # Deserialize and validate
        workflow_restored = json.loads(json_str)
        qc_ir_restored = AnalysisStep.from_dict(workflow_restored[0])
        filter_ir_restored = AnalysisStep.from_dict(workflow_restored[1])

        assert qc_ir_restored.operation == qc_ir.operation
        assert filter_ir_restored.operation == filter_ir.operation

    def test_parameter_override_workflow(self):
        """Test parameter override in workflow execution."""
        ir = AnalysisStep(
            operation="test.operation",
            tool_name="test",
            description="test",
            library="test",
            code_template="""
min_genes = {{ min_genes }}
max_pct = {{ max_pct }}
result = process(min_genes, max_pct)
""",
            imports=["import test"],
            parameters={"min_genes": 200, "max_pct": 20.0},
            parameter_schema={
                "min_genes": ParameterSpec(
                    param_type="int",
                    papermill_injectable=True,
                    default_value=200,
                    required=False,
                ),
                "max_pct": ParameterSpec(
                    param_type="float",
                    papermill_injectable=True,
                    default_value=20.0,
                    required=False,
                ),
            },
        )

        # Original render
        code1 = ir.render()
        assert "200" in code1
        assert "20.0" in code1

        # Override parameters
        code2 = ir.render(min_genes=500, max_pct=15.0)
        assert "500" in code2
        assert "15.0" in code2
        assert "200" not in code2
        assert "20.0" not in code2
