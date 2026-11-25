"""
Intermediate Representation (IR) for reproducible analysis operations.

This module provides the Service-Emitted IR architecture that enables
automatic, deterministic Jupyter notebook generation. Services emit
AnalysisStep objects that contain everything needed to generate
executable code without manual mapping registries.
"""

import logging
import math
import operator
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from jinja2.sandbox import SandboxedEnvironment, SecurityError

logger = logging.getLogger(__name__)

# Shared Jinja2 environment for rendering AnalysisStep code templates.
#
# ``autoescape=False`` because templates produce Python source, not HTML
# (matches the PromptComposer environment in lobster/prompts/composer.py).
#
# The ``repr`` filter is required: analysis templates across the codebase use
# ``{{ value | repr }}`` to emit Python literals safely (e.g. quoted strings,
# lists). jinja2 has no built-in ``repr`` filter, so a bare ``Template(...)``
# raises "No filter named 'repr'" at parse time (jinja >= 3.1) for every such
# template — which is every genomics/GWAS export and the scRNA sub-clustering
# export. Registering it here fixes all call sites at the AnalysisStep level.
#
# ``| repr`` param contract: values passed to templates are expected to be
# JSON/literal-shaped (str, int, float, bool, None, list/dict of those). A
# custom object with a hostile ``__repr__`` could inject into a generated
# notebook cell; current callers only pass literal-shaped params. A
# python-literal-hardening filter is tracked as separate future work.
# Generates Python source, not HTML; HTML escaping would corrupt Python literals.
# Sandboxed: text that reaches a template through an f-string must not be able to
# reach this process's objects (its environment holds the provider keys).
_MAX_TEMPLATE_INT_BITS = 333  # about 10**100
_MAX_TEMPLATE_SEQUENCE = 100_000


class _BoundedTemplateEnvironment(SandboxedEnvironment):
    """Template environment that refuses ``**`` and ``*`` results too large to build.

    Python computes ``10**10**10`` or ``'a' * 10**12`` without a limit, so a
    template holding one hangs or exhausts the memory of the exporting process.
    Intercepted operators are also never folded at compile time.
    """

    intercepted_binops = frozenset({"*", "**"})

    def call_binop(self, context, operator, left, right):
        lint, rint = _as_int(left), _as_int(right)
        if operator == "**":
            if (
                lint is not None
                and rint is not None
                and rint > 0
                and abs(lint) > 1
                # rint is checked alone first: a huge int cannot become a float
                and (
                    rint > _MAX_TEMPLATE_INT_BITS
                    or rint * math.log2(abs(lint)) > _MAX_TEMPLATE_INT_BITS
                )
            ):
                raise SecurityError("the power in this template is too large")
        elif lint is not None and rint is not None:
            if (
                abs(lint).bit_length() + abs(rint).bit_length() - 1
                > _MAX_TEMPLATE_INT_BITS
            ):
                raise SecurityError("the product in this template is too large")
        else:
            sequence, count = (left, rint) if rint is not None else (right, lint)
            if (
                count is not None
                and isinstance(sequence, (str, bytes, list, tuple))
                and len(sequence) * count > _MAX_TEMPLATE_SEQUENCE
            ):
                raise SecurityError("the repeated value in this template is too long")
        return super().call_binop(context, operator, left, right)


def _as_int(value: Any) -> Optional[int]:
    """The int an integer operand stands for (NumPy integers too, via ``__index__``)."""
    try:
        return operator.index(value)
    except TypeError:
        return None


_JINJA_ENV = _BoundedTemplateEnvironment(autoescape=False)  # nosec B701


def _python_literal(value: Any) -> str:
    """Render ``value`` as Python source that evaluates back to it.

    Identical to ``repr`` for plain str/int/float/bool/None and containers of
    them. It differs only where ``repr`` produces source that does not run in a
    notebook: non-finite floats (``inf``/``nan`` are undefined names), NumPy
    scalars (``np.float64(1.0)`` needs numpy imported under that alias) and
    NumPy arrays (``array([...])``), which become plain Python values/lists.
    """
    if type(value).__module__ == "numpy" and hasattr(value, "tolist"):
        # ndarray or numpy scalar (np.float64 subclasses float, so check first).
        # Only bool/int/float/complex/str convert to plain Python; datetimes,
        # objects and extended precision (whose tolist() stays numpy) keep repr.
        if getattr(getattr(value, "dtype", None), "kind", "") not in "biufcUS":
            return repr(value)
        converted = value.tolist()
        if type(converted).__module__ == "numpy":
            return repr(value)
        return _python_literal(converted)
    if isinstance(value, float):
        if math.isnan(value):
            return "float('nan')"
        if math.isinf(value):
            return "float('inf')" if value > 0 else "float('-inf')"
        return repr(value)
    if isinstance(value, (str, bytes, bool, int)) or value is None:
        return repr(value)
    if isinstance(value, dict):
        items = ", ".join(
            f"{_python_literal(k)}: {_python_literal(v)}" for k, v in value.items()
        )
        return "{" + items + "}"
    if isinstance(value, list):
        return "[" + ", ".join(_python_literal(v) for v in value) + "]"
    if isinstance(value, tuple):
        if len(value) == 1:
            return "(" + _python_literal(value[0]) + ",)"
        return "(" + ", ".join(_python_literal(v) for v in value) + ")"
    return repr(value)


_JINJA_ENV.filters["repr"] = _python_literal
# Templates also use ``| tojson``, whose built-in JSON is not Python: True, None
# and NaN would render as the undefined names true, null and NaN.
_JINJA_ENV.filters["tojson"] = _python_literal


def _comment(value: Any) -> str:
    """Render text as Python comment lines: every line, split at each break
    ``str.splitlines`` knows (\\r, \\x0b, \\x85, \\u2028 among them), starts
    with ``# `` so no part of it runs."""
    return "\n".join("# " + line for line in str(value).splitlines()) or "#"


_JINJA_ENV.filters["comment"] = _comment


@dataclass
class ParameterSpec:
    """
    Type and behavior specification for analysis parameters.

    This class defines how a parameter should be handled during
    notebook generation, including whether it can be overridden
    via Papermill parameter injection.

    Attributes:
        param_type: Python type as string (e.g., "int", "float", "List[str]", "Path")
        papermill_injectable: Whether this parameter can be overridden by Papermill
        default_value: Default value if not specified
        required: Whether this parameter must be provided
        validation_rule: Optional validation expression (e.g., "min_genes > 0")
        description: Human-readable parameter description

    Example:
        >>> spec = ParameterSpec(
        ...     param_type="int",
        ...     papermill_injectable=True,
        ...     default_value=200,
        ...     required=False,
        ...     validation_rule="min_genes > 0",
        ...     description="Minimum genes per cell threshold"
        ... )
    """

    param_type: str
    papermill_injectable: bool
    default_value: Any
    required: bool
    validation_rule: Optional[str] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to JSON-compatible dictionary.

        Returns:
            Dictionary with all parameter spec fields

        Raises:
            TypeError: If default_value is not JSON-serializable
        """
        import json

        # Validate JSON serializability of default_value
        try:
            json.dumps({"value": self.default_value})
        except TypeError as e:
            raise TypeError(
                f"ParameterSpec default_value is not JSON-serializable: "
                f"{type(self.default_value).__name__} = {self.default_value!r}. "
                f"Parameter: '{self.description or self.param_type}'. "
                f"Error: {e}"
            ) from e

        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParameterSpec":
        """
        Deserialize from dictionary.

        Args:
            data: Dictionary with parameter spec fields

        Returns:
            ParameterSpec instance
        """
        import dataclasses

        valid_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"ParameterSpec(type={self.param_type}, "
            f"injectable={self.papermill_injectable}, "
            f"default={self.default_value})"
        )


@dataclass
class AnalysisStep:
    """
    Intermediate Representation for reproducible analysis operations.

    This is the canonical, single source of truth for how a service
    operation should be represented in an exported Jupyter notebook.
    Services emit AnalysisStep objects alongside their results to
    enable automatic notebook generation without manual code mapping.

    Attributes:
        operation: Fully-qualified operation name (e.g., "scanpy.pp.normalize_total")
        tool_name: Original tool name for provenance linking
        description: Human-readable description (appears in notebook markdown)
        library: Required library name (e.g., "scanpy", "pandas")
        code_template: Jinja2 template with {{ variable }} placeholders
        imports: List of import statements (e.g., ["import scanpy as sc"])
        parameters: Actual parameter values used in this execution
        parameter_schema: Type info and Papermill flags for each parameter
        input_entities: List of input data references
        output_entities: List of output data references
        execution_context: Random seeds, versions, timestamps, etc.
        validates_on_export: Whether to run AST syntax validation on export
        requires_validation: Whether full execution validation is recommended
        exportable: Whether to include in a curated notebook export (default True)
        replayable: Whether this step is required to reproduce the result
            (default True)

    Example:
        >>> ir = AnalysisStep(
        ...     operation="scanpy.pp.normalize_total",
        ...     tool_name="normalize",
        ...     description="Total-count normalize expression data",
        ...     library="scanpy",
        ...     code_template="sc.pp.normalize_total(adata, target_sum={{ target }})",
        ...     imports=["import scanpy as sc"],
        ...     parameters={"target": 1e4},
        ...     parameter_schema={
        ...         "target": ParameterSpec(
        ...             param_type="float",
        ...             papermill_injectable=True,
        ...             default_value=1e4,
        ...             required=False,
        ...             description="Target sum for normalization"
        ...         )
        ...     },
        ...     input_entities=["adata"],
        ...     output_entities=["adata"],
        ...     execution_context={"scanpy_version": "1.9.3"}
        ... )
    """

    # Identity
    operation: str
    tool_name: str
    description: str

    # Code generation
    library: str
    code_template: str
    imports: List[str]

    # Parameters
    parameters: Dict[str, Any]
    parameter_schema: Dict[str, ParameterSpec]

    # Data flow
    input_entities: List[str] = field(default_factory=list)
    output_entities: List[str] = field(default_factory=list)

    # Metadata
    execution_context: Dict[str, Any] = field(default_factory=dict)

    # Validation flags
    validates_on_export: bool = True
    requires_validation: bool = False

    # Export control — curation. Answers "should this appear in a notebook a
    # user might publish?" Services commonly derive it from a `persist` flag.
    exportable: bool = True  # Whether to include in notebook export

    # Replay control — reproducibility. Answers the unrelated question "is this
    # step required to reproduce the result?" Deliberately independent of
    # `exportable`/`persist`: a step can be too ad-hoc to publish and still be
    # indispensable to a faithful replay. Default True, so anything that
    # computes or mutates state is replayable unless a service opts out, and IR
    # rehydrated from provenance written before this field existed replays.
    replayable: bool = True

    # Helper code (shared functions injected into notebook helper cell)
    helper_code: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to JSON-compatible dictionary.

        This method handles the conversion of ParameterSpec objects
        within parameter_schema to dictionaries for JSON storage.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        data = asdict(self)

        # Convert ParameterSpec objects in parameter_schema
        data["parameter_schema"] = {
            k: v.to_dict() if isinstance(v, ParameterSpec) else v
            for k, v in self.parameter_schema.items()
        }

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisStep":
        """
        Deserialize from dictionary.

        This method reconstructs ParameterSpec objects from
        dictionaries in parameter_schema.

        Args:
            data: Dictionary representation from to_dict()

        Returns:
            AnalysisStep instance

        Raises:
            ValueError: If required fields are missing or invalid
        """
        # Validate required fields
        required_fields = [
            "operation",
            "tool_name",
            "description",
            "library",
            "code_template",
            "imports",
            "parameters",
            "parameter_schema",
        ]

        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")

        # Convert parameter_schema dicts back to ParameterSpec objects
        # Only convert dicts that match ParameterSpec format (have 'param_type' field)
        # Some IRs use JSON Schema-style dicts with 'type' instead — leave those as-is
        if "parameter_schema" in data:
            converted = {}
            for k, v in data["parameter_schema"].items():
                if isinstance(v, dict) and "param_type" in v:
                    converted[k] = ParameterSpec.from_dict(v)
                else:
                    converted[k] = v
            data["parameter_schema"] = converted

        return cls(**data)

    def validate_template(self) -> bool:
        """
        Validate that the code template uses correct Jinja2 syntax.

        Returns:
            True if template is valid

        Raises:
            ValueError: If template has invalid syntax
        """
        try:
            _JINJA_ENV.from_string(self.code_template)
            return True
        except Exception as e:
            raise ValueError(f"Invalid Jinja2 template: {e}") from e

    def render(self, **override_params) -> str:
        """
        Render code template with parameters.

        Args:
            **override_params: Optional parameter overrides

        Returns:
            Rendered Python code string

        Raises:
            ValueError: If template rendering fails
        """
        try:
            # Merge parameters with overrides
            params = {**self.parameters, **override_params}

            template = _JINJA_ENV.from_string(self.code_template)
            code = template.render(**params)

            return code
        except Exception as e:
            logger.error(f"Failed to render template for {self.operation}: {e}")
            raise ValueError(f"Template rendering failed: {e}") from e

    def validate_rendered_code(self, **override_params) -> bool:
        """
        Render and validate generated code syntax.

        Args:
            **override_params: Optional parameter overrides

        Returns:
            True if rendered code has valid Python syntax

        Raises:
            SyntaxError: If generated code is invalid
        """
        import ast

        code = self.render(**override_params)

        try:
            ast.parse(code)
            return True
        except SyntaxError as e:
            logger.error(f"Invalid generated code for {self.operation}: {e}")
            raise

    def get_papermill_parameters(self) -> Dict[str, Any]:
        """
        Extract parameters that can be overridden via Papermill.

        Returns:
            Dictionary of injectable parameter names and current values
        """
        result = {}
        for param_name, spec in self.parameter_schema.items():
            if isinstance(spec, ParameterSpec):
                if spec.papermill_injectable:
                    result[param_name] = self.parameters.get(
                        param_name, spec.default_value
                    )
            # Skip non-ParameterSpec entries (e.g., JSON Schema dicts)
        return result

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"AnalysisStep(operation={self.operation}, "
            f"tool={self.tool_name}, "
            f"params={len(self.parameters)})"
        )


# Utility functions for IR validation


def validate_ir_list(irs: List[Dict[str, Any]]) -> List[str]:
    """
    Validate a list of IR dictionaries.

    Args:
        irs: List of IR dictionaries from provenance

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    for idx, ir_dict in enumerate(irs):
        try:
            ir = AnalysisStep.from_dict(ir_dict)

            # Validate template syntax
            try:
                ir.validate_template()
            except ValueError as e:
                errors.append(f"IR {idx} ({ir.operation}): Invalid template - {e}")

            # Check for empty required fields
            if not ir.operation:
                errors.append(f"IR {idx}: Empty operation field")
            if not ir.code_template:
                errors.append(f"IR {idx}: Empty code_template field")
            if not ir.imports:
                errors.append(f"IR {idx}: No imports specified")

        except Exception as e:
            errors.append(f"IR {idx}: Deserialization failed - {e}")

    return errors


def extract_unique_imports(irs: List[AnalysisStep]) -> List[str]:
    """
    Extract and deduplicate imports from multiple IR objects.

    Imports are sorted in standard order: stdlib → third-party → local

    Args:
        irs: List of AnalysisStep objects

    Returns:
        Sorted list of unique import statements
    """
    imports = set()

    for ir in irs:
        imports.update(ir.imports)

    # Standard library modules for sorting
    stdlib_modules = {
        "os",
        "sys",
        "pathlib",
        "datetime",
        "json",
        "re",
        "math",
        "time",
        "typing",
    }

    def import_sort_key(import_str: str) -> tuple:
        """Sort key: stdlib (0) → third-party (1) → local (2)."""
        # Extract module name
        if import_str.startswith("from"):
            parts = import_str.split()
            module = parts[1].split(".")[0] if len(parts) > 1 else ""
        elif import_str.startswith("import"):
            parts = import_str.split()
            module = parts[1].split(".")[0] if len(parts) > 1 else ""
        else:
            module = ""

        # Categorize
        if module in stdlib_modules:
            return (0, import_str)  # stdlib first
        elif module.startswith("lobster"):
            return (2, import_str)  # local last
        else:
            return (1, import_str)  # third-party middle

    return sorted(imports, key=import_sort_key)


def extract_unique_helper_code(irs: List[AnalysisStep]) -> List[str]:
    """
    Extract and deduplicate helper code from multiple IR objects.

    Preserves insertion order (unlike set-based dedup) since helpers
    may depend on each other. Deduplication uses exact string matching.

    Args:
        irs: List of AnalysisStep objects

    Returns:
        Ordered list of unique helper code strings
    """
    seen: set = set()
    helpers: List[str] = []

    for ir in irs:
        for code in ir.helper_code:
            if code not in seen:
                seen.add(code)
                helpers.append(code)

    return helpers


def create_minimal_ir(
    operation: str, tool_name: str, code: str, library: str = "unknown"
) -> AnalysisStep:
    """
    Create a minimal AnalysisStep for operations without full IR support.

    This is a fallback for tools that haven't been updated yet.
    The generated notebook will have a TODO comment.

    Args:
        operation: Operation name
        tool_name: Tool name for provenance
        code: Basic code to include
        library: Library name

    Returns:
        Minimal AnalysisStep with TODO markers
    """
    return AnalysisStep(
        operation=operation,
        tool_name=tool_name,
        description=f"TODO: Add proper IR support for {operation}",
        library=library,
        code_template=f"# TODO: Manual review needed for {operation}\n{code}",
        imports=[],
        parameters={},
        parameter_schema={},
        input_entities=[],
        output_entities=[],
        execution_context={"created": datetime.now().isoformat(), "ir_type": "minimal"},
        validates_on_export=False,
        requires_validation=True,
    )


def create_data_loading_ir(
    input_param_name: str = "input_data",
    description: str = "Load single-cell RNA-seq data from H5AD file",
) -> AnalysisStep:
    """
    Create IR for loading AnnData from H5AD file.

    Args:
        input_param_name: Name of parameter containing input file path
        description: Human-readable description of the loading operation

    Returns:
        AnalysisStep with data loading code template
    """
    # Parameter schema with Papermill flag
    parameter_schema = {
        input_param_name: ParameterSpec(
            param_type="str",
            papermill_injectable=True,
            default_value="dataset.h5ad",
            required=True,
            validation_rule=f"{input_param_name}.endswith('.h5ad')",
            description="Path to input H5AD file",
        ),
    }

    # Use variable reference (not Jinja2 substitution) since input_data
    # is a Papermill-injected parameter defined in the parameters cell
    code_template = f"""# Load input data
import anndata as ad

adata = ad.read_h5ad({input_param_name})
print(f"Loaded data: {{adata.n_obs}} cells × {{adata.n_vars}} genes")
"""

    return AnalysisStep(
        operation="anndata.read_h5ad",
        tool_name="load_dataset",
        description=description,
        library="anndata",
        code_template=code_template,
        imports=["import anndata as ad"],
        parameters={input_param_name: "dataset.h5ad"},
        parameter_schema=parameter_schema,
        input_entities=["file"],
        output_entities=["adata"],
        execution_context={
            "operation_type": "data_io",
            "io_direction": "input",
            "file_format": "h5ad",
        },
        validates_on_export=False,  # Data loading doesn't require validation
        requires_validation=False,
    )


def _is_plain_literal(value: Any) -> bool:
    """True if ``value`` is built only from exact str/int/float/bool/None/list/dict.

    Exact types only (subclasses such as NumPy scalars or ``str`` subclasses are
    rejected), and dict keys must be ``str``. For such values ``_python_literal``
    yields source that evaluates back to an equal value.
    """
    value_type = type(value)
    if value is None or value_type in (str, int, float, bool):
        return True
    if value_type is list:
        return all(_is_plain_literal(item) for item in value)
    if value_type is dict:
        return all(
            type(key) is str and _is_plain_literal(item) for key, item in value.items()
        )
    return False


def create_adapter_loading_ir(
    adapter_name: str,
    adapter_kwargs: Optional[Dict[str, Any]] = None,
    input_param_name: str = "input_data",
    description: str = "Load input data through its modality adapter",
    file_format: Optional[str] = None,
) -> AnalysisStep:
    """
    Create IR for loading a file through a named modality adapter.

    Adapters change the data on load (metadata, QC columns, preprocessing), so
    a plain ``read_h5ad`` would start the notebook from a different object than
    the session. The rendered code calls
    ``lobster.core.adapters.factory.load_with_adapter``, the same constructor
    the data manager registers its adapters with.

    The adapter name and keyword arguments are passed as template parameters
    and rendered with the ``repr`` filter as Python literals. Only plain
    str/int/float/bool/None/list/dict values are accepted; if any keyword
    argument is something else, the IR refuses to reproduce the load and its
    code raises instead of silently loading a different object.

    Args:
        adapter_name: Registered adapter name (e.g. "transcriptomics_single_cell")
        adapter_kwargs: Keyword arguments the session passed to ``from_source``
        input_param_name: Name of the Papermill parameter holding the file path
        description: Human-readable description of the loading operation
        file_format: File extension of the loaded file, if known

    Returns:
        AnalysisStep with the adapter loading code template
    """
    if type(adapter_name) is not str:
        raise TypeError("adapter_name must be a str")
    if not input_param_name.isidentifier():
        raise ValueError(f"Invalid input parameter name: {input_param_name!r}")

    adapter_kwargs = dict(adapter_kwargs or {})
    unrenderable = sorted(
        str(key)
        for key, value in adapter_kwargs.items()
        if type(key) is not str or not _is_plain_literal(value)
    )

    parameter_schema = {
        input_param_name: ParameterSpec(
            param_type="str",
            papermill_injectable=True,
            default_value="dataset.h5ad",
            required=True,
            validation_rule=(
                f"{input_param_name}.endswith('.h5ad')"
                if file_format == "h5ad"
                else None
            ),
            description="Path to input data file",
        ),
        "adapter": ParameterSpec(
            param_type="str",
            papermill_injectable=False,
            default_value=adapter_name,
            required=True,
            description="Adapter the session loaded the file with",
        ),
    }
    parameters: Dict[str, Any] = {
        input_param_name: "dataset.h5ad",
        "adapter": adapter_name,
    }

    if unrenderable:
        parameters["unrenderable_kwargs"] = unrenderable
        code_template = (
            "# Load input data\n"
            "# The session loaded this file through the {{ adapter | repr }} adapter\n"
            "# with arguments that cannot be written as Python literals, so the\n"
            "# load cannot be reproduced here.\n"
            "raise NotImplementedError(\n"
            '    "Input load through adapter {!r} not reproducible: arguments {!r} "\n'
            '    "are not plain Python literals".format(\n'
            "        {{ adapter | repr }}, {{ unrenderable_kwargs | repr }}\n"
            "    )\n"
            ")\n"
        )
    else:
        parameters["adapter_kwargs"] = adapter_kwargs
        parameter_schema["adapter_kwargs"] = ParameterSpec(
            param_type="Dict[str, Any]",
            papermill_injectable=False,
            default_value=adapter_kwargs,
            required=False,
            description="Keyword arguments the session passed to the adapter",
        )
        # input_param_name is a variable reference (validated identifier), not
        # Jinja2 substitution: it is the Papermill-injected path parameter.
        code_template = (
            "# Load input data through the adapter the session used, so the\n"
            "# notebook starts from the same object (adapters add metadata and\n"
            "# QC columns on load).\n"
            "from lobster.core.adapters.factory import load_with_adapter\n"
            "\n"
            "adata = load_with_adapter(\n"
            "    {{ adapter | repr }},\n"
            f"    {input_param_name},\n"
            "{% if adapter_kwargs %}"
            "    **{{ adapter_kwargs | repr }},\n"
            "{% endif %}"
            ")\n"
            'print(f"Loaded data: {adata.n_obs} observations × {adata.n_vars} variables")\n'
        )

    return AnalysisStep(
        operation="lobster.core.adapters.factory.load_with_adapter",
        tool_name="load_dataset",
        description=description,
        library="lobster",
        code_template=code_template,
        imports=[
            "import anndata as ad",
            "from lobster.core.adapters.factory import load_with_adapter",
        ],
        parameters=parameters,
        parameter_schema=parameter_schema,
        input_entities=["file"],
        output_entities=["adata"],
        execution_context={
            "operation_type": "data_io",
            "io_direction": "input",
            "file_format": file_format,
            "adapter": adapter_name,
        },
        validates_on_export=False,  # Data loading doesn't require validation
        requires_validation=False,
    )


def create_data_saving_ir(
    output_prefix_param: str = "output_prefix",
    filename_suffix: str = "processed",
    description: str = "Save processed single-cell data to H5AD file",
    compression: str = "gzip",
) -> AnalysisStep:
    """
    Create IR for saving AnnData to H5AD file.

    Args:
        output_prefix_param: Name of parameter containing output file prefix
        filename_suffix: Suffix to append to output filename
        description: Human-readable description of the saving operation
        compression: H5AD compression method ('gzip', 'lzf', None)

    Returns:
        AnalysisStep with data saving code template
    """
    # Parameter schema with Papermill flags
    parameter_schema = {
        output_prefix_param: ParameterSpec(
            param_type="str",
            papermill_injectable=True,
            default_value="results",
            required=True,
            validation_rule=f"len({output_prefix_param}) > 0",
            description="Prefix for output files",
        ),
        "compression": ParameterSpec(
            param_type="str",
            papermill_injectable=True,
            default_value=compression,
            required=False,
            validation_rule="compression in ['gzip', 'lzf', None, 'none']",
            description="Compression method for H5AD file",
        ),
    }

    # Jinja2 template with parameter placeholders. filename_suffix reaches Jinja
    # as a parameter so it is written as a Python literal, not as source text.
    code_template = f"""# Save processed data
output_path = f"{{{output_prefix_param}}}_{{ {{{{ filename_suffix | repr }}}} }}.h5ad"
adata.write_h5ad(output_path, compression={{% if compression %}}{{{{ compression | tojson }}}}{{% else %}}None{{% endif %}})
print(f"Saved processed data to: {{output_path}}")
"""

    return AnalysisStep(
        operation="anndata.write_h5ad",
        tool_name="save_dataset",
        description=description,
        library="anndata",
        code_template=code_template,
        imports=["import anndata as ad"],
        parameters={
            output_prefix_param: "results",
            "compression": compression,
            "filename_suffix": filename_suffix,
        },
        parameter_schema=parameter_schema,
        input_entities=["adata"],
        output_entities=["file"],
        execution_context={
            "operation_type": "data_io",
            "io_direction": "output",
            "file_format": "h5ad",
            "compression": compression,
        },
        validates_on_export=False,  # Data saving doesn't require validation
        requires_validation=False,
    )
