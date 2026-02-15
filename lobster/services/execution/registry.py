"""
Custom code execution services registry for Lobster AI agents.

This module provides services for executing arbitrary Python code with
workspace context injection, safety validation, and W3C-PROV compliance.
It also provides SDK delegation for complex reasoning tasks.

Note: This module exists because PEP 420 namespace packages cannot have __init__.py.
Import directly from this module instead of from the package.

Examples:
    >>> from lobster.services.execution.registry import (
    ...     CustomCodeExecutionService,
    ...     CodeExecutionError,
    ... )
"""

# SDK Delegation Service (premium feature - graceful fallback if unavailable)
from lobster.core.component_registry import component_registry
from lobster.services.execution.custom_code_execution_service import (
    CodeExecutionError,
    CodeValidationError,
    CustomCodeExecutionService,
)
from lobster.services.execution.execution_context_builder import (
    ExecutionContextBuilder,
)

SDKDelegationService = component_registry.get_service("sdk_delegation")
HAS_SDK_DELEGATION = SDKDelegationService is not None
# For the exception class, use registry or fallback to base Exception
SDKDelegationError = component_registry.get_service("sdk_delegation_error") or Exception

__all__ = [
    "CustomCodeExecutionService",
    "CodeExecutionError",
    "CodeValidationError",
    "ExecutionContextBuilder",
    "SDKDelegationService",
    "SDKDelegationError",
    "HAS_SDK_DELEGATION",
]
