"""
scVI Embedding Service

Deep learning-based dimensionality reduction and batch correction using scVI
(single-cell Variational Inference). This service provides state-of-the-art
embeddings for single-cell RNA-seq data.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from lobster.core.provenance.analysis_ir import AnalysisStep, ParameterSpec

# Import GPU detector without ML dependencies
from lobster.tools.gpu_detector import GPUDetector, format_installation_message

logger = logging.getLogger(__name__)

# scvi imported conditionally in methods (optional dependency)


class ScviEmbeddingService:
    """
    Service for training scVI embeddings on single-cell data.

    This service handles:
    - Dependency checking and installation guidance
    - scVI model training with optimal parameters
    - Embedding generation and storage
    - GPU/CPU device management
    - Batch correction capabilities
    """

    def __init__(self):
        self.availability_info = GPUDetector.check_scvi_availability()
        self.device = None
        self.model = None

    def check_availability(self) -> Dict[str, Any]:
        """
        Check if scVI dependencies are available.

        Returns:
            Dictionary with availability status and installation guidance
        """
        return self.availability_info.copy()

    def get_installation_message(self) -> str:
        """Get user-friendly installation guidance message."""
        return format_installation_message(self.availability_info)

    def _import_scvi_dependencies(self):
        """
        Import scVI dependencies with helpful error messages.

        Raises:
            ImportError: If dependencies are not available with installation guidance
        """
        if not self.availability_info["ready_for_scvi"]:
            message = self.get_installation_message()
            raise ImportError(
                f"scVI dependencies not available.\n{message}\n"
                f"Install ML extras with: pip install lobster-ai[ml]"
            )

        try:
            import scvi
            import torch

            return torch
        except ImportError as e:
            raise ImportError(
                f"Failed to import scVI dependencies: {e}\n"
                f"Install ML extras with: pip install lobster-ai[ml]"
            )

    def setup_device(self, force_cpu: bool = False) -> str:
        """
        Set up the optimal compute device for training.

        Args:
            force_cpu: Force CPU usage even if GPU is available

        Returns:
            Device string used ("cuda", "mps", or "cpu")
        """
        torch = self._import_scvi_dependencies()

        if force_cpu:
            device = "cpu"
        else:
            hardware_rec = self.availability_info.get(
                "hardware_recommendation",
                GPUDetector.get_hardware_recommendation(),
            )
            recommended_device = hardware_rec["device"]

            # Verify the device is actually available
            if recommended_device == "cuda" and torch.cuda.is_available():
                device = "cuda"
            elif recommended_device == "mps" and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        logger.info(f"Using device: {device}")
        return device

    def get_embeddings(self, adata, model=None) -> Any:
        """
        Extract scVI embeddings from AnnData or a trained model.

        If embeddings already exist in adata.obsm['X_scvi'], return those.
        Otherwise, use the provided model (or self.model) to compute them.

        Args:
            adata: AnnData object (may contain pre-computed embeddings)
            model: Optional scVI model to use for extraction

        Returns:
            numpy array of embeddings

        Raises:
            ValueError: If no embeddings found and no model available
        """
        # Check for pre-existing embeddings
        if "X_scvi" in adata.obsm:
            return adata.obsm["X_scvi"]

        # Use provided model or stored model
        active_model = model or self.model
        if active_model is None:
            raise ValueError(
                "No scVI embeddings found in adata.obsm['X_scvi'] and no model available. "
                "Train a model first with train_scvi_embedding()."
            )

        embeddings = active_model.get_latent_representation()
        return embeddings

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current scVI model.

        Returns:
            Dictionary with model status, device, and architecture details
        """
        if self.model is None:
            return {"model_available": False}

        info: Dict[str, Any] = {
            "model_available": True,
            "device": self.device,
        }

        try:
            info["n_latent"] = self.model.module.n_latent
        except AttributeError:
            info["n_latent"] = None

        try:
            info["is_trained"] = self.model.is_trained_
        except AttributeError:
            info["is_trained"] = None

        return info

    def _create_scvi_ir(
        self,
        batch_key: Optional[str],
        n_layers: int,
        n_latent: int,
        gene_likelihood: str,
        max_epochs: int,
        early_stopping: bool,
        early_stopping_patience: int,
        force_cpu: bool,
    ) -> AnalysisStep:
        """
        Create Intermediate Representation for scVI embedding training.

        Args:
            batch_key: Key in adata.obs for batch information
            n_layers: Number of layers in the neural network
            n_latent: Number of latent dimensions (embedding size)
            gene_likelihood: Gene likelihood distribution
            max_epochs: Maximum training epochs
            early_stopping: Whether to use early stopping
            early_stopping_patience: Patience for early stopping
            force_cpu: Force CPU usage

        Returns:
            AnalysisStep with full scVI training code template
        """
        parameter_schema = {
            "batch_key": ParameterSpec(
                param_type="Optional[str]",
                papermill_injectable=True,
                default_value=batch_key,
                required=False,
                description="Key in adata.obs for batch information",
            ),
            "n_layers": ParameterSpec(
                param_type="int",
                papermill_injectable=True,
                default_value=n_layers,
                required=False,
                validation_rule="n_layers > 0",
                description="Number of hidden layers in the neural network",
            ),
            "n_latent": ParameterSpec(
                param_type="int",
                papermill_injectable=True,
                default_value=n_latent,
                required=False,
                validation_rule="n_latent > 0",
                description="Number of latent dimensions (embedding size)",
            ),
            "gene_likelihood": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value=gene_likelihood,
                required=False,
                description="Gene likelihood distribution: 'zinb', 'nb', or 'poisson'",
            ),
            "max_epochs": ParameterSpec(
                param_type="int",
                papermill_injectable=True,
                default_value=max_epochs,
                required=False,
                validation_rule="max_epochs > 0",
                description="Maximum number of training epochs",
            ),
            "early_stopping": ParameterSpec(
                param_type="bool",
                papermill_injectable=True,
                default_value=early_stopping,
                required=False,
                description="Whether to use early stopping",
            ),
            "early_stopping_patience": ParameterSpec(
                param_type="int",
                papermill_injectable=True,
                default_value=early_stopping_patience,
                required=False,
                validation_rule="early_stopping_patience > 0",
                description="Number of epochs to wait before early stopping",
            ),
            "force_cpu": ParameterSpec(
                param_type="bool",
                papermill_injectable=True,
                default_value=force_cpu,
                required=False,
                description="Force CPU usage even if GPU is available",
            ),
        }

        code_template = """# scVI embedding training with deep learning
import scvi
import torch
import numpy as np

# Setup device
{% if force_cpu %}
device = "cpu"
print("Forcing CPU usage")
{% else %}
# Auto-detect best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")
{% endif %}

# Setup AnnData for scVI
print("Setting up scVI data...")
scvi.model.SCVI.setup_anndata(
    adata,
    layer=None,  # Use X
    batch_key={{ "'" + batch_key + "'" if batch_key else "None" }},
    categorical_covariate_keys=None,
    continuous_covariate_keys=None,
)

# Create scVI model
print(f"Creating scVI model with {{ n_latent }} latent dimensions...")
model = scvi.model.SCVI(
    adata,
    n_layers={{ n_layers }},
    n_latent={{ n_latent }},
    gene_likelihood="{{ gene_likelihood }}",
    use_observed_lib_size=True,
)

# Train model with error handling
print(f"Training scVI model for up to {{ max_epochs }} epochs on {device}...")
training_config = {
    "max_epochs": {{ max_epochs }},
    "accelerator": device,
    "devices": "auto",
    "early_stopping": {{ "True" if early_stopping else "False" }},
}

try:
    model.train(**training_config)
    print("Training completed successfully")
except RuntimeError as e:
    # Handle device errors with CPU fallback
    if device != "cpu" and ("device" in str(e).lower() or "cuda" in str(e).lower()):
        print(f"Device error encountered, falling back to CPU: {e}")
        training_config["accelerator"] = "cpu"
        training_config["devices"] = "auto"
        model.train(**training_config)
        device = "cpu"
        print("Successfully completed training on CPU after device error")
    else:
        raise

# Extract embeddings
print("Extracting embeddings...")
embeddings = model.get_latent_representation()
adata.obsm['X_scvi'] = embeddings

print(f"scVI training complete:")
print(f"  - Latent dimensions: {{ n_latent }}")
print(f"  - Cells: {adata.n_obs}")
print(f"  - Genes: {adata.n_vars}")
print(f"  - Device: {device}")
print(f"  - Embedding shape: {embeddings.shape}")

# Full training code with GPU fallback logic available in service
# Note: Complex error handling for convergence/numerical issues
# is handled at the service level
"""

        return AnalysisStep(
            operation="scvi.model.SCVI.train",
            tool_name="train_scvi_embedding",
            description=f"Train scVI model with {n_latent} latent dimensions for deep learning embeddings",
            library="scvi",
            code_template=code_template,
            imports=["import scvi", "import torch", "import numpy as np"],
            parameters={
                "batch_key": batch_key,
                "n_layers": n_layers,
                "n_latent": n_latent,
                "gene_likelihood": gene_likelihood,
                "max_epochs": max_epochs,
                "early_stopping": early_stopping,
                "early_stopping_patience": early_stopping_patience,
                "force_cpu": force_cpu,
            },
            parameter_schema=parameter_schema,
            input_entities=["adata"],
            output_entities=["adata"],
            execution_context={
                "operation_type": "embedding",
                "device": "auto-detect",
                "n_latent": n_latent,
                "scvi_version": "1.0.0",
                "timestamp": datetime.now().isoformat(),
            },
            validates_on_export=True,
            requires_validation=False,
        )

    def train_scvi_embedding(
        self,
        adata,
        batch_key: Optional[str] = None,
        categorical_covariate_keys: Optional[List[str]] = None,
        continuous_covariate_keys: Optional[List[str]] = None,
        layer: Optional[str] = None,
        n_layers: int = 2,
        n_latent: int = 10,
        gene_likelihood: str = "zinb",
        max_epochs: Optional[int] = 100,
        early_stopping: bool = True,
        early_stopping_patience: int = 5,
        save_path: Optional[str] = None,
        force_cpu: bool = False,
        **model_kwargs,
    ) -> Tuple[Any, Dict[str, Any], AnalysisStep]:
        """
        Train scVI model and generate embeddings.

        Args:
            adata: AnnData object with single-cell data
            batch_key: Key in adata.obs for batch information
            categorical_covariate_keys: List of categorical covariate keys
            continuous_covariate_keys: List of continuous covariate keys
            layer: Layer to use for training (None for X)
            n_latent: Number of latent dimensions (embedding size)
            max_epochs: Maximum training epochs (auto-calculated if None)
            early_stopping: Whether to use early stopping
            save_path: Path to save trained model
            force_cpu: Force CPU usage
            train_kwargs: Dictionary of advanced training parameters (learning_rate, batch_size, etc.)
            **model_kwargs: Additional arguments for scVI model (architecture parameters)

        Returns:
            Tuple of (adata, stats_dict, AnalysisStep)
        """
        torch = self._import_scvi_dependencies()

        # Setup device
        device = self.setup_device(force_cpu=force_cpu)

        # Prepare data for scVI
        logger.info("Setting up scVI data...")

        # Import scvi module (already validated by _import_scvi_dependencies)
        import scvi

        # Setup AnnData for scVI
        scvi.model.SCVI.setup_anndata(
            adata,
            layer=layer,
            batch_key=batch_key,
            categorical_covariate_keys=categorical_covariate_keys,
            continuous_covariate_keys=continuous_covariate_keys,
        )

        # Create model
        logger.info(f"Creating scVI model with {n_latent} latent dimensions...")

        # Extract use_observed_lib_size from model_kwargs if present, default True
        use_observed_lib_size = model_kwargs.pop("use_observed_lib_size", True)

        model = scvi.model.SCVI(
            adata,
            n_layers=n_layers,
            n_latent=n_latent,
            gene_likelihood=gene_likelihood,
            use_observed_lib_size=use_observed_lib_size,
            # **model_kwargs #FIXME needs to be readjusted once ready
        )

        # Train model with comprehensive error handling
        logger.info(f"Training scVI model for up to {max_epochs} epochs on {device}...")

        # get model to device
        # model.to_device(device)

        training_config = {
            "max_epochs": max_epochs,
            "accelerator": device,
            "devices": "auto",
            "early_stopping": early_stopping,
        }

        # Execute training with comprehensive error handling
        try:
            print(training_config)
            model.train(**training_config)

            # Training completed successfully - store model and extract embeddings
            self.model = model
            logger.info("Training completed successfully, extracting embeddings...")

            # Extract embeddings and store in AnnData
            embeddings = model.get_latent_representation()
            adata.obsm["X_scvi"] = embeddings
            logger.info(
                f"Stored embeddings in adata.obsm['X_scvi']: {embeddings.shape}"
            )

            # Create comprehensive training info
            training_info = {
                "device": device,
                "n_latent": n_latent,
                "n_cells": adata.n_obs,
                "n_genes": adata.n_vars,
                "max_epochs": max_epochs,
                "embedding_shape": embeddings.shape,
                "model_saved": save_path is not None,
                "batch_key": batch_key,
                "accelerator": device,  # Keep for backward compatibility
                "early_stopping": early_stopping,
            }

            # Save model if path provided
            if save_path:
                try:
                    model.save(save_path, overwrite=True)
                    training_info["model_saved"] = True
                    training_info["model_path"] = save_path
                    logger.info(f"Model saved to: {save_path}")
                except Exception as save_error:
                    logger.warning(f"Failed to save model: {save_error}")
                    training_info["model_saved"] = False

            # Create IR for provenance
            ir = self._create_scvi_ir(
                batch_key=batch_key,
                n_layers=n_layers,
                n_latent=n_latent,
                gene_likelihood=gene_likelihood,
                max_epochs=max_epochs,
                early_stopping=early_stopping,
                early_stopping_patience=early_stopping_patience,
                force_cpu=force_cpu,
            )

            return adata, training_info, ir
        except RuntimeError as e:
            error_msg = str(e).lower()

            # Handle device placement errors
            if "device" in error_msg or "cuda" in error_msg:
                logger.error(f"Device error during training: {e}")
                if device != "cpu" and not force_cpu:
                    logger.info(
                        "Attempting to retry training on CPU due to device error..."
                    )
                    try:
                        training_config["accelerator"] = "cpu"
                        training_config["devices"] = "auto"
                        model.train(**training_config)
                        device = "cpu"
                        logger.info(
                            "Successfully completed training on CPU after device error"
                        )

                        # Training successful on CPU - process like normal success case
                        self.model = model
                        embeddings = model.get_latent_representation()
                        adata.obsm["X_scvi"] = embeddings
                        logger.info(
                            f"Stored embeddings in adata.obsm['X_scvi']: {embeddings.shape}"
                        )

                        training_info = {
                            "device": device,
                            "n_latent": n_latent,
                            "n_cells": adata.n_obs,
                            "n_genes": adata.n_vars,
                            "max_epochs": max_epochs,
                            "embedding_shape": embeddings.shape,
                            "model_saved": save_path is not None,
                            "batch_key": batch_key,
                            "accelerator": device,
                            "early_stopping": early_stopping,
                            "fallback_to_cpu": True,
                        }

                        if save_path:
                            try:
                                model.save(save_path, overwrite=True)
                                training_info["model_saved"] = True
                                training_info["model_path"] = save_path
                            except Exception as save_error:
                                logger.warning(f"Failed to save model: {save_error}")
                                training_info["model_saved"] = False

                        # Create IR for provenance (CPU fallback case)
                        ir = self._create_scvi_ir(
                            batch_key=batch_key,
                            n_layers=n_layers,
                            n_latent=n_latent,
                            gene_likelihood=gene_likelihood,
                            max_epochs=max_epochs,
                            early_stopping=early_stopping,
                            early_stopping_patience=early_stopping_patience,
                            force_cpu=force_cpu,
                        )

                        return adata, training_info, ir

                    except Exception as cpu_error:
                        error_info = {
                            "device": "failed",
                            "n_latent": n_latent,
                            "n_cells": adata.n_obs,
                            "n_genes": adata.n_vars,
                            "max_epochs": max_epochs,
                            "error": f"Training failed on both {device} (device error) and CPU: {cpu_error}",
                            "error_type": "device_and_cpu_failure",
                        }
                        return None, error_info, None
                else:
                    error_info = {
                        "device": device,
                        "n_latent": n_latent,
                        "n_cells": adata.n_obs,
                        "n_genes": adata.n_vars,
                        "max_epochs": max_epochs,
                        "error": f"Device error during training: {e}",
                        "error_type": "device_error",
                    }
                    return None, error_info, None

            # Handle convergence or numerical issues
            elif "nan" in error_msg or "inf" in error_msg or "convergence" in error_msg:
                error_info = {
                    "device": device,
                    "n_latent": n_latent,
                    "n_cells": adata.n_obs,
                    "n_genes": adata.n_vars,
                    "max_epochs": max_epochs,
                    "error": f"Training convergence error - try reducing learning rate or batch size: {e}",
                    "error_type": "convergence_error",
                }
                return None, error_info, None

            # Handle other runtime errors
            else:
                error_info = {
                    "device": device,
                    "n_latent": n_latent,
                    "n_cells": adata.n_obs,
                    "n_genes": adata.n_vars,
                    "max_epochs": max_epochs,
                    "error": f"Training failed with runtime error: {e}",
                    "error_type": "runtime_error",
                }
                return None, error_info, None

        except ValueError as e:
            # Handle parameter validation errors
            error_info = {
                "device": device if "device" in locals() else "unknown",
                "n_latent": n_latent,
                "n_cells": adata.n_obs,
                "n_genes": adata.n_vars,
                "max_epochs": max_epochs,
                "error": f"Invalid training parameters: {e}",
                "error_type": "parameter_validation_error",
            }
            return None, error_info, None

        except Exception as e:
            # Handle any other unexpected errors
            logger.error(f"Unexpected error during scVI training: {e}")
            error_info = {
                "device": device if "device" in locals() else "unknown",
                "n_latent": n_latent,
                "n_cells": adata.n_obs,
                "n_genes": adata.n_vars,
                "max_epochs": max_epochs,
                "error": f"Unexpected error during scVI training: {e}",
                "error_type": "unexpected_error",
            }
            return None, error_info, None
