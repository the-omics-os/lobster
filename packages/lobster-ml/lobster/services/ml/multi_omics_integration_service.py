"""
Multi-Omics Integration Service for MOFA-based factor analysis.

Provides MOFA (Multi-Omics Factor Analysis) for extracting shared
latent factors across modalities (RNA, protein, metabolites, etc.).
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from anndata import AnnData

from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = ["MultiOmicsIntegrationService", "ZeroOverlapError"]


class ZeroOverlapError(ValueError):
    """Raised when no samples overlap across all modalities."""

    pass


class MultiOmicsIntegrationService:
    """
    Stateless service for multi-omics integration on AnnData objects.

    Implements MOFA (Multi-Omics Factor Analysis) for extracting shared
    latent factors across modalities.

    All methods return the standard lobster 3-tuple:
    (AnnData, stats_dict, AnalysisStep)
    """

    def compute_sample_overlap(self, modalities: Dict[str, AnnData]) -> Dict[str, Any]:
        """
        Compute sample overlap statistics across modalities.

        Args:
            modalities: Dict mapping modality names to AnnData objects

        Returns:
            Dict with:
                - overlap_matrix: DataFrame (samples x modalities, boolean)
                - complete_cases: List of sample IDs present in all modalities
                - overlap_fractions: Dict of threshold -> fraction of samples
                - per_modality_counts: Dict of modality -> sample count
        """
        if not modalities:
            raise ValueError("At least one modality required")

        # Collect all unique sample IDs across modalities
        all_samples = set()
        modality_samples = {}
        for name, adata in modalities.items():
            samples = set(adata.obs_names)
            modality_samples[name] = samples
            all_samples.update(samples)

        # Build presence/absence matrix
        overlap_matrix = pd.DataFrame(
            False, index=sorted(all_samples), columns=sorted(modalities.keys())
        )
        for name, samples in modality_samples.items():
            overlap_matrix.loc[list(samples), name] = True

        # Compute complete cases (intersection)
        complete_cases = sorted(
            set.intersection(*[modality_samples[name] for name in modalities.keys()])
        )

        # Compute overlap fractions at different thresholds
        overlap_fractions = {}
        n_modalities = len(modalities)
        for threshold in [1.0, 0.9, 0.8, 0.7, 0.5]:
            min_modalities = int(np.ceil(threshold * n_modalities))
            n_samples_meeting = (overlap_matrix.sum(axis=1) >= min_modalities).sum()
            overlap_fractions[threshold] = n_samples_meeting / len(all_samples)

        # Per-modality counts
        per_modality_counts = {
            name: len(samples) for name, samples in modality_samples.items()
        }

        return {
            "overlap_matrix": overlap_matrix,
            "complete_cases": complete_cases,
            "overlap_fractions": overlap_fractions,
            "per_modality_counts": per_modality_counts,
        }

    def _estimate_mofa_memory(
        self,
        modalities: Dict[str, AnnData],
        n_factors: int,
        overhead_multiplier: float = 3.0,
    ) -> Dict[str, float]:
        """
        Estimate memory requirements for MOFA training.

        Formula: (Data + Factors + Weights) * overhead_multiplier
        - Data: N * sum(D_m) * 8 bytes
        - Factors (Z): N * K * 8 bytes
        - Weights (W): sum(D_m) * K * 8 bytes

        Args:
            modalities: Dict mapping modality names to AnnData objects
            n_factors: Number of latent factors
            overhead_multiplier: Multiplier for overhead (default 3.0)

        Returns:
            Dict with: data_gb, factors_gb, weights_gb, base_gb, total_gb, overhead_multiplier
        """
        # Get sample count (use first modality, overlap validation happens separately)
        n_samples = list(modalities.values())[0].n_obs

        # Sum of feature dimensions across modalities
        total_features = sum(adata.n_vars for adata in modalities.values())

        # Data matrix memory
        data_bytes = n_samples * total_features * 8
        data_gb = data_bytes / (1024**3)

        # Factors memory (N x K)
        factors_bytes = n_samples * n_factors * 8
        factors_gb = factors_bytes / (1024**3)

        # Weights memory (sum(D_m) x K)
        weights_bytes = total_features * n_factors * 8
        weights_gb = weights_bytes / (1024**3)

        # Base memory
        base_gb = data_gb + factors_gb + weights_gb

        # Total with overhead
        total_gb = base_gb * overhead_multiplier

        return {
            "data_gb": data_gb,
            "factors_gb": factors_gb,
            "weights_gb": weights_gb,
            "base_gb": base_gb,
            "total_gb": total_gb,
            "overhead_multiplier": overhead_multiplier,
        }

    def _validate_overlap(
        self, overlap_stats: Dict, min_overlap_fraction: float
    ) -> None:
        """
        Validate sample overlap meets minimum requirements.

        Args:
            overlap_stats: Output from compute_sample_overlap()
            min_overlap_fraction: Minimum fraction of samples required

        Raises:
            ZeroOverlapError: If no samples overlap across all modalities
            ValueError: If overlap fraction below minimum
        """
        complete_cases = overlap_stats["complete_cases"]

        if len(complete_cases) == 0:
            raise ZeroOverlapError(
                "No samples found in all modalities. Multi-omics integration "
                "requires at least one sample present in all input datasets. "
                "Check sample IDs (obs_names) for consistency."
            )

        overlap_fractions = overlap_stats["overlap_fractions"]
        actual_fraction = overlap_fractions[1.0]  # complete cases fraction

        if actual_fraction < min_overlap_fraction:
            raise ValueError(
                f"Sample overlap ({actual_fraction:.2%}) below minimum threshold "
                f"({min_overlap_fraction:.2%}). Only {len(complete_cases)} samples "
                f"present in all modalities. Consider:\n"
                f"  1. Relaxing min_overlap_fraction\n"
                f"  2. Filtering modalities to improve overlap\n"
                f"  3. Using imputation to fill missing data"
            )

    def _check_gpu_availability(self) -> bool:
        """
        Check if GPU is available for MOFA training.

        Returns:
            True if GPU available, False otherwise
        """
        try:
            from mofapy2.core.gpu_utils import gpu_utils

            return gpu_utils.gpu_available
        except (ImportError, AttributeError):
            return False

    def _check_mofapy2_available(self) -> bool:
        """
        Check if mofapy2 is installed.

        Returns:
            True if mofapy2 available, False otherwise
        """
        try:
            from mofapy2.run.entry_point import entry_point

            return True
        except ImportError:
            return False

    def integrate_modalities(
        self,
        modalities: Dict[str, AnnData],
        n_factors: int = 15,
        min_overlap_fraction: float = 1.0,
        memory_threshold_gb: float = 2.0,
        gpu_mode: bool = False,
        convergence_mode: str = "fast",
        n_iterations: int = 1000,
        random_state: int = 42,
    ) -> Tuple[AnnData, Dict[str, Any], AnalysisStep]:
        """
        Integrate multiple omics modalities using MOFA.

        Args:
            modalities: Dict mapping modality names to AnnData objects
            n_factors: Number of latent factors to extract (default 15)
            min_overlap_fraction: Minimum sample overlap fraction (default 1.0)
            memory_threshold_gb: Memory threshold in GB (default 2.0)
            gpu_mode: Use GPU acceleration if available (default False)
            convergence_mode: MOFA convergence mode: fast/medium/slow (default "fast")
            n_iterations: Maximum training iterations (default 1000)
            random_state: Random seed (default 42)

        Returns:
            Tuple of (augmented_adata, stats_dict, analysis_step)
            - augmented_adata: Primary modality with factors in obsm['X_mofa']
            - stats_dict: Integration statistics
            - analysis_step: Provenance IR

        Raises:
            ImportError: If mofapy2 not installed
            ZeroOverlapError: If no sample overlap
            ValueError: If overlap below threshold or memory too high
        """
        # Check mofapy2 availability
        if not self._check_mofapy2_available():
            raise ImportError(
                "mofapy2 required for MOFA integration. "
                "Install with: pip install mofapy2>=0.7.0"
            )

        from mofapy2.run.entry_point import entry_point

        logger.info(f"Integrating {len(modalities)} modalities with MOFA")

        # 1. Compute overlap statistics
        overlap_stats = self.compute_sample_overlap(modalities)
        complete_cases = overlap_stats["complete_cases"]
        n_complete = len(complete_cases)

        logger.info(f"Found {n_complete} samples in all modalities")

        # 2. Validate overlap
        self._validate_overlap(overlap_stats, min_overlap_fraction)

        # 3. Estimate memory
        memory_estimate = self._estimate_mofa_memory(modalities, n_factors)
        total_gb = memory_estimate["total_gb"]

        logger.info(f"Estimated memory: {total_gb:.2f} GB")

        # 4. Check memory against threshold
        if total_gb > memory_threshold_gb * 2:
            raise ValueError(
                f"Estimated memory ({total_gb:.2f} GB) exceeds 2x threshold "
                f"({memory_threshold_gb * 2:.2f} GB). Reduce n_factors or filter features."
            )
        elif total_gb > memory_threshold_gb:
            logger.warning(
                f"Estimated memory ({total_gb:.2f} GB) exceeds threshold "
                f"({memory_threshold_gb:.2f} GB). Training may be slow."
            )

        # 5. Check GPU availability
        if gpu_mode and not self._check_gpu_availability():
            logger.warning(
                "GPU mode requested but no GPU available. Falling back to CPU."
            )
            gpu_mode = False

        # 6. Convert modalities to MOFA-compatible format
        # Align all modalities to complete_cases samples
        aligned_modalities = {}
        for name, adata in modalities.items():
            # Subset to complete cases
            adata_aligned = adata[complete_cases, :].copy()
            aligned_modalities[name] = adata_aligned

        # 7. Initialize and train MOFA
        logger.info("Initializing MOFA model...")

        # Create entry point
        ent = entry_point()

        # Set data options
        ent.set_data_options(scale_views=False, scale_groups=False, center_groups=True)

        # Set data matrices
        for view_name, adata in aligned_modalities.items():
            # Convert to numpy array, transpose to (features x samples)
            data_matrix = adata.X.T
            if hasattr(data_matrix, "toarray"):
                data_matrix = data_matrix.toarray()

            ent.set_data_matrix(
                data=data_matrix, views_names=[view_name], groups_names=["group1"]
            )

        # Set model options
        ent.set_model_options(
            factors=n_factors,
            ard_factors=True,
            ard_weights=True,
            spikeslab_weights=True,
        )

        # Set training options
        ent.set_train_options(
            iter=n_iterations,
            convergence_mode=convergence_mode,
            gpu_mode=gpu_mode,
            seed=random_state,
            verbose=False,
        )

        # Build and train model
        logger.info("Training MOFA model...")
        ent.build()
        ent.run()
        logger.info("MOFA training complete")

        # 8. Extract factors and loadings
        factors = ent.model.nodes["Z"]["group1"].getExpectation()  # (samples, factors)

        # Extract variance explained per view
        variance_explained = {}
        for view_idx, view_name in enumerate(sorted(aligned_modalities.keys())):
            r2 = ent.model.calculate_variance_explained(
                views=[view_name], groups=["group1"]
            )
            variance_explained[view_name] = float(r2.values[0])

        # 9. Store results in primary modality (first in dict)
        primary_name = list(modalities.keys())[0]
        primary_adata = aligned_modalities[primary_name].copy()

        # Store factors
        primary_adata.obsm["X_mofa"] = factors

        # Store loadings for primary modality
        primary_view_name = list(aligned_modalities.keys())[0]
        loadings = ent.model.nodes["W"][
            primary_view_name
        ].getExpectation()  # (features, factors)
        primary_adata.varm["LFs"] = loadings

        # Store metadata
        primary_adata.uns["integration_method"] = "MOFA"
        primary_adata.uns["integration_params"] = {
            "n_factors": n_factors,
            "min_overlap_fraction": min_overlap_fraction,
            "memory_threshold_gb": memory_threshold_gb,
            "gpu_mode": gpu_mode,
            "convergence_mode": convergence_mode,
            "n_iterations": n_iterations,
            "random_state": random_state,
        }
        primary_adata.uns["integration_input_modalities"] = list(modalities.keys())
        primary_adata.uns["integration_n_factors"] = n_factors
        primary_adata.uns["integration_variance_explained"] = variance_explained

        # 10. Build stats dict
        n_original = overlap_stats["per_modality_counts"][primary_name]
        n_dropped = n_original - n_complete

        stats = {
            "n_factors": n_factors,
            "n_modalities": len(modalities),
            "n_samples_used": n_complete,
            "n_samples_dropped": n_dropped,
            "overlap_fraction": n_complete / n_original,
            "variance_explained": variance_explained,
            "memory_used_gb": total_gb,
        }

        # 11. Build IR
        ir = self._create_integration_ir(
            modality_names=list(modalities.keys()),
            n_factors=n_factors,
            min_overlap_fraction=min_overlap_fraction,
            memory_threshold_gb=memory_threshold_gb,
            gpu_mode=gpu_mode,
            convergence_mode=convergence_mode,
            n_iterations=n_iterations,
            random_state=random_state,
        )

        logger.info(
            f"Integration complete: {n_factors} factors from {len(modalities)} modalities"
        )

        return primary_adata, stats, ir

    def _create_integration_ir(
        self,
        modality_names: List[str],
        n_factors: int,
        min_overlap_fraction: float,
        memory_threshold_gb: float,
        gpu_mode: bool,
        convergence_mode: str,
        n_iterations: int,
        random_state: int,
    ) -> AnalysisStep:
        """
        Create IR for MOFA integration.

        Args:
            modality_names: List of modality names
            n_factors: Number of latent factors
            min_overlap_fraction: Minimum sample overlap fraction
            memory_threshold_gb: Memory threshold in GB
            gpu_mode: Use GPU acceleration
            convergence_mode: MOFA convergence mode
            n_iterations: Maximum training iterations
            random_state: Random seed

        Returns:
            AnalysisStep with executable MOFA integration code
        """
        parameter_schema = {
            "modality_names": ParameterSpec(
                param_type="list",
                papermill_injectable=True,
                default_value=modality_names,
                required=True,
                description="List of modality names to integrate",
            ),
            "n_factors": ParameterSpec(
                param_type="int",
                papermill_injectable=True,
                default_value=n_factors,
                required=False,
                validation_rule="n_factors > 0 and n_factors <= 50",
                description="Number of latent factors to extract",
            ),
            "min_overlap_fraction": ParameterSpec(
                param_type="float",
                papermill_injectable=True,
                default_value=min_overlap_fraction,
                required=False,
                validation_rule="0.0 <= min_overlap_fraction <= 1.0",
                description="Minimum sample overlap fraction",
            ),
            "memory_threshold_gb": ParameterSpec(
                param_type="float",
                papermill_injectable=True,
                default_value=memory_threshold_gb,
                required=False,
                description="Memory threshold in GB",
            ),
            "gpu_mode": ParameterSpec(
                param_type="bool",
                papermill_injectable=True,
                default_value=gpu_mode,
                required=False,
                description="Use GPU acceleration if available",
            ),
            "convergence_mode": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value=convergence_mode,
                required=False,
                validation_rule="convergence_mode in ['fast', 'medium', 'slow']",
                description="MOFA convergence mode",
            ),
            "n_iterations": ParameterSpec(
                param_type="int",
                papermill_injectable=True,
                default_value=n_iterations,
                required=False,
                description="Maximum training iterations",
            ),
            "random_state": ParameterSpec(
                param_type="int",
                papermill_injectable=True,
                default_value=random_state,
                required=False,
                description="Random seed for reproducibility",
            ),
        }

        code_template = """
# Multi-Omics Factor Analysis (MOFA) Integration
# Integrates {{ modality_names|length }} modalities into {{ n_factors }} shared latent factors

import numpy as np
import pandas as pd
from anndata import AnnData
from mofapy2.run.entry_point import entry_point
import h5py
import tempfile

# Parameters
modality_names = {{ modality_names }}
n_factors = {{ n_factors }}
min_overlap_fraction = {{ min_overlap_fraction }}
memory_threshold_gb = {{ memory_threshold_gb }}
gpu_mode = {{ gpu_mode }}
convergence_mode = "{{ convergence_mode }}"
n_iterations = {{ n_iterations }}
random_state = {{ random_state }}

print(f"\\n{'='*60}")
print(f"MOFA Multi-Omics Integration")
print(f"{'='*60}")
print(f"Modalities: {modality_names}")
print(f"Factors: {n_factors}")
print(f"Min overlap: {min_overlap_fraction:.1%}")
print(f"Memory threshold: {memory_threshold_gb:.2f} GB")
print(f"GPU mode: {gpu_mode}")
print(f"Convergence: {convergence_mode}")

# Load modalities from data_manager
print(f"\\n1. Loading modalities...")
modalities = {}
for name in modality_names:
    adata = data_manager.get_modality(name)
    modalities[name] = adata
    print(f"  {name}: {adata.n_obs} samples × {adata.n_vars} features")

# Compute sample overlap
print(f"\\n2. Computing sample overlap...")
all_samples = set()
modality_samples = {}
for name, adata in modalities.items():
    samples = set(adata.obs_names)
    modality_samples[name] = samples
    all_samples.update(samples)

# Build overlap matrix
overlap_matrix = pd.DataFrame(
    False,
    index=sorted(all_samples),
    columns=sorted(modalities.keys())
)
for name, samples in modality_samples.items():
    overlap_matrix.loc[list(samples), name] = True

# Complete cases (intersection)
complete_cases = sorted(
    set.intersection(*[modality_samples[name] for name in modalities.keys()])
)

print(f"  Total unique samples: {len(all_samples)}")
print(f"  Complete cases (all modalities): {len(complete_cases)}")
print(f"  Overlap fraction: {len(complete_cases)/len(all_samples):.1%}")

# Overlap at thresholds
n_modalities = len(modalities)
for threshold in [1.0, 0.9, 0.8, 0.7, 0.5]:
    min_mods = int(np.ceil(threshold * n_modalities))
    n_meeting = (overlap_matrix.sum(axis=1) >= min_mods).sum()
    frac = n_meeting / len(all_samples)
    print(f"  {threshold:.0%} modalities: {n_meeting} samples ({frac:.1%})")

# Validate overlap
if len(complete_cases) == 0:
    raise ValueError(
        "No samples found in all modalities. Check sample IDs for consistency."
    )

actual_overlap = len(complete_cases) / len(all_samples)
if actual_overlap < min_overlap_fraction:
    raise ValueError(
        f"Sample overlap ({actual_overlap:.2%}) below threshold "
        f"({min_overlap_fraction:.2%})"
    )

# Estimate memory
print(f"\\n3. Estimating memory requirements...")
n_samples = len(complete_cases)
total_features = sum(adata.n_vars for adata in modalities.values())

data_gb = (n_samples * total_features * 8) / (1024**3)
factors_gb = (n_samples * n_factors * 8) / (1024**3)
weights_gb = (total_features * n_factors * 8) / (1024**3)
base_gb = data_gb + factors_gb + weights_gb
total_gb = base_gb * 3.0  # 3x overhead

print(f"  Data: {data_gb:.3f} GB")
print(f"  Factors: {factors_gb:.3f} GB")
print(f"  Weights: {weights_gb:.3f} GB")
print(f"  Base: {base_gb:.3f} GB")
print(f"  Total (3x overhead): {total_gb:.3f} GB")

if total_gb > memory_threshold_gb * 2:
    raise ValueError(
        f"Estimated memory ({total_gb:.2f} GB) exceeds 2x threshold "
        f"({memory_threshold_gb * 2:.2f} GB)"
    )
elif total_gb > memory_threshold_gb:
    print(f"  WARNING: Exceeds threshold ({memory_threshold_gb:.2f} GB)")

# Check GPU availability
print(f"\\n4. Checking GPU availability...")
try:
    from mofapy2.core.gpu_utils import gpu_utils
    gpu_available = gpu_utils.gpu_available
except (ImportError, AttributeError):
    gpu_available = False

print(f"  GPU available: {gpu_available}")
if gpu_mode and not gpu_available:
    print(f"  WARNING: GPU mode requested but not available. Using CPU.")
    gpu_mode = False

# Align modalities to complete cases
print(f"\\n5. Aligning modalities to complete cases...")
aligned_modalities = {}
for name, adata in modalities.items():
    adata_aligned = adata[complete_cases, :].copy()
    aligned_modalities[name] = adata_aligned
    print(f"  {name}: {adata_aligned.n_obs} samples × {adata_aligned.n_vars} features")

# Initialize MOFA
print(f"\\n6. Initializing MOFA model...")
ent = entry_point()

# Set data options
ent.set_data_options(
    scale_views=False,
    scale_groups=False,
    center_groups=True
)

# Set data matrices
for view_name, adata in aligned_modalities.items():
    data_matrix = adata.X.T
    if hasattr(data_matrix, 'toarray'):
        data_matrix = data_matrix.toarray()

    ent.set_data_matrix(
        data=data_matrix,
        views_names=[view_name],
        groups_names=["group1"]
    )
    print(f"  Set data for {view_name}")

# Set model options
ent.set_model_options(
    factors=n_factors,
    ard_factors=True,
    ard_weights=True,
    spikeslab_weights=True
)
print(f"  Model: {n_factors} factors with ARD regularization")

# Set training options
ent.set_train_options(
    iter=n_iterations,
    convergence_mode=convergence_mode,
    gpu_mode=gpu_mode,
    seed=random_state,
    verbose=False
)
print(f"  Training: {n_iterations} iterations, {convergence_mode} convergence")

# Build and train
print(f"\\n7. Training MOFA model...")
ent.build()
ent.run()
print(f"  Training complete")

# Extract factors
print(f"\\n8. Extracting factors and loadings...")
factors = ent.model.nodes["Z"]["group1"].getExpectation()
print(f"  Factors shape: {factors.shape}")

# Extract variance explained per view
variance_explained = {}
for view_idx, view_name in enumerate(sorted(aligned_modalities.keys())):
    r2 = ent.model.calculate_variance_explained(
        views=[view_name],
        groups=["group1"]
    )
    variance_explained[view_name] = float(r2.values[0])
    print(f"  {view_name}: {variance_explained[view_name]:.1%} variance explained")

# Store results in primary modality
print(f"\\n9. Storing results...")
primary_name = modality_names[0]
primary_adata = aligned_modalities[primary_name].copy()

# Store factors
primary_adata.obsm['X_mofa'] = factors
print(f"  Stored factors in obsm['X_mofa']")

# Store loadings for primary modality
loadings = ent.model.nodes["W"][primary_name].getExpectation()
primary_adata.varm['LFs'] = loadings
print(f"  Stored loadings in varm['LFs']")

# Store metadata
primary_adata.uns['integration_method'] = 'MOFA'
primary_adata.uns['integration_params'] = {
    'n_factors': n_factors,
    'min_overlap_fraction': min_overlap_fraction,
    'memory_threshold_gb': memory_threshold_gb,
    'gpu_mode': gpu_mode,
    'convergence_mode': convergence_mode,
    'n_iterations': n_iterations,
    'random_state': random_state,
}
primary_adata.uns['integration_input_modalities'] = modality_names
primary_adata.uns['integration_n_factors'] = n_factors
primary_adata.uns['integration_variance_explained'] = variance_explained

# Summary statistics
n_original = modalities[primary_name].n_obs
n_dropped = n_original - n_samples

print(f"\\n{'='*60}")
print(f"Integration Summary")
print(f"{'='*60}")
print(f"Factors extracted: {n_factors}")
print(f"Modalities integrated: {len(modalities)}")
print(f"Samples used: {n_samples}")
print(f"Samples dropped: {n_dropped}")
print(f"Overlap fraction: {n_samples/n_original:.1%}")
print(f"Memory used: {total_gb:.2f} GB")
print(f"\\nVariance explained by modality:")
for view, r2 in variance_explained.items():
    print(f"  {view}: {r2:.1%}")

# Store in data_manager
integrated_name = f"{primary_name}_mofa_integrated"
data_manager.modalities[integrated_name] = primary_adata
print(f"\\nStored as: {integrated_name}")
""".strip()

        return AnalysisStep(
            operation="mofapy2.multi_omics_integration",
            tool_name="integrate_modalities",
            description=f"Multi-omics factor analysis integrating {len(modality_names)} modalities into {n_factors} shared latent factors",
            library="mofapy2",
            code_template=code_template,
            parameters={
                "modality_names": modality_names,
                "n_factors": n_factors,
                "min_overlap_fraction": min_overlap_fraction,
                "memory_threshold_gb": memory_threshold_gb,
                "gpu_mode": gpu_mode,
                "convergence_mode": convergence_mode,
                "n_iterations": n_iterations,
                "random_state": random_state,
            },
            parameter_schema=parameter_schema,
            imports=[
                "import numpy as np",
                "import pandas as pd",
                "from anndata import AnnData",
                "from mofapy2.run.entry_point import entry_point",
                "import h5py",
                "import tempfile",
            ],
            input_entities=modality_names,
            output_entities=["X_mofa", "LFs", "integration_metadata"],
        )
