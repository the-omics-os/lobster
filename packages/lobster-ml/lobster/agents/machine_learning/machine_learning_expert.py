"""
Machine Learning Expert Agent for ML model training with biological data.

This agent focuses on preparing biological data for machine learning tasks,
providing ML-specific tools and workflows for transcriptomics and proteomics data
using the modular DataManagerV2 system.
"""

# Agent configuration for entry point discovery (must be at top, before heavy imports)
from lobster.agents.machine_learning.config import ML_EXPERT_CONFIG

AGENT_CONFIG = ML_EXPERT_CONFIG

# === Heavy imports below ===
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from lobster.agents.machine_learning.prompts import create_ml_expert_prompt
from lobster.agents.machine_learning.state import MachineLearningExpertState
from lobster.config.llm_factory import create_llm
from lobster.config.settings import get_settings
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.services.ml.ml_preparation_service import MLPreparationService
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class MachineLearningError(Exception):
    """Base exception for machine learning operations."""

    pass


class ModalityNotFoundError(MachineLearningError):
    """Raised when requested modality doesn't exist."""

    pass


class DataPreparationError(MachineLearningError):
    """Raised when data preparation fails."""

    pass


def machine_learning_expert(
    data_manager: DataManagerV2,
    callback_handler=None,
    agent_name: str = "machine_learning_expert_agent",
    delegation_tools: List = None,
    workspace_path: Optional[Path] = None,
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
):
    """
    Create machine learning expert agent using DataManagerV2.

    Args:
        data_manager: DataManagerV2 instance for data access
        callback_handler: Optional callback for LLM events
        agent_name: Name for this agent instance
        delegation_tools: List of delegation tools for child agent handoffs
        workspace_path: Optional workspace path for LLM configuration
    """
    settings = get_settings()
    model_params = settings.get_agent_llm_params(agent_name)
    llm = create_llm(
        agent_name,
        model_params,
        provider_override=provider_override,
        model_override=model_override,
        workspace_path=workspace_path,
    )

    # Normalize callbacks to a flat list
    if callback_handler and hasattr(llm, "with_config"):
        callbacks = (
            callback_handler
            if isinstance(callback_handler, list)
            else [callback_handler]
        )
        llm = llm.with_config(callbacks=callbacks)

    # Store ML-specific results and metadata
    ml_results = {"summary": "", "details": {}}

    # -------------------------
    # DATA STATUS AND INSPECTION TOOLS
    # -------------------------
    @tool
    def check_ml_ready_modalities(modality_type: str = "all") -> str:
        """
        Check which modalities are ready for machine learning tasks.

        Args:
            modality_type: Filter by type ("transcriptomics", "proteomics", "all")

        Returns:
            str: Summary of ML-ready modalities with their characteristics
        """
        try:
            modalities = data_manager.list_modalities()
            if not modalities:
                return "No modalities loaded. Please ask the data expert to load biological datasets first."

            ml_ready_modalities = []

            ml_service = MLPreparationService()
            for mod_name in modalities:
                adata = data_manager.get_modality(mod_name)
                mod_type = ml_service._detect_modality_type(mod_name)

                # Check if modality matches requested type
                if modality_type != "all":
                    if (
                        modality_type == "transcriptomics"
                        and "rna" not in mod_type.lower()
                    ):
                        continue
                    elif (
                        modality_type == "proteomics"
                        and "proteomics" not in mod_type.lower()
                    ):
                        continue

                # Assess ML readiness
                ml_info = {
                    "name": mod_name,
                    "type": mod_type,
                    "shape": adata.shape,
                    "has_labels": any(
                        col in adata.obs.columns
                        for col in [
                            "condition",
                            "cell_type",
                            "treatment",
                            "group",
                            "label",
                            "class",
                        ]
                    ),
                    "normalized": "normalized" in mod_name or adata.X.max() <= 100,
                    "filtered": "filtered" in mod_name,
                    "clustered": any(
                        c in adata.obs.columns
                        for c in ["leiden", "louvain", "seurat_clusters", "cluster"]
                    ) or any(
                        c.startswith(("leiden_", "louvain_", "RNA_snn_res"))
                        for c in adata.obs.columns
                    ),
                }

                # Check for batch information
                ml_info["has_batch"] = any(
                    col in adata.obs.columns for col in ["batch", "sample", "donor"]
                )

                # Check data sparsity
                if hasattr(adata.X, "nnz"):
                    ml_info["sparsity"] = 1 - (
                        adata.X.nnz / (adata.X.shape[0] * adata.X.shape[1])
                    )
                else:
                    ml_info["sparsity"] = (
                        np.mean(adata.X == 0) if adata.X.size > 0 else 0
                    )

                ml_ready_modalities.append(ml_info)

            # Format response
            response = (
                f"Found {len(ml_ready_modalities)} modalities suitable for ML:\n\n"
            )

            for info in ml_ready_modalities:
                response += f"**{info['name']}** ({info['type']}):\n"
                response += f"  - Shape: {info['shape'][0]} samples x {info['shape'][1]} features\n"
                response += (
                    f"  - Labels available: {'Yes' if info['has_labels'] else 'No'}\n"
                )
                response += f"  - Normalized: {'Yes' if info['normalized'] else 'No'}\n"
                response += f"  - Filtered: {'Yes' if info['filtered'] else 'No'}\n"
                response += f"  - Sparsity: {info['sparsity']:.1%}\n"

                if info["has_batch"]:
                    response += "  - Batch info: Yes (consider batch correction)\n"

                # ML recommendations
                if info["shape"][0] < 50:
                    response += (
                        "  - Warning: Small sample size - consider regularization\n"
                    )
                elif info["shape"][0] > 10000:
                    response += "  - Note: Large dataset - suitable for deep learning\n"

                response += "\n"

            ml_results["details"]["ml_ready_check"] = response
            return response

        except Exception as e:
            logger.error(f"Error checking ML-ready modalities: {e}")
            return f"Error checking ML-ready modalities: {str(e)}"

    check_ml_ready_modalities.metadata = {"categories": ["UTILITY"], "provenance": False}
    check_ml_ready_modalities.tags = ["UTILITY"]

    @tool
    def prepare_ml_features(
        modality_name: str,
        feature_selection: str = "highly_variable",
        n_features: int = 2000,
        scale: bool = True,
        handle_zeros: str = "keep",
        save_result: bool = True,
    ) -> str:
        """
        Prepare features from biological data for machine learning.

        Args:
            modality_name: Name of the modality to process
            feature_selection: Method for feature selection
                              ("highly_variable", "pca", "all", "marker_genes")
            n_features: Number of features to select
            scale: Whether to scale features (z-score normalization)
            handle_zeros: How to handle zero values ("keep", "remove", "impute")
            save_result: Whether to save the prepared modality

        Returns:
            str: Summary of feature preparation
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
                )

            # Get the modality
            adata = data_manager.get_modality(modality_name)
            logger.info(f"Preparing ML features from '{modality_name}': {adata.shape}")

            # Copy to avoid modifying original
            import anndata

            adata_ml = adata.copy()

            # Feature selection
            if feature_selection == "highly_variable":
                try:
                    import scanpy as sc

                    if "highly_variable" not in adata_ml.var.columns:
                        sc.pp.highly_variable_genes(adata_ml, n_top_genes=n_features)
                    adata_ml = adata_ml[:, adata_ml.var.highly_variable]
                except ImportError:
                    # Fallback: variance-based selection
                    variances = np.var(adata_ml.X, axis=0)
                    if hasattr(variances, "A1"):
                        variances = variances.A1
                    top_var_indices = np.argsort(variances)[-n_features:]
                    adata_ml = adata_ml[:, top_var_indices]

            elif feature_selection == "pca":
                try:
                    import scanpy as sc

                    if "X_pca" not in adata_ml.obsm:
                        sc.pp.pca(
                            adata_ml, n_comps=min(n_features, adata_ml.shape[1] - 1)
                        )
                    pca_data = adata_ml.obsm["X_pca"][:, :n_features]
                    adata_ml = anndata.AnnData(
                        X=pca_data,
                        obs=adata_ml.obs.copy(),
                        var=pd.DataFrame(
                            index=[f"PC{i + 1}" for i in range(pca_data.shape[1])]
                        ),
                    )
                except ImportError:
                    logger.warning(
                        "Scanpy not available, using variance-based selection instead of PCA"
                    )
                    variances = np.var(adata_ml.X, axis=0)
                    if hasattr(variances, "A1"):
                        variances = variances.A1
                    top_var_indices = np.argsort(variances)[-n_features:]
                    adata_ml = adata_ml[:, top_var_indices]

            elif feature_selection == "marker_genes":
                if "rank_genes_groups" in adata_ml.uns:
                    marker_genes = []
                    groups_key = adata_ml.uns["rank_genes_groups"]["params"]["groupby"]
                    for group in adata_ml.obs[groups_key].unique():
                        genes = adata_ml.uns["rank_genes_groups"]["names"][group][
                            : n_features // 10
                        ]
                        marker_genes.extend(genes)
                    marker_genes = list(set(marker_genes))[:n_features]
                    available_markers = [
                        g for g in marker_genes if g in adata_ml.var_names
                    ]
                    if available_markers:
                        adata_ml = adata_ml[:, available_markers]
                    else:
                        logger.warning(
                            "No marker genes found, using highly variable genes"
                        )
                else:
                    logger.warning("No marker genes found, using highly variable genes")

            # Handle zeros
            if handle_zeros == "remove":
                from scipy.sparse import issparse

                if hasattr(adata_ml.X, "nnz"):
                    X_csc = adata_ml.X.tocsc() if issparse(adata_ml.X) else adata_ml.X
                    non_zeros_per_col = np.diff(X_csc.indptr)
                    zero_prop = 1.0 - (non_zeros_per_col / adata_ml.shape[0])
                else:
                    zero_prop = np.mean(adata_ml.X == 0, axis=0)
                    if hasattr(zero_prop, "A1"):
                        zero_prop = zero_prop.A1
                keep_features = zero_prop < 0.9
                adata_ml = adata_ml[:, keep_features]

            elif handle_zeros == "impute":
                if hasattr(adata_ml.X, "toarray"):
                    X_dense = adata_ml.X.toarray()
                else:
                    X_dense = adata_ml.X.copy()

                for j in range(X_dense.shape[1]):
                    col = X_dense[:, j]
                    if np.any(col == 0):
                        non_zero_mean = (
                            np.mean(col[col != 0]) if np.any(col != 0) else 0
                        )
                        col[col == 0] = non_zero_mean
                        X_dense[:, j] = col

                adata_ml.X = X_dense

            # Scale features
            if scale:
                try:
                    import scanpy as sc

                    sc.pp.scale(adata_ml, zero_center=True)
                except ImportError:
                    if hasattr(adata_ml.X, "toarray"):
                        X_dense = adata_ml.X.toarray()
                    else:
                        X_dense = adata_ml.X.copy()

                    X_dense = (X_dense - np.mean(X_dense, axis=0)) / (
                        np.std(X_dense, axis=0) + 1e-8
                    )
                    adata_ml.X = X_dense

            # Add ML metadata
            adata_ml.uns["ml_preprocessing"] = {
                "source_modality": modality_name,
                "feature_selection": feature_selection,
                "n_features_selected": adata_ml.shape[1],
                "scaled": scale,
                "zero_handling": handle_zeros,
                "original_shape": adata.shape,
            }

            # Save as new modality
            ml_modality_name = f"{modality_name}_ml_features"
            data_manager.store_modality(
                name=ml_modality_name,
                adata=adata_ml,
                parent_name=modality_name,
                step_summary=f"Prepared ML features: {adata_ml.shape[1]} features",
            )

            if save_result:
                save_path = f"{modality_name}_ml_features.h5ad"
                data_manager.save_modality(ml_modality_name, save_path)

            # Log the operation
            data_manager.log_tool_usage(
                tool_name="prepare_ml_features",
                parameters={
                    "modality_name": modality_name,
                    "feature_selection": feature_selection,
                    "n_features": n_features,
                    "scale": scale,
                    "handle_zeros": handle_zeros,
                },
                description=f"Prepared ML features: {adata_ml.shape}",
                ir=None,
            )

            # Calculate sparsity
            if hasattr(adata_ml.X, "nnz"):
                sparsity = 1.0 - (
                    adata_ml.X.nnz / (adata_ml.shape[0] * adata_ml.shape[1])
                )
            else:
                sparsity = np.mean(adata_ml.X == 0)

            response = f"""Successfully prepared features for machine learning from '{modality_name}'!

**Feature Preparation Results:**
- Original shape: {adata.shape[0]} samples x {adata.shape[1]} features
- ML-ready shape: {adata_ml.shape[0]} samples x {adata_ml.shape[1]} features
- Feature selection: {feature_selection}
- Features scaled: {"Yes" if scale else "No"}
- Zero handling: {handle_zeros}

**Feature Statistics:**
- Sparsity: {sparsity:.1%} zeros
- Value range: [{np.min(adata_ml.X):.2f}, {np.max(adata_ml.X):.2f}]

**New modality created**: '{ml_modality_name}'"""

            if save_result:
                response += f"\n**Saved to**: {save_path}"

            response += (
                "\n\nNext steps: create train/test splits or export for ML frameworks."
            )

            ml_results["details"]["feature_preparation"] = response
            return response

        except (ModalityNotFoundError, DataPreparationError) as e:
            logger.error(f"Error preparing ML features: {e}")
            return f"Error preparing ML features: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in ML feature preparation: {e}")
            return f"Unexpected error: {str(e)}"

    prepare_ml_features.metadata = {"categories": ["PREPROCESS"], "provenance": True}
    prepare_ml_features.tags = ["PREPROCESS"]

    @tool
    def create_ml_splits(
        modality_name: str,
        test_size: float = 0.2,
        validation_size: float = 0.1,
        stratify_by: Optional[str] = None,
        random_state: int = 42,
        save_result: bool = True,
    ) -> str:
        """
        Create train/test/validation splits for machine learning.

        Args:
            modality_name: Name of the modality to split
            test_size: Proportion of data for testing (0-1)
            validation_size: Proportion of training data for validation (0-1)
            stratify_by: Column name for stratified splitting (e.g., 'cell_type', 'condition')
            random_state: Random seed for reproducibility
            save_result: Whether to save the splits

        Returns:
            str: Summary of data splitting
        """
        try:
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
                )

            adata = data_manager.get_modality(modality_name)

            if stratify_by and stratify_by not in adata.obs.columns:
                available_cols = list(adata.obs.columns)
                return (
                    f"Stratification column '{stratify_by}' not found.\n\n"
                    f"Available columns: {available_cols}"
                )

            from sklearn.model_selection import train_test_split

            n_samples = adata.n_obs
            indices = np.arange(n_samples)
            stratify_labels = adata.obs[stratify_by].values if stratify_by else None

            # First split: train+val vs test
            train_val_idx, test_idx = train_test_split(
                indices,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify_labels,
            )

            # Second split: train vs val
            if validation_size > 0:
                stratify_labels_tv = (
                    stratify_labels[train_val_idx]
                    if stratify_labels is not None
                    else None
                )
                train_idx, val_idx = train_test_split(
                    train_val_idx,
                    test_size=validation_size,
                    random_state=random_state,
                    stratify=stratify_labels_tv,
                )
            else:
                train_idx = train_val_idx
                val_idx = np.array([])

            # Create split annotations
            adata.obs["ml_split"] = "unassigned"
            adata.obs.loc[adata.obs_names[train_idx], "ml_split"] = "train"
            adata.obs.loc[adata.obs_names[test_idx], "ml_split"] = "test"
            if len(val_idx) > 0:
                adata.obs.loc[adata.obs_names[val_idx], "ml_split"] = "validation"

            # Create separate modalities for each split
            adata_train = adata[train_idx, :].copy()
            adata_test = adata[test_idx, :].copy()

            train_modality = f"{modality_name}_train"
            test_modality = f"{modality_name}_test"

            data_manager.store_modality(
                name=train_modality,
                adata=adata_train,
                parent_name=modality_name,
                step_summary=f"Train split: {len(train_idx)} samples",
            )
            data_manager.store_modality(
                name=test_modality,
                adata=adata_test,
                parent_name=modality_name,
                step_summary=f"Test split: {len(test_idx)} samples",
            )

            if len(val_idx) > 0:
                adata_val = adata[val_idx, :].copy()
                val_modality = f"{modality_name}_validation"
                data_manager.store_modality(
                    name=val_modality,
                    adata=adata_val,
                    parent_name=modality_name,
                    step_summary=f"Validation split: {len(val_idx)} samples",
                )

            data_manager.store_modality(
                name=modality_name,
                adata=adata,
                step_summary="Added ML split info",
            )

            if save_result:
                data_manager.save_modality(train_modality, f"{train_modality}.h5ad")
                data_manager.save_modality(test_modality, f"{test_modality}.h5ad")
                if len(val_idx) > 0:
                    data_manager.save_modality(val_modality, f"{val_modality}.h5ad")

            data_manager.log_tool_usage(
                tool_name="create_ml_splits",
                parameters={
                    "modality_name": modality_name,
                    "test_size": test_size,
                    "validation_size": validation_size,
                    "stratify_by": stratify_by,
                    "random_state": random_state,
                },
                description=f"Created ML splits: train={len(train_idx)}, test={len(test_idx)}, val={len(val_idx)}",
                ir=None,
            )

            split_stats = {
                "train": len(train_idx),
                "test": len(test_idx),
                "validation": len(val_idx) if len(val_idx) > 0 else 0,
            }

            response = f"""Successfully created ML data splits for '{modality_name}'!

**Split Statistics:**
- Training set: {split_stats["train"]} samples ({split_stats["train"] / n_samples * 100:.1f}%)
- Test set: {split_stats["test"]} samples ({split_stats["test"] / n_samples * 100:.1f}%)"""

            if split_stats["validation"] > 0:
                response += f"\n- Validation set: {split_stats['validation']} samples ({split_stats['validation'] / n_samples * 100:.1f}%)"

            if stratify_by:
                response += f"\n- Stratified by: {stratify_by}"

            response += "\n\n**New modalities created**:"
            response += f"\n- '{train_modality}' (training data)"
            response += f"\n- '{test_modality}' (test data)"
            if split_stats["validation"] > 0:
                response += f"\n- '{val_modality}' (validation data)"

            ml_results["details"]["data_splitting"] = response
            return response

        except Exception as e:
            logger.error(f"Error creating ML splits: {e}")
            return f"Error creating ML splits: {str(e)}"

    create_ml_splits.metadata = {"categories": ["PREPROCESS"], "provenance": True}
    create_ml_splits.tags = ["PREPROCESS"]

    @tool
    def export_for_ml_framework(
        modality_name: str,
        format: str = "numpy",
        include_labels: bool = True,
        label_column: Optional[str] = None,
        output_dir: str = "ml_exports",
    ) -> str:
        """
        Export biological data in formats suitable for ML frameworks.

        Args:
            modality_name: Name of the modality to export
            format: Export format ("numpy", "csv", "pytorch", "tensorflow")
            include_labels: Whether to export labels/targets
            label_column: Column name containing labels
            output_dir: Directory for exported files

        Returns:
            str: Summary of exported files
        """
        try:
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
                )

            adata = data_manager.get_modality(modality_name)

            import os

            export_path = data_manager.exports_dir / output_dir
            export_path.mkdir(exist_ok=True)

            # Get features (X matrix)
            if hasattr(adata.X, "toarray"):
                X = adata.X.toarray()
            else:
                X = adata.X.copy()

            # Get labels if requested
            y = None
            if include_labels and label_column:
                if label_column not in adata.obs.columns:
                    _label_candidates = {
                        "condition", "cell_type", "treatment", "group",
                        "leiden", "louvain", "seurat_clusters",
                    }
                    potential_labels = [
                        col
                        for col in adata.obs.columns
                        if col in _label_candidates
                    ]
                    if potential_labels:
                        label_column = potential_labels[0]
                        logger.info(f"Using '{label_column}' as label column")
                    else:
                        include_labels = False
                        logger.warning("No suitable label column found")

                if include_labels:
                    labels = adata.obs[label_column]
                    if labels.dtype == "object" or labels.dtype.name == "category":
                        from sklearn.preprocessing import LabelEncoder

                        le = LabelEncoder()
                        y = le.fit_transform(labels)

                        label_mapping = dict(
                            zip(le.classes_, le.transform(le.classes_))
                        )
                        import json

                        with open(
                            export_path / f"{modality_name}_label_mapping.json", "w"
                        ) as f:
                            json.dump(
                                {str(k): int(v) for k, v in label_mapping.items()},
                                f,
                                indent=2,
                            )
                    else:
                        y = labels.values

            exported_files = []

            if format == "numpy":
                np.save(export_path / f"{modality_name}_features.npy", X)
                exported_files.append(f"{modality_name}_features.npy")

                if y is not None:
                    np.save(export_path / f"{modality_name}_labels.npy", y)
                    exported_files.append(f"{modality_name}_labels.npy")

                np.save(
                    export_path / f"{modality_name}_feature_names.npy",
                    adata.var_names.values,
                )
                exported_files.append(f"{modality_name}_feature_names.npy")

                np.save(
                    export_path / f"{modality_name}_sample_names.npy",
                    adata.obs_names.values,
                )
                exported_files.append(f"{modality_name}_sample_names.npy")

            elif format == "csv":
                feature_df = pd.DataFrame(
                    X, index=adata.obs_names, columns=adata.var_names
                )

                if y is not None:
                    feature_df["_label"] = y

                csv_path = export_path / f"{modality_name}_ml_data.csv"
                feature_df.to_csv(csv_path)
                exported_files.append(f"{modality_name}_ml_data.csv")

            elif format == "pytorch":
                try:
                    import torch

                    X_tensor = torch.FloatTensor(X)
                    torch.save(X_tensor, export_path / f"{modality_name}_features.pt")
                    exported_files.append(f"{modality_name}_features.pt")

                    if y is not None:
                        y_tensor = torch.LongTensor(y)
                        torch.save(y_tensor, export_path / f"{modality_name}_labels.pt")
                        exported_files.append(f"{modality_name}_labels.pt")

                    metadata = {
                        "shape": list(X.shape),
                        "feature_names": adata.var_names.tolist(),
                        "sample_names": adata.obs_names.tolist(),
                        "label_column": label_column if y is not None else None,
                    }
                    torch.save(metadata, export_path / f"{modality_name}_metadata.pt")
                    exported_files.append(f"{modality_name}_metadata.pt")

                except ImportError:
                    return "PyTorch not installed. Please install torch to export in PyTorch format."

            elif format == "tensorflow":
                try:
                    save_dict = {"features": X}
                    if y is not None:
                        save_dict["labels"] = y
                    save_dict["feature_names"] = adata.var_names.values
                    save_dict["sample_names"] = adata.obs_names.values

                    np.savez_compressed(
                        export_path / f"{modality_name}_tf_data.npz", **save_dict
                    )
                    exported_files.append(f"{modality_name}_tf_data.npz")

                except Exception as e:
                    return f"Error exporting for TensorFlow: {str(e)}"

            data_manager.log_tool_usage(
                tool_name="export_for_ml_framework",
                parameters={
                    "modality_name": modality_name,
                    "format": format,
                    "include_labels": include_labels,
                    "label_column": label_column,
                    "output_dir": output_dir,
                },
                description=f"Exported {len(exported_files)} files for {format}",
            )

            response = f"""Successfully exported '{modality_name}' for {format}!

**Export Summary:**
- Data shape: {X.shape[0]} samples x {X.shape[1]} features
- Format: {format}
- Labels included: {"Yes" if y is not None else "No"}"""

            if y is not None:
                response += f"\n- Label column: {label_column}"
                response += f"\n- Number of classes: {len(np.unique(y))}"

            response += f"\n\n**Exported files ({len(exported_files)}):**"
            for file in exported_files:
                file_path = export_path / file
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                response += f"\n- {file} ({size_mb:.1f} MB)"

            response += f"\n\n**Export directory**: {export_path}"

            ml_results["details"]["ml_export"] = response
            return response

        except Exception as e:
            logger.error(f"Error exporting for ML framework: {e}")
            return f"Error exporting for ML framework: {str(e)}"

    export_for_ml_framework.metadata = {"categories": ["UTILITY"], "provenance": False}
    export_for_ml_framework.tags = ["UTILITY"]

    @tool
    def create_ml_analysis_summary() -> str:
        """Create a comprehensive summary of all ML preprocessing steps performed."""
        try:
            if not ml_results["details"]:
                return "No ML preprocessing steps have been performed yet. Run some ML tools first."

            summary = "# Machine Learning Data Preparation Summary\n\n"

            for step, details in ml_results["details"].items():
                summary += f"## {step.replace('_', ' ').title()}\n"
                summary += f"{details}\n\n"

            modalities = data_manager.list_modalities()
            if modalities:
                ml_modalities = [
                    mod
                    for mod in modalities
                    if "ml_features" in mod.lower()
                    or "train" in mod.lower()
                    or "test" in mod.lower()
                ]

                summary += "## Current ML-Ready Modalities\n"
                summary += f"ML-prepared modalities ({len(ml_modalities)}): {', '.join(ml_modalities)}\n\n"

                summary += "### ML Modality Details:\n"
                for mod_name in ml_modalities:
                    try:
                        adata = data_manager.get_modality(mod_name)
                        summary += f"- **{mod_name}**: {adata.n_obs} samples x {adata.n_vars} features\n"

                        key_cols = [
                            col
                            for col in adata.obs.columns
                            if col.lower()
                            in [
                                "ml_split",
                                "condition",
                                "cell_type",
                                "treatment",
                                "group",
                            ]
                        ]
                        if key_cols:
                            summary += f"  - ML annotations: {', '.join(key_cols)}\n"
                    except Exception:
                        summary += f"- **{mod_name}**: Error accessing modality\n"

            ml_results["summary"] = summary
            logger.info(
                f"Created ML analysis summary with {len(ml_results['details'])} processing steps"
            )
            return summary

        except Exception as e:
            logger.error(f"Error creating ML analysis summary: {e}")
            return f"Error creating ML summary: {str(e)}"

    create_ml_analysis_summary.metadata = {"categories": ["UTILITY"], "provenance": False}
    create_ml_analysis_summary.tags = ["UTILITY"]

    # -------------------------
    # DEEP LEARNING EMBEDDING TOOLS (scVI Integration)
    # -------------------------
    @tool
    def check_scvi_availability() -> str:
        """
        Check if scVI dependencies are available and provide installation instructions.

        Returns:
            str: Status message with availability and installation guidance
        """
        try:
            from lobster.services.analysis.scvi_embedding_service import (
                ScviEmbeddingService,
            )

            service = ScviEmbeddingService()
            availability_info = service.check_availability()

            if availability_info["ready_for_scvi"]:
                device = availability_info["hardware_recommendation"]["device"]
                info = availability_info["hardware_recommendation"]["info"]

                return f"""scVI is ready for deep learning embeddings!

**Hardware Configuration:**
- Compute device: {device.upper()}
- Hardware info: {info}

**Available Features:**
- Deep learning-based dimensionality reduction
- Batch correction with scVI models
- State-of-the-art single-cell embeddings
- GPU acceleration (if available)

You can now use `train_scvi_embedding()` to create deep learning embeddings."""
            else:
                hardware_rec = availability_info["hardware_recommendation"]
                missing = []
                if not availability_info["torch_available"]:
                    missing.append("PyTorch")
                if not availability_info["scvi_available"]:
                    missing.append("scVI")

                return f"""scVI dependencies not available

**Missing Dependencies:** {", ".join(missing)}

**Installation Instructions:**
{hardware_rec["command"]}

**Hardware Detected:** {hardware_rec["info"]}

After installation, restart your session and run this tool again."""

        except Exception as e:
            logger.error(f"Error checking scVI availability: {e}")
            return f"Error checking scVI availability: {str(e)}"

    check_scvi_availability.metadata = {"categories": ["UTILITY"], "provenance": False}
    check_scvi_availability.tags = ["UTILITY"]

    @tool
    def train_scvi_embedding(
        modality_name: str,
        n_latent: int = 10,
        n_layers: int = 2,
        n_hidden: int = 128,
        max_epochs: int = 400,
        batch_key: Optional[str] = None,
        use_gpu: bool = True,
        save_model: bool = True,
        batch_size: int = 128,
        early_stopping_patience: int = 10,
        dropout_rate: float = 0.1,
        gene_likelihood: str = "zinb",
    ) -> str:
        """
        Train scVI model for deep learning-based embedding and dimensionality reduction.

        Args:
            modality_name: Name of the modality to process
            n_latent: Number of latent dimensions (embedding size, default: 10)
            n_layers: Number of hidden layers in the neural network (default: 2)
            n_hidden: Number of hidden units per layer (default: 128)
            max_epochs: Maximum training epochs (default: 400)
            batch_key: Column name for batch correction (optional)
            use_gpu: Whether to use GPU if available (default: True)
            save_model: Whether to save the trained model (default: True)
            batch_size: Training batch size (default: 128)
            early_stopping_patience: Epochs to wait before early stopping (default: 10)
            dropout_rate: Dropout rate for regularization (default: 0.1)
            gene_likelihood: Gene expression likelihood ("nb", "zinb", "poisson", default: "zinb")

        Returns:
            str: Summary of scVI training results with embedding information
        """
        try:
            from lobster.services.analysis.scvi_embedding_service import (
                ScviEmbeddingService,
            )

            service = ScviEmbeddingService()
            if not service.check_availability()["ready_for_scvi"]:
                return check_scvi_availability()

            if modality_name not in data_manager.list_modalities():
                available = data_manager.list_modalities()
                return f"Modality '{modality_name}' not found.\n\nAvailable modalities: {', '.join(available)}"

            adata = data_manager.get_modality(modality_name)
            logger.info(f"Training scVI embedding for '{modality_name}': {adata.shape}")

            if batch_key and batch_key not in adata.obs.columns:
                available_cols = [
                    col
                    for col in adata.obs.columns
                    if any(
                        keyword in col.lower()
                        for keyword in ["batch", "sample", "donor", "replicate"]
                    )
                ]
                return f"Batch key '{batch_key}' not found.\n\nAvailable batch-related columns: {available_cols}"

            model_save_path = None
            if save_model:
                if data_manager.workspace_path:
                    models_dir = Path(data_manager.workspace_path) / "models"
                    models_dir.mkdir(exist_ok=True)
                    model_save_path = str(models_dir / f"{modality_name}_scvi_model")
                else:
                    model_save_path = f"{modality_name}_scvi_model"

            model_kwargs = {
                "n_layers": n_layers,
                "n_hidden": n_hidden,
                "max_epochs": max_epochs,
                "early_stopping_patience": early_stopping_patience,
                "batch_size": batch_size,
                "dropout_rate": dropout_rate,
                "gene_likelihood": gene_likelihood,
            }

            result = service.train_scvi_embedding(
                adata=adata,
                batch_key=batch_key,
                force_cpu=not use_gpu,
                save_path=model_save_path,
                **model_kwargs,
            )

            # Handle error case (returns 2-tuple with None, error_info)
            if len(result) == 2:
                model, training_info = result
                error_type = training_info.get("error_type", "unknown_error")
                error_msg = training_info.get("error", "Training failed")
                return f"scVI training failed: {error_type} - {error_msg}"

            # Success case (returns 3-tuple: adata, stats, ir)
            adata, training_info, ir = result

            data_manager.modalities[modality_name] = adata

            data_manager.log_tool_usage(
                tool_name="train_scvi_embedding",
                parameters={
                    "modality_name": modality_name,
                    "n_latent": n_latent,
                    "max_epochs": max_epochs,
                    "batch_key": batch_key,
                    "device": training_info.get("device"),
                },
                description=f"Trained scVI model with {n_latent} latent dimensions",
                ir=ir,
            )

            response = f"""Successfully trained scVI embedding for '{modality_name}'!

**Deep Learning Model:**
- Architecture: scVI (single-cell Variational Inference)
- Latent dimensions: {training_info["n_latent"]}
- Hidden layers: {n_layers} x {n_hidden} units
- Training device: {training_info["device"].upper()}

**Training Results:**
- Dataset: {training_info["n_cells"]:,} cells x {training_info["n_genes"]:,} genes
- Embedding shape: {training_info["embedding_shape"]}
- Embeddings stored in: obsm['X_scvi']"""

            if batch_key:
                response += f"\n- Batch correction: Yes (using '{batch_key}')"

            if training_info["model_saved"]:
                response += "\n- Model saved: Yes"

            response += """

**Next Steps:**
The scVI embeddings are now available in modality.obsm['X_scvi'] and can be used for:
- Clustering with custom embeddings (set use_rep='X_scvi')
- Visualization (UMAP/t-SNE on scVI space)
- Batch-corrected downstream analysis"""

            ml_results["details"]["scvi_training"] = response
            return response

        except ImportError as e:
            return f"scVI dependencies not available: {str(e)}\n\nRun `check_scvi_availability()` for installation instructions."
        except Exception as e:
            logger.error(f"Error training scVI embedding: {e}")
            return f"Error training scVI embedding: {str(e)}"

    train_scvi_embedding.metadata = {"categories": ["ANALYZE"], "provenance": True}
    train_scvi_embedding.tags = ["ANALYZE"]

    # -------------------------
    # TOOL REGISTRY
    # -------------------------
    base_tools = [
        check_ml_ready_modalities,
        prepare_ml_features,
        create_ml_splits,
        export_for_ml_framework,
        create_ml_analysis_summary,
        check_scvi_availability,
        train_scvi_embedding,
    ]

    tools = base_tools + (delegation_tools or [])

    # -------------------------
    # SYSTEM PROMPT
    # -------------------------
    system_prompt = create_ml_expert_prompt()

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        name=agent_name,
        state_schema=MachineLearningExpertState,
    )
