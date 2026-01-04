"""
ML Preparation Service for machine learning workflow support.

This service provides stateless ML data preparation operations including:
- ML readiness assessment
- Feature engineering (selection, normalization, scaling)
- Train/validation/test splits (stratified)
- Framework-specific export (sklearn, pytorch, tensorflow)
- ML workflow summaries

All methods return 3-tuples: (result, stats, AnalysisStep) for provenance tracking.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

from lobster.core.analysis_ir import AnalysisStep

logger = logging.getLogger(__name__)


class MLPreparationService:
    """
    Stateless service for ML data preparation workflows.

    This service operates on AnnData objects and returns processed results
    without maintaining internal state. All methods follow the 3-tuple pattern:
    (result, stats, AnalysisStep) for consistency with Lobster's service architecture.
    """

    def __init__(self):
        """Initialize MLPreparationService (stateless)."""
        pass

    # =========================================================================
    # PUBLIC API: ML Readiness Assessment
    # =========================================================================

    def check_ml_readiness(
        self, adata: "AnnData", modality_name: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any], AnalysisStep]:
        """
        Check if data is ready for machine learning workflows.

        Args:
            adata: AnnData object to assess
            modality_name: Name of the modality for context

        Returns:
            Tuple of (readiness_report, stats, AnalysisStep)
        """
        # Basic data structure checks
        checks = {
            "has_expression_data": adata.X is not None,
            "sufficient_samples": adata.n_obs >= 10,
            "sufficient_features": adata.n_vars >= 50,
            "no_missing_values": not np.any(
                pd.isna(adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X)
            ),
            "has_metadata": len(adata.obs.columns) > 0,
            "numeric_data": True,  # Assume true for now
        }

        # Advanced checks based on modality type
        modality_type = self._detect_modality_type(modality_name)

        if modality_type in ["single_cell_rna_seq", "bulk_rna_seq"]:
            # Transcriptomics-specific checks
            checks.update(
                {
                    "gene_symbols_available": "gene_symbols" in adata.var.columns
                    or adata.var.index.dtype == "object",
                    "count_data": (
                        np.all(adata.X >= 0) if hasattr(adata.X, "__iter__") else True
                    ),
                    "reasonable_gene_count": 500 <= adata.n_vars <= 50000,
                }
            )
        elif "proteomics" in modality_type:
            # Proteomics-specific checks
            checks.update(
                {
                    "protein_identifiers": len(adata.var.index) > 0,
                    "reasonable_protein_count": 10 <= adata.n_vars <= 10000,
                    "positive_values": (
                        np.all(adata.X >= 0) if hasattr(adata.X, "__iter__") else True
                    ),
                }
            )

        # Calculate overall readiness score
        passed_checks = sum(checks.values())
        total_checks = len(checks)
        readiness_score = passed_checks / total_checks

        # Determine readiness level
        if readiness_score >= 0.9:
            readiness_level = "excellent"
        elif readiness_score >= 0.75:
            readiness_level = "good"
        elif readiness_score >= 0.5:
            readiness_level = "fair"
        else:
            readiness_level = "poor"

        # Compile results
        result = {
            "modality_type": modality_type,
            "shape": adata.shape,
            "readiness_score": readiness_score,
            "readiness_level": readiness_level,
            "checks": checks,
            "recommendations": self._generate_ml_recommendations(checks, modality_type),
        }

        stats = {
            "modality_name": modality_name,
            "readiness_score": readiness_score,
            "readiness_level": readiness_level,
            "passed_checks": passed_checks,
            "total_checks": total_checks,
            "timestamp": datetime.now().isoformat(),
        }

        ir = self._create_readiness_ir(modality_name, readiness_score, readiness_level)

        logger.info(
            f"ML readiness check for {modality_name}: {readiness_level} "
            f"({readiness_score:.2%})"
        )

        return result, stats, ir

    # =========================================================================
    # PUBLIC API: Feature Preparation
    # =========================================================================

    def prepare_ml_features(
        self,
        adata: "AnnData",
        modality_name: str,
        feature_selection: str = "variance",
        n_features: int = 2000,
        normalization: str = "log1p",
        scaling: str = "standard",
    ) -> Tuple["AnnData", Dict[str, Any], AnalysisStep]:
        """
        Prepare ML-ready feature matrices from biological data.

        Args:
            adata: AnnData object to process
            modality_name: Name of the modality
            feature_selection: Method ('variance', 'correlation', 'chi2', 'mutual_info')
            n_features: Number of features to select
            normalization: Method ('log1p', 'cpm', 'none')
            scaling: Method ('standard', 'minmax', 'robust', 'none')

        Returns:
            Tuple of (processed_adata, stats, AnalysisStep)
        """
        try:
            from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
        except ImportError:
            raise ImportError(
                "scikit-learn is required for ML feature preparation. "
                "Install with: pip install scikit-learn"
            )

        # Import anndata for creating new object
        try:
            import anndata as ad
        except ImportError:
            raise ImportError("anndata is required. Install with: pip install anndata")

        # Try to import scanpy
        try:
            import scanpy as sc

            scanpy_available = True
        except ImportError:
            scanpy_available = False
            logger.warning("Scanpy not available - using basic processing")

        adata_copy = adata.copy()
        processing_steps = []

        # Step 1: Normalization
        if normalization == "log1p":
            if scanpy_available:
                sc.pp.normalize_total(adata_copy, target_sum=1e4)
                sc.pp.log1p(adata_copy)
            else:
                X = (
                    adata_copy.X.toarray()
                    if hasattr(adata_copy.X, "toarray")
                    else adata_copy.X
                )
                X = np.log1p(X / np.sum(X, axis=1, keepdims=True) * 1e4)
                adata_copy.X = X
            processing_steps.append(f"Applied {normalization} normalization")
        elif normalization == "cpm":
            if scanpy_available:
                sc.pp.normalize_total(adata_copy, target_sum=1e6)
            else:
                X = (
                    adata_copy.X.toarray()
                    if hasattr(adata_copy.X, "toarray")
                    else adata_copy.X
                )
                X = X / np.sum(X, axis=1, keepdims=True) * 1e6
                adata_copy.X = X
            processing_steps.append("Applied CPM normalization")

        # Step 2: Feature Selection
        X = (
            adata_copy.X.toarray()
            if hasattr(adata_copy.X, "toarray")
            else adata_copy.X
        )

        if feature_selection == "variance" and scanpy_available:
            sc.pp.highly_variable_genes(
                adata_copy, n_top_genes=min(n_features, adata_copy.n_vars)
            )
            selected_features = adata_copy.var["highly_variable"]
            selected_indices = np.where(selected_features)[0]
        elif feature_selection == "variance":
            variances = np.var(X, axis=0)
            selected_indices = np.argsort(variances)[-n_features:]
        else:
            variances = np.var(X, axis=0)
            selected_indices = np.argsort(variances)[-n_features:]

        X_selected = X[:, selected_indices]
        selected_feature_names = adata_copy.var_names[selected_indices]
        processing_steps.append(
            f"Selected {len(selected_indices)} features using {feature_selection}"
        )

        # Step 3: Scaling
        if scaling == "standard":
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_selected)
            processing_steps.append("Applied standard scaling")
        elif scaling == "minmax":
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X_selected)
            processing_steps.append("Applied min-max scaling")
        elif scaling == "robust":
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X_selected)
            processing_steps.append("Applied robust scaling")
        else:
            X_scaled = X_selected
            scaler = None

        # Create new AnnData with processed features
        adata_processed = ad.AnnData(
            X=X_scaled,
            obs=adata_copy.obs.copy(),
            var=pd.DataFrame(index=selected_feature_names),
        )

        # Add processing metadata
        adata_processed.uns["ml_processing"] = {
            "source_modality": modality_name,
            "feature_selection": feature_selection,
            "n_features_selected": len(selected_indices),
            "normalization": normalization,
            "scaling": scaling,
            "processing_steps": processing_steps,
            "selected_indices": selected_indices.tolist(),
            "original_feature_names": list(adata.var_names),
            "timestamp": datetime.now().isoformat(),
        }

        stats = {
            "source_modality": modality_name,
            "original_shape": adata.shape,
            "processed_shape": adata_processed.shape,
            "n_features_selected": len(selected_indices),
            "processing_steps": processing_steps,
            "timestamp": datetime.now().isoformat(),
        }

        ir = self._create_feature_prep_ir(
            modality_name,
            feature_selection,
            n_features,
            normalization,
            scaling,
        )

        logger.info(
            f"ML feature preparation: {modality_name} "
            f"{adata.shape} -> {adata_processed.shape}"
        )

        return adata_processed, stats, ir

    # =========================================================================
    # PUBLIC API: Train/Validation/Test Splits
    # =========================================================================

    def create_ml_splits(
        self,
        adata: "AnnData",
        modality_name: str,
        target_column: Optional[str] = None,
        test_size: float = 0.2,
        validation_size: float = 0.1,
        stratify: bool = True,
        random_state: int = 42,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], AnalysisStep]:
        """
        Create stratified train/validation/test splits for ML workflows.

        Args:
            adata: AnnData object to split
            modality_name: Name of the modality
            target_column: Column in obs for stratification (None = random)
            test_size: Proportion for test set
            validation_size: Proportion for validation set
            stratify: Whether to stratify by target_column
            random_state: Random seed

        Returns:
            Tuple of (split_metadata, stats, AnalysisStep)
        """
        try:
            from sklearn.model_selection import train_test_split
        except ImportError:
            raise ImportError(
                "scikit-learn required for ML splits. Install with: pip install scikit-learn"
            )

        n_samples = adata.n_obs
        sample_indices = np.arange(n_samples)

        # Prepare target and stratification
        if target_column and target_column in adata.obs.columns:
            target = adata.obs[target_column].values
            if stratify:
                unique_classes, counts = np.unique(target, return_counts=True)
                min_class_count = np.min(counts)
                min_required = max(2, int(1 / min(test_size, validation_size)))

                if min_class_count < min_required:
                    logger.warning(
                        f"Insufficient samples for stratification. "
                        f"Min class: {min_class_count}, need: {min_required}"
                    )
                    stratify = False
                    target = None
        else:
            target = None
            stratify = False

        # Create train/temp split
        if stratify and target is not None:
            train_idx, temp_idx = train_test_split(
                sample_indices,
                test_size=test_size + validation_size,
                stratify=target,
                random_state=random_state,
            )
            temp_target = target[temp_idx]
        else:
            train_idx, temp_idx = train_test_split(
                sample_indices,
                test_size=test_size + validation_size,
                random_state=random_state,
            )
            temp_target = None

        # Create validation/test split
        if len(temp_idx) > 1 and validation_size > 0:
            val_test_ratio = test_size / (test_size + validation_size)

            if stratify and temp_target is not None:
                val_idx, test_idx = train_test_split(
                    temp_idx,
                    test_size=val_test_ratio,
                    stratify=temp_target,
                    random_state=random_state,
                )
            else:
                val_idx, test_idx = train_test_split(
                    temp_idx, test_size=val_test_ratio, random_state=random_state
                )
        else:
            test_idx = temp_idx
            val_idx = np.array([])

        # Create split metadata
        splits = {
            "train": {
                "indices": train_idx.tolist(),
                "size": len(train_idx),
                "proportion": len(train_idx) / n_samples,
            },
            "validation": (
                {
                    "indices": val_idx.tolist(),
                    "size": len(val_idx),
                    "proportion": len(val_idx) / n_samples,
                }
                if len(val_idx) > 0
                else None
            ),
            "test": {
                "indices": test_idx.tolist(),
                "size": len(test_idx),
                "proportion": len(test_idx) / n_samples,
            },
        }

        # Add target distribution
        if target is not None:
            for split_name, split_info in splits.items():
                if split_info is not None:
                    split_indices = split_info["indices"]
                    split_target = target[split_indices]
                    unique, counts = np.unique(split_target, return_counts=True)
                    split_info["target_distribution"] = dict(zip(unique, counts.tolist()))

        result = {
            "modality": modality_name,
            "target_column": target_column,
            "stratified": stratify,
            "test_size": test_size,
            "validation_size": validation_size,
            "random_state": random_state,
            "n_samples": n_samples,
            "splits": splits,
            "timestamp": datetime.now().isoformat(),
        }

        stats = {
            "modality_name": modality_name,
            "train_size": len(train_idx),
            "validation_size": len(val_idx),
            "test_size": len(test_idx),
            "stratified": stratify,
            "timestamp": datetime.now().isoformat(),
        }

        ir = self._create_splits_ir(
            modality_name, target_column, test_size, validation_size, stratify, random_state
        )

        logger.info(
            f"ML splits created for {modality_name}: "
            f"train({len(train_idx)}) / val({len(val_idx)}) / test({len(test_idx)})"
        )

        return result, stats, ir

    # =========================================================================
    # PUBLIC API: Framework Export
    # =========================================================================

    def export_for_ml_framework(
        self,
        adata: "AnnData",
        modality_name: str,
        output_dir: Path,
        framework: str = "sklearn",
        split: Optional[str] = None,
        target_column: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], AnalysisStep]:
        """
        Export data in formats suitable for ML frameworks.

        Args:
            adata: AnnData object to export
            modality_name: Name of the modality
            output_dir: Directory to save exports
            framework: Target framework ('sklearn', 'pytorch', 'tensorflow', 'xgboost')
            split: Specific split ('train', 'validation', 'test', or None for all)
            target_column: Target column for supervised learning

        Returns:
            Tuple of (export_info, stats, AnalysisStep)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get data matrix
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X

        # Get target
        y = None
        if target_column and target_column in adata.obs.columns:
            y = adata.obs[target_column].values

        # Get splits
        splits_info = adata.uns.get("ml_splits", {})
        splits = splits_info.get("splits", {})

        result = {
            "modality": modality_name,
            "framework": framework,
            "shape": X.shape,
            "has_target": y is not None,
            "target_column": target_column,
            "export_timestamp": datetime.now().isoformat(),
            "files": {},
        }

        # Framework-specific export
        if framework == "sklearn":
            if split is None:
                np.save(output_dir / f"{modality_name}_X.npy", X)
                result["files"]["features"] = str(output_dir / f"{modality_name}_X.npy")
                if y is not None:
                    np.save(output_dir / f"{modality_name}_y.npy", y)
                    result["files"]["target"] = str(output_dir / f"{modality_name}_y.npy")
            else:
                if split in splits:
                    indices = splits[split]["indices"]
                    X_split = X[indices]
                    np.save(output_dir / f"{modality_name}_{split}_X.npy", X_split)
                    result["files"][f"{split}_features"] = str(
                        output_dir / f"{modality_name}_{split}_X.npy"
                    )
                    if y is not None:
                        y_split = y[indices]
                        np.save(output_dir / f"{modality_name}_{split}_y.npy", y_split)
                        result["files"][f"{split}_target"] = str(
                            output_dir / f"{modality_name}_{split}_y.npy"
                        )

        elif framework == "pytorch":
            try:
                import torch

                if split is None:
                    X_tensor = torch.FloatTensor(X)
                    torch.save(X_tensor, output_dir / f"{modality_name}_X.pt")
                    result["files"]["features"] = str(output_dir / f"{modality_name}_X.pt")
                    if y is not None:
                        y_tensor = (
                            torch.LongTensor(y)
                            if y.dtype.kind in ["i", "u"]
                            else torch.FloatTensor(y)
                        )
                        torch.save(y_tensor, output_dir / f"{modality_name}_y.pt")
                        result["files"]["target"] = str(output_dir / f"{modality_name}_y.pt")
                else:
                    if split in splits:
                        indices = splits[split]["indices"]
                        X_split = X[indices]
                        X_tensor = torch.FloatTensor(X_split)
                        torch.save(X_tensor, output_dir / f"{modality_name}_{split}_X.pt")
                        result["files"][f"{split}_features"] = str(
                            output_dir / f"{modality_name}_{split}_X.pt"
                        )
                        if y is not None:
                            y_split = y[indices]
                            y_tensor = (
                                torch.LongTensor(y_split)
                                if y_split.dtype.kind in ["i", "u"]
                                else torch.FloatTensor(y_split)
                            )
                            torch.save(y_tensor, output_dir / f"{modality_name}_{split}_y.pt")
                            result["files"][f"{split}_target"] = str(
                                output_dir / f"{modality_name}_{split}_y.pt"
                            )
            except ImportError:
                logger.warning("PyTorch not available, falling back to NumPy")
                framework = "sklearn"

        # Export metadata
        metadata = {
            "feature_names": list(adata.var_names),
            "sample_names": list(adata.obs_names),
            "shape": X.shape,
            "modality_metadata": dict(adata.obs.dtypes.astype(str)),
            "processing_info": adata.uns.get("ml_processing", {}),
            "splits_info": splits_info,
        }

        metadata_path = output_dir / f"{modality_name}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        result["files"]["metadata"] = str(metadata_path)

        stats = {
            "modality_name": modality_name,
            "framework": framework,
            "files_created": len(result["files"]),
            "output_dir": str(output_dir),
            "timestamp": datetime.now().isoformat(),
        }

        ir = self._create_export_ir(modality_name, framework, split, target_column)

        logger.info(
            f"ML export completed: {modality_name} -> {framework} "
            f"({len(result['files'])} files)"
        )

        return result, stats, ir

    # =========================================================================
    # PUBLIC API: ML Summary
    # =========================================================================

    def get_ml_summary(
        self, adata: "AnnData", modality_name: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any], None]:
        """
        Get comprehensive ML workflow summary.

        Args:
            adata: AnnData object to summarize
            modality_name: Name of the modality

        Returns:
            Tuple of (summary, stats, None) - No IR for query operations
        """
        summary = {
            "modality_name": modality_name,
            "modality_type": self._detect_modality_type(modality_name),
            "shape": adata.shape,
            "data_type": str(adata.X.dtype) if hasattr(adata.X, "dtype") else "unknown",
        }

        # Feature processing info
        if "ml_processing" in adata.uns:
            processing_info = adata.uns["ml_processing"]
            summary["feature_processing"] = {
                "processed": True,
                "source_modality": processing_info.get("source_modality"),
                "n_features_selected": processing_info.get("n_features_selected"),
                "processing_steps": processing_info.get("processing_steps", []),
            }
        else:
            summary["feature_processing"] = {"processed": False}

        # Splits info
        if "ml_splits" in adata.uns:
            splits_info = adata.uns["ml_splits"]
            splits = splits_info.get("splits", {})
            summary["splits"] = {
                "created": True,
                "stratified": splits_info.get("stratified", False),
                "target_column": splits_info.get("target_column"),
                "train_size": splits.get("train", {}).get("size", 0),
                "validation_size": (
                    splits.get("validation", {}).get("size", 0)
                    if splits.get("validation")
                    else 0
                ),
                "test_size": splits.get("test", {}).get("size", 0),
            }
        else:
            summary["splits"] = {"created": False}

        # Metadata columns
        categorical_columns = []
        numerical_columns = []
        for col in adata.obs.columns:
            if adata.obs[col].dtype.kind in ["O", "S"]:
                categorical_columns.append(col)
            elif adata.obs[col].dtype.kind in ["i", "u", "f"]:
                numerical_columns.append(col)

        summary["metadata"] = {
            "categorical_columns": categorical_columns,
            "numerical_columns": numerical_columns,
            "total_metadata_columns": len(adata.obs.columns),
        }

        stats = {
            "modality_name": modality_name,
            "has_processing": "ml_processing" in adata.uns,
            "has_splits": "ml_splits" in adata.uns,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"ML summary generated for {modality_name}")

        return summary, stats, None  # No IR for query operations

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _detect_modality_type(self, modality_name: str) -> str:
        """Detect modality type from name."""
        name_lower = modality_name.lower()

        if "transcriptomics" in name_lower or "rna" in name_lower or "geo" in name_lower:
            if "single_cell" in name_lower or "sc" in name_lower:
                return "single_cell_rna_seq"
            else:
                return "bulk_rna_seq"
        elif "proteomics" in name_lower or "protein" in name_lower:
            if "ms" in name_lower or "mass" in name_lower:
                return "mass_spectrometry_proteomics"
            else:
                return "affinity_proteomics"

        return "unknown"

    def _generate_ml_recommendations(
        self, checks: Dict[str, bool], modality_type: str
    ) -> List[str]:
        """Generate ML-specific recommendations."""
        recommendations = []

        if not checks.get("sufficient_samples", True):
            recommendations.append(
                "Consider data augmentation or collecting more samples (minimum 10 recommended)"
            )

        if not checks.get("sufficient_features", True):
            recommendations.append("Feature count is low - consider feature selection strategies")

        if not checks.get("no_missing_values", True):
            recommendations.append("Handle missing values through imputation or removal")

        if not checks.get("has_metadata", True):
            recommendations.append("Add sample metadata for supervised learning tasks")

        if modality_type in ["single_cell_rna_seq", "bulk_rna_seq"]:
            if not checks.get("reasonable_gene_count", True):
                recommendations.append("Gene count outside typical range - verify data quality")
            if not checks.get("count_data", True):
                recommendations.append(
                    "Negative values detected - ensure proper preprocessing for count data"
                )

        if len(recommendations) == 0:
            recommendations.append("Data appears ML-ready!")

        return recommendations

    # =========================================================================
    # PROVENANCE IR GENERATION
    # =========================================================================

    def _create_readiness_ir(
        self, modality_name: str, readiness_score: float, readiness_level: str
    ) -> AnalysisStep:
        """Create IR for ML readiness check."""
        return AnalysisStep(
            operation="ml_preparation.check_ml_readiness",
            tool_name="MLPreparationService.check_ml_readiness",
            description=f"ML readiness check for {modality_name}: {readiness_level}",
            library="lobster",
            imports=[
                "from lobster.services.ml.ml_preparation_service import MLPreparationService"
            ],
            code_template="""# Check ML readiness
service = MLPreparationService()
result, stats, ir = service.check_ml_readiness(adata, modality_name="{{ modality_name }}")
print(f"Readiness: {result['readiness_level']} ({result['readiness_score']:.2%})")
print(f"Recommendations: {result['recommendations']}")
""",
            parameters={
                "modality_name": modality_name,
                "readiness_score": readiness_score,
                "readiness_level": readiness_level,
            },
            parameter_schema={},
            input_entities=[modality_name],
            output_entities=["readiness_report"],
        )

    def _create_feature_prep_ir(
        self,
        modality_name: str,
        feature_selection: str,
        n_features: int,
        normalization: str,
        scaling: str,
    ) -> AnalysisStep:
        """Create IR for feature preparation."""
        return AnalysisStep(
            operation="ml_preparation.prepare_ml_features",
            tool_name="MLPreparationService.prepare_ml_features",
            description=f"ML feature preparation: {modality_name}",
            library="lobster",
            imports=[
                "from lobster.services.ml.ml_preparation_service import MLPreparationService"
            ],
            code_template="""# Prepare ML features
service = MLPreparationService()
processed_adata, stats, ir = service.prepare_ml_features(
    adata,
    modality_name="{{ modality_name }}",
    feature_selection="{{ feature_selection }}",
    n_features={{ n_features }},
    normalization="{{ normalization }}",
    scaling="{{ scaling }}"
)
print(f"Shape: {stats['original_shape']} -> {stats['processed_shape']}")
""",
            parameters={
                "modality_name": modality_name,
                "feature_selection": feature_selection,
                "n_features": n_features,
                "normalization": normalization,
                "scaling": scaling,
            },
            parameter_schema={},
            input_entities=[modality_name],
            output_entities=[f"{modality_name}_ml_features"],
        )

    def _create_splits_ir(
        self,
        modality_name: str,
        target_column: Optional[str],
        test_size: float,
        validation_size: float,
        stratify: bool,
        random_state: int,
    ) -> AnalysisStep:
        """Create IR for ML splits."""
        return AnalysisStep(
            operation="ml_preparation.create_ml_splits",
            tool_name="MLPreparationService.create_ml_splits",
            description=f"Create ML splits for {modality_name}",
            library="lobster",
            imports=[
                "from lobster.services.ml.ml_preparation_service import MLPreparationService"
            ],
            code_template="""# Create ML splits
service = MLPreparationService()
split_metadata, stats, ir = service.create_ml_splits(
    adata,
    modality_name="{{ modality_name }}",
    target_column={{ target_column if target_column else 'None' }},
    test_size={{ test_size }},
    validation_size={{ validation_size }},
    stratify={{ stratify }},
    random_state={{ random_state }}
)
print(f"Train: {stats['train_size']}, Val: {stats['validation_size']}, Test: {stats['test_size']}")
""",
            parameters={
                "modality_name": modality_name,
                "target_column": target_column,
                "test_size": test_size,
                "validation_size": validation_size,
                "stratify": stratify,
                "random_state": random_state,
            },
            parameter_schema={},
            input_entities=[modality_name],
            output_entities=["ml_splits"],
        )

    def _create_export_ir(
        self,
        modality_name: str,
        framework: str,
        split: Optional[str],
        target_column: Optional[str],
    ) -> AnalysisStep:
        """Create IR for framework export."""
        return AnalysisStep(
            operation="ml_preparation.export_for_ml_framework",
            tool_name="MLPreparationService.export_for_ml_framework",
            description=f"Export {modality_name} for {framework}",
            library="lobster",
            imports=[
                "from lobster.services.ml.ml_preparation_service import MLPreparationService",
                "from pathlib import Path",
            ],
            code_template="""# Export for ML framework
service = MLPreparationService()
export_info, stats, ir = service.export_for_ml_framework(
    adata,
    modality_name="{{ modality_name }}",
    output_dir=Path("{{ output_dir }}"),
    framework="{{ framework }}",
    split={{ split if split else 'None' }},
    target_column={{ target_column if target_column else 'None' }}
)
print(f"Exported {stats['files_created']} files to {stats['output_dir']}")
""",
            parameters={
                "modality_name": modality_name,
                "framework": framework,
                "split": split,
                "target_column": target_column,
                "output_dir": "exports/ml_exports",
            },
            parameter_schema={},
            input_entities=[modality_name],
            output_entities=["ml_export_files"],
        )
