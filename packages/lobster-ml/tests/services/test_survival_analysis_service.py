"""
Tests for SurvivalAnalysisService scientific correctness.

Covers SAV-01 through SAV-05 fixes:
- SAV-01: Cox model alpha selection
- SAV-02: Kaplan-Meier median survival
- SAV-03: C-index reporting
- SAV-04: Bootstrap terminology
- SAV-05: Censored-before-horizon
"""

import numpy as np
import pytest
from anndata import AnnData

from lobster.services.analysis.survival_analysis_service import SurvivalAnalysisService


@pytest.fixture
def service():
    """Create service instance."""
    return SurvivalAnalysisService()


@pytest.fixture
def survival_adata():
    """Create test AnnData with survival data."""
    np.random.seed(42)
    n_samples = 100
    n_features = 20

    # Generate features
    X = np.random.randn(n_samples, n_features)

    # Generate survival data
    times = np.random.exponential(10, n_samples)
    events = np.random.binomial(1, 0.7, n_samples).astype(bool)

    adata = AnnData(X=X)
    adata.obs["time"] = times
    adata.obs["event"] = events
    adata.var_names = [f"gene_{i}" for i in range(n_features)]
    adata.obs_names = [f"sample_{i}" for i in range(n_samples)]

    return adata


class TestEdgeCaseValidation:
    """Tests for edge case handling (Plan 01)."""

    def test_all_censored_raises_error(self, service):
        """All censored data should raise ValueError."""
        events = np.array([False, False, False, False])
        times = np.array([1.0, 2.0, 3.0, 4.0])

        with pytest.raises(ValueError, match="No events observed"):
            service._validate_survival_data(events, times)

    def test_invalid_times_raises_error(self, service):
        """Zero or negative times should raise ValueError."""
        events = np.array([True, True, False])
        times = np.array([1.0, 0.0, 3.0])  # 0.0 is invalid

        with pytest.raises(ValueError, match="invalid survival times"):
            service._validate_survival_data(events, times)

    def test_too_few_events_raises_error(self, service):
        """Fewer than 20 events should raise ValueError."""
        events = np.array([True] * 10 + [False] * 90)
        times = np.random.exponential(10, 100)

        with pytest.raises(ValueError, match="minimum: 20"):
            service._validate_survival_data(events, times)

    def test_low_event_rate_warning(self, service):
        """Low event rate (<10%) should return warning."""
        events = np.array([True] * 25 + [False] * 475)  # 5% event rate
        times = np.random.exponential(10, 500)

        warnings = service._validate_survival_data(events, times)
        assert len(warnings) == 1
        assert "Low event rate" in warnings[0]


class TestMedianSurvival:
    """Tests for Kaplan-Meier median survival (SAV-02)."""

    def test_median_survival_first_crossing(self, service):
        """Median should be first time survival drops to 0.5."""
        # Survival: 1.0, 0.8, 0.6, 0.4, 0.2 at times 1, 2, 3, 4, 5
        # First time <= 0.5 is at time 4 (survival 0.4)
        km_time = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        km_survival = np.array([1.0, 0.8, 0.6, 0.4, 0.2])

        median = service._calculate_median_survival(km_time, km_survival)
        assert median == 4.0

    def test_median_not_reached(self, service):
        """Median should be None if survival never drops to 0.5."""
        km_time = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        km_survival = np.array([1.0, 0.9, 0.8, 0.7, 0.6])

        median = service._calculate_median_survival(km_time, km_survival)
        assert median is None

    def test_median_exactly_half(self, service):
        """Median should handle exactly 0.5 survival."""
        km_time = np.array([1.0, 2.0, 3.0])
        km_survival = np.array([1.0, 0.5, 0.3])

        median = service._calculate_median_survival(km_time, km_survival)
        assert median == 2.0  # First time survival <= 0.5


class TestCensoringAwareness:
    """Tests for censored-before-horizon handling (SAV-05)."""

    def test_censored_before_horizon_warning(self, service):
        """Should warn when censored-before-horizon samples included."""
        times = np.array([5.0, 3.0, 15.0])  # Sample 1 censored at 3
        events = np.array([True, False, True])
        time_horizon = 10.0

        _, _, warnings = service._create_binary_outcome(
            times, events, time_horizon, exclude_early_censored=False
        )

        assert len(warnings) == 1
        assert "censored before" in warnings[0].lower()

    def test_censored_before_horizon_exclusion(self, service):
        """Should exclude samples when exclude_early_censored=True."""
        times = np.array([5.0, 3.0, 15.0])
        events = np.array([True, False, True])
        time_horizon = 10.0

        binary, mask, warnings = service._create_binary_outcome(
            times, events, time_horizon, exclude_early_censored=True
        )

        # Should exclude sample 1 (censored at 3 < horizon 10)
        assert mask.sum() == 2
        assert not mask[1]  # Sample 1 excluded
        assert len(binary) == 2


class TestSubsamplingTerminology:
    """Tests for correct terminology (SAV-04)."""

    def test_parameter_names(self, service):
        """Should use n_iterations and subsample_fraction, not bootstrap."""
        import inspect

        sig = inspect.signature(service.optimize_threshold)
        params = list(sig.parameters.keys())

        assert "n_iterations" in params
        assert "subsample_fraction" in params
        assert "n_bootstrap" not in params
        assert "bootstrap_fraction" not in params


class TestCoxModelAlphaSelection:
    """Tests for Cox model training (SAV-01)."""

    def test_unregularized_default(self, service, survival_adata):
        """Default should use unregularized CoxPHSurvivalAnalysis."""
        if not service._sksurv_available:
            pytest.skip("scikit-survival not installed")

        adata, stats, ir = service.train_cox_model(
            survival_adata,
            time_column="time",
            event_column="event",
            regularized=False,
        )

        assert stats["regularized"] is False
        assert stats.get("best_alpha") is None
        assert "cox_risk_score" in adata.obs.columns

    def test_regularized_with_cv(self, service, survival_adata):
        """Regularized should use CV-based alpha selection."""
        if not service._sksurv_available:
            pytest.skip("scikit-survival not installed")

        adata, stats, ir = service.train_cox_model(
            survival_adata,
            time_column="time",
            event_column="event",
            regularized=True,
            cv_folds=3,  # Use 3 for speed
        )

        assert stats["regularized"] is True
        assert stats["best_alpha"] is not None
        assert stats["cv_folds"] == 3


class TestCIndexReporting:
    """Tests for C-index evaluation (SAV-03)."""

    def test_training_cindex_warning(self, service, survival_adata):
        """Should warn when only training C-index available."""
        if not service._sksurv_available:
            pytest.skip("scikit-survival not installed")

        adata, stats, ir = service.train_cox_model(
            survival_adata,
            time_column="time",
            event_column="event",
            regularized=False,  # No CV C-index
            test_adata=None,  # No test set
        )

        assert "c_index_train" in stats or "c_index" in stats
        assert "warnings" in stats
        # Should have warning about training C-index
        warnings_text = " ".join(stats["warnings"])
        assert "training" in warnings_text.lower() or "biased" in warnings_text.lower()
