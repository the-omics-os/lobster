"""Sparse deviance tests: avoid densification while preserving ranking and correctness.

Sparse input must remain sparse throughout calculation. The tests compare ranking and
scores with a dense reference, and check that the implementation does not materialize
full-matrix intermediates.
"""

import numpy as np
import pytest
import scipy.sparse as spr

from lobster.utils.deviance import calculate_deviance


def _counts(n_cells, n_genes, density, seed):
    """Integer-ish UMI counts, shaped like real scRNA-seq sparsity."""
    X = spr.random(n_cells, n_genes, density=density, format="csr", random_state=seed)
    X.data = np.ceil(X.data * 20)
    return X


def _previous_implementation(count_matrix):
    """Keep a copy of the previous calculation as a ranking oracle.

    This intentionally remains separate from the implementation under test so future
    changes do not silently update both sides of the comparison.
    """
    X = count_matrix.toarray() if spr.issparse(count_matrix) else count_matrix.copy()
    X = np.maximum(X, 1e-10)
    cell_totals = X.sum(axis=1, keepdims=True)
    gene_totals = X.sum(axis=0)
    total_counts = X.sum()
    p_null = np.maximum(gene_totals / total_counts, 1e-10)
    expected = np.maximum(cell_totals @ p_null.reshape(1, -1), 1e-10)
    mask = X > 0
    terms = np.zeros_like(X)
    terms[mask] = 2 * X[mask] * np.log(X[mask] / expected[mask])
    return terms.sum(axis=0)


class TestNeverDensifies:
    """The load-bearing test. Everything else is correctness; this is the bug."""

    def test_toarray_is_never_called_on_a_sparse_input(self, monkeypatch):
        """Monkeypatch `toarray` to raise so densification is impossible.

        A memory assertion would be flaky under a loaded CI box; making densification
        *impossible* is deterministic and states the invariant directly.
        """
        X = _counts(400, 300, density=0.05, seed=0)

        def explode(*args, **kwargs):
            raise AssertionError(
                "calculate_deviance densified a sparse matrix; sparse inputs must not be expanded."
            )

        monkeypatch.setattr(spr.csr_matrix, "toarray", explode, raising=True)
        monkeypatch.setattr(spr.coo_matrix, "toarray", explode, raising=True)

        scores = calculate_deviance(X)

        assert scores.shape == (300,)
        assert np.isfinite(scores).all()

    def test_todense_is_also_not_used(self, monkeypatch):
        """`.todense()` is the other spelling; block it too so neither route reopens."""
        X = _counts(400, 300, density=0.05, seed=1)

        def explode(*args, **kwargs):
            raise AssertionError("calculate_deviance called .todense()")

        monkeypatch.setattr(spr.csr_matrix, "todense", explode, raising=True)

        assert np.isfinite(calculate_deviance(X)).all()

    def test_expected_is_not_materialised_as_a_full_outer_product(self):
        """`expected` was `cell_totals @ p_null.reshape(1, -1)` -- (cells x genes).

        That array is never smaller than X no matter how X is stored, so it was the
        decisive allocation. A tall-thin matrix makes the regression obvious: the outer
        product would be 40,000 x 2,000 = 640 MB of float64, while the stored entries
        are ~0.2% of that.

        Note the bound here is relative to the dense size, not an absolute byte budget.
        The memory check is relative to the dense size so it catches full-matrix
        intermediates without relying on an absolute platform-specific limit.
        """
        import tracemalloc

        X = _counts(40_000, 2_000, density=0.002, seed=2)
        dense_bytes = 40_000 * 2_000 * 8

        tracemalloc.start()
        calculate_deviance(X)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Generous bound: an order of magnitude under the dense size still fails loudly
        # if a full-matrix temporary comes back.
        assert peak < dense_bytes / 10, (
            f"peak allocation {peak / 2**20:.0f} MiB approaches the dense size "
            f"{dense_bytes / 2**20:.0f} MiB; a whole-matrix temporary may be back"
        )


class TestRankingPreserved:
    """The selected gene set must match the dense reference."""

    @pytest.mark.parametrize(
        "n_cells,n_genes,density,seed",
        [
            (2_000, 1_500, 0.0055, 0),  # sparse single-cell-like input
            (1_000, 800, 0.02, 1),
            (3_000, 1_000, 0.01, 2),
            (500, 400, 0.10, 3),  # denser, e.g. post-filtering
        ],
    )
    def test_top_gene_set_matches_previous_implementation(
        self, n_cells, n_genes, density, seed
    ):
        X = _counts(n_cells, n_genes, density, seed)

        want = _previous_implementation(X)
        got = calculate_deviance(X)

        k = min(2_000, n_genes)
        want_top = set(np.argsort(want)[::-1][:k])
        got_top = set(np.argsort(got)[::-1][:k])
        assert want_top == got_top, f"gene set changed: {len(want_top & got_top)}/{k}"

    def test_scores_agree_to_float_tolerance(self):
        """Not bit-identical, and cannot be: the old version added a spurious term.

        Flooring zeros to 1e-10 rather than skipping them meant every unobserved entry
        contributed `2 * 1e-10 * log(1e-10 / E)` -- small, non-zero, and negative. The
        true term is exactly 0. So the sparse form is not an approximation of the old
        one; it is the reference, and the old one was the approximation.
        """
        X = _counts(2_000, 1_000, density=0.0055, seed=4)

        rel = np.abs(_previous_implementation(X) - calculate_deviance(X)) / np.maximum(
            np.abs(_previous_implementation(X)), 1e-12
        )
        assert rel.max() < 1e-6

    def test_chunking_does_not_change_the_result(self):
        """`chunk` bounds memory only. Any value must give the same scores."""
        X = _counts(1_000, 500, density=0.02, seed=5)

        reference = calculate_deviance(X, chunk=1_000_000)  # single block
        for chunk in (None, 1, 7, 100, 999, 1_000, 1_001):
            assert np.allclose(
                calculate_deviance(X, chunk=chunk), reference, rtol=1e-12, atol=0.0
            ), f"chunk={chunk} changed the result"


class TestBlockSizeAdaptsToDensity:
    """Peak memory depends on nonzeros per block, so rows are the wrong unit.

    Fixed row counts do not bound memory consistently across densities.
    The default derives the row count from the input's own density.
    """

    def test_denser_input_gets_fewer_rows_per_block(self, monkeypatch):
        """Same shape, 10x the density -> roughly 10x fewer rows per block."""
        from lobster.utils import deviance as mod

        seen: list[int] = []
        real_getitem = spr.csr_matrix.__getitem__

        def spy(self, key):
            if (
                isinstance(key, slice)
                and key.stop is not None
                and key.start is not None
            ):
                seen.append(key.stop - key.start)
            return real_getitem(self, key)

        monkeypatch.setattr(spr.csr_matrix, "__getitem__", spy)
        monkeypatch.setattr(mod, "TARGET_BLOCK_NNZ", 100_000)

        seen.clear()
        calculate_deviance(_counts(4_000, 1_000, density=0.01, seed=0))
        sparse_rows = seen[0]

        seen.clear()
        calculate_deviance(_counts(4_000, 1_000, density=0.10, seed=0))
        dense_rows = seen[0]

        assert dense_rows < sparse_rows, (
            f"denser matrix used {dense_rows} rows/block vs {sparse_rows} for the "
            f"sparser one; block size is not adapting to density"
        )

    def test_block_size_is_at_least_one_row(self):
        """A single row wider than the nnz target must not yield chunk=0 and loop forever."""
        from lobster.utils import deviance as mod

        wide = _counts(3, 5_000, density=0.9, seed=1)
        original = mod.TARGET_BLOCK_NNZ
        try:
            mod.TARGET_BLOCK_NNZ = 10  # far below one row's nnz
            scores = calculate_deviance(wide)
        finally:
            mod.TARGET_BLOCK_NNZ = original

        assert scores.shape == (5_000,)
        assert np.isfinite(scores).all()

    def test_dense_and_sparse_inputs_agree(self):
        X = _counts(300, 200, density=0.08, seed=6)

        assert np.allclose(
            calculate_deviance(X), calculate_deviance(X.toarray()), rtol=1e-12
        )


class TestDegenerateInputsStayFinite:
    """Zero totals and non-positive values must not produce non-finite scores.

    Sparse inputs can contain explicit zeros. Calculation must handle them without
    introducing NaN values.
    """

    def test_explicit_stored_zeros_do_not_produce_nan(self):
        """A sparse matrix may store explicit zeros -- routine after filtering.

        Built via the CSR triple rather than by assigning 0 into a `lil` matrix: writing
        zero to an already-zero position is a no-op, so that route produces no stored
        zero and the test would pass vacuously. The guard below catches exactly that.
        """
        X = spr.csr_matrix(
            (
                np.array([5.0, 0.0, 3.0, 2.0, 4.0]),  # element 1 is a STORED zero
                np.array([0, 1, 2, 0, 1]),
                np.array([0, 3, 5]),
            ),
            shape=(2, 3),
        )
        assert (X.data == 0).any(), "fixture must actually contain a stored zero"

        scores = calculate_deviance(X)

        assert np.isfinite(scores).all()
        assert np.allclose(scores, _previous_implementation(X), rtol=1e-6)

    def test_negative_values_do_not_produce_nan(self):
        """Reached when log-transformed or scaled data is passed by mistake.

        The old floor clamped these to 1e-10; skipping them is the exact analogue, since
        a non-positive count has no deviance contribution. What matters is that the
        function does not emit NaN into a gene ranking.
        """
        X = spr.csr_matrix(np.array([[-1.0, 2.0], [3.0, -4.0]]))

        scores = calculate_deviance(X)

        assert np.isfinite(scores).all()

    def test_all_zero_matrix_returns_zeros(self):
        """`gene_totals.sum() == 0` would divide by zero without the guard."""
        scores = calculate_deviance(spr.csr_matrix((4, 3)))

        assert scores.shape == (3,)
        assert np.array_equal(scores, np.zeros(3))

    @pytest.mark.parametrize("shape", [(0, 5), (5, 0), (0, 0)])
    def test_empty_axes(self, shape):
        scores = calculate_deviance(spr.csr_matrix(shape))
        assert scores.shape == (shape[1],)
        assert np.isfinite(scores).all()

    def test_single_cell_and_single_gene(self):
        assert calculate_deviance(
            spr.csr_matrix(np.array([[3.0, 0.0, 7.0]]))
        ).shape == (3,)
        assert calculate_deviance(spr.csr_matrix(np.array([[3.0], [7.0]]))).shape == (
            1,
        )


class TestGeneratedTemplateMatchesLibrary:
    """The library fix alone is insufficient -- the IR template emits its own copy.

    `_create_deviance_selection_ir` carries a Jinja2 `code_template` that defined the
    *same* densifying `calculate_deviance` into every agent-generated analysis script. A
    fix touching only `utils/deviance.py` would leave generated notebooks OOM-ing, and
    would mean the harness supplies the failure mode to the model rather than observing
    it. So the emitted code is executed here and compared against the library.
    """

    @staticmethod
    def _render():
        from jinja2 import Template

        from lobster.services.quality.preprocessing_service import PreprocessingService

        ir = PreprocessingService()._create_deviance_selection_ir(n_top_genes=2000)
        return Template(ir.code_template).render(n_top_genes=2000)

    def test_template_is_valid_python(self):
        import ast

        ast.parse(self._render())

    def test_template_contains_no_densification(self):
        rendered = self._render()
        code_lines = [
            line
            for line in rendered.splitlines()
            if not line.lstrip().startswith("#")  # warnings mention .toarray() by name
        ]
        code = "\n".join(code_lines)

        for pattern in (".toarray()", ".todense()", "np.maximum(X, 1e-10)"):
            assert pattern not in code, f"template still emits {pattern}"

    def test_emitted_function_matches_the_library(self):
        """Same inputs, same scores -- otherwise notebooks and runtime diverge."""
        namespace: dict = {}
        helper = self._render().split("# Calculate binomial deviance")[0]
        exec(
            helper, namespace
        )  # nosec B102 # Execute repository-generated code on synthetic fixtures to validate replay. noqa: S102 - executing our own template is the test
        emitted = namespace["calculate_deviance"]

        X = _counts(1_000, 600, density=0.02, seed=8)

        assert np.allclose(emitted(X), calculate_deviance(X), rtol=1e-12, atol=0.0)
