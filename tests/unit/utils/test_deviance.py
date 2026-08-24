"""Regression tests for binomial deviance feature selection (bug #27).

``calculate_deviance`` historically floored the observed matrix
(``X = np.maximum(X, 1e-10)``) before masking with ``X > 0``. The floor made
every element positive, so the mask selected the *whole* matrix: the sparse path
silently densified (5.1x-17x measured memory amplification, TiB-scale working
sets on real GEO datasets, repeated OOM kills of benchmark campaigns) and every
structural zero contributed a small spurious negative term.

The primary guard here is ``test_only_nonzero_entries_reach_log``: it counts the
elements handed to ``np.log`` and requires at most ``nnz`` (see that test for why
the bound is an inequality). That is the defect itself, asserted deterministically
rather than via memory measurement.

Note which baseline each test discriminates against. The two ``np.log`` guards,
``test_matches_independent_reference``, ``test_all_zero_gene_scores_zero`` and
``test_integer_counts_do_not_truncate`` fail against the original floor-then-mask
implementation. The float32, large-integer, duplicate-coordinate and return-dtype
tests do NOT -- the original densified via ``.toarray()``, which incidentally
summed duplicate coordinates and promoted dtypes, so it was immune to those three.
Those tests guard the sparse rewrite against its own regressions, which is a
different job from guarding bug #27, and they were written because a review found
all three defects live in the rewrite.

That "do NOT fail against the original" claim is **differentially verified**, not
inferred: all four were run against the pre-fix implementation and all four pass
it. This matters methodologically. "A test not shown to fail against the bug is
not a guard" has a mirror that is easy to miss -- a test asserted NOT to be a
guard is also just a claim, and downgrading coverage needs the same differential
run as trusting it. Reading the assertions and reasoning about what the old
``.toarray()`` path incidentally did (it summed duplicates and promoted dtypes)
would have been the same move as reading a green test and inferring adequacy.
"""

import numpy as np
import pytest
import scipy.sparse as spr

from lobster.utils import deviance as deviance_module
from lobster.utils.deviance import calculate_deviance


def reference_deviance(matrix) -> np.ndarray:
    """Independent oracle: 2*x*log(x/E[x]) summed over observed nonzeros only.

    Written from the Townes et al. (2019) definition rather than derived from the
    implementation under test, so agreement is meaningful.

    Carries NO epsilon floor, deliberately. An earlier version of this oracle
    copied the production ``np.maximum(..., 1e-10)`` on both ``p_null`` and
    ``expected``, which made it structurally incapable of detecting those floors
    -- it agreed with the implementation precisely where the implementation was
    wrong. An oracle that shares the code's debatable decisions is not an oracle.

    No floor is needed: an observed positive count at (cell, gene) guarantees
    ``gene_totals[gene] > 0`` and ``cell_totals[cell] > 0``, so ``expected > 0``
    strictly, and entries with ``observed <= 0`` are skipped before any division.
    """
    dense = matrix.toarray() if spr.issparse(matrix) else np.asarray(matrix)
    dense = dense.astype(np.float64)
    n_cells, n_genes = dense.shape
    cell_totals = dense.sum(axis=1)
    gene_totals = dense.sum(axis=0)
    total = dense.sum()
    if total <= 0:
        return np.zeros(n_genes, dtype=np.float64)
    p_null = gene_totals / total
    scores = np.zeros(n_genes, dtype=np.float64)
    for cell in range(n_cells):
        for gene in range(n_genes):
            observed = dense[cell, gene]
            if observed <= 0:
                continue  # x*log(x/mu) -> 0 as x -> 0
            expected = cell_totals[cell] * p_null[gene]
            scores[gene] += 2.0 * observed * np.log(observed / expected)
    return scores


def sparse_counts(n_cells=60, n_genes=40, density=0.08, seed=0):
    matrix = spr.random(
        n_cells, n_genes, density=density, format="csr", random_state=seed
    )
    matrix.data = np.ceil(matrix.data * 30)
    return matrix


def test_only_nonzero_entries_reach_log(monkeypatch):
    """THE bug #27 guard: the mask must not saturate.

    Under the floor-then-mask defect every element is positive, so ``np.log``
    receives ``n_cells * n_genes`` values instead of ``nnz``.

    The bound is an inequality on purpose. Exceeding ``nnz`` is the defect and is
    what this test exists to catch. Going *below* ``nnz`` would be a legitimate
    future optimisation (e.g. skipping genes whose total count is zero, whose
    deviance is identically zero), so pinning exact equality would turn this red
    for a correct change. Under-counting entries that should contribute is
    already caught by ``test_matches_independent_reference``, which compares
    values rather than work. The current implementation touches exactly ``nnz``.
    """
    matrix = sparse_counts()
    nnz = int((matrix.toarray() > 0).sum())
    n_elements = matrix.shape[0] * matrix.shape[1]
    assert 0 < nnz < n_elements, "fixture must be genuinely sparse to be a guard"

    sizes = []
    real_log = np.log

    def spy(values, *args, **kwargs):
        sizes.append(np.size(values))
        return real_log(values, *args, **kwargs)

    monkeypatch.setattr(np, "log", spy)
    calculate_deviance(matrix)
    monkeypatch.setattr(np, "log", real_log)

    assert sum(sizes) <= nnz, (
        f"np.log saw {sum(sizes)} elements, which exceeds nnz={nnz}; "
        f"{n_elements} means the mask saturated (bug #27 regressed)"
    )


def test_dense_path_also_restricted_to_nonzeros(monkeypatch):
    """Same guard on the dense entry point, which has no sparsity to lean on."""
    dense = sparse_counts().toarray()
    nnz = int((dense > 0).sum())

    sizes = []
    real_log = np.log

    def spy(values, *args, **kwargs):
        sizes.append(np.size(values))
        return real_log(values, *args, **kwargs)

    monkeypatch.setattr(np, "log", spy)
    calculate_deviance(dense)
    monkeypatch.setattr(np, "log", real_log)

    # Inequality for the same reason as the sparse guard above.
    assert sum(sizes) <= nnz, f"np.log saw {sum(sizes)} elements, exceeds nnz={nnz}"


def test_matches_independent_reference():
    matrix = sparse_counts()
    np.testing.assert_allclose(
        calculate_deviance(matrix), reference_deviance(matrix), rtol=1e-12, atol=1e-12
    )


def test_sparse_and_dense_agree():
    matrix = sparse_counts()
    np.testing.assert_allclose(
        calculate_deviance(matrix),
        calculate_deviance(matrix.toarray()),
        rtol=1e-12,
        atol=1e-12,
    )


def test_dense_row_blocking_is_exercised(monkeypatch):
    """Force many row blocks; result must be identical to the single-block path."""
    dense = sparse_counts().toarray()
    expected = calculate_deviance(dense)

    # One row per block: exercises the accumulation loop and the row offsets.
    monkeypatch.setattr(deviance_module, "_DENSE_BLOCK_ELEMENTS", 1)
    np.testing.assert_allclose(
        calculate_deviance(dense), expected, rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("fmt", ["csr", "csc", "coo"])
def test_sparse_formats_agree(fmt):
    matrix = sparse_counts()
    np.testing.assert_allclose(
        calculate_deviance(matrix.asformat(fmt)),
        calculate_deviance(matrix),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("ctor_name", ["csr_array", "csc_array", "coo_array"])
def test_sparse_array_api_agrees_with_matrix_api(ctor_name):
    """The sparse *array* API, which every other test here skips.

    ``spr.issparse`` is True for both spmatrix and sparray, so a caller handing us
    an ``adata.X`` built by a newer scipy reaches the same branch -- but every
    other fixture in this module is built by ``spr.csr_matrix`` or ``spr.random``,
    both of which return the legacy *matrix* type. The array path was therefore
    entirely unexercised.

    It matters because the implementation reads ``entries.row`` / ``entries.col``
    off the COO. On ``coo_array`` the forward API is ``.coords``; ``.row``/``.col``
    are 2-D conveniences that scipy is free to retire. The engine declares
    ``scipy>=1.10.0`` with **no upper bound**, so the resolved version in any given
    install is unknown and this is the guard that would catch a removal instead of
    letting it surface as an AttributeError in a user's analysis.
    """
    dense = sparse_counts().toarray()
    array_input = getattr(spr, ctor_name)(dense)
    assert not isinstance(array_input, spr.spmatrix), "fixture must be a sparray"

    np.testing.assert_allclose(
        calculate_deviance(array_input),
        calculate_deviance(dense),
        rtol=1e-12,
        atol=1e-12,
    )


def test_noncanonical_sparse_array_duplicates_are_summed():
    """``sum_duplicates`` / ``has_canonical_format`` on the array type too.

    Same nonlinearity argument as the matrix-typed test above, but the flag is a
    separate code path on ``csr_array`` and a conservative ``False`` (extra copy)
    is safe here while a wrong ``True`` would silently split a count.
    """
    noncanonical = spr.csr_array(
        (
            np.array([2.0, 3.0, 1.0]),
            np.array([0, 0, 1]),
            np.array([0, 2, 3]),
        ),
        shape=(2, 2),
    )
    assert not noncanonical.has_canonical_format, "fixture must store duplicates"

    data_snapshot = noncanonical.data.copy()
    np.testing.assert_allclose(
        calculate_deviance(noncanonical),
        calculate_deviance(noncanonical.toarray()),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_array_equal(noncanonical.data, data_snapshot)


def test_explicitly_stored_zeros_are_ignored():
    """A stored 0 must behave exactly like a structural 0, not reach log()."""
    dense = np.array([[5.0, 0.0, 2.0], [0.0, 3.0, 1.0]])
    # Hand-built CSR carrying an explicit stored zero at (0, 1).
    with_stored_zero = spr.csr_matrix(
        (
            np.array([5.0, 0.0, 2.0, 3.0, 1.0]),
            np.array([0, 1, 2, 1, 2]),
            np.array([0, 3, 5]),
        ),
        shape=dense.shape,
    )
    np.testing.assert_array_equal(with_stored_zero.toarray(), dense)
    assert with_stored_zero.nnz == 5, "fixture must actually store the zero"
    np.testing.assert_allclose(
        calculate_deviance(with_stored_zero),
        calculate_deviance(dense),
        rtol=1e-12,
        atol=1e-12,
    )


def test_ultra_rare_gene_shares_are_not_floored():
    """A legitimate positive p_j below _EPS must not be replaced.

    Reachable on real data, not a contrived limit: at ~10M cells x 3000 counts the
    grand total is 3e10, so a gene detected in a single cell has a true share of
    3.3e-11 and gets floored to 1e-10 -- inflating its expected count and
    understating its deviance. Ultra-rare genes are exactly the high-deviance ones
    selection is meant to surface. The fixture below compresses that arithmetic
    into a small matrix so the test is fast.
    """
    counts = np.array(
        [[10**15, 1, 1, 1], [1, 1, 0, 1], [1, 0, 1, 1]],
        dtype=np.int64,
    )
    np.testing.assert_allclose(
        calculate_deviance(counts),
        reference_deviance(counts),
        rtol=1e-9,
        atol=1e-9,
    )


class TestInputDomainIsEnforced:
    """Negative and non-finite input must RAISE, not warn, drop, or clamp.

    Coupled to the removal of the ``1e-10`` denominator floors: with negatives
    admitted, a gene holding ``+5`` and ``-5`` has ``gene_total == 0`` while still
    owning observed nonzero entries, so ``p_null`` is 0 for a gene that IS indexed
    and ``expected`` becomes 0 -- a division by zero that the floors were hiding.
    Rejecting negatives is what makes the floors provably unnecessary rather than
    merely unfashionable, which is why these tests and that removal are one change.
    """

    @pytest.mark.parametrize("sparse", [False, True])
    def test_negative_values_raise(self, sparse):
        counts = np.array([[3.0, -1.0], [2.0, 4.0]])
        matrix = spr.csr_matrix(counts) if sparse else counts
        with pytest.raises(ValueError, match="negative"):
            calculate_deviance(matrix)

    @pytest.mark.parametrize("sparse", [False, True])
    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_non_finite_values_raise(self, sparse, bad):
        counts = np.array([[3.0, 1.0], [2.0, 4.0]])
        counts[0, 1] = bad
        matrix = spr.csr_matrix(counts) if sparse else counts
        with pytest.raises(ValueError, match="non-finite"):
            calculate_deviance(matrix)

    def test_nan_is_not_masked_by_the_negative_check(self):
        """Ordering guard: ``np.min`` of a NaN array is NaN and ``NaN < 0`` is False.

        If the dense path checked the minimum before checking finiteness, a NaN
        matrix would pass both tests and reach ``np.log``. This asserts the checks
        run in the order that makes the second one reachable.
        """
        counts = np.array([[3.0, np.nan], [2.0, 4.0]])
        assert not (float(np.nanmin(counts)) < 0), "fixture has no negative to catch"
        with pytest.raises(ValueError, match="non-finite"):
            calculate_deviance(counts)

    def test_negative_stored_in_a_sparse_container_is_not_skipped(self):
        """A negative must be rejected even where a structural zero is fine.

        The validator runs on the stored values, so it has to distinguish "absent,
        therefore zero, therefore legal" from "stored and negative".
        """
        counts = np.array([[5.0, 0.0, 2.0], [0.0, -3.0, 1.0]])
        matrix = spr.csr_matrix(counts)
        assert matrix.nnz == 4
        with pytest.raises(ValueError, match="negative"):
            calculate_deviance(matrix)

    def test_message_names_the_statistic_and_the_expected_domain(self):
        """The person who hits this needs to know their pipeline is upstream-wrong."""
        with pytest.raises(ValueError) as excinfo:
            calculate_deviance(np.array([[1.0, -2.5], [3.0, 4.0]]))
        message = str(excinfo.value)
        assert "-2.5" in message, "must name the offending value"
        assert "raw counts" in message.lower(), "must state the expected domain"

    def test_log_normalized_input_is_rejected(self):
        """The realistic failure: ``adata.X`` after ``sc.pp.log1p`` on scaled data.

        This is the motivating case for raising rather than warning. Such input
        previously returned a plausible finite ranking.
        """
        counts = np.array([[5.0, 0.0, 2.0], [0.0, 3.0, 1.0], [1.0, 1.0, 0.0]])
        scaled = (counts - counts.mean(axis=0)) / counts.std(axis=0)
        assert scaled.min() < 0, "fixture must actually contain negatives"
        with pytest.raises(ValueError, match="negative"):
            calculate_deviance(scaled)

    @pytest.mark.parametrize("sparse", [False, True])
    def test_legitimate_zero_total_gene_is_not_rejected(self, sparse):
        """Guard against over-rejection: an all-zero gene is legal count data.

        ``p_null`` for that gene is exactly 0 now that the floor is gone, which is
        safe only because the gene owns no observed positive entry and is therefore
        never indexed. If that reasoning were wrong this would raise or return NaN.
        """
        counts = np.array([[4.0, 0.0, 2.0], [1.0, 0.0, 3.0]])
        matrix = spr.csr_matrix(counts) if sparse else counts
        scores = calculate_deviance(matrix)
        assert scores[1] == 0.0
        assert np.all(np.isfinite(scores))

    def test_all_zero_input_is_legal_and_returns_zeros(self):
        """Zero is a count. Degenerate totals must not be confused with bad input."""
        for matrix in (np.zeros((3, 4)), spr.csr_matrix((3, 4))):
            np.testing.assert_array_equal(calculate_deviance(matrix), np.zeros(4))


def test_float32_sparse_matches_float32_dense():
    """Marginals must accumulate in float64, not in the stored dtype.

    scipy's sparse ``.sum()`` reduces in the stored dtype and casts afterwards, so
    a float32 matrix lost precision in the cell totals and the sparse and dense
    paths disagreed (9.68 dense vs 15.68 sparse on this fixture). Passing
    ``dtype=`` to ``sum()`` does not fix it -- it casts after the damage.
    """
    dense = np.ones((4, 4), dtype=np.float32)
    dense[0, 0] = 1e8  # 1e8 + 1 is not representable in float32

    np.testing.assert_allclose(
        calculate_deviance(spr.csr_matrix(dense)),
        calculate_deviance(dense),
        rtol=1e-12,
        atol=1e-12,
    )


def test_large_integer_totals_do_not_wrap():
    """int64 row/gene totals must not overflow before reaching float64."""
    huge = np.iinfo(np.int64).max
    matrix = spr.csr_matrix(np.array([[huge], [huge]], dtype=np.int64))

    scores = calculate_deviance(matrix)
    assert np.all(np.isfinite(scores)), "an in-dtype sum wrapped to a negative total"
    # One gene holds every count, so it matches the null exactly: deviance is 0.
    np.testing.assert_allclose(scores, np.zeros(1), rtol=1e-12, atol=1e-12)


def test_duplicate_sparse_coordinates_are_summed():
    """A non-canonical container storing (cell, gene) twice must sum, not split.

    ``x*log(x)`` is nonlinear, so ``f(2) + f(3) != f(5)``. ``tocsr()`` returns an
    existing CSR unchanged, so duplicates survive into the COO expansion unless
    handled. Before the fix this gave -4.91 where the dense path gave 1.82.
    """
    # Row 0 stores column 0 twice (2.0 and 3.0); densely that is a single 5.0.
    noncanonical = spr.csr_matrix(
        (
            np.array([2.0, 3.0, 1.0]),
            np.array([0, 0, 1]),
            np.array([0, 2, 3]),
        ),
        shape=(2, 2),
    )
    assert not noncanonical.has_canonical_format, "fixture must store duplicates"

    data_snapshot = noncanonical.data.copy()
    np.testing.assert_allclose(
        calculate_deviance(noncanonical),
        calculate_deviance(noncanonical.toarray()),
        rtol=1e-12,
        atol=1e-12,
    )
    # sum_duplicates() mutates in place, so the caller's matrix must be copied.
    np.testing.assert_array_equal(noncanonical.data, data_snapshot)


def test_return_dtype_is_float64_including_degenerate_shapes():
    """``np.bincount`` ignores weight dtype when empty and returns int64."""
    for matrix in (
        spr.csr_matrix((0, 5)),
        spr.csr_matrix((4, 3)),
        np.zeros((0, 5)),
        np.zeros((4, 3)),
        sparse_counts(),
    ):
        assert calculate_deviance(matrix).dtype == np.float64


def test_integer_counts_do_not_truncate():
    """The old floor implicitly promoted int -> float; the fix must do so too."""
    dense_int = np.array([[3, 0, 7], [0, 5, 1]], dtype=np.int64)
    np.testing.assert_allclose(
        calculate_deviance(dense_int),
        reference_deviance(dense_int),
        rtol=1e-12,
        atol=1e-12,
    )
    assert calculate_deviance(dense_int).dtype == np.float64


def test_input_is_not_mutated():
    dense = sparse_counts().toarray()
    snapshot = dense.copy()
    calculate_deviance(dense)
    np.testing.assert_array_equal(dense, snapshot)

    matrix = sparse_counts()
    data_snapshot = matrix.data.copy()
    calculate_deviance(matrix)
    np.testing.assert_array_equal(matrix.data, data_snapshot)


def test_all_zero_matrix_returns_zeros():
    for matrix in (np.zeros((4, 3)), spr.csr_matrix((4, 3))):
        scores = calculate_deviance(matrix)
        assert scores.shape == (3,)
        np.testing.assert_array_equal(scores, np.zeros(3))


def test_no_cells_returns_zeros_per_gene():
    for matrix in (np.zeros((0, 5)), spr.csr_matrix((0, 5))):
        np.testing.assert_array_equal(calculate_deviance(matrix), np.zeros(5))


def test_all_zero_gene_scores_zero():
    dense = np.array([[4.0, 0.0, 2.0], [1.0, 0.0, 3.0]])
    assert calculate_deviance(dense)[1] == 0.0


def test_shape_and_ranking_preserved_on_larger_matrix():
    """Selection behaviour, which is the only property downstream consumes."""
    matrix = sparse_counts(n_cells=200, n_genes=120, density=0.06, seed=7)
    scores = calculate_deviance(matrix)
    assert scores.shape == (120,)
    top = np.argsort(scores)[::-1][:20]
    reference_top = np.argsort(reference_deviance(matrix))[::-1][:20]
    assert set(top) == set(reference_top)
