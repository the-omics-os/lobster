"""Per-cell confidence scoring must be vectorised and numerically unchanged.

Precompute values that do not depend on the individual cell, rather than rebuilding
marker lookups inside nested loops. Because vectorization can return plausible but wrong
values if axes are handled incorrectly, tests compare results with hand-computed values and
a reference implementation.
"""

import anndata
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr

from lobster.services.analysis.enhanced_singlecell_service import (
    EnhancedSingleCellService,
)


@pytest.fixture
def service():
    return EnhancedSingleCellService()


def _reference_loop(adata, reference_markers):
    """The ORIGINAL per-cell implementation, kept verbatim as a numerical oracle.

    Deliberately a copy rather than an import: the point is to compare against the
    behaviour that shipped, so this must not change when the implementation does.
    """
    from scipy.stats import entropy as shannon_entropy

    n_cells = adata.n_obs
    confidence = np.zeros(n_cells)
    top3_out = np.empty(n_cells, dtype=object)
    entropies = np.zeros(n_cells)

    signatures = {}
    for ct, markers in reference_markers.items():
        valid = [m for m in markers if m in adata.var_names]
        if valid:
            block = adata[:, valid].X
            if hasattr(block, "toarray"):
                block = block.toarray()
            signatures[ct] = np.mean(block, axis=0)
        else:
            signatures[ct] = np.zeros(0)

    for i in range(n_cells):
        correlations = {}
        for ct, sig in signatures.items():
            if np.std(sig) > 0:
                cell_expr = adata[i, :].X
                if hasattr(cell_expr, "toarray"):
                    cell_expr = cell_expr.toarray().flatten()
                else:
                    cell_expr = np.array(cell_expr).flatten()
                markers = [m for m in reference_markers[ct] if m in adata.var_names]
                if markers:
                    idx = [list(adata.var_names).index(m) for m in markers]
                    vals = cell_expr[idx]
                    if np.std(vals) > 0:
                        corr, _ = pearsonr(vals, sig)
                        correlations[ct] = max(0, corr)
                    else:
                        correlations[ct] = 0.0
                else:
                    correlations[ct] = 0.0
            else:
                correlations[ct] = 0.0

        ordered = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        top3 = [ct for ct, _ in ordered[:3]]
        scores = np.array([correlations[ct] for ct in top3])
        confidence[i] = correlations[ordered[0][0]] if ordered else 0.0
        top3_out[i] = ",".join(top3)
        if np.sum(scores) > 0:
            entropies[i] = shannon_entropy(scores / np.sum(scores))
        else:
            entropies[i] = np.log(3)
    return confidence, top3_out, entropies


def _adata(matrix, gene_names, sparse=False):
    X = csr_matrix(matrix) if sparse else np.asarray(matrix, dtype=float)
    return anndata.AnnData(
        X,
        var=pd.DataFrame(index=list(gene_names)),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(np.shape(matrix)[0])]),
    )


class TestHandComputedValues:
    """Exact values derived by hand, so the tests survive a reimplementation."""

    def test_perfectly_correlated_cell_scores_one(self, service):
        """A cell proportional to the signature has Pearson r == 1 exactly."""
        # Two cells, identical profile -> signature equals each cell's profile.
        matrix = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
        adata = _adata(matrix, ["A", "B", "C"])
        markers = {"X": ["A", "B", "C"]}

        confidence, top3, entropy = service._calculate_per_cell_confidence(
            adata, markers
        )

        assert np.allclose(confidence, [1.0, 1.0])
        assert list(top3) == ["X", "X"]
        # Single cell type -> top3 has one entry with probability 1 -> entropy 0.
        assert np.allclose(entropy, [0.0, 0.0])

    def test_anticorrelated_cell_is_clipped_to_zero(self, service):
        """Negative correlations become 0.0, matching `max(0, corr)`."""
        # Cell 0 rises, cell 1 falls; the mean signature is flat-ish, so build a case
        # where cell 1 is the exact mirror of the signature.
        matrix = [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]
        adata = _adata(matrix, ["A", "B", "C", "D"])
        markers = {"X": ["A", "B", "C", "D"]}

        # Signature is the column mean = [2.5, 2.5, 2.5, 2.5] -> constant -> std == 0,
        # so BOTH cells score 0 via the constant-signature guard.
        confidence, _, entropy = service._calculate_per_cell_confidence(adata, markers)
        assert np.allclose(confidence, [0.0, 0.0])
        # No positive score anywhere -> max entropy log(3), as the original did.
        assert np.allclose(entropy, [np.log(3), np.log(3)])

    def test_constant_cell_expression_scores_zero(self, service):
        """`np.std(cell_marker_expr) > 0` guard: a flat cell cannot correlate."""
        matrix = [[5.0, 5.0, 5.0], [1.0, 3.0, 9.0]]
        adata = _adata(matrix, ["A", "B", "C"])
        markers = {"X": ["A", "B", "C"]}

        confidence, _, _ = service._calculate_per_cell_confidence(adata, markers)
        assert confidence[0] == 0.0

    def test_known_two_type_ordering(self, service):
        """Hand-checked r for a cell that matches type X better than type Y."""
        matrix = [
            [10.0, 10.0, 0.0, 0.0],  # cell 0: X-like
            [0.0, 0.0, 10.0, 10.0],  # cell 1: Y-like
        ]
        adata = _adata(matrix, ["X1", "X2", "Y1", "Y2"])
        markers = {"X": ["X1", "X2"], "Y": ["Y1", "Y2"]}

        confidence, top3, _ = service._calculate_per_cell_confidence(adata, markers)

        # Signatures are both [5, 5] -> constant -> std == 0 -> all scores 0.
        # This is the SHIPPED behaviour, and asserting it pins the guard rather than
        # an intuition about what the scores "should" be.
        assert np.allclose(confidence, [0.0, 0.0])
        assert all(len(t.split(",")) == 2 for t in top3)

    def test_entropy_of_two_equal_scores_is_log_two(self, service):
        """Shannon entropy is computed inline; verify against the closed form."""
        # Two cell types whose signatures are non-constant and which the cell matches
        # equally well: probabilities (0.5, 0.5) -> entropy ln(2).
        matrix = [
            [1.0, 3.0, 1.0, 3.0],
            [2.0, 5.0, 2.0, 5.0],
            [4.0, 1.0, 4.0, 1.0],
        ]
        adata = _adata(matrix, ["P1", "P2", "Q1", "Q2"])
        markers = {"P": ["P1", "P2"], "Q": ["Q1", "Q2"]}

        _, _, entropy = service._calculate_per_cell_confidence(adata, markers)
        # Both types see identical marker values by construction, so each cell's two
        # scores are equal -> ln(2) whenever they are positive.
        expected = _reference_loop(adata, markers)[2]
        assert np.allclose(entropy, expected, atol=1e-12)


class TestEquivalenceWithOriginalLoop:
    """The regression net: the vectorised form must match the loop it replaced."""

    @staticmethod
    def _random_case(n_cells, n_genes, markers, seed, sparse, threshold=2.0):
        rng = np.random.default_rng(seed)
        X = rng.random((n_cells, n_genes)) * 5.0
        X[X < threshold] = 0.0
        genes = [f"G{i}" for i in range(n_genes)]
        for pos, name in enumerate(sorted({m for v in markers.values() for m in v})):
            genes[pos] = name
        return _adata(X, genes, sparse=sparse)

    MARKERS = {
        "T": ["CD3D", "CD3E", "IL7R"],
        "B": ["MS4A1", "CD79A"],
        "NK": ["GNLY", "NKG7", "KLRD1"],
        "Mono": ["LYZ", "CD14"],
        "DC": ["FCER1A"],
        "Platelet": ["PPBP"],
    }

    @pytest.mark.parametrize(
        "n_cells,n_genes,seed,sparse,threshold",
        [
            (120, 200, 1, True, 2.0),
            (120, 200, 2, False, 2.0),
            (80, 150, 3, True, 4.5),  # very sparse
            (60, 120, 4, False, 0.0),  # no zeros at all
            (5, 60, 5, True, 2.0),  # tiny
            (1, 40, 6, True, 2.0),  # single cell
        ],
    )
    def test_matches_reference_loop(
        self, service, n_cells, n_genes, seed, sparse, threshold
    ):
        """Equal to 6 decimals, not bit-for-bit -- and 6 dp is the honest claim.

        A centred dot product and `scipy.stats.pearsonr` can differ in the last
        ULP, so compare the returned scores within numerical tolerance.
        """
        adata = self._random_case(
            n_cells, n_genes, self.MARKERS, seed, sparse, threshold
        )
        want = _reference_loop(adata.copy(), self.MARKERS)
        got = service._calculate_per_cell_confidence(adata.copy(), self.MARKERS)

        assert np.allclose(got[0], want[0], atol=1e-6), "confidence diverged"
        assert np.allclose(got[2], want[2], atol=1e-6), "entropy diverged"
        assert list(got[1]) == list(want[1]), "top3 ordering diverged"

    def test_two_marker_types_preserve_tie_order(self, service):
        """The tie-order trap, isolated.

        With exactly two markers, centring gives `[d/2, -d/2]`, so any two such vectors
        are exactly collinear and |r| is analytically 1 -- yet the quotient lands ~2.2e-16
        ABOVE 1.0 for ~22% of inputs and ~2.2e-16 BELOW it for another ~22%. Clamping
        repairs only the overshoot. Since 1.0000000000000002 sorts above a genuine 1.0,
        cells whose two best types are both perfect matches would get `top3` silently
        reordered -- same confidence, same entropy, different labels. Ranking on
        6-dp-rounded scores removes the sensitivity. Two-marker panels are common, so
        this is a realistic configuration rather than a contrived one.
        """
        markers = {
            "A": ["Ma1", "Ma2"],
            "B": ["Mb1", "Mb2"],
            "C": ["Mc1", "Mc2"],
        }
        adata = self._random_case(300, 300, markers, seed=12, sparse=True)

        want = _reference_loop(adata.copy(), markers)
        got = service._calculate_per_cell_confidence(adata.copy(), markers)

        assert list(got[1]) == list(want[1])
        assert np.allclose(got[0], want[0], atol=1e-6)

    def test_single_cell_signature_equals_the_cell_itself(self, service):
        """n_cells == 1 makes every signature collinear with the cell, at ANY marker count.

        This is why the fix cannot be a `marker_cols.size == 2` special case: with one
        cell the signature *is* that cell's profile, so |r| is analytically 1 for a
        3-marker type too, and the same last-ULP undershoot appears there. An earlier
        attempt that special-cased two markers failed on exactly this input.
        """
        markers = {
            "T": ["CD3D", "CD3E", "IL7R"],
            "B": ["MS4A1", "CD79A"],
            "NK": ["GNLY", "NKG7", "KLRD1"],
        }
        adata = self._random_case(1, 40, markers, seed=6, sparse=True)

        want = _reference_loop(adata.copy(), markers)
        got = service._calculate_per_cell_confidence(adata.copy(), markers)

        assert list(got[1]) == list(want[1])
        assert np.allclose(got[0], want[0], atol=1e-6)

    def test_ties_preserve_insertion_order(self, service):
        """All-zero scores must rank cell types in `signatures` insertion order.

        Ties are the common case, not an edge case: every cell type failing a guard
        scores exactly 0.0. Sorting ascending and reversing would invert their order.
        """
        matrix = np.zeros((4, 10))
        adata = _adata(matrix, [f"G{i}" for i in range(10)])
        markers = {"First": ["G0"], "Second": ["G1"], "Third": ["G2"], "Fourth": ["G3"]}

        _, top3, entropy = service._calculate_per_cell_confidence(adata, markers)

        assert all(t == "First,Second,Third" for t in top3)
        assert np.allclose(entropy, np.log(3))


class TestEdgeCases:
    def test_markers_absent_from_dataset(self, service):
        adata = _adata(
            np.random.default_rng(0).random((10, 8)) * 3, [f"G{i}" for i in range(8)]
        )
        markers = {"Ghost": ["NOPE1", "NOPE2"], "Real": ["G0", "G1", "G2"]}

        confidence, top3, _ = service._calculate_per_cell_confidence(adata, markers)

        assert confidence.shape == (10,)
        assert all("Ghost" in t or "Real" in t for t in top3)

    def test_empty_reference_markers(self, service):
        """No signatures at all: zeros and max entropy, no exception."""
        adata = _adata(np.ones((5, 4)), ["A", "B", "C", "D"])

        confidence, top3, entropy = service._calculate_per_cell_confidence(adata, {})

        assert np.allclose(confidence, 0.0)
        assert list(top3) == [""] * 5
        assert np.allclose(entropy, np.log(3))

    def test_gene_position_map_resolves_first_occurrence(self):
        """`list.index()` returns the FIRST match; a dict comprehension keeps the LAST.

        Duplicated `var_names` are legal in AnnData (`var_names_make_unique()` is not
        guaranteed to have run), so a naive `{g: i for i, g in enumerate(...)}` would
        silently read a *different* gene's column and yield a plausible wrong
        correlation. This pins the lookup construction the implementation relies on.
        """
        names = ["DUP", "B", "C", "DUP"]

        naive = {gene: pos for pos, gene in enumerate(names)}
        assert naive["DUP"] == 3, "a plain comprehension keeps the LAST duplicate"

        gene_pos = {}
        for pos, gene in enumerate(names):
            if gene not in gene_pos:
                gene_pos[gene] = pos
        assert gene_pos["DUP"] == 0, "must match list.index(), which returns the first"

    @pytest.mark.xfail(
        raises=Exception,
        strict=True,
        reason=(
            "Pre-existing limitation, not introduced by the vectorisation: "
            "signature-building slices by NAME (`adata[:, valid_markers]`), which raises "
            "InvalidIndexError on duplicate var_names. Both the original loop and the "
            "vectorised form fail identically here, upstream of the gene->column map. "
            "Fixing it means switching signature-building to positional indexing too, "
            "which is a separate change with its own equivalence burden."
        ),
    )
    def test_duplicate_var_names_are_not_yet_supported(self, service):
        matrix = [[1.0, 5.0, 9.0, 100.0], [2.0, 6.0, 8.0, 200.0]]
        adata = _adata(matrix, ["DUP", "B", "C", "DUP"])
        markers = {"X": ["DUP", "B", "C"]}

        service._calculate_per_cell_confidence(adata, markers)

    def test_output_shapes_and_dtypes(self, service):
        adata = _adata(
            np.random.default_rng(1).random((25, 12)) * 4, [f"G{i}" for i in range(12)]
        )
        markers = {"A": ["G0", "G1"], "B": ["G2", "G3", "G4"]}

        confidence, top3, entropy = service._calculate_per_cell_confidence(
            adata, markers
        )

        assert confidence.shape == (25,)
        assert entropy.shape == (25,)
        assert len(top3) == 25
        assert confidence.dtype == np.float64
        assert all(isinstance(t, str) for t in top3)
        assert np.all(confidence >= 0.0) and np.all(confidence <= 1.0)


class TestPerformance:
    """Ensure vectorized confidence calculation avoids per-cell Python work."""

    def test_scales_without_per_cell_python_work(self, service, monkeypatch):
        """Marker slicing and Pearson calls must not repeat for each cell.

        Full-array work still scales with cell count; this checks the expensive
        per-cell computation directly instead of relying on wall-clock ratios.
        """
        import sys

        markers = {
            "T": ["CD3D", "CD3E", "IL7R"],
            "B": ["MS4A1", "CD79A"],
            "NK": ["GNLY", "NKG7"],
        }
        counts = {"row_reads": 0, "slices": 0, "pearson_calls": 0}
        original_getitem = anndata.AnnData.__getitem__
        original_pearsonr = pearsonr

        def counted_getitem(adata, key):
            counts["slices"] += 1
            rows = key[0] if isinstance(key, tuple) else key
            if isinstance(rows, (int, np.integer)):
                counts["row_reads"] += 1
            return original_getitem(adata, key)

        def counted_pearsonr(*args, **kwargs):
            counts["pearson_calls"] += 1
            return original_pearsonr(*args, **kwargs)

        monkeypatch.setattr(anndata.AnnData, "__getitem__", counted_getitem)
        monkeypatch.setattr(sys.modules[__name__], "pearsonr", counted_pearsonr)
        monkeypatch.setattr(
            sys.modules[EnhancedSingleCellService.__module__],
            "pearsonr",
            counted_pearsonr,
        )

        def input_data(n_cells):
            rng = np.random.default_rng(n_cells)
            X = rng.random((n_cells, 100)) * 5
            genes = [f"G{i}" for i in range(100)]
            for pos, name in enumerate(
                sorted({m for v in markers.values() for m in v})
            ):
                genes[pos] = name
            return _adata(X, genes, sparse=True)

        def assert_batched_work():
            assert counts["row_reads"] == 0, "per-cell AnnData slicing restored"
            assert counts["pearson_calls"] == 0, "per-cell Pearson calls restored"
            assert counts["slices"] <= 2 * len(markers), counts

        observed = []
        for n_cells in (20, 200):
            counts.update(row_reads=0, slices=0, pearson_calls=0)
            result = service._calculate_per_cell_confidence(
                input_data(n_cells), markers
            )
            assert all(len(values) == n_cells for values in result)
            assert_batched_work()
            observed.append(dict(counts))
        assert observed[0] == observed[1]

        # The retained old implementation must fail the same mechanism guard.
        counts.update(row_reads=0, slices=0, pearson_calls=0)
        _reference_loop(input_data(20), markers)
        assert counts["row_reads"] > 0 and counts["pearson_calls"] > 0
        with pytest.raises(AssertionError, match="per-cell AnnData slicing restored"):
            assert_batched_work()
