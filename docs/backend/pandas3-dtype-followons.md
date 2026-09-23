# pandas-3 dtype follow-on hardening — deferred-items registry

**Status:** tracked, not yet implemented. **Owner:** engine (lobster-ga5r).
**Created:** 2026-08-17, alongside branch `fix/pandas3-followon-hardening`.

> Why this file exists: a deferred defect that is not written down is an
> undiscovered defect with extra steps. This is the single address for the
> pandas-3 dtype follow-on work so it does not live only in ephemeral chat.
> (Intended for `.planning/PANDAS3_DTYPE_FIX/`, but `**/.planning/` is
> gitignored — so it lives here, tracked, to actually survive.)

## Context

The pandas-3 production dtype fix shipped at engine commit
`04d833bab92714babf1c5089feee36edab6fee82` (deployed task def 440; prod runs
pandas 2.3.3 / anndata 0.13.2). Branch `fix/pandas3-dtype-corruption` is frozen
at that SHA permanently as the production audit artifact — do not advance it.

The **follow-on hardening batch** (`fix/pandas3-followon-hardening`) contains:
in-scope = the categorical-string flag test closer, the h5ad object-path
conservative coercion (`_numeric_coercion_is_lossless`), the adapter
`_is_numeric_string_column` full-column scan, the `is_arrow_dtype` StringDtype
fix, and the `is_object_dtype` gate refinement. Everything below is **deferred**
out of that batch.

Deferral bar: **no regression, and the authorized scope is closed** — not "every
adjacent defect is fixed." Each item below was verified to be pre-existing at
`04d833ba`; the follow-on batch is strictly better-or-equal on all of them.

---

## Deferred defects (all pre-existing at `04d833ba`, none introduced by the batch)

### D1 — Identifier-vs-measurement is undecidable from value alone

- **Where:** `lobster/core/backends/h5ad_backend.py` sanitize object-branch
  (`_numeric_coercion_is_lossless`); conceptually also every value-based numeric
  detector.
- **Mechanism:** a canonical numeric string like `"1"` is coerced to `1` whether
  it denotes a measurement or an identifier; `"1e3"` is preserved as a string
  whether it denotes an ID or a measurement. No value-only predicate can
  distinguish the two meanings. `_numeric_coercion_is_lossless` is a
  *spelling-preservation* heuristic (it guarantees an accepted coercion never
  changes a value's text), **not** a complete identifier classifier — its
  docstring says so.
- **Not a regression:** pre-fix, unconditional `pd.to_numeric` coerced `"1"` the
  same way *and also* corrupted `"001"`/`"1e3"`; the batch preserves those. Strict
  improvement.
- **Real fix:** type-preserving persistence — do not infer numeric meaning for
  values that are already strings. Numeric-string normalization belongs at
  ingestion (adapters with an explicit intensity role) or behind an explicit
  column-role / schema signal. Architectural; not a heuristic tweak.

### D2 — `pd.to_numeric` returning an object dtype → unwritable, and the helper accepts it

- **Where:** `lobster/core/backends/h5ad_backend.py` object-branch coercion (the
  single `pd.to_numeric(df[col])` site) + `_numeric_coercion_is_lossless`.
- **Mechanism:** for out-of-native-range integers (e.g. `"18446744073709551616"`
  = uint64 max + 1 under pandas 3, or a mixed signed/unsigned python-int object
  column under both versions) `pd.to_numeric` returns an **object** Series of
  python ints. The helper sees `Integral` values, accepts the exact spelling,
  and assigns an object Series back to `adata.var`/`obs`; anndata 0.13.2 then
  fails to write it (`Can't implicitly convert non-string objects to strings`).
- **Not a regression:** pre-fix unconditional `pd.to_numeric` produced the same
  object Series and the same write failure. Not observed in api-u0z6's
  production-corpus enumeration (8,570 objects); no known omics var/obs column
  carries 20-digit integer identifiers.
- **Real fix:** accept a coercion only when its result has a **non-object numeric
  dtype**; treat an object-dtype `to_numeric` result as unusable, and have the
  caller run the non-string stringification fallback whenever coercion is absent
  **or** unusable (a bare helper `False` is insufficient — the current final
  `else` leaves non-string objects untouched). Needs a caller restructure + a
  cross-version test.

### D3 — Non-string object scalars are assumed lossless (bytes, Decimal)

- **Where:** `lobster/core/backends/h5ad_backend.py` `_numeric_coercion_is_lossless`,
  the `if not isinstance(orig, str): continue` fast path.
- **Mechanism:** the fast path skips the round-trip check for any non-string
  object. `b"001"` → `1` (bytes identity lost; `bytes` is not `str`);
  `Decimal("1.0000000000000000001")` → `1.0` (precision discarded).
- **Not a regression:** pre-fix unconditional `pd.to_numeric` coerced these
  identically. Rare in var/obs.
- **Real fix:** restrict the fast path to scalar types whose conversion is
  demonstrably exact (python `int`/`float`); treat `bytes`/`bytearray` as text
  (decode + compare spelling); compare `Decimal`/`Fraction` exactly or
  conservatively stringify. Combine with D2's non-object-result check.

### D4 — Numeric-looking identifier metadata mis-classified as intensity

- **Where:** `lobster/core/adapters/proteomics_adapter.py`
  `_separate_intensity_metadata_columns` / `_is_numeric_string_column`
  (`:349-376`, `:380-415`) and `lobster/core/adapters/metabolomics_adapter.py`
  (`:253-280`, `:284-319`). Standardizers already recognize exact `id`
  (`proteomics_adapter.py:431`, `metabolomics_adapter.py:335`) but classification
  never routes the column there.
- **Mechanism:** a column named `id` (or `sample_code`) holding `["001","002",
  "003"]` is fully numeric-coercible, so `_is_numeric_string_column` returns
  `True`, the column is kept as intensity, and downstream
  `pd.to_numeric(errors="coerce")` rewrites it to `[1,2,3]` — leading zeros lost.
- **Not a regression:** the classifier never had name-based ID detection; the
  full-column scan (this batch) did not introduce this. Position-independent but
  semantically wrong either way.
- **Real fix:** recognize tokenized identifier names before value-based
  detection — exact `id` and separator-delimited `_id`. **Do NOT use a raw
  `"id" in name` substring:** `"lipid"` contains `"id"`, and this guard lives in
  the **metabolomics** adapter where lipid names are the expected content, not a
  hypothetical edge. A naive `col.lower().endswith("id")` hits `"lipid"` on the
  first real dataset. Ambiguous numeric-text columns ultimately require the
  explicit `intensity_columns` / schema signal (see D1); value coercibility alone
  cannot prove meaning.

---

## D5 — AST lint against reintroducing type-identity dtype comparisons (engine CI)

Owner: engine (this repo's CI). Replaces the removed grep. Requirements
(hardened after the corpus review below):

- Detect `ast.Compare` nodes comparing a dtype against a type-identity literal
  (`== "object"`, `== object`, `.dtype == "category"`, etc.).
- Allowlist keyed by **normalized node** `(file, symbol, normalized-comparison)`,
  **not** by line number and **not** by whole symbol — suppressing a symbol hides
  a second bad comparison next to a legitimate one.
- Explicit **per-class policy** for tests, stubs, vendored code, generated files,
  and code-template strings — **no blanket allowlist** for `lobster-ml` or
  `tests/**`.
- Stated contract for parser version and parse errors: fail-closed on an
  unparseable file blocks every PR; skip-on-error is a hole. Choose and document.
- **Must render or separately parse executable code-template strings** — a plain
  AST walk silently ignores comparisons that live inside template strings (see
  the 6 template sites below). This is the sharpest structural risk: AST produces
  a false negative exactly where a grep would have caught it.

Legitimate allowlisted entry: `h5ad_backend.py` object-gate — now
`pd.api.types.is_object_dtype(...)`, which states intent in code and needs **no**
allowlist entry. That is the pattern to prefer.

## D6 — 15 live type-identity comparisons in `packages/lobster-ml/**`

Same bug class as the shipped fix (`.values.dtype == object` and
`.dtype == "object"` both evaluate False for a pandas-3 default string Series /
Index). **Not live in production** — `lobster-ml` is not installed in the Cloud
image — hence a finding, not a P0. Found by api-u0z6's Codex review.

Nine live `ast.Compare` nodes:
```
feature_selection_service.py:1109,1334,1339
ml_preparation_service.py:81
ml_preprocessing_service.py:48,1155,1394
cross_validation_service.py:658
machine_learning_expert.py:674
```

Six inside executable code-template strings (invisible to a plain AST walk):
```
feature_selection_service.py:280,566,570
ml_preparation_service.py:823
ml_preprocessing_service.py:526
cross_validation_service.py:271
```

Fix uses the canonical predicates in `lobster/core/utils/dtype_guards.py`
(`is_text_dtype`, `boolean_flag_mask`) where semantic, or `is_object_dtype` where
the question is genuinely representation-level.
