"""OLD-vs-NEW guard comparison — runs INSIDE the production image (pandas 3).

Reproduces the live contaminant/reverse corruption with the code that is on
production today and proves the hardened guard fixes it, in the exact runtime —
no downgrade, no version ambiguity. Prints one row per flag encoding and exits
nonzero if the hardened guard is ever wrong. Emits a JSON summary as the last
line so the pytest orchestrator can assert on it.
"""
import json
import sys

import numpy as np
import pandas as pd
from pandas.api import types as pdt

FLAG = [True, False, True, False]  # correct keep = {F1, F3} = 2


def make(enc):
    plus = {True: "+", False: ""}
    tf = {True: "True", False: "False"}
    tens = {True: "1", False: "0"}
    return {
        "text_plus": lambda: pd.Series([plus[v] for v in FLAG]),
        "text_10": lambda: pd.Series([tens[v] for v in FLAG]),
        "text_tf": lambda: pd.Series([tf[v] for v in FLAG]),
        "bool": lambda: pd.Series(FLAG, dtype=bool),
        "numeric": lambda: pd.Series([1 if v else 0 for v in FLAG], dtype="int64"),
        "cat_text": lambda: pd.Series(pd.Categorical([plus[v] for v in FLAG])),
    }[enc]()


def OLD_mask(col):
    """The guard currently on production (shared_tools.py:663/675)."""
    if col.dtype == object or col.dtype.name == "category":
        return col.fillna("").astype(str).isin(["+", "True", "true", "1", "yes", "Yes"])
    return col.fillna(False).astype(bool)


_TRUTHY = {"+", "true", "1", "yes"}


def NEW_mask(col):
    """Mirror of lobster.core.utils.dtype_guards.boolean_flag_mask (inlined
    because the repo is not importable inside the production image)."""
    def _t(s):
        return s.fillna("").astype(str).str.strip().str.lower().isin(_TRUTHY)

    def _n(s):
        return pd.to_numeric(s, errors="coerce").astype("float64").fillna(0.0).ne(0.0)

    def _b(s):
        return s.astype("boolean").fillna(False).astype(bool)

    if isinstance(col.dtype, pd.CategoricalDtype):
        c = col.cat.categories.dtype
        v = col.astype(object)
        return _b(v) if pdt.is_bool_dtype(c) else _n(v) if pdt.is_numeric_dtype(c) else _t(v)
    if pdt.is_bool_dtype(col):
        return _b(col)
    if pdt.is_numeric_dtype(col):
        return _n(col)
    if pdt.is_string_dtype(col):
        return _t(col)
    coerced = pd.to_numeric(col, errors="coerce")
    non_missing = col.notna()
    if non_missing.sum() == 0 or bool((coerced.notna() | ~non_missing).all()):
        return _n(col)
    return _t(col)


rows = []
new_wrong = 0
old_emptied = 0
for enc in ["text_plus", "text_10", "text_tf", "bool", "numeric", "cat_text"]:
    col = make(enc)
    old_keep = int((~OLD_mask(col)).sum())
    new_keep = int((~NEW_mask(col)).sum())
    if new_keep != 2:
        new_wrong += 1
    if old_keep == 0:
        old_emptied += 1
    rows.append({"encoding": enc, "dtype": str(col.dtype),
                 "old_kept": old_keep, "new_kept": new_keep})
    print(f"{enc:10s} dtype={str(col.dtype):9s} OLD={old_keep}/4 NEW={new_keep}/4")

summary = {"pandas": pd.__version__, "rows": rows,
           "new_wrong": new_wrong, "old_emptied": old_emptied}
print("SUMMARY_JSON " + json.dumps(summary))
sys.exit(1 if new_wrong else 0)
