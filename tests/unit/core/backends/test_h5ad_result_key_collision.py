"""Result-key collisions must not silently discard a payload during H5AD save."""

import errno
import os
import stat
from collections import OrderedDict
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from lobster.core.backends import h5ad_backend as backend_module
from lobster.core.backends.h5ad_backend import H5ADBackend
from lobster.core.utils.h5ad_utils import sanitize_dict, sanitize_value

RESULT_KEYS = ("de_results_A/B", "de_results_A__B")
# Backend I/O tests do not depend on provider configuration or mocked LLMs.
pytestmark = pytest.mark.no_auto_config


def _adata():
    data = ad.AnnData(X=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
    data.obs_names = ["cell_0", "cell_1"]
    data.var_names = ["ACTB", "GAPDH", "MALAT1"]
    return data


def _colliding_results(reverse):
    data = _adata()
    payloads = {
        RESULT_KEYS[0]: np.array([1.25, 2.5, 3.75]),
        RESULT_KEYS[1]: np.array([-1.25, -2.5, -3.75]),
    }
    keys = RESULT_KEYS[::-1] if reverse else RESULT_KEYS
    for key in keys:
        data.uns[key] = payloads[key]
    data.uns["de_current_results_key"] = RESULT_KEYS[0]
    return data


def _assert_source_unchanged(actual, original):
    np.testing.assert_array_equal(actual.X, original.X)
    assert actual.obs_names.tolist() == original.obs_names.tolist()
    assert actual.var_names.tolist() == original.var_names.tolist()
    assert list(actual.uns) == list(original.uns)
    assert (
        actual.uns["de_current_results_key"] == original.uns["de_current_results_key"]
    )
    for key in RESULT_KEYS:
        np.testing.assert_array_equal(actual.uns[key], original.uns[key])


def test_backend_comes_from_this_checkout():
    root = Path(__file__).resolve().parents[4]
    assert Path(backend_module.__file__).resolve().is_relative_to(root)


@pytest.mark.parametrize("reverse", [False, True], ids=["slash_first", "safe_first"])
def test_colliding_result_keys_refuse_before_creating_a_file(tmp_path, reverse):
    original = _colliding_results(reverse)
    snapshot = original.copy()
    target = tmp_path / "results.h5ad"

    try:
        with pytest.raises(ValueError, match="(?i)colli"):
            H5ADBackend(base_path=tmp_path).save(original, target)
    finally:
        _assert_source_unchanged(original, snapshot)

    assert not target.exists(), "refused results must not create a partial H5AD"


@pytest.mark.parametrize("reverse", [False, True], ids=["slash_first", "safe_first"])
def test_collision_refusal_preserves_an_existing_destination(tmp_path, reverse):
    backend = H5ADBackend(base_path=tmp_path)
    target = tmp_path / "results.h5ad"
    previous = _adata()
    previous.uns["existing_result"] = np.array([9.0, 8.0, 7.0])
    backend.save(previous, target)
    previous_bytes = target.read_bytes()

    original = _colliding_results(reverse)
    snapshot = original.copy()
    error = None
    try:
        backend.save(original, target)
    except ValueError as caught:
        error = caught
    finally:
        _assert_source_unchanged(original, snapshot)

    assert target.exists(), "a pre-write refusal must not unlink an existing file"
    assert target.read_bytes() == previous_bytes, "refused save replaced existing data"
    restored = ad.read_h5ad(target)
    np.testing.assert_array_equal(restored.X, previous.X)
    np.testing.assert_array_equal(
        restored.uns["existing_result"], previous.uns["existing_result"]
    )
    assert error is not None, "colliding result keys must be refused visibly"
    assert "colli" in str(error).lower()


def _existing_destination(backend, target):
    previous = _adata()
    previous.uns["existing_result"] = np.array([9.0, 8.0, 7.0])
    backend.save(previous, target)
    return previous, target.read_bytes()


def _assert_existing_destination(target, previous, previous_bytes):
    assert target.exists(), "a failed save must not unlink the previous file"
    assert target.read_bytes() == previous_bytes, "a failed save replaced old bytes"
    restored = ad.read_h5ad(target)
    np.testing.assert_array_equal(restored.X, previous.X)
    np.testing.assert_array_equal(
        restored.uns["existing_result"], previous.uns["existing_result"]
    )


def test_pre_write_failure_preserves_existing_data(tmp_path, monkeypatch):
    backend = H5ADBackend(base_path=tmp_path)
    target = tmp_path / "results.h5ad"
    previous, previous_bytes = _existing_destination(backend, target)
    previous_entries = set(tmp_path.iterdir())
    incoming = _adata()
    snapshot = incoming.copy()

    def refuse_conversion(adata):
        raise ValueError("unsupported serialization dtype")

    monkeypatch.setattr(backend, "_convert_arrow_to_standard", refuse_conversion)
    with pytest.raises(ValueError, match="unsupported serialization dtype"):
        backend.save(incoming, target)

    _assert_existing_destination(target, previous, previous_bytes)
    assert set(tmp_path.iterdir()) == previous_entries
    np.testing.assert_array_equal(incoming.X, snapshot.X)
    assert incoming.obs_names.tolist() == snapshot.obs_names.tolist()
    assert incoming.var_names.tolist() == snapshot.var_names.tolist()


def test_mid_write_failure_preserves_existing_data(tmp_path, monkeypatch):
    backend = H5ADBackend(base_path=tmp_path)
    target = tmp_path / "results.h5ad"
    previous, previous_bytes = _existing_destination(backend, target)
    previous_entries = set(tmp_path.iterdir())
    incoming = _adata()
    snapshot = incoming.copy()

    def partial_write_then_fail(adata, filename, **kwargs):
        # Controlled I/O fault: never fill the real filesystem to simulate ENOSPC.
        Path(filename).write_bytes(b"partial HDF5 output")
        raise OSError(errno.ENOSPC, "simulated write exhaustion")

    monkeypatch.setattr(ad.AnnData, "write_h5ad", partial_write_then_fail)
    with pytest.raises(ValueError, match="simulated write exhaustion"):
        backend.save(incoming, target)

    _assert_existing_destination(target, previous, previous_bytes)
    assert set(tmp_path.iterdir()) == previous_entries
    np.testing.assert_array_equal(incoming.X, snapshot.X)
    assert incoming.obs_names.tolist() == snapshot.obs_names.tolist()
    assert incoming.var_names.tolist() == snapshot.var_names.tolist()


@pytest.mark.parametrize("existing", [False, True])
def test_replace_failure_preserves_previous_destination(
    tmp_path, monkeypatch, existing
):
    backend = H5ADBackend(base_path=tmp_path)
    target = tmp_path / "results.h5ad"
    if existing:
        previous, previous_bytes = _existing_destination(backend, target)
    previous_entries = set(tmp_path.iterdir())

    def refuse_replace(source, destination, **kwargs):
        assert Path(source).parent == target.parent
        assert Path(source) != target
        assert Path(source).is_file()
        assert Path(destination) == target
        raise PermissionError("simulated replacement denial")

    monkeypatch.setattr(os, "replace", refuse_replace)
    with pytest.raises(ValueError, match="simulated replacement denial"):
        backend.save(_adata(), target)

    if existing:
        _assert_existing_destination(target, previous, previous_bytes)
    else:
        assert not target.exists()
    assert set(tmp_path.iterdir()) == previous_entries


@pytest.mark.parametrize("existing", [False, True])
def test_pre_commit_logging_failure_preserves_final_path(
    tmp_path, monkeypatch, existing
):
    backend = H5ADBackend(base_path=tmp_path)
    target = tmp_path / "results.h5ad"
    if existing:
        previous, previous_bytes = _existing_destination(backend, target)
    reached = []

    def fail_logging(*args, **kwargs):
        reached.append(True)
        raise RuntimeError("simulated pre-commit logging failure")

    monkeypatch.setattr(backend, "_log_operation", fail_logging)
    with pytest.raises(ValueError, match="simulated pre-commit logging failure"):
        backend.save(_adata(), target)

    assert reached == [True]
    if existing:
        _assert_existing_destination(target, previous, previous_bytes)
    else:
        assert not target.exists()
    assert set(tmp_path.iterdir()) == ({target} if existing else set())


@pytest.mark.parametrize("existing", [False, True])
def test_pre_commit_stat_failure_preserves_final_path(tmp_path, monkeypatch, existing):
    backend = H5ADBackend(base_path=tmp_path)
    target = tmp_path / "results.h5ad"
    if existing:
        previous, previous_bytes = _existing_destination(backend, target)
    original_write = ad.AnnData.write_h5ad
    original_stat = Path.stat
    completed = []

    def write_then_mark(data, filename, **kwargs):
        result = original_write(data, filename, **kwargs)
        completed.append(Path(filename))
        return result

    def fail_staged_stat(path, *args, **kwargs):
        if path in completed:
            raise OSError("simulated staged stat failure")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(ad.AnnData, "write_h5ad", write_then_mark)
    monkeypatch.setattr(Path, "stat", fail_staged_stat)
    with pytest.raises(ValueError, match="simulated staged stat failure"):
        backend.save(_adata(), target)

    assert completed
    if existing:
        _assert_existing_destination(target, previous, previous_bytes)
    else:
        assert not target.exists()
    assert set(tmp_path.iterdir()) == ({target} if existing else set())


@pytest.mark.parametrize("existing", [False, True])
def test_post_commit_bookkeeping_error_reports_saved_data(
    tmp_path, monkeypatch, existing
):
    backend = H5ADBackend(base_path=tmp_path)
    target = tmp_path / "results.h5ad"
    if existing:
        _existing_destination(backend, target)
    incoming = _adata()
    incoming.X = incoming.X + 5

    def fail_committed_bookkeeping(operation, *args, **kwargs):
        if operation == "save":
            raise RuntimeError("simulated committed bookkeeping failure")

    monkeypatch.setattr(backend, "_log_operation", fail_committed_bookkeeping)
    with pytest.raises(ValueError) as error:
        backend.save(incoming, target)

    message = str(error.value).lower()
    assert "data was saved at" in message
    assert str(target) in str(error.value)
    assert "bookkeeping failed" in message
    restored = ad.read_h5ad(target)
    np.testing.assert_array_equal(restored.X, incoming.X)
    assert restored.obs_names.tolist() == incoming.obs_names.tolist()
    assert restored.var_names.tolist() == incoming.var_names.tolist()
    assert set(tmp_path.iterdir()) == {target}


def test_replacement_preserves_existing_mode_bits(tmp_path):
    backend = H5ADBackend(base_path=tmp_path)
    target = tmp_path / "results.h5ad"
    _existing_destination(backend, target)
    target.chmod(0o664)
    backend.save(_adata(), target)
    assert stat.S_IMODE(target.stat().st_mode) == 0o664


@pytest.mark.parametrize("mask", [0o022, 0o027])
def test_fresh_mode_matches_non_atomic_writer_default(tmp_path, mask):
    target = tmp_path / "results.h5ad"
    previous_mask = os.umask(mask)
    try:
        H5ADBackend(base_path=tmp_path).save(_adata(), target)
    finally:
        os.umask(previous_mask)
    assert stat.S_IMODE(target.stat().st_mode) == (0o666 & ~mask)


def test_read_only_target_is_refused_before_preparation(tmp_path, monkeypatch):
    backend = H5ADBackend(base_path=tmp_path)
    target = tmp_path / "results.h5ad"
    previous, previous_bytes = _existing_destination(backend, target)
    target.chmod(0o444)
    original_converter = backend._convert_arrow_to_standard
    prepared = []

    def observe_preparation(data):
        prepared.append(True)
        return original_converter(data)

    monkeypatch.setattr(backend, "_convert_arrow_to_standard", observe_preparation)
    try:
        with pytest.raises(ValueError, match="(?i)(read.only|permission)"):
            backend.save(_adata(), target)
        assert not prepared
        _assert_existing_destination(target, previous, previous_bytes)
        assert stat.S_IMODE(target.stat().st_mode) == 0o444
    finally:
        target.chmod(0o600)


@pytest.mark.parametrize("relative", [False, True])
def test_symlink_target_is_preserved_when_real_path_is_replaced(tmp_path, relative):
    backend = H5ADBackend(base_path=tmp_path)
    real = tmp_path / "real.h5ad"
    _existing_destination(backend, real)
    target = tmp_path / "link.h5ad"
    target.symlink_to(real.name)
    incoming = _adata()
    incoming.X = incoming.X + 5
    backend.save(incoming, target.name if relative else target)
    assert target.is_symlink()
    assert target.readlink() == Path(real.name)
    restored = ad.read_h5ad(real)
    np.testing.assert_array_equal(restored.X, incoming.X)
    assert set(tmp_path.iterdir()) == {target, real}


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("mapping_type", [dict, OrderedDict])
def test_nested_dictionary_collision_refuses_without_mutation(
    tmp_path, reverse, mapping_type
):
    backend = H5ADBackend(base_path=tmp_path)
    target = tmp_path / "results.h5ad"
    previous, previous_bytes = _existing_destination(backend, target)
    incoming = _adata()
    keys = RESULT_KEYS[::-1] if reverse else RESULT_KEYS
    payload = mapping_type(
        (key, np.array([index, index + 1])) for index, key in enumerate(keys)
    )
    incoming.uns["nested"] = {"results": payload}
    snapshot = incoming.copy()

    with pytest.raises(ValueError, match="(?i)colli"):
        backend.save(incoming, target)

    _assert_existing_destination(target, previous, previous_bytes)
    assert list(incoming.uns["nested"]["results"]) == list(
        snapshot.uns["nested"]["results"]
    )
    for key in keys:
        np.testing.assert_array_equal(
            incoming.uns["nested"]["results"][key],
            snapshot.uns["nested"]["results"][key],
        )
    assert set(tmp_path.iterdir()) == {target}


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("slot", ["obsm", "varm", "layers"])
def test_other_slot_collision_refuses_without_mutation(tmp_path, reverse, slot):
    backend = H5ADBackend(base_path=tmp_path)
    target = tmp_path / "results.h5ad"
    previous, previous_bytes = _existing_destination(backend, target)
    incoming = _adata()
    shape = (
        incoming.shape
        if slot == "layers"
        else (incoming.n_vars if slot == "varm" else incoming.n_obs, 1)
    )
    keys = RESULT_KEYS[::-1] if reverse else RESULT_KEYS
    for index, key in enumerate(keys):
        getattr(incoming, slot)[key] = np.full(shape, index + 1, dtype=np.float32)
    snapshot = incoming.copy()

    with pytest.raises(ValueError, match="(?i)colli"):
        backend.save(incoming, target)

    _assert_existing_destination(target, previous, previous_bytes)
    assert list(getattr(incoming, slot)) == list(getattr(snapshot, slot))
    for key in keys:
        np.testing.assert_array_equal(
            getattr(incoming, slot)[key], getattr(snapshot, slot)[key]
        )
    assert set(tmp_path.iterdir()) == {target}


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("slot", ["obs", "var"])
def test_column_normalization_collision_refuses_without_mutation(
    tmp_path, reverse, slot
):
    backend = H5ADBackend(base_path=tmp_path)
    target = tmp_path / "results.h5ad"
    previous, previous_bytes = _existing_destination(backend, target)
    incoming = _adata()
    frame = getattr(incoming, slot)
    keys = RESULT_KEYS[::-1] if reverse else RESULT_KEYS
    for index, key in enumerate(keys):
        frame[key] = np.full(len(frame), index + 1)
    snapshot = frame.copy(deep=True)

    with pytest.raises(ValueError, match="(?i)colli"):
        backend.save(incoming, target)

    _assert_existing_destination(target, previous, previous_bytes)
    pd.testing.assert_frame_equal(getattr(incoming, slot), snapshot)
    assert set(tmp_path.iterdir()) == {target}


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("sanitize", [sanitize_dict, sanitize_value])
def test_dictionary_utility_collision_refuses_without_mutation(reverse, sanitize):
    keys = RESULT_KEYS[::-1] if reverse else RESULT_KEYS
    payload = {key: np.array([index, index + 1]) for index, key in enumerate(keys)}
    snapshot = {key: value.copy() for key, value in payload.items()}
    with pytest.raises(ValueError, match="(?i)colli"):
        sanitize(payload)
    assert list(payload) == list(snapshot)
    for key in keys:
        np.testing.assert_array_equal(payload[key], snapshot[key])


@pytest.mark.parametrize("existing", [False, True])
def test_pre_commit_state_restoration_failure_preserves_final_path(
    tmp_path, monkeypatch, existing
):
    backend = H5ADBackend(base_path=tmp_path)
    target = tmp_path / "results.h5ad"
    if existing:
        previous, previous_bytes = _existing_destination(backend, target)
    before_settings = (
        pd.options.future.infer_string,
        ad.settings.allow_write_nullable_strings,
    )
    wrapper_type = type(pd.options.future)
    original_setter = wrapper_type.__setattr__
    original_log = backend._log_operation
    ready = []
    failed = []

    def observe_prepared(operation, *args, **kwargs):
        original_log(operation, *args, **kwargs)
        if operation == "save_prepared":
            ready.append(True)

    def fail_restore_once(wrapper, name, value):
        if name == "infer_string" and ready and not failed:
            failed.append(True)
            raise RuntimeError("simulated pre-commit state restoration failure")
        return original_setter(wrapper, name, value)

    monkeypatch.setattr(backend, "_log_operation", observe_prepared)
    monkeypatch.setattr(wrapper_type, "__setattr__", fail_restore_once)
    with pytest.raises(
        ValueError, match="simulated pre-commit state restoration failure"
    ):
        backend.save(_adata(), target)

    assert failed == [True]
    if existing:
        _assert_existing_destination(target, previous, previous_bytes)
    else:
        assert not target.exists()
    assert before_settings == (
        pd.options.future.infer_string,
        ad.settings.allow_write_nullable_strings,
    )
    assert set(tmp_path.iterdir()) == ({target} if existing else set())


@pytest.mark.parametrize("existing", [False, True])
def test_staged_chmod_failure_preserves_final_path(tmp_path, monkeypatch, existing):
    backend = H5ADBackend(base_path=tmp_path)
    target = tmp_path / "results.h5ad"
    if existing:
        previous, previous_bytes = _existing_destination(backend, target)
    original_chmod = Path.chmod

    def refuse_staged_chmod(path, *args, **kwargs):
        if path.name.startswith(".lobster-h5ad-"):
            raise PermissionError("simulated staged mode failure")
        return original_chmod(path, *args, **kwargs)

    # Existing targets require explicit mode restoration; for a fresh target,
    # mode failure is injected at staged stat, whose outcome must be identical.
    if existing:
        monkeypatch.setattr(Path, "chmod", refuse_staged_chmod)
    else:
        original_stat = Path.stat

        def refuse_staged_stat(path, *args, **kwargs):
            if path.name.startswith(".lobster-h5ad-"):
                raise PermissionError("simulated staged mode failure")
            return original_stat(path, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", refuse_staged_stat)
    with pytest.raises(ValueError, match="simulated staged mode failure"):
        backend.save(_adata(), target)
    if existing:
        _assert_existing_destination(target, previous, previous_bytes)
    else:
        assert not target.exists()
    assert set(tmp_path.iterdir()) == ({target} if existing else set())


def test_exclusive_staging_creation_never_unlinks_a_preexisting_file(
    tmp_path, monkeypatch
):
    backend = H5ADBackend(base_path=tmp_path)
    target = tmp_path / "results.h5ad"
    previous, previous_bytes = _existing_destination(backend, target)
    other = tmp_path / ".lobster-h5ad-fixed.h5ad"
    other.write_bytes(b"file not created by this save call")

    class FixedNonce:
        hex = "fixed"

    monkeypatch.setattr(backend_module.uuid, "uuid4", lambda: FixedNonce())
    with pytest.raises(ValueError, match="(?i)(exists|file)"):
        backend.save(_adata(), target)
    assert other.read_bytes() == b"file not created by this save call"
    _assert_existing_destination(target, previous, previous_bytes)
    assert set(tmp_path.iterdir()) == {target, other}


@pytest.mark.parametrize("existing", [False, True])
def test_staging_is_private_and_owner_writable_at_native_writer_entry(
    tmp_path, monkeypatch, existing
):
    backend = H5ADBackend(base_path=tmp_path)
    target = tmp_path / "results.h5ad"
    if existing:
        _existing_destination(backend, target)
        target.chmod(0o600)
    original_write = ad.AnnData.write_h5ad
    original_open = os.open
    observed_modes = []
    creation_modes = []

    def observe_creation(path, flags, mode=0o777, **kwargs):
        if (
            Path(path).name.startswith(".lobster-h5ad-")
            and Path(path).suffix == ".h5ad"
        ):
            creation_modes.append(mode)
        return original_open(path, flags, mode, **kwargs)

    def observe_staging(data, filename, **kwargs):
        mode = stat.S_IMODE(Path(filename).stat().st_mode)
        observed_modes.append(mode)
        assert mode == 0o600, "native writer must receive a private owner-RW stage"
        return original_write(data, filename, **kwargs)

    monkeypatch.setattr(ad.AnnData, "write_h5ad", observe_staging)
    monkeypatch.setattr(os, "open", observe_creation)
    before_mask = os.umask(0o022)
    try:
        backend.save(_adata(), target)
    finally:
        os.umask(before_mask)
    assert observed_modes == [0o600]
    assert creation_modes == [0o600]
    assert stat.S_IMODE(target.stat().st_mode) == (0o600 if existing else 0o644)
    assert set(tmp_path.iterdir()) == {target}


def test_fresh_owner_restrictive_umask_native_round_trip(tmp_path):
    # A privileged writer would not establish the permission-reopen regression.
    assert os.geteuid() != 0, "this permission control requires a non-root writer"
    backend = H5ADBackend(base_path=tmp_path)
    target = tmp_path / "results.h5ad"
    incoming = _adata()
    before_mask = os.umask(0o222)
    try:
        backend.save(incoming, target)
    finally:
        os.umask(before_mask)
    assert stat.S_IMODE(target.stat().st_mode) == 0o444
    restored = ad.read_h5ad(target)
    np.testing.assert_array_equal(restored.X, incoming.X)
    assert restored.obs_names.tolist() == incoming.obs_names.tolist()
    assert restored.var_names.tolist() == incoming.var_names.tolist()
    assert set(tmp_path.iterdir()) == {target}


@pytest.mark.parametrize(
    "existing,operation", [(False, "fstat"), (False, "fchmod"), (True, "fchmod")]
)
def test_staging_descriptor_fault_preserves_final_and_closes_descriptor(
    tmp_path, monkeypatch, existing, operation
):
    backend = H5ADBackend(base_path=tmp_path)
    target = tmp_path / "results.h5ad"
    if existing:
        previous, previous_bytes = _existing_destination(backend, target)
    original_open = os.open
    original_fstat = os.fstat
    original_operation = getattr(os, operation)
    descriptors = []
    faults = []

    def observe_open(path, flags, *args, **kwargs):
        descriptor = original_open(path, flags, *args, **kwargs)
        if Path(path).name.startswith(".lobster-h5ad-"):
            descriptors.append(descriptor)
        return descriptor

    def refuse_owned_operation(descriptor, *args, **kwargs):
        if descriptor in descriptors:
            faults.append(descriptor)
            raise OSError("simulated staging descriptor failure")
        return original_operation(descriptor, *args, **kwargs)

    monkeypatch.setattr(os, "open", observe_open)
    monkeypatch.setattr(os, operation, refuse_owned_operation)
    with pytest.raises(ValueError, match="simulated staging descriptor failure"):
        backend.save(_adata(), target)
    assert len(faults) == 1
    for descriptor in set(descriptors):
        with pytest.raises(OSError) as error:
            original_fstat(descriptor)
        assert error.value.errno == errno.EBADF
    if existing:
        _assert_existing_destination(target, previous, previous_bytes)
    else:
        assert not target.exists()
    assert set(tmp_path.iterdir()) == ({target} if existing else set())


def test_exclusive_mode_probe_never_unlinks_a_preexisting_file(tmp_path, monkeypatch):
    target = tmp_path / "results.h5ad"
    other = tmp_path / ".lobster-h5ad-fixed.mode"
    other.write_bytes(b"not owned by this save call")

    class FixedNonce:
        hex = "fixed"

    monkeypatch.setattr(backend_module.uuid, "uuid4", lambda: FixedNonce())
    with pytest.raises(ValueError, match="(?i)(exists|file)"):
        H5ADBackend(base_path=tmp_path).save(_adata(), target)
    assert other.read_bytes() == b"not owned by this save call"
    assert not target.exists()
    assert set(tmp_path.iterdir()) == {other}


def test_save_never_changes_the_process_umask(tmp_path, monkeypatch):
    target = tmp_path / "results.h5ad"

    def refuse_umask_change(*args, **kwargs):
        raise AssertionError("saving must not alter the process-wide umask")

    monkeypatch.setattr(os, "umask", refuse_umask_change)
    H5ADBackend(base_path=tmp_path).save(_adata(), target)
    np.testing.assert_array_equal(ad.read_h5ad(target).X, _adata().X)
    assert set(tmp_path.iterdir()) == {target}


def test_staging_permission_preparation_without_descriptor_chmod(tmp_path, monkeypatch):
    # Simulate API absence on this native profile; this is not a platform test.
    monkeypatch.delattr(os, "fchmod")
    target = tmp_path / "results.h5ad"
    before_mask = os.umask(0o222)
    try:
        H5ADBackend(base_path=tmp_path).save(_adata(), target)
    finally:
        os.umask(before_mask)
    assert stat.S_IMODE(target.stat().st_mode) == 0o444
    np.testing.assert_array_equal(ad.read_h5ad(target).X, _adata().X)
    assert set(tmp_path.iterdir()) == {target}


def test_distinct_safe_result_keys_round_trip_without_loss(tmp_path):
    original = _adata()
    original.uns["de_results_A_vs_B"] = np.array([1.25, 2.5, 3.75])
    original.uns["de_results_C_vs_D"] = np.array([-1.25, -2.5, -3.75])
    original.uns["de_current_results_key"] = "de_results_C_vs_D"
    snapshot = original.copy()
    target = tmp_path / "results.h5ad"
    H5ADBackend(base_path=tmp_path).save(original, target)

    restored = ad.read_h5ad(target)
    np.testing.assert_array_equal(restored.X, snapshot.X)
    assert restored.obs_names.tolist() == snapshot.obs_names.tolist()
    assert restored.var_names.tolist() == snapshot.var_names.tolist()
    assert set(restored.uns) == set(snapshot.uns)
    assert restored.uns["de_current_results_key"] == "de_results_C_vs_D"
    for key in ("de_results_A_vs_B", "de_results_C_vs_D"):
        np.testing.assert_array_equal(restored.uns[key], snapshot.uns[key])
        np.testing.assert_array_equal(original.uns[key], snapshot.uns[key])
    np.testing.assert_array_equal(original.X, snapshot.X)
