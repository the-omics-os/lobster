"""Integration tests never write the user's own config folder."""

import os

from lobster.config import global_config
from tests.conftest import _user_files


def test_a_test_gets_a_config_folder_of_its_own(tmp_path_factory):
    assert global_config.CONFIG_DIR.is_relative_to(tmp_path_factory.getbasetemp())


def _with_old_times(path, st):
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))


def test_the_guard_sees_a_file_rewritten_with_its_old_times(tmp_path):
    real = tmp_path / "projects.json"
    real.write_text("AAAA")
    before = _user_files(tmp_path)
    st = os.stat(real)

    real.write_text("BBBB")
    _with_old_times(real, st)

    assert _user_files(tmp_path) != before


def test_the_guard_sees_a_file_replaced_by_one_with_its_old_times(tmp_path):
    real = tmp_path / "projects.json"
    real.write_text("AAAA")
    before = _user_files(tmp_path)
    st = os.stat(real)

    new = tmp_path / "projects.tmp"
    new.write_text("CCCC")
    _with_old_times(new, st)
    os.replace(new, real)

    assert _user_files(tmp_path) != before
