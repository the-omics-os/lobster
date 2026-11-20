"""Integration tests never write the user's own config folder."""

from lobster.config import global_config


def test_a_test_gets_a_config_folder_of_its_own(tmp_path_factory):
    assert global_config.CONFIG_DIR.is_relative_to(tmp_path_factory.getbasetemp())
