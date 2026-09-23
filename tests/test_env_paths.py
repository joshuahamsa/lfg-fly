import os

import pytest

from lfg_fly import env, paths


def test_thread_caps_are_set_on_import():
    for var in env.THREAD_VARS:
        assert os.environ[var] == "1"


def test_data_dir_follows_env(_data_dir):
    assert paths.data_dir() == _data_dir
    assert paths.raw_dir() == _data_dir / "raw"
    assert paths.graph_dir() == _data_dir / "graph"
    assert paths.cache_dir() == _data_dir / "cache"


def test_network_dir_is_scoped(_data_dir):
    assert paths.network_dir("mainnet") == _data_dir / "mainnet"
    assert paths.network_dir("testnet") == _data_dir / "testnet"
    with pytest.raises(ValueError):
        paths.network_dir("devnet")


def test_repo_root_contains_pyproject():
    assert (paths.repo_root() / "pyproject.toml").exists()


def test_cli_help_exits_zero(capsys):
    from lfg_fly import cli

    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    assert "fly" in capsys.readouterr().out
