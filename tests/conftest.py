import lfg_fly  # noqa: F401,I001  (must be first: applies the CPU thread caps)
import pytest


@pytest.fixture(autouse=True)
def _data_dir(tmp_path, monkeypatch):
    """No test may touch ~/fly-data: every test gets its own FLY_DATA_DIR."""
    monkeypatch.setenv("FLY_DATA_DIR", str(tmp_path / "fly-data"))
    return tmp_path / "fly-data"


@pytest.fixture(autouse=True)
def _no_dotenv(tmp_path, monkeypatch):
    """No test may read the operator's ~/lfg-fly/.env: it holds FLY_REGULAR_SEED, FLY_ENABLED
    and the live FLY_API_BASE, and the CLI modules load it straight into os.environ (where a
    monkeypatch cannot undo it). Every CLI's DOTENV points at a file that does not exist; a
    test that wants a dotenv sets its own."""
    from lfg_fly.body import cli_daily, cli_move, cli_setup

    for module in (cli_daily, cli_move, cli_setup):
        monkeypatch.setattr(module, "DOTENV", tmp_path / "no-such-env")
