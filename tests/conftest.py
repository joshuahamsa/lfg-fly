import lfg_fly  # noqa: F401,I001  (must be first: applies the CPU thread caps)
import pytest


@pytest.fixture(autouse=True)
def _data_dir(tmp_path, monkeypatch):
    """No test may touch ~/fly-data: every test gets its own FLY_DATA_DIR."""
    monkeypatch.setenv("FLY_DATA_DIR", str(tmp_path / "fly-data"))
    return tmp_path / "fly-data"
