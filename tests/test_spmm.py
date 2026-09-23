import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _csr(n_rows=300, n_cols=250, seed=0):
    """A CSR matrix with empty rows, one long row and signed integer weights."""
    rng = np.random.default_rng(seed)
    length = rng.integers(0, 40, n_rows)
    length[::17] = 0
    length[5] = 3000  # longer than any column tile; repeats columns, as CSR allows
    crow = np.concatenate([[0], np.cumsum(length)]).astype(np.int64)
    col = rng.integers(0, n_cols, crow[-1]).astype(np.int64)
    val = (rng.integers(1, 30, crow[-1]) * rng.choice([-1, 1], crow[-1])).astype(np.float32)
    dense = np.zeros((n_rows, n_cols), np.float64)
    np.add.at(dense, (np.repeat(np.arange(n_rows), length), col), val)
    return crow, col, val, dense


@pytest.mark.parametrize("b", [1, 5, 8, 9, 33, 130, 300])
def test_product_matches_float64_and_repeats_bit_for_bit(b):
    from lfg_fly.brain.spmm import CudaCsr

    crow, col, val, dense = _csr()
    W = CudaCsr(crow, col, val, dense.shape, "cuda")
    x = np.random.default_rng(b).uniform(0, 3, (dense.shape[1], b)).astype(np.float32)
    xt = torch.as_tensor(x, device="cuda")
    got = W @ xt
    ref = dense @ x.astype(np.float64)
    scale = np.abs(dense) @ np.abs(x.astype(np.float64))  # the fp32 error bound's scale
    assert got.shape == (dense.shape[0], b) and got.dtype == torch.float32
    assert np.all(np.abs(got.cpu().numpy() - ref) <= 1e-5 * scale + 1e-6)
    for _ in range(5):
        assert torch.equal(W @ xt, got)


def test_rejects_inputs_it_would_read_out_of_bounds():
    from lfg_fly.brain.spmm import CudaCsr

    crow, col, val, dense = _csr()
    W = CudaCsr(crow, col, val, dense.shape, "cuda")
    with pytest.raises(ValueError):
        W @ torch.zeros(dense.shape[1] - 1, 4, device="cuda")  # too few rows
    with pytest.raises(ValueError):
        W @ torch.zeros(dense.shape[1], 4, device="cuda", dtype=torch.float64)
    with pytest.raises(ValueError):
        W @ torch.zeros(dense.shape[1], 4)  # on the CPU
    with pytest.raises(ValueError):
        CudaCsr(crow, np.where(col == col.max(), dense.shape[1], col), val, dense.shape, "cuda")
