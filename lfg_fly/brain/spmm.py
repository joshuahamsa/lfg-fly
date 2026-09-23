"""A CSR matrix on CUDA whose product with a dense matrix is bit-reproducible.

torch's `W @ x` for a sparse CSR `W` on CUDA goes to cuSPARSE SpMM, which changes
its summation order from run to run. The rate brain's `W @ h` sums non-integer
products, so the same setting gave readout features differing by up to ~4e-7, and
the Phase 0 probe turned that into held-out swings of about +-0.006. Measured on
the RTX 2060 SUPER (N = 165k, 10.5M edges, 512 columns, ~60 ms a product):
`torch.use_deterministic_algorithms(True)` leaves this op nondeterministic without
raising; cuSPARSE's CSR_ALG3, documented as deterministic, still varied in both
layouts; COO_ALG2 is deterministic but takes 0.3-2 s a product.

These kernels are deterministic by construction: each output element is summed by
exactly one thread, in the CSR's own nonzero order, or, for narrow x, by one warp
with a fixed lane split and a fixed shuffle tree. Run to run on one device they are
bit-identical, at cuSPARSE's speed. They are compiled once per process with NVRTC
(which ships with torch's CUDA wheels) and launched through the CUDA driver API.
"""

from __future__ import annotations

import ctypes
import functools
import os

import numpy as np
import torch

_SOURCE = r"""
extern "C" __global__ void spmm_rows(const long long* __restrict__ crow,
                                     const long long* __restrict__ col,
                                     const float* __restrict__ val,
                                     const float* __restrict__ x,
                                     float* __restrict__ out,
                                     const int* __restrict__ order,
                                     int b)
{
    // one block per row (longest rows first), one thread per column of x
    const int r = order[blockIdx.x];
    const int j = blockIdx.y * blockDim.x + threadIdx.x;
    if (j >= b) return;
    const long long end = crow[r + 1];
    float acc = 0.0f;
    #pragma unroll 4
    for (long long e = crow[r]; e < end; ++e)
        acc = fmaf(val[e], x[col[e] * b + j], acc);
    out[(long long)r * b + j] = acc;
}

extern "C" __global__ void spmm_warp(const long long* __restrict__ crow,
                                     const long long* __restrict__ col,
                                     const float* __restrict__ val,
                                     const float* __restrict__ x,
                                     float* __restrict__ out,
                                     const int* __restrict__ order,
                                     int rows, int b)
{
    // one warp per (row, column of x): lane k sums nonzeros k, k+32, ..., then a fixed tree
    const long long w = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (w >= (long long)rows * b) return;
    const int r = order[w / b];
    const int j = (int)(w % b);
    const long long end = crow[r + 1];
    float acc = 0.0f;
    for (long long e = crow[r] + lane; e < end; e += 32)
        acc = fmaf(val[e], x[col[e] * b + j], acc);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    if (lane == 0) out[(long long)r * b + j] = acc;
}
"""
WARP_MAX_COLUMNS = 8  # x this narrow goes to spmm_warp (faster there; spmm_rows above)
ROW_THREADS = 128

_modules: dict[int, dict[str, ctypes.c_void_p]] = {}


def _nvrtc() -> ctypes.CDLL:
    try:
        import nvidia.cuda_nvrtc

        return ctypes.CDLL(os.path.join(nvidia.cuda_nvrtc.__path__[0], "lib", "libnvrtc.so.12"))
    except (ImportError, OSError):
        return ctypes.CDLL("libnvrtc.so.12")


def _check_nvrtc(lib, status: int, what: str, prog=None) -> None:
    if status == 0:
        return
    lib.nvrtcGetErrorString.restype = ctypes.c_char_p
    detail = ""
    if prog is not None:
        size = ctypes.c_size_t()
        lib.nvrtcGetProgramLogSize(prog, ctypes.byref(size))
        log = ctypes.create_string_buffer(size.value)
        lib.nvrtcGetProgramLog(prog, log)
        detail = "\n" + log.value.decode(errors="replace")
    raise RuntimeError(f"NVRTC {what} failed: {lib.nvrtcGetErrorString(status).decode()}{detail}")


def _check_cu(status: int, what: str) -> None:
    if status == 0:
        return
    msg = ctypes.c_char_p()
    _driver().cuGetErrorString(status, ctypes.byref(msg))
    raise RuntimeError(f"CUDA {what} failed: {status} {(msg.value or b'').decode()}")


@functools.cache
def _driver() -> ctypes.CDLL:
    return ctypes.CDLL("libcuda.so.1")


def _cubin(major: int, minor: int) -> ctypes.Array:
    lib = _nvrtc()
    prog = ctypes.c_void_p()
    _check_nvrtc(lib, lib.nvrtcCreateProgram(ctypes.byref(prog), _SOURCE.encode(), b"spmm.cu",
                                             0, None, None), "create")
    try:
        opts = (ctypes.c_char_p * 1)(f"--gpu-architecture=sm_{major}{minor}".encode())
        _check_nvrtc(lib, lib.nvrtcCompileProgram(prog, 1, opts), "compile", prog)
        size = ctypes.c_size_t()
        _check_nvrtc(lib, lib.nvrtcGetCUBINSize(prog, ctypes.byref(size)), "cubin size")
        cubin = ctypes.create_string_buffer(size.value)
        _check_nvrtc(lib, lib.nvrtcGetCUBIN(prog, cubin), "cubin")
    finally:
        lib.nvrtcDestroyProgram(ctypes.byref(prog))
    return cubin


def _functions(device: torch.device) -> dict[str, ctypes.c_void_p]:
    """The compiled kernels, loaded into torch's context on `device` (once per process)."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _modules:
        cu = _driver()
        with torch.cuda.device(index):
            torch.zeros(1, device=f"cuda:{index}")  # makes torch's primary context current
            _check_cu(cu.cuInit(0), "init")
            ctx = ctypes.c_void_p()
            _check_cu(cu.cuCtxGetCurrent(ctypes.byref(ctx)), "context")
            if not ctx.value:
                raise RuntimeError(f"no current CUDA context on cuda:{index}")
            module = ctypes.c_void_p()
            cubin = _cubin(*torch.cuda.get_device_capability(index))
            _check_cu(cu.cuModuleLoadData(ctypes.byref(module), cubin), "module load")
            fns = {}
            for name in ("spmm_rows", "spmm_warp"):
                fn = ctypes.c_void_p()
                _check_cu(cu.cuModuleGetFunction(ctypes.byref(fn), module, name.encode()),
                          f"get {name}")
                fns[name] = fn
        _modules[index] = fns
    return _modules[index]


def _launch(fn, grid: tuple[int, int], block: int, args: tuple, stream: int) -> None:
    held = [ctypes.c_uint64(a.data_ptr()) if isinstance(a, torch.Tensor) else ctypes.c_int(a)
            for a in args]
    params = (ctypes.c_void_p * len(held))(*[ctypes.cast(ctypes.pointer(h), ctypes.c_void_p)
                                             for h in held])
    u = ctypes.c_uint
    _check_cu(_driver().cuLaunchKernel(fn, u(grid[0]), u(grid[1]), u(1), u(block), u(1), u(1),
                                       u(0), ctypes.c_void_p(stream), params, None), "launch")


class CudaCsr:
    """A float32 CSR matrix on a CUDA device; `W @ x` is bit-reproducible run to run."""

    def __init__(self, crow, col, val, shape: tuple[int, int], device="cuda"):
        crow, col = np.asarray(crow, np.int64), np.asarray(col, np.int64)
        val = np.asarray(val, np.float32)
        rows, cols = (int(s) for s in shape)
        if (len(crow) != rows + 1 or crow[0] != 0 or np.any(np.diff(crow) < 0)
                or crow[-1] != len(col) or len(col) != len(val)):
            raise ValueError("malformed CSR: crow must rise from 0 to nnz, with len(col) == nnz")
        if len(col) and (col.min() < 0 or col.max() >= cols):
            raise ValueError(f"CSR column index out of range [0, {cols})")
        if max(rows, cols) >= 2**31:
            raise ValueError("the kernels index rows and columns with 32-bit ints")
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError(f"CudaCsr needs a CUDA device, got {self.device}")
        self.shape = (rows, cols)
        self.crow = torch.as_tensor(crow, device=self.device)
        self.col = torch.as_tensor(col, device=self.device)
        self.val = torch.as_tensor(val, device=self.device)
        # longest rows first, so hub neurons don't set the tail of the launch
        order = np.argsort(-np.diff(crow), kind="stable").astype(np.int32)
        self.order = torch.as_tensor(order, device=self.device)
        self._fns = _functions(self.device)

    def __matmul__(self, x: torch.Tensor) -> torch.Tensor:
        rows, cols = self.shape
        if x.dim() != 2 or x.shape[0] != cols:
            raise ValueError(f"x must be [{cols}, b], got {list(x.shape)}")
        if x.dtype != torch.float32:
            raise ValueError(f"x must be float32, got {x.dtype}")
        if x.device != self.crow.device:
            raise ValueError(f"x is on {x.device}, the matrix on {self.crow.device}")
        b = x.shape[1]
        if b >= 2**31:
            raise ValueError(f"x has too many columns ({b})")
        x = x.contiguous()
        out = torch.empty(rows, b, device=x.device, dtype=torch.float32)
        if rows == 0 or b == 0:
            return out
        stream = torch.cuda.current_stream(x.device).cuda_stream
        with torch.cuda.device(x.device):
            if b <= WARP_MAX_COLUMNS:
                threads = 256
                grid = ((rows * b * 32 + threads - 1) // threads, 1)
                _launch(self._fns["spmm_warp"], grid, threads,
                        (self.crow, self.col, self.val, x, out, self.order, rows, b), stream)
            else:
                threads = min(ROW_THREADS, (b + 31) // 32 * 32)
                _launch(self._fns["spmm_rows"], (rows, (b + threads - 1) // threads), threads,
                        (self.crow, self.col, self.val, x, out, self.order, b), stream)
        return out
