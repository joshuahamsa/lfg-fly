#!/usr/bin/env bash
# Build the fly's venv: torch for the RTX 2060 SUPER (driver 535 -> CUDA <= 12.2 -> cu121 wheels).
set -euo pipefail
cd "$(dirname "$0")/.."
uv venv .venv --python 3.10
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1
uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
