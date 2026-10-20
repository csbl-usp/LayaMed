# 1. Environment

[← 0. Overview](00-overview.md) · [Guide index](../README.md#step-by-step-guide) · [2. Data →](02-data.md)

## Option A: Google Colab

Nothing to install. Open `notebooks/layamed_colab.ipynb` on Colab and enable the GPU in **Runtime → Change runtime type → GPU**. Cell 1 clones the repository and installs `laya==0.4.7` with `--no-deps`, so pip does not replace Colab's CUDA build of PyTorch.

| GPU | Precision | Training batch | Notebook time |
|---|---|---|---|
| T4 (free) | fp16 | 4 × 2 (gradient checkpointing) | ~30 min |
| L4 / A100 | bf16 | 8 | ~10 min |

## Option B: on your computer (Windows, Linux or macOS with an NVIDIA GPU)

Requirements: an NVIDIA GPU with 6 GB of VRAM or more (to train; 2 GB is enough just to run the interface), [Miniforge](https://github.com/conda-forge/miniforge) or another conda, and Git.

Run one command at a time (PowerShell on Windows):

```powershell
git clone https://github.com/csbl-usp/LayaMed.git
```

```powershell
cd LayaMed
```

```powershell
conda create -n layamed python=3.11 -y
```

```powershell
conda activate layamed
```

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
```

```powershell
python -m pip install -r requirements.txt
```

```powershell
python -m pip install laya==0.4.7 --no-deps
```

```powershell
python -c "import torch, laya; print(torch.cuda.is_available(), laya.__version__)"
```

The last command should print `True 0.4.7`.

### Why `laya==0.4.7` is pinned

The `laya` package changes often, and the version affects how the sequence is built, how the text is truncated and how temperatures are applied. Training and inference need the **same** version. The notebook checks it with `assert laya.__version__ == "0.4.7"`.

### Notes

- **No flash-attention or Triton on Windows:** Laya uses SDPA, which ships with PyTorch. Do not use `torch.compile`.
- **Duplicate OpenMP (conda + pip):** if numpy/scipy come from conda (MKL) and torch from pip, each ships a different copy of OpenMP. With torch loaded, numpy linear algebra (`@`, `np.linalg`, `scipy.optimize`) can crash Python (`OMP: Error #15`). That is why this repository does matrix math in torch and the recalibration uses element-wise operations only. Installing everything with pip (as above) avoids the problem.
- **bf16 × fp16:** Ampere or newer GPUs (RTX 30/40, A100, L4) use bf16; older ones (T4, RTX 20) use fp16. The code chooses automatically.

---

[← 0. Overview](00-overview.md) · [Guide index](../README.md#step-by-step-guide) · [2. Data →](02-data.md)
