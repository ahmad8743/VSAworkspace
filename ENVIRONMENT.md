# Recreating the VSA environment

Recorded from the working environment: Python **3.12.7**, PyTorch **2.6.0+cu124**, torchvision **0.21.0+cu124**, torchaudio **2.6.0+cu124**. Use Python 3.12.7 for the closest match. The NVIDIA driver must support the CUDA 12.4 PyTorch build. These files describe packages; the `.venv` folder remains ignored.

## LRA experiment only (recommended starting point)

After cloning, open PowerShell in the repository root. Install Python 3.12.7 first if it is unavailable. `py -3.12` uses the installed Python 3.12 interpreter; check its patch version with `py -3.12 --version`.

```powershell
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements-lra.txt
$env:PYTHONPATH = Join-Path $PWD 'zoology'
& .\.venv\Scripts\python.exe .\lra_retrieval.py smoke
& .\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

This is sufficient for the standalone LRA runner: it imports the local HRR module directly, without installing Zoology's optional mixer dependencies. Set `PYTHONPATH` again in each new terminal session. See `LRA_RETRIEVAL_README.md` for dataset preparation and training commands. Datasets are downloaded separately.

## Full current environment

`requirements.txt` pins every package reported by `pip freeze`, excluding the editable Zoology entry, which is installed from your new checkout instead of the old checkout's Git commit.

Start with the same Python environment creation command above, then run:

```powershell
& .\.venv\Scripts\python.exe -m pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -m pip install setuptools==84.0.0 wheel==0.48.0
& .\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e ./zoology
& .\.venv\Scripts\python.exe -m pip check
```

PyTorch is installed first because several GPU extensions and Zoology's setup script import it during installation. The local editable install makes subsequent changes to this checkout take effect without reinstalling Zoology.

The full snapshot is a record of the current environment, not a platform-independent lockfile or a tested clean installation. Packages such as `causal_conv1d`, `flash-linear-attention`, and `fla-core` can require compatible CUDA tooling, operating systems, or prebuilt wheels. Their exact installed binary builds are not captured by `pip freeze`. If installation fails on those packages, the minimal LRA environment above avoids them. Do not change the LRA mixer comparison settings merely to fix installation.

On Linux, use `python3.12 -m venv .venv`, `.venv/bin/python` in place of the Windows interpreter path, and `export PYTHONPATH="$PWD/zoology"`. CUDA wheel availability and GPU extensions still depend on the destination platform.

Activation is optional: these commands explicitly use the environment's interpreter. If desired, activate it in PowerShell with `.\.venv\Scripts\Activate.ps1`.