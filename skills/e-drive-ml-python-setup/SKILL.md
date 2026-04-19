---
name: e-drive-ml-python-setup
description: Set up Python machine learning and AI repositories on Windows so environments, caches, temp files, model downloads, dataset downloads, and other writable runtime state stay on the E drive instead of C. Use when preparing a repo for local ML work, training, inference, notebook use, Hugging Face usage, Torch usage, or other Python workflows that would otherwise spill cache or temp data into the user profile.
---

# E-Drive ML Python Setup

Set up the repository so writable runtime state stays inside the repo or another
explicit `E:` location.

## Goals

- Keep caches, temp files, model downloads, dataset downloads, logs, and Python
  writable state off `C:`.
- Prefer repo-local directories such as `.local/`, `.venv/`, and `.vendor/`.
- Make the setup reproducible for future runs in PowerShell and VS Code.
- Validate the result with real commands instead of assuming environment changes
  worked.

## Default Layout

Use this layout unless the repo already has a better-established pattern:

```text
.venv/                    Python virtual environment
.local/
  cache/
    huggingface/
    torch/
    pip/
    matplotlib/
    jupyter/
    wandb/
  pycache/
  tmp/
  userprofile/
    AppData/
      Local/
        Temp/
      Roaming/
.vendor/                  vendored source deps when upstream install is broken
scripts/
  setup_windows_e_drive.ps1
sitecustomize.py          optional global Python bootstrap for repo-local env
```

## Environment Rules

Create repo-local paths first, then set environment variables before install or
runtime validation.

Prefer these variables when relevant:

```powershell
$env:VIRTUAL_ENV
$env:TMP
$env:TEMP
$env:TMPDIR
$env:HF_HOME
$env:HUGGINGFACE_HUB_CACHE
$env:HF_DATASETS_CACHE
$env:TRANSFORMERS_CACHE
$env:TORCH_HOME
$env:MPLCONFIGDIR
$env:WANDB_DIR
$env:JUPYTER_CONFIG_DIR
$env:JUPYTER_DATA_DIR
$env:IPYTHONDIR
$env:PYTHONPYCACHEPREFIX
$env:PIP_CACHE_DIR
$env:HOME
$env:USERPROFILE
$env:APPDATA
$env:LOCALAPPDATA
```

Map them to repo-local locations under `E:`. For example:

- temp: `.local/userprofile/AppData/Local/Temp/`
- Hugging Face: `.local/cache/huggingface/`
- Torch: `.local/cache/torch/`
- pip: `.local/cache/pip/`
- Python bytecode: `.local/pycache/`
- user profile style dirs: `.local/userprofile/`

If the repo already has an environment bootstrap module, extend it instead of
creating a competing mechanism.

## Recommended Workflow

### 1. Inspect first

Check:

- whether the repo already uses `.venv`, Conda, Poetry, uv, or custom bootstrap
- whether scripts import `tempfile`, Hugging Face, Torch, Jupyter, wandb, or
  other tools that create user-profile caches
- whether requirements include packages that download models or datasets at
  runtime
- whether the project already has Windows-specific setup notes

### 2. Add repo-local bootstrap

Prefer one or both of these:

- a Python bootstrap module such as `repo_env.py` or `project_env.py`
- a PowerShell bootstrap script such as `scripts/setup_windows_e_drive.ps1`

The PowerShell script should:

- create `.venv` if missing
- create `.local/` subdirectories
- export the needed environment variables for the current session
- print or run the recommended install commands

The Python bootstrap module should:

- resolve repo-local directories
- set missing environment variables
- expose helper functions for shared cache/tmp paths
- be imported early by scripts that touch downloads, temp files, or training

### 3. Redirect temp usage in code

Do not rely only on shell environment variables when code creates temp files or
runtime workspaces. Update the code so temp-heavy paths are explicit.

Common fixes:

- pass `cache_dir=` to dataset/model loaders
- pass `dir=` to `tempfile.mkdtemp`, `TemporaryDirectory`, and `NamedTemporaryFile`
- redirect planner scratch folders, training outputs, notebook state, and
  checkpoints if they would otherwise go elsewhere

### 4. Keep downloads local

For Hugging Face and similar tooling:

- set cache variables before import/use
- prefer explicit `cache_dir=` where supported
- if offline or semi-offline validation matters, resolve local cached snapshots
  directly rather than depending on runtime hub lookups

For pip and build tooling:

- set `PIP_CACHE_DIR`
- keep editable installs, wheel builds, and vendored source checkouts inside
  the repo or another approved `E:` path

### 5. Handle broken upstream packages safely

If a dependency fails on the target Python/Windows stack:

- prefer a local vendored source checkout under `.vendor/`
- patch only what is necessary for compatibility
- document the reason in the repo
- avoid global machine-level hacks when a repo-local workaround will do

### 6. Validate with real commands

Do not stop at file edits. Validate with actual commands such as:

- environment activation
- `python -c` import checks
- dataset load smoke test
- model/tokenizer load smoke test
- one dry run or small training/inference command

Confirm that outputs and caches land under `E:` paths.

## PowerShell Pattern

When creating a bootstrap script, prefer straightforward repo-relative path
construction and explicit env vars over clever shell tricks.

Key expectations:

- support repeated runs without breaking existing setups
- do not require admin privileges
- keep activation and install commands obvious
- work well from VS Code integrated terminal

## VS Code Notes

- If VS Code prints the activation command when opening a terminal, that is
  usually normal shell integration behavior.
- Do not treat the printed activation line itself as a failure.
- Investigate only if activation actually fails or points to the wrong drive.

## Validation Checklist

- `.venv` exists on `E:`
- `.local` exists on `E:`
- temp variables point to `E:` paths
- Hugging Face cache points to `E:` paths
- pip cache points to `E:` paths
- Python bytecode cache points to `E:` paths
- a dataset/model smoke test succeeds
- a dry run for the project succeeds
- no new cache directories appeared under `C:\Users\...`

## Output Expectations

When using this skill, produce:

- the repo-local bootstrap changes
- a short explanation of what was redirected
- the validation commands that were run
- any remaining risk, especially around package builds, CUDA, or provider
  downloads

## Caution Areas

- Do not delete user data or caches unless explicitly asked.
- Do not move large caches across drives unless the user asked for that.
- Do not assume every library respects the same cache variables.
- Do not assume `sitecustomize.py` alone is enough; verify script entry points.
- Do not assume Windows, WSL, and Linux use identical cache behavior.
- Do not break an existing repo-specific environment model just to force this
  pattern.

## Adaptation Guidance

Keep the approach generic:

- preserve an existing repo's naming and structure when reasonable
- use repo-local helpers instead of hardcoding this repo's module names
- only add vendored dependencies when normal installation is actually broken
- prefer the smallest set of environment variables and code changes that fully
  keep writable state on `E:`
