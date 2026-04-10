# Audio-JEPA Implementation Context (Handoff Doc)

Last updated: 2026-04-09 (afternoon, America/Vancouver)  
Workspace: `/scratch/lovenya/Audio-JEPA`  
Shared dataset root: `/scratch/lovenya/datasets/AudioSet`

## 1) Goal

### Immediate goal (time-constrained)
- Understand the paper deeply.
- Have a working, resumable implementation and training pipeline.
- Avoid blocking on full-scale data/training.

### End goal
- Reproduce Audio-JEPA more faithfully (larger data coverage, longer training).
- Optionally scale to 4x H100 for faster throughput.

## 2) Current Status Snapshot

### Data download progress (raw WAV)
- `balanced_train`: **831** wav files downloaded.
- `eval`: **162** wav files downloaded.
- `unbalanced_train`: **0** downloaded so far.

### HDF5 artifacts currently present
Location: `/scratch/lovenya/datasets/AudioSet/hdf5`
- `balanced_train_soxrhq.h5` (~427M, built from current balanced subset).
- `eval_soxrhq.h5` (~83M, from eval smoke subset).
- `full_unbal_bal_train_wav.h5` (~427M, currently linked to balanced subset build).
- `silent_files_*.csv` produced for each file.

Symlinks in `/scratch/lovenya/datasets/AudioSet` point to these HDF5/CSV files so existing Hydra config resolves without extra path overrides.

### Training submission status
- Previous long job (`33712704`) was cancelled.
- Script simplified to demo format and updated to:
  - `--time=01:00:00`
  - `--gpus=h100:1`
  - `--mem=64G`
  - concise logging (`%x-%j.out` / `%x-%j.err`)
  - checkpoint auto-resume from `logs/train/runs/audiojepa_demo_1h/checkpoints/last.ckpt`
- Dry-run check passed:
  - `sbatch --test-only scripts/train_on_AS2M.sh`
  - Slurm predicted full-H100 node in `gpubase_bygpu_b1`.
- Current real demo job:
  - Job ID: `33713716`
  - Name: `ajepa-demo`
  - Command: `sbatch scripts/train_on_AS2M.sh`
  - Current state at last check: `PENDING` with `Reason=None` on `gpubase_bygpu_b1,gpubackfill`.

## 3) Key Code/Script Changes Made

### Repo bug fix (checkpoint serialization)
File: `src/data/audioset_datamodule.py`
- Fixed `save_hyperparameters(...)` ignore list to also ignore `val_dataset`.
- Final ignore list now includes:
  - `mask_collator`
  - `train_dataset`
  - `val_dataset`
  - `eval_dataset`

Why: checkpoint save crashed with `TypeError: h5py objects cannot be pickled` when dataset object leaked into hparams.

### Training script modernization
File: `scripts/train_on_AS2M.sh`
- Converted from `.slurm` to `.sh`.
- Added resumability behavior:
  - auto-resume from `logs/train/runs/${RUN_NAME}/checkpoints/last.ckpt` when present.
  - frequent checkpointing via `callbacks.model_checkpoint.every_n_train_steps`.
  - fixed working defaults for H100 (editable via env vars).
- Added non-interactive-safe overrides:
  - `extras.enforce_tags=False`
  - `tags='[h100,resumable]'`

### New dataset pipeline scripts
Added:
- `scripts/audioset_prepare_manifests.py`
- `scripts/audioset_download.py`
- `scripts/audioset_build_hdf5.py`
- `scripts/run_audioset_download.sh`
- `scripts/run_audioset_build_hdf5.sh`

Behavior:
- Manifest generation from official AudioSet CSV.
- Resumable download (skip existing files automatically).
- HDF5 build in the format expected by current Audio-JEPA dataloader.

## 4) Failures Encountered and What They Mean

### A) Filesystem/network sandbox failures
- Errors like `Read-only file system` and DNS failures happened inside Codex sandbox.
- Resolved by rerunning those commands with escalated permissions.
- Not a project bug.
- Slurm controller access showed the same pattern from sandbox (`Unable to contact slurm controller`), and works when rerun with elevated execution.

### B) YouTube anti-bot failures on eval
- Initial eval run failed 100% with `Sign in to confirm you’re not a bot`.
- Mitigation implemented: downloader now defaults to `--extractor-args youtube:player_client=android`.
- This improved eval smoke success substantially.

### C) Hydra strict config override
- `model.encoder.use_flash_attn=False` failed without `+`.
- Correct usage: `+model.encoder.use_flash_attn=False`.
- Not a bug; Hydra structured-config behavior.

### D) Non-interactive tag prompt
- Run failed with `EOFError` because config wanted interactive tag input.
- Fixed by setting `extras.enforce_tags=False` and explicit tags.

### E) HDF5 chunk size bug in helper script
- In smoke mode, target chunk `(256, 527)` was invalid when fewer rows existed.
- Fixed in `audioset_build_hdf5.py` using `target_chunk_rows = min(n, 256)`.

## 5) Resumability Strategy (Important)

### Downloads
- `audioset_download.py` resumes by checking file existence (`*.wav`).
- Wrapper supports chunking:
  - `max_items` (window size)
  - `skip_items` (window offset)
- If node dies, rerun same command safely.

### HDF5 build
- Rebuild is deterministic from current wav set.
- Can be re-run any time after more clips are downloaded.

### Training
- `train_on_AS2M.sh`:
  - writes to stable run dir `logs/train/runs/${RUN_NAME}`
  - auto-resumes from `last.ckpt` if present.

## 6) Recommended Next Steps (Priority Order)

1. Continue chunked `balanced_train` download (finish this split first).
2. Continue chunked `eval` download.
3. Start chunked `unbalanced_train` (long-running, optional for immediate milestone).
4. Rebuild HDF5 after each meaningful download increment.
5. Start 1x H100 training first (functional milestone), then scale to 4x H100 if available.

## 7) Practical Command Cookbook

Run from repo root:

```bash
cd /scratch/lovenya/Audio-JEPA
```

### Prepare manifests (already done, but idempotent)
```bash
ajepa_env/bin/python scripts/audioset_prepare_manifests.py \
  --class-csv /scratch/lovenya/datasets/AudioSet/metadata/class_labels_indices.csv \
  --balanced-csv /scratch/lovenya/datasets/AudioSet/metadata/balanced_train_segments.csv \
  --eval-csv /scratch/lovenya/datasets/AudioSet/metadata/eval_segments.csv \
  --unbalanced-csv /scratch/lovenya/datasets/AudioSet/metadata/unbalanced_train_segments.csv \
  --out-dir /scratch/lovenya/datasets/AudioSet/manifests
```

### Resumable download examples
```bash
./scripts/run_audioset_download.sh balanced_train 1000 0
./scripts/run_audioset_download.sh balanced_train 1000 1000
./scripts/run_audioset_download.sh eval 1000 0
./scripts/run_audioset_download.sh unbalanced_train 5000 0
```

### Rebuild HDF5 from current downloaded clips
```bash
./scripts/run_audioset_build_hdf5.sh balanced_train 1000
./scripts/run_audioset_build_hdf5.sh eval 1000
./scripts/run_audioset_build_hdf5.sh unbalanced_train 5000
```

### Submit resumable training
```bash
RUN_NAME=audiojepa_h100 CKPT_STEPS=1000 sbatch scripts/train_on_AS2M.sh
```

Requeue/resume:
```bash
RUN_NAME=audiojepa_h100 RESUME_CKPT=auto sbatch scripts/train_on_AS2M.sh
```

## 8) TamIA Migration Plan (No Re-download)

Do **not** re-download from internet if avoidable.

### What to transfer
- `/scratch/lovenya/datasets/AudioSet/raw` (wav files)
- `/scratch/lovenya/datasets/AudioSet/metadata`
- `/scratch/lovenya/datasets/AudioSet/manifests`
- `/scratch/lovenya/datasets/AudioSet/hdf5`
- `/scratch/lovenya/datasets/AudioSet/logs`
- repository `/scratch/lovenya/Audio-JEPA` (excluding local env if desired)

### What to recreate on TamIA
- Python env (`ajepa_env`) and module loads (better than copying venv binaries).

### Transfer note
- Preferred: cluster-supported fast transfer tool (e.g., Globus) if available.
- Alternative: `rsync` from login node to login node.

## 9) Scope Guidance (Given Limited Time)

- 4x H100 is **not required** for immediate implementation validation.
- 1x H100 is enough to validate model/data/training correctness.
- Full paper-scale reproduction requires much more time due dataset acquisition volume.

## 10) Open Items

- Keep improving download success ratio (YouTube churn is unavoidable).
- Decide whether to include cookies-based download mode for higher recall.
- Decide minimum dataset coverage target for your milestone (e.g., balanced+eval only vs partial unbalanced).
