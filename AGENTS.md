# AGENTS.md

PyTorch audio-embedding / metric-learning experiments. No package manifest, tests, CI, or git; scripts are run directly and configured by top-of-file constants.

## Environment

- Use the `dl4` conda env: `/mnt/sdb1/anaconda/envs/dl4/bin/python` (Python 3.11, torch 2.9.1+cu128, CUDA works). The `cpython-311` files in `__pycache__` match this env.
- Do NOT use system `python3` (no torch) or `/home/mkr/anaconda3/envs/dl3` (broken here: `bin/python` is a Windows-style link file and `libffi.so.8` is truncated). `.vscode/settings.json` still points at that stale py3.9 path.
- `dl4` lacks `seaborn`, so `test.py` cannot import; the training scripts do not need it.

## Running

- Only `train_melodyst.py` matches the current `my_sampler.YesDataset` signature. From the repo root:
  `/mnt/sdb1/anaconda/envs/dl4/bin/python train_melodyst.py`
- `train.py`, `test.py`, and `train_vae.py` are stale: they use the old dataset signature, reference a nonexistent `4instruments/` tree, or need missing `.pth` checkpoints. Do not assume they run.
- Data and model paths are hardcoded to `/mnt/sdb1/projects/dl3/...`; run there or edit the constants.
- No lint/test config exists. Verify changes with `python -m py_compile ...` or by shrinking `epochs` / `batch_size` / `sound_len` at the top of the script.

## Data / pipeline

- Active config in `train_melodyst.py` is `PROJECT='Medley-solos-DB'`: `data/8instruments/Medley-solos-DB_metadata.csv` plus WAVs, 8 instrument classes (label = `instrument_id` 0-7).
- `data/melodyst` is a separate song-level set (19 classes in `train/labels.csv`); its constants are commented out at the top of `train_melodyst.py`. The `classes` list in that file is the song list, NOT the active 8-instrument labels.
- `YesDataset` has no defaults for the LMDB/split arguments; pass every argument as shown in `train_melodyst.py`.
- `train_melodyst.py` ignores the CSV's `validation` split: it trains on `subset=='training'` and `random_split`s that 80/20 for train/val; test is `subset=='test'`.
- Audio is resampled by librosa, padded/truncated to `sound_len`, converted to MelSpectrogram (dB) or MFCC, then padded to 1024 time frames.

## Gotchas

- LMDB caches (`data/**/cache/*.lmdb`) are filled during dataset construction, keyed by row index after filtering, and never invalidated. Delete the cache dir to rebuild. Do not share one `lmdb_audio_path` between train and test: `_fill_database` only writes missing keys, so the later split silently reuses the first split's cached audio.
- `my_net.wrn_nn` flattens after `AdaptiveAvgPool2d((2, 2))` and feeds `inner_dim*2*2` to its head. Changing `n_mels`, `sound_len`, or layer widths requires a matching `inner_dim` or checkpoint loading fails.
- The metadata CSV has a duplicate header row at line 9337 (`subset,instrument,...`). Filtering by `subset` drops it; do not assume every row is a sample.
- VAE classes in `my_net.py` hardcode `.cuda()` (encoder `Normal`, reparameterization), so they are CUDA-only. `wrn_nn` is not.
- No `.pth` checkpoints are present; `FIRST_RUN=False` makes `train_melodyst.py` load `saved_path` and fail. Keep it `True` unless you supply one.
- `train_melodyst.py` writes a checkpoint dict (`model_state_dict`, `optimizer_state_dict`, `epochs`, `lr`) to `path` in the repo root after every run, even a quick test run.
