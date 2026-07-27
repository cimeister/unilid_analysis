# UniLID Analysis — Setup and Infrastructure

## Cluster Environment

This runs on **CSCS Clariden** (GH200 GPUs, Slingshot interconnect). The analysis is CPU-only but uses SLURM for memory-intensive jobs.

- **SLURM account**: `a139`, partition: `normal`, max walltime: 12 hours
- **Python**: `/users/cmeister747/.pyenv/versions/3.11.5/bin/python3` (3.11.5 via pyenv)
- **No venv** — packages installed directly into the pyenv Python

## Filesystem Layout

```
/users/cmeister747/unilid_analysis/          # Home dir — code + small outputs
  analysis/                                   # Analysis scripts (the main pipeline)
    config.py                                 # All constants: bins, paths, model names
    metrics.py                                # Accuracy, macro F1, macro FPR computation
    format_utils.py                           # Markdown + LaTeX table formatters, confusion matrix plots
    sample_data.py                            # Uniform sampler (seed=42, without replacement)
    tables.py                                 # Tables 1-4: overall, by length, resource, script
    comparison.py                             # Tables 5-7: error overlap, per-script winner, divergences
    confusion.py                              # Confusion matrices for 7 language clusters
    length_bias.py                            # Token count bias + normalization counterfactual
    normalized_predict.py                     # Full re-classification with normalized scores
    run_all.py                                # CLI entry point
  outputs/
    tables/                                   # Generated .md + .tex files
    figures/                                  # Confusion matrix PNGs, histograms
  UNILID/                                     # UniLID source code (git repo with submodules)
    unilid/model_io.py                        # UnilidModel class (predict, predict_normalized)
    tokenizers/                               # Custom Rust tokenizer fork (modified for normalization)
  slurm_*.sh                                  # SLURM submission scripts

/capstor/scratch/cscs/cmeister747/unilid_analysis/   # Scratch — large data
  glotlid_unilid/                             # All data files (moved here from home to save space)
    glotlid_correct_test.txt                  # 45.6M lines, 6.7 GB, fastText format
    glotlid_correct_test.txt.zip              # Zipped version (must unzip before use)
    glotlidc_y_pred.txt                       # UniLID predictions (45.6M lines)
    deepseek_v3.2_glotlid_y_pred.txt          # UniLID-DeepSeek predictions
    qwen3_8b_glotlid_y_pred.txt               # UniLID-Qwen predictions
    marg_y_pred.txt                           # UniLID-Marg predictions
    fasttext_y_pred.txt                       # fastText predictions
    glotlid_train_counts.json                 # {lang_script: n_training_samples}
  glotlidc.unilid                             # UniLID model file (744 MB)
  sample_500k_all.pkl                         # Cached 500k sample (no raw texts)
  sample_45627k_all.pkl                       # Cached full-dataset sample (no raw texts)
  logs/                                       # SLURM stdout/stderr
```

## Important Gotchas

### 1. Data location moved to scratch
All data files were moved from `~/unilid_analysis/glotlid_unilid/` to scratch because home dir ran out of space. `config.py` was updated: `DATA_DIR` now points to scratch. If data disappears (scratch is not backed up), re-download from Google Drive:
- Test data + predictions: `gdown --folder "https://drive.google.com/drive/folders/19sRPRiFHX8Lk3vZWlNGl0zzA88eAZ3Yx"`
- UniLID-Marg predictions: `gdown --folder "https://drive.google.com/drive/folders/1vm1R4p00ixTTQmWpkFwWpW6yGaDbI4aY"` → copy `glotlidc_y_pred.txt` as `marg_y_pred.txt`
- fastText predictions: `gdown --folder "https://drive.google.com/drive/folders/1ivBVoQNAGPGjlWogqIsBMHzKyJD86an9"` → copy `glotlid_fasttext_e100_sanity_y_pred.txt` as `fasttext_y_pred.txt`
- After download, unzip `glotlid_correct_test.txt.zip`

### 2. Test file must be unzipped
`glotlid_correct_test.txt.zip` must be unzipped before running any analysis. The unzipped file is 6.7 GB. This has been forgotten multiple times.

### 3. Sample pickle does NOT store raw texts
The sample pickle stores `y_true`, `text_lengths`, `train_counts`, and predictions for all 5 models — but NOT raw texts. This was a deliberate decision after the full-dataset pickle with texts OOM'd at 128 GB. Scripts that need raw texts (length_bias, normalized_predict) stream the test file directly.

### 4. Memory requirements for SLURM jobs
| Job | Memory needed | Reason |
|-----|--------------|--------|
| Tables (run_all.py) | 64 GB | 45.6M string arrays + Counter-based metrics |
| Length bias | 400 GB | 1,895 per-language Tokenizer objects (~250 GB for cached weights in the HF tokenizer) |
| Normalized predict | 400 GB | Same — loads UnilidModel which caches all 1,940 weight sets in Rust |

The 128 GB and even 256 GB requests OOM'd for the tokenizer-heavy jobs. The tokenizer cache (1,895 language tokenizers with 100k vocab each) takes ~250 GB in memory.

### 5. Macro F1 bug (fixed)
The original `compute_metrics` averaged F1 over `set(y_true) | set(y_pred)` instead of just `set(y_true)`. This penalized models that predicted labels not in the true set with phantom zero-F1 terms. Fixed to match sklearn convention (average over true labels only).

### 6. Custom Rust tokenizer build
The normalized scoring methods (`best_of_cached_weight_sets_normalized`) were added to the Rust tokenizer fork. To rebuild after changes:
```bash
cd UNILID/tokenizers/bindings/python
VIRTUAL_ENV=/users/cmeister747/.pyenv/versions/3.11.5 maturin develop --release
```
The submodule must be initialized first: `cd UNILID && git submodule update --init tokenizers`

### 7. Login node memory limits
The login nodes have limited memory (~32 GB usable). Any job that loads the full model and caches its weights must run via SLURM, not on the login node.

Measured footprint, correcting the "~250 GB" figure that appears in older docstrings: the 1,940 x 100,000 float32 matrix is about 776 MB, and the transient during `set_weight_sets(matrix.tolist())` is roughly 6-7 GB per call. Scoring jobs therefore run comfortably at 100 GB, and full training runs request 400 GB. The login node can host small scoring passes (tens of thousands of lines) but was killed by the out-of-memory killer (exit code 137) on the 373,230-line CommonLID pass, which is the practical boundary: anything at that scale or above goes to SLURM.

### 8. The trainer binary and the fp64 fix (2026-07-27)
`~/.local/bin/spm_train` is a build of the forked SentencePiece (`cimeister/sentencepiece`, branch `fixed-vocab-em`) that performs fixed-vocabulary EM with pruning disabled. As of 2026-07-27 the installed binary includes commits `d0208d9` (double-precision forward-backward in the training expectation step) and `c5921a2` (hard failure on non-finite expected counts). The pre-fix binary is preserved as `~/.local/bin/spm_train.pre_fp64` and should be used only to reproduce historical training runs.

Rebuild after changing the fork:
```bash
git clone --branch fixed-vocab-em https://github.com/cimeister/sentencepiece.git
cd sentencepiece && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DSPM_ENABLE_SHARED=OFF
make -j 16 spm_train
cp src/spm_train ~/.local/bin/spm_train      # keep a backup of the previous binary
```
Acceptance test after any trainer change: retrain `azj_Latn` alone against the 131k base tokenizer and check that its row has tens of thousands of entries above the row minimum (the pre-fix binary produced 7). Then run `python -m analysis.degeneracy_scan` on any newly packed model before evaluating it. Background and the failure mechanism are in `EXPERIMENTAL_SETUP.md`, "Per-language training pipeline and the trainer fix".

### 9. Long training lines are not truncated
The pipeline passes `--max_sentence_length=1000000`, well above the upstream default of 4,192 bytes, because silently discarding training lines is worse than keeping them. This is safe only with the fp64 trainer above; with the pre-fix binary it was the trigger for a silent model collapse. If the flag is ever lowered, record which lines are dropped and in which corpora.

## How to Run

### Standard tables (500k sample, uses cached pickle)
```bash
cd /users/cmeister747/unilid_analysis
python -m analysis.run_all --sample-size 500000 --format both
```

### Full dataset tables (no resample needed if pickle exists)
```bash
python -m analysis.run_all --sample-size 45627279 --format both
# Or via SLURM:
sbatch slurm_tables.sh
```

### Length bias analysis (streams full test file, needs SLURM)
```bash
sbatch slurm_length_bias.sh   # 400 GB, 12h walltime
```

### Normalized scoring comparison (needs SLURM for memory)
```bash
sbatch slurm_normalized.sh    # 400 GB, 2h walltime
```

### Resample (needed when adding new models or after code changes to sample_data.py)
```bash
python -m analysis.run_all --resample --sample-size 500000 --format both
```

## Dependencies
- `numpy`, `matplotlib`, `scipy` — standard scientific stack
- `gdown` — Google Drive downloads
- `tokenizers` — custom build from `UNILID/tokenizers/` (includes normalization methods)
- `maturin` — for building the Rust tokenizer bindings
