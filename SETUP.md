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
The login nodes have limited memory (~32 GB usable). The UniLID model with cached weights takes ~250 GB. Any job that loads the full model must run via SLURM, not on the login node.

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
