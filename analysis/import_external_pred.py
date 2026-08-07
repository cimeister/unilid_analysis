"""Import a full-length external prediction file into an int16 memmap (plan item E1,
"Recover fasttext_y_pred.txt and glotlidc_y_pred.txt ... import to int16 memmaps").

Each source file holds one bare `lang_Script` label per line (no `__label__` prefix),
exactly TOTAL_LINES lines, already aligned with the GlotLID test file line-for-line.
This script maps every label to its index in the model's canonical 1,940-language
list (analysis.transfer_sweep._load_model_data) and writes an int16 memmap of shape
(TOTAL_LINES,) to SCRATCH_DIR, the same directory that already holds pred_baseline.npy
and the other Exp 16 prediction memmaps, so downstream scripts (analysis/paper_eval.py)
can load it exactly like any other configuration.

Two sources, both already read once by analysis/sample_data.py to build the seed-42
sample pickle:
- fasttext: DATA_DIR/fasttext_y_pred.txt (analysis.config.PRED_FILES["fastText"]) ->
  SCRATCH_DIR/pred_fasttext.npy
- glotlidc_file: DATA_DIR/glotlidc_y_pred.txt (analysis.config.PRED_FILES["UniLID"]) ->
  SCRATCH_DIR/pred_glotlidc_file.npy

No silent fallback: any label not in the 1,940-language set aborts immediately, naming
the offending label, the 0-indexed line, and the source file. A truncated or
over-length source file also aborts (line-count mismatch against TOTAL_LINES).

Chunked and resumable (CHUNK_LINES=500_000, mirrors analysis/full_test_eval.py):
completed chunks are tracked in progress_import_{source}.json (atomic replace); a
fingerprint over the source path, source file size, chunk size, and the language list
(fingerprint_import_{source}.json) aborts a resume if any of those changed since the
prior run, rather than silently mixing label indices from two different sources. The
memmap is filled with UNSEEN before any chunk is written.

Verification gates, run after the import completes, before this script prints
"IMPORT OK":
  (a) BLOCKING: the seed-42 500k-line sample (re-derived via
      analysis.full_test_eval._sample_line_indices, the same code path
      analysis/full_test_eval.py uses) must agree 100% with the corresponding
      predictions already stored in sample_500k_all.pkl (fastText predictions for
      source=fasttext, UniLID/GlotLID-C predictions for source=glotlidc_file) -- the
      same source files were already read once by analysis/sample_data.py, so this
      checks the newly built memmap against an independent earlier read of the same
      file. Any disagreement aborts.
  (b) NON-BLOCKING comparability measurements (printed and recorded in the report,
      never gate the import): fastText's full-kept-pool macro F1 against the paper's
      reported value; glotlidc_file's agreement rate with pred_baseline.npy on the
      full kept pool.

Output: outputs/tables/import_{source}.md (source file, size, sha256 computed during
the streaming pass, gates passed, comparability measurements, git commit).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess

import numpy as np

from analysis.config import CACHE_DIR, DEFAULT_SAMPLE_SIZE, PRED_FILES, TOTAL_LINES
from analysis.full_test_eval import CHUNK_LINES, SCRATCH_DIR, UNSEEN, _sample_line_indices
from analysis.metric_decomposition import EXPECTED_KEPT, _per_lang_stats
from analysis.sample_data import load_sample
from analysis.transfer_sweep import _load_model_data

# Source registry: source file (DATA_DIR, already read once by analysis/sample_data.py
# to build sample_500k_all.pkl), output memmap name (in SCRATCH_DIR, alongside
# pred_baseline.npy etc.), and the pickle key (analysis.config.PRED_FILES) whose
# predictions the blocking gate checks this import against.
SOURCES = {
    "fasttext": {
        "src_path": PRED_FILES["fastText"],
        "out_npy": "pred_fasttext.npy",
        "pickle_key": "pred_fastText",
    },
    "glotlidc_file": {
        "src_path": PRED_FILES["UniLID"],
        "out_npy": "pred_glotlidc_file.npy",
        "pickle_key": "pred_UniLID",
    },
}

# Non-blocking comparability reference (E1 pre-registration, EXPERIMENTS_PLAN.md
# "Camera-ready evaluation program"): the paper's reported fastText macro F1. The paper
# team's own metrics JSON (outputs/diagnostic/paper_team/fasttext_folder/
# glotlid_fasttext_e100_sanity_metrics.json) records macro_f1 0.9443269 computed over
# total_samples 45,627,279 (all lines, including the 250,000 retired validation lines),
# while this script measures on the 45,377,279-line kept pool; the two instruments
# differ and the report must say so.
PAPER_FASTTEXT_F1 = 0.944
PAPER_FASTTEXT_TOTAL_LINES = 45_627_279
# Flag threshold for the fastText macro F1 comparability check (informational; does
# not abort the import).
COMPARABILITY_TOL = 0.005
# Recorded context for the glotlidc_file agreement check: self-agreement of the
# re-scored baseline against itself (informational only, not a gate).
PAPER_GLOTLIDC_BASELINE_SELF_AGREEMENT_CONTEXT = 0.9951

OUT_DIR = "outputs"


def _fingerprint(source_path: str, langs: list) -> dict:
    """Identity of everything that determines this source's memmap contents. A resume
    with ANY mismatch (different source file, edited language list, changed chunking)
    must abort rather than mix label indices from two different runs."""
    st = os.stat(source_path)
    return {
        "source_path": source_path,
        "source_bytes": st.st_size,
        "langs_sha256": hashlib.sha256("|".join(langs).encode()).hexdigest(),
        "chunk_lines": CHUNK_LINES,
        "total_lines": TOTAL_LINES,
    }


def run(source: str, out_dir: str = OUT_DIR) -> str:
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; must be one of {sorted(SOURCES)}")
    cfg = SOURCES[source]
    src_path = cfg["src_path"]
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"source prediction file missing: {src_path}")

    os.makedirs(SCRATCH_DIR, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)

    weights, langs, lang_to_idx = _load_model_data()
    del weights
    n_lang = len(langs)

    fp = _fingerprint(src_path, langs)
    fp_path = os.path.join(SCRATCH_DIR, f"fingerprint_import_{source}.json")
    if os.path.exists(fp_path):
        with open(fp_path) as f:
            prev = json.load(f)
        if prev != fp:
            bad = sorted(k for k in fp if prev.get(k) != fp[k])
            raise RuntimeError(
                f"scratch state for import source {source!r} in {SCRATCH_DIR} was "
                f"produced under a different configuration (mismatched: {bad}); clear "
                "the progress/fingerprint files for this source or restore the "
                "original source file before resuming")
    else:
        with open(fp_path + ".tmp", "w") as f:
            json.dump(fp, f)
        os.replace(fp_path + ".tmp", fp_path)

    out_path = os.path.join(SCRATCH_DIR, cfg["out_npy"])
    if os.path.exists(out_path):
        mm = np.lib.format.open_memmap(out_path, mode="r+")
        if mm.shape != (TOTAL_LINES,):
            raise RuntimeError(f"existing memmap {out_path} has shape {mm.shape}, "
                               f"expected ({TOTAL_LINES},)")
        if mm.dtype != np.int16:
            raise RuntimeError(f"existing memmap {out_path} has dtype {mm.dtype}, "
                               "expected int16")
    else:
        mm = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.int16,
                                       shape=(TOTAL_LINES,))
        mm[:] = UNSEEN
        mm.flush()

    progress_path = os.path.join(SCRATCH_DIR, f"progress_import_{source}.json")
    done_chunks = set()
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            done_chunks = set(json.load(f))

    n_chunks = (TOTAL_LINES + CHUNK_LINES - 1) // CHUNK_LINES
    file_hash = hashlib.sha256()
    with open(src_path, "rb") as fh:
        for chunk in range(n_chunks):
            lo = chunk * CHUNK_LINES
            hi = min(lo + CHUNK_LINES, TOTAL_LINES)
            active = chunk not in done_chunks
            if active:
                idx = np.empty(hi - lo, dtype=np.int16)
            for j in range(hi - lo):
                raw = fh.readline()
                if raw == b"":
                    raise RuntimeError(
                        f"{src_path} ended prematurely at line {lo + j:,}; expected "
                        f"{TOTAL_LINES:,} lines")
                file_hash.update(raw)
                if active:
                    label = raw.decode("utf-8").strip()
                    li = lang_to_idx.get(label)
                    if li is None:
                        raise RuntimeError(
                            f"{src_path}: line {lo + j:,} has label {label!r}, not in "
                            f"the model's {n_lang} languages")
                    idx[j] = li
            if active:
                mm[lo:hi] = idx
                mm.flush()
                done_chunks.add(chunk)
                with open(progress_path + ".tmp", "w") as f:
                    json.dump(sorted(done_chunks), f)
                os.replace(progress_path + ".tmp", progress_path)
                print(f"chunk {chunk + 1}/{n_chunks} done ({hi - lo:,} lines)",
                      flush=True)
        extra = fh.readline()
        if extra != b"":
            raise RuntimeError(f"{src_path} has more than {TOTAL_LINES:,} lines "
                               "(unexpected trailing content after the last chunk)")

    source_bytes = os.stat(src_path).st_size
    source_sha256 = file_hash.hexdigest()

    # --- post-pass coverage gate (one condition: no UNSEEN entries remain, which is
    # equivalent to all TOTAL_LINES lines having been written) ---
    mm_arr = np.asarray(mm)
    n_unseen = int((mm_arr == UNSEEN).sum())
    if n_unseen != 0:
        raise RuntimeError(f"{n_unseen:,} UNSEEN entries remain in {out_path} after "
                           "the import pass")
    print(f"Coverage gate passed: all {TOTAL_LINES:,} lines written, zero UNSEEN "
          "remain.")

    # --- (a) BLOCKING: seed-42 sample gate ---
    sample_idx = _sample_line_indices()
    sample_pkl_path = os.path.join(CACHE_DIR,
                                   f"sample_{DEFAULT_SAMPLE_SIZE // 1000}k_all.pkl")
    if not os.path.exists(sample_pkl_path):
        # load_sample regenerates a missing pickle from the source files themselves,
        # which would turn this gate into a comparison of the import against a fresh
        # read of the same file in the same session. The gate is only meaningful
        # against the independent earlier read, so a missing pickle aborts.
        raise FileNotFoundError(
            f"sample pickle missing: {sample_pkl_path}; the blocking gate needs the "
            "independent earlier read and must not regenerate it")
    sample_pkl = load_sample(DEFAULT_SAMPLE_SIZE)
    pickle_key = cfg["pickle_key"]
    if pickle_key not in sample_pkl:
        raise RuntimeError(f"sample pickle is missing key {pickle_key!r}; keys "
                           f"present: {sorted(k for k in sample_pkl if k.startswith('pred_'))}")
    pickle_labels = np.asarray(sample_pkl[pickle_key])
    if len(pickle_labels) != len(sample_idx):
        raise RuntimeError(f"sample pickle has {len(pickle_labels):,} {pickle_key!r} "
                           f"entries, expected {len(sample_idx):,} matching the "
                           "re-derived seed-42 sample")
    imported_idx = mm_arr[sample_idx]
    imported_labels = np.array([langs[i] for i in imported_idx])
    mismatch = imported_labels != pickle_labels
    n_mismatch = int(mismatch.sum())
    if n_mismatch:
        bad = np.where(mismatch)[0][:10]
        detail = "; ".join(
            f"(line_index={int(sample_idx[k])}, pickle_label={pickle_labels[k]!r}, "
            f"imported_label={imported_labels[k]!r})" for k in bad)
        raise RuntimeError(
            f"blocking seed-42 sample gate FAILED: {n_mismatch:,} of "
            f"{len(sample_idx):,} sampled lines disagree between {out_path} and "
            f"{pickle_key!r} in sample_500k_all.pkl; first 10 disagreements: {detail}")
    print(f"Blocking sample gate passed: 100% agreement ({len(sample_idx):,} lines) "
          f"between {out_path} and {pickle_key!r} in sample_500k_all.pkl.")

    # --- (b) NON-BLOCKING comparability measurements ---
    y_path = os.path.join(SCRATCH_DIR, "y_true.npy")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"required artifact missing: {y_path}")
    y = np.asarray(np.lib.format.open_memmap(y_path, mode="r"))
    if y.shape != (TOTAL_LINES,):
        raise RuntimeError(f"y_true memmap {y_path} has shape {y.shape}, expected "
                           f"({TOTAL_LINES},)")
    kept = y >= 0
    n_kept = int(kept.sum())
    if n_kept != EXPECTED_KEPT:
        raise RuntimeError(f"full kept pool in {y_path} has {n_kept:,} lines != "
                           f"EXPECTED_KEPT ({EXPECTED_KEPT:,})")
    yk = y[kept]

    comparability_lines = []
    if source == "fasttext":
        pred_k = mm_arr[kept]
        _, _, f1, _, _, _ = _per_lang_stats(pred_k, yk, n_lang)
        macro_f1 = float(f1.mean())
        diff = abs(macro_f1 - PAPER_FASTTEXT_F1)
        flagged = diff > COMPARABILITY_TOL
        msg = (f"fastText macro F1 on the {n_kept:,}-line kept pool: {macro_f1:.6f}. "
               f"The paper reports {PAPER_FASTTEXT_F1}, computed on all "
               f"{PAPER_FASTTEXT_TOTAL_LINES:,} lines (paper team metrics JSON, "
               f"total_samples field), so the two instruments differ by the 250,000 "
               f"retired validation lines. Absolute difference {diff:.4f} "
               f"(flag threshold {COMPARABILITY_TOL}).")
        if flagged:
            msg += (
                " FLAGGED: outside the threshold. Pre-registered branch: do not "
                "discard the import; candidate causes are the instrument difference "
                "above (kept pool vs all lines) and the label basis (the paper's "
                "script table covers 1,938 languages against this pipeline's 1,940); "
                "resolve with the paper team (ask item 7) before interpreting.")
        print(f"Non-blocking: {msg}")
        comparability_lines.append(msg)
    else:
        baseline_path = os.path.join(SCRATCH_DIR, "pred_baseline.npy")
        if not os.path.exists(baseline_path):
            raise FileNotFoundError(f"required artifact missing: {baseline_path}")
        pred_baseline = np.asarray(np.lib.format.open_memmap(baseline_path, mode="r"))
        agreement = float((mm_arr[kept] == pred_baseline[kept]).mean())
        msg = (f"glotlidc_file agreement with pred_baseline.npy on the full kept "
               f"pool ({n_kept:,} lines): {agreement:.4f} (recorded context: "
               "re-scored baseline self-agreement "
               f"{PAPER_GLOTLIDC_BASELINE_SELF_AGREEMENT_CONTEXT}).")
        print(f"Non-blocking: {msg}")
        comparability_lines.append(msg)

    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError) as e:
        raise RuntimeError(f"git rev-parse HEAD failed: {e}") from e

    # --- report ---
    L = [
        f"# Import external prediction file: {source}\n",
        f"Source file: {src_path}",
        f"Size: {source_bytes:,} bytes",
        f"sha256: {source_sha256}",
        f"Output memmap: {out_path} (int16, shape ({TOTAL_LINES:,},))",
        f"Git commit: {git_commit}\n",
        "## Gates passed\n",
        f"- Coverage: all {TOTAL_LINES:,} lines written, zero UNSEEN entries remain.",
        f"- Blocking seed-42 sample gate: 100% agreement ({len(sample_idx):,} "
        f"lines) between the imported memmap and {pickle_key!r} in "
        "sample_500k_all.pkl.",
        "\n## Comparability measurements (non-blocking)\n",
    ]
    L.extend(f"- {line}" for line in comparability_lines)

    md_path = os.path.join(out_dir, "tables", f"import_{source}.md")
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nWrote {md_path}")
    print("IMPORT OK")
    return md_path


def main():
    parser = argparse.ArgumentParser(
        description="Import an external full-length prediction file into an int16 "
                    "memmap aligned with y_true.npy.")
    parser.add_argument("--source", required=True, choices=sorted(SOURCES),
                        help="which external prediction file to import")
    args = parser.parse_args()
    run(args.source)


if __name__ == "__main__":
    main()
