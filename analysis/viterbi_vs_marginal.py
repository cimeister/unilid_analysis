"""The tab:viterbi_vs_marginal table: Viterbi decoding against exact marginalization.

The published table reports accuracy and macro F1 for both decoders on GlotLID-C
and had no reproducible generator: nothing in the analysis chain calls the
forward (marginalizing) scorer over the evaluation pool. This is that generator.

Viterbi scores a language by its single best segmentation; marginalization sums
over all of them with the forward algorithm. Both are base-mode only: the
package refuses forward scoring on a calibrated model, because that would apply
the unseen-token constant without the re-examination, which is neither base nor
calibrated inference.

Resumable and chunked in the manner of analysis/full_test_eval.py: predictions go
to int16 memmaps keyed by absolute line index, with completed chunks tracked in a
progress file, and the fingerprint covers the model so a resume cannot mix two
models' chunks. The output root is resolved through analysis.model_context, which
refuses to write a non-default model's arrays into the released model's
directory.

Marginalization is the more expensive decoder (the paper reports roughly 2x), so
budget for the forward pass accordingly.

  python -m analysis.viterbi_vs_marginal --scratch-dir DIR -o outputs/tables
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "UNILID"))

from analysis.config import DEFAULT_SAMPLE_SIZE, TEST_FILE, TOTAL_LINES
from analysis.full_test_eval import (CHUNK_LINES, EMPTY, EXCLUDED, UNSEEN,
                                     _parse_line, _sample_line_indices)
from analysis.hierarchical_pool import VAL_MASK
from analysis.metrics import compute_metrics
from analysis.model_context import add_arguments, resolve
from analysis.transfer_sweep import _load_model_data, _load_unilid_model

DECODERS = ["viterbi", "marginal"]
SCORE_BATCH = 20_000
OUT_DIR = "outputs/tables"


def _fingerprint(ctx, langs) -> dict:
    return {"model_path": ctx.model_path, "model_sha256": ctx.sha256(),
            "langs_sha256": hashlib.sha256("|".join(langs).encode()).hexdigest(),
            "chunk_lines": CHUNK_LINES, "total_lines": TOTAL_LINES,
            "decoders": DECODERS}


def run(model_path: str = None, scratch_dir: str = None,
        out_dir: str = OUT_DIR) -> str:
    from unilid.model_io import UnilidModel

    ctx = resolve(model_path, scratch_dir,
                  purpose="Viterbi against marginalization scoring")
    scratch = ctx.scratch_dir
    if os.path.abspath(scratch) == os.path.abspath(ctx.default_scratch_dir):
        raise RuntimeError(
            f"pass an explicit --scratch-dir: this script writes "
            f"pred_viterbi.npy and pred_marginal.npy, and {scratch} is the "
            f"released model's memmap directory whose entries are symlinks into "
            f"the durable store")
    os.makedirs(scratch, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    print(f"decoder comparison against {ctx.describe()}", flush=True)

    weights, langs, _lang_to_idx = _load_model_data(ctx.model_path)
    del weights
    n_lang = len(langs)

    fp = _fingerprint(ctx, langs)
    fp_path = os.path.join(scratch, "fingerprint_decoders.json")
    if os.path.exists(fp_path):
        prev = json.loads(Path(fp_path).read_text())
        if prev != fp:
            bad = sorted(k for k in fp if prev.get(k) != fp[k])
            raise RuntimeError(
                f"scratch state in {scratch} was produced under a different "
                f"configuration (mismatched: {bad}); clear it or restore the "
                f"inputs before resuming")
    else:
        Path(fp_path + ".tmp").write_text(json.dumps(fp))
        os.replace(fp_path + ".tmp", fp_path)

    # The pool is the full test set minus the 250k validation lines, the same
    # definition analysis/full_test_eval.py uses.
    sample_idx = _sample_line_indices()
    parity_val = (np.arange(DEFAULT_SAMPLE_SIZE) % 2) == 0
    if not np.array_equal(np.load(VAL_MASK), parity_val):
        raise RuntimeError("saved val_mask.npy does not match the position-parity split")
    val_lines = set(sample_idx[parity_val].tolist())

    paths = {d: os.path.join(scratch, f"pred_{d}.npy") for d in DECODERS}
    paths["y_true"] = os.path.join(scratch, "y_true.npy")
    mm = {}
    for name, p in paths.items():
        if os.path.exists(p):
            mm[name] = np.lib.format.open_memmap(p, mode="r+")
            if mm[name].shape != (TOTAL_LINES,):
                raise RuntimeError(f"existing memmap {p} has shape {mm[name].shape}")
        else:
            mm[name] = np.lib.format.open_memmap(p, mode="w+", dtype=np.int16,
                                                 shape=(TOTAL_LINES,))
            mm[name][:] = UNSEEN
            mm[name].flush()

    progress_path = os.path.join(scratch, "progress_decoders.json")
    done = set(json.loads(Path(progress_path).read_text())) \
        if os.path.exists(progress_path) else set()

    lang_to_pos = {l: i for i, l in enumerate(langs)}
    n_chunks = (TOTAL_LINES + CHUNK_LINES - 1) // CHUNK_LINES
    model = None
    with open(TEST_FILE) as fh:
        for chunk in range(n_chunks):
            lo = chunk * CHUNK_LINES
            hi = min(lo + CHUNK_LINES, TOTAL_LINES)
            if chunk in done:
                for _ in range(hi - lo):
                    fh.readline()
                continue
            if model is None:
                print("Loading model (base mode; forward scoring needs it)...",
                      flush=True)
                model = UnilidModel(ctx.model_path, calibrated=False)

            lines = [fh.readline() for _ in range(hi - lo)]
            keep_pos, texts, yt = [], [], []
            for j, line in enumerate(lines):
                i = lo + j
                if i in val_lines:
                    mm["y_true"][i] = EXCLUDED
                    for d in DECODERS:
                        mm[d][i] = EXCLUDED
                    continue
                label, text = _parse_line(line)
                pos = lang_to_pos.get(label)
                if pos is None:
                    mm["y_true"][i] = EXCLUDED
                    for d in DECODERS:
                        mm[d][i] = EXCLUDED
                    continue
                keep_pos.append(i)
                texts.append(text)
                yt.append(pos)
            if not keep_pos:
                done.add(chunk)
                continue
            keep_pos = np.array(keep_pos, dtype=np.int64)
            mm["y_true"][keep_pos] = np.asarray(yt, dtype=np.int16)

            for d in DECODERS:
                out = np.full(len(texts), EMPTY, dtype=np.int16)
                for start in range(0, len(texts), SCORE_BATCH):
                    part = texts[start:start + SCORE_BATCH]
                    got = model.predict_batch(part, forward=(d == "marginal"))
                    for k, (lang, _t, _s) in enumerate(got):
                        if lang:
                            out[start + k] = lang_to_pos[lang]
                mm[d][keep_pos] = out

            for name in mm:
                mm[name].flush()
            done.add(chunk)
            Path(progress_path + ".tmp").write_text(json.dumps(sorted(done)))
            os.replace(progress_path + ".tmp", progress_path)
            print(f"chunk {chunk + 1}/{n_chunks} done ({hi - lo} lines)", flush=True)

    y = np.asarray(mm["y_true"])
    kept = y >= 0
    yk = y[kept]
    rows = []
    for d in DECODERS:
        pk = np.asarray(mm[d])[kept]
        m = compute_metrics(yk, pk)
        rows.append({"decoder": d, "accuracy": m["accuracy"],
                     "macro_f1": m["macro_f1"]})
        print(f"{d:9} accuracy {m['accuracy']:.4f}  macro F1 {m['macro_f1']:.4f}")

    lines_out = ["# tab:viterbi_vs_marginal: Viterbi decoding against exact "
                 "marginalization", "",
                 f"Model: `{ctx.model_path}`. Pool: {int(kept.sum()):,} lines "
                 f"(full test set minus the 250,000 validation lines).", "",
                 "| Decoding | Accuracy | Macro F1 |", "|---|---|---|"]
    for r in rows:
        name = "UniLID (Viterbi)" if r["decoder"] == "viterbi" \
            else "UniLID (Marginalization)"
        lines_out.append(f"| {name} | {r['accuracy']:.3f} | {r['macro_f1']:.3f} |")
    d_f1 = rows[1]["macro_f1"] - rows[0]["macro_f1"]
    lines_out += ["", f"Marginalization changes macro F1 by {d_f1:+.4f} and "
                      f"accuracy by {rows[1]['accuracy'] - rows[0]['accuracy']:+.4f}."]

    md_path = os.path.join(out_dir, "viterbi_vs_marginal.md")
    Path(md_path).write_text("\n".join(lines_out) + "\n")
    Path(os.path.join(out_dir, "viterbi_vs_marginal.json")).write_text(
        json.dumps({"model": ctx.model_path, "n_lines": int(kept.sum()),
                    "rows": rows}, indent=2))
    print("\n".join(lines_out))
    print(f"\nWrote {md_path}")
    return md_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out-dir", default=OUT_DIR)
    add_arguments(ap)
    a = ap.parse_args(argv)
    run(model_path=a.model_path, scratch_dir=a.scratch_dir, out_dir=a.out_dir)


if __name__ == "__main__":
    main()
