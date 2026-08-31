"""Build the per-subset training corpus directories for the CLD3-subset models.

WHY THIS EXISTS
---------------
`tab:lid_main`'s right half evaluates "the subsets of the benchmarks that \\cld
has label coverage for" -- 83 / 80 / 77 *languages*, bare ISO 639-3. The author
answered on 2026-08-31 that the plain \\unilid row's CLD3 columns came from
models whose BASE TOKENIZER was trained on that subset of languages, i.e. a
from-scratch subset vocabulary rather than a restricted argmax over the full
1,940-label model (outputs/rerelease/cld_subset_gate_2026-08-31.md section 10).

Regenerating those columns for the corrected generation therefore needs one
training corpus per subset. The subset definitions are lists of bare ISO codes
and the shared corpus draw is a directory of `lang_Script_train.txt` files, so
the mapping is one-to-many: every `lang_Script` corpus whose bare ISO is in the
subset is included. That is the same `label.split("_", 1)[0]` collapse the
evaluation uses (`analysis/cld_subset_eval.py`, and the `--lang-only` flag of
unilid_resources/eval_glotlid.py:38), so the model's label set and the evaluated
label set are built by one rule rather than two.

The corpora are SYMLINKED, not copied: `UNILID/train.py --corpus-dir DIR
--reuse-corpus` only opens `*_train.txt` under the directory
(train.py:236-259), and 1.3 GB per subset copied three times buys nothing.

NO SILENT FALLBACKS
-------------------
Every subset code must map to at least one corpus file, and every corpus file
must exist and be non-empty. A code with no corpus is an abort: a silently
dropped language would shrink the model's label set and make
`analysis/cld_subset_eval.py`'s own subset check fail much later, after hours of
training. An existing output directory is an abort unless --force, so a second
run can never leave a stale link from a previous subset definition in place.

  python -m analysis.build_cld3_subset_corpus --subset 83 --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The shared corpus draw every retrain since the Apertus 200k run has used.
# The monolithic train.txt the released model was built from is gone; the author
# ruled on 2026-08-31 that a fresh draw from this directory is acceptable
# ("Just approximately reproduce them").
SHARED_CORPUS_DIR = ("/capstor/scratch/cscs/cmeister747/unilid_analysis/"
                     "results_apertus200k/corpus")

# subset key -> (definition file, expected number of ISO codes, expected number
# of lang_Script corpora). The corpus counts are asserted, not discovered: they
# are the counts recorded in outputs/rerelease/cld_subset_gate_2026-08-31.md
# section 1 for the same collapse over the same 1,940-label draw, so a change in
# either the subset files or the draw shows up here instead of downstream.
SUBSETS = {
    "83": ("unilid_resources/glotlidc_cld3subset_83.txt", 83, 99),
    "80": ("unilid_resources/udhr_cld3subset_80.txt", 80, 94),
    "77": ("unilid_resources/flores_cld3subset_77.txt", 77, 93),
}

CORPUS_SUFFIX = "_train.txt"

# train.py's --max-base-samples-per-lang default (UNILID/train.py:331): the base
# tokenizer is fitted on the first this-many lines of each language, not on the
# whole corpus. Duplicated here rather than imported because train.py is a
# script, not an importable module. It is used ONLY to predict the base-fit size
# for job sizing; the authoritative figure is the line total the base tokenizer's
# own sidecar records, and analysis/cld3_regenerated_report.py compares the two.
TRAIN_PY_MAX_BASE_SAMPLES_PER_LANG = 10_000


def _read_codes(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        codes = [ln.strip() for ln in f if ln.strip()]
    if len(codes) != len(set(codes)):
        raise SystemExit(f"FATAL: duplicate entries in {path}")
    return codes


def corpora_by_iso(corpus_dir: str) -> dict[str, list[str]]:
    if not os.path.isdir(corpus_dir):
        raise SystemExit(f"FATAL: shared corpus draw missing at {corpus_dir}")
    by_iso: dict[str, list[str]] = {}
    n = 0
    for name in os.listdir(corpus_dir):
        if not name.endswith(CORPUS_SUFFIX):
            continue
        label = name[: -len(CORPUS_SUFFIX)]
        by_iso.setdefault(label.split("_", 1)[0], []).append(label)
        n += 1
    if not n:
        raise SystemExit(f"FATAL: no *{CORPUS_SUFFIX} files under {corpus_dir}")
    for v in by_iso.values():
        v.sort()
    return by_iso


def build(subset: str, out_dir: str, corpus_dir: str, force: bool) -> dict:
    def_rel, n_codes_expected, n_corpora_expected = SUBSETS[subset]
    def_path = os.path.join(REPO_ROOT, def_rel)
    if not os.path.exists(def_path):
        raise SystemExit(f"FATAL: subset definition missing at {def_path}")
    codes = _read_codes(def_path)
    if len(codes) != n_codes_expected:
        raise SystemExit(
            f"FATAL: {def_rel} carries {len(codes)} codes, expected "
            f"{n_codes_expected} (SUBSETS[{subset!r}])")

    by_iso = corpora_by_iso(corpus_dir)
    missing = [c for c in codes if c not in by_iso]
    if missing:
        raise SystemExit(
            f"FATAL: {len(missing)} of the {len(codes)} codes in {def_rel} have "
            f"no corpus under {corpus_dir}: {missing}")

    labels = sorted(l for c in codes for l in by_iso[c])
    if len(labels) != n_corpora_expected:
        raise SystemExit(
            f"FATAL: subset {subset} maps to {len(labels)} corpora, expected "
            f"{n_corpora_expected} (SUBSETS[{subset!r}])")

    if os.path.exists(out_dir):
        if not force:
            raise SystemExit(
                f"FATAL: {out_dir} already exists. Refusing to add links to a "
                f"directory that may hold a different subset's corpora; pass "
                f"--force to rebuild it from scratch.")
        for name in os.listdir(out_dir):
            p = os.path.join(out_dir, name)
            if os.path.islink(p):
                os.unlink(p)
            else:
                raise SystemExit(
                    f"FATAL: {p} is not a symlink; refusing to delete it.")
    os.makedirs(out_dir, exist_ok=True)

    lines_per_label = {}
    base_lines_per_label = {}
    total_bytes = 0
    cap = TRAIN_PY_MAX_BASE_SAMPLES_PER_LANG
    for label in labels:
        src = os.path.join(corpus_dir, label + CORPUS_SUFFIX)
        if not os.path.isfile(src):
            raise SystemExit(f"FATAL: corpus file missing at {src}")
        size = os.path.getsize(src)
        if size == 0:
            raise SystemExit(f"FATAL: corpus file empty at {src}")
        total_bytes += size
        # Two counts in one pass, each matching the code that will consume it:
        #   n     -- every line, as reuse_corpus_from_dir counts them
        #            (train.py:250-258), which is what feeds row estimation;
        #   n_base-- non-blank lines capped at the sampling cap, matching what
        #            sample_corpus takes (train.py:212-230, which skips
        #            `if not line.strip()`), which is what the base fit sees.
        #            EQUIVALENT, not identical: this loop reads bytes and blanks
        #            on bytes.strip() (ASCII whitespace) while sample_corpus
        #            reads text with universal newlines and blanks on
        #            str.strip() (Unicode). A bare CR, or a line of only U+00A0
        #            or U+0085, would diverge. Checked on this draw: no corpus
        #            contains a bare CR, and both rules give the same per-label
        #            count for all 99 corpora. The report cross-checks this
        #            figure against the base tokenizer's own sidecar, so a
        #            divergence aborts the record rather than skewing it.
        # Counting blanks into the base figure would make the report's
        # manifest-vs-sidecar cross-check fail on any corpus with a blank line.
        n = 0
        n_base = 0
        with open(src, "rb") as fh:
            for raw in fh:
                n += 1
                if n_base < cap and raw.strip():
                    n_base += 1
        if n == 0:
            raise SystemExit(f"FATAL: corpus file has no lines at {src}")
        if n_base == 0:
            raise SystemExit(
                f"FATAL: every line of {src} is blank; the base vocabulary "
                f"would be fitted on nothing for this language")
        lines_per_label[label] = n
        base_lines_per_label[label] = n_base
        os.symlink(src, os.path.join(out_dir, label + CORPUS_SUFFIX))

    return {
        "subset": subset,
        "definition_file": def_rel,
        "n_codes": len(codes),
        "codes": codes,
        "shared_corpus_dir": corpus_dir,
        "out_dir": os.path.abspath(out_dir),
        "n_corpora": len(labels),
        "labels": labels,
        "lines_per_label": lines_per_label,
        "total_lines": sum(lines_per_label.values()),
        "total_bytes": total_bytes,
        "max_base_samples_per_lang": TRAIN_PY_MAX_BASE_SAMPLES_PER_LANG,
        "base_lines_per_label": base_lines_per_label,
        "base_sample_lines": sum(base_lines_per_label.values()),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--subset", required=True, choices=sorted(SUBSETS))
    p.add_argument("--out", required=True, help="corpus directory to create")
    p.add_argument("--corpus-dir", default=SHARED_CORPUS_DIR)
    p.add_argument("--force", action="store_true",
                   help="rebuild an existing output directory (symlinks only)")
    p.add_argument("--manifest", default=None, help="write the manifest JSON here")
    a = p.parse_args(argv)
    info = build(a.subset, a.out, a.corpus_dir, a.force)
    print(f"subset {info['subset']}: {info['n_codes']} ISO codes -> "
          f"{info['n_corpora']} corpora, {info['total_lines']:,} lines, "
          f"{info['total_bytes'] / 1e9:.2f} GB -> {info['out_dir']}")
    print(f"  base-tokenizer sample at "
          f"{info['max_base_samples_per_lang']:,} lines/lang: "
          f"{info['base_sample_lines']:,} lines")
    if a.manifest:
        os.makedirs(os.path.dirname(os.path.abspath(a.manifest)), exist_ok=True)
        with open(a.manifest, "w") as f:
            json.dump(info, f, indent=2)
        print(f"Wrote {a.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
