"""NULL ARM driver for the WiLI transformation gate.

Promoted 2026-08-24 from scratchpad/fp32null_verdict.py, the throwaway driver
that produced outputs/rerelease/wili_fp32null_verdict.json (arm `fp32null`,
training job 3157851). Nothing about that arm's arithmetic changed in the move:
the per-arm specifics that were literals in the scratchpad copy now live in the
ARMS registry below, and re-running `--arm fp32null` writes the same artifact
with the same keys.

The gate (analysis/wili_transform_gate.py) compared the fp64 retrain of
wili_100k_500 against the special-token-transformed STORED release model and
reported 107/235 languages differing beyond the transferred thresholds. That
comparison crosses several changes at once -- the trainer's special-token
handling (undone by the transformation), the spm_train build (fp32 -> patched
fp64), and whatever else the published run did differently. A FAIL therefore
localizes a difference without attributing it.

A NULL ARM is a retrain that holds everything fixed against the fp64 arm except
ONE named thing, so that the gate FAIL can be attributed. Each registered arm
names what it changes:

  fp32null          the unpatched fp32 spm_train instead of the patched fp64
                    build. Isolates the trainer BUILD.
  fp32null_cap4192  the same unpatched fp32 build AND sentencepiece's upstream
                    default max_sentence_length of 4192 instead of the
                    pipeline's 1,000,000. Two things differ from the fp64 arm,
                    but exactly ONE -- the cap -- differs from fp32null, so the
                    cap's effect is read off the fp32null/cap4192 pair.

Three comparisons, all using wili_transform_gate's own arithmetic (row_metrics,
row_checks, real_token_mask, compare_ordered_vocab, correct):

  (a) null arm vs transformed stored   delta = null - stored_T
  (b) fp64     vs null arm             delta = fp64 - null
  (o) fp64     vs transformed stored   delta = fp64 - stored_T   (the gate run,
                                       recomputed here so all three come from
                                       one process and one corpus manifest)

The sign conventions are chosen so that delta_o = delta_a + delta_b exactly at
the row level, which is checked numerically rather than assumed. row_metrics
weights by the REFERENCE side's probabilities, so the mass-weighted columns of
(a), (b) and (o) use different weights and are NOT additive; only signed_mean is.

Nothing here re-derives thresholds. The same transferred thresholds are applied
to all three arms so the failing-set comparison is like for like.

Writes one artifact per arm; analysis/wili_null_arm_augment.py adds the corpus
check, the floor decomposition, the sentence-cap split and the WiLI test cells
to that same file afterwards.

  python -m analysis.wili_null_arm_verdict
  python -m analysis.wili_null_arm_verdict --arm fp32null_cap4192
  python -m analysis.wili_null_arm_verdict \
      --null-model /capstor/scratch/cscs/cmeister747/unilid_analysis/wili_100k_500_fp32null_cap4192.unilid

An unregistered --null-model is accepted only with --tag, --job-id, --job-log
and --out supplied as well: those are what make the artifact traceable, and
guessing any of them would put an unverified claim in the record.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import traceback
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from analysis.wili_transform_gate import (  # noqa: E402
    MAX_ABS_SIGNED_MEAN, MAX_MASS_WEIGHTED_DIFF, MIN_CORRELATION,
    SCRATCH, TOP_N, WILI_LABELS,
    compare_ordered_vocab, corpus_manifest, die, load_container, provenance,
    read_label_set, real_token_mask, require_same_label_set, resolve_stored,
    row_checks, row_metrics,
)
from analysis.correct_special_token_mass import correct  # noqa: E402

NAME = "wili_100k_500"
FP64 = SCRATCH / "wili_100k_500_fp64.unilid"
CORPUS_DIR = SCRATCH / "wili_corpus_shared"
OUT_DIR = REPO / "outputs" / "rerelease"
GATE_JSON = OUT_DIR / f"wili_transform_gate_{NAME}.json"
# Home for the temporary transformed container (a copy of the stored model,
# ~100 MB). Cluster scratch, not the repo and not session scratch: the repo must
# not carry model-sized temporaries, and a session directory disappears with the
# session that made it. Override with --work-dir.
DEFAULT_WORK_DIR = SCRATCH / "wili_null_arm_work"

# The unpatched fp32 spm_train every registered null arm is required to have
# used, and the patched fp64 build none of them may have touched. Both were
# measured from the binaries themselves (see slurm_wili_train_fp32null.sh).
UNPATCHED_FP32_SHA256 = "4dd4a2e9e35b2731bca8b7124e101e5a3c65474cfb22f591f83b41135013217f"
PATCHED_FP64_SHA256 = "ff5744ae620074154e558dcf1d48d5f3e32934047b40e9f48f9333f0d7fc94d2"
# The base tokenizer every arm's rows must have been estimated over: the fp64
# arm's, copied into each null arm's results directory by its submit script.
BASE_TOKENIZER_SHA256 = "5fa5342a011dca96b170785c220d21505ae91844b2050a5199959c62aeadfbe8"

# --- the registered null arms ---------------------------------------------
# Everything here is a claim about a training run that already happened, so
# every field is either read back out of that job's log by verify_null_job_log()
# or recorded as provenance next to the number it explains. Adding an arm means
# adding its submit script and its job id, not editing code below.
ARMS = {
    "fp32null": {
        "model": SCRATCH / "wili_100k_500_fp32null.unilid",
        "job_id": 3157851,
        "job_log": SCRATCH / "logs" / "wili_w-wili100k-fp32null_3157851.out",
        "submit_script": "slurm_wili_train_fp32null.sh",
        "out": OUT_DIR / "wili_fp32null_verdict.json",
        "max_sentence_length": 1_000_000,
        "trainer_code": "UNILID 0.3.0, identical to the fp64 arm",
        "changes_vs_fp64": ("the spm_train BUILD only: the unpatched fp32 "
                            "binary instead of the patched fp64 one"),
        "arm_a_description": ("fp32 null arm: fp32null retrain vs transformed "
                              "stored model"),
        "arm_b_description": ("pure build effect: fp64 retrain vs fp32null "
                              "retrain (both on the corrected scale; no "
                              "transformation involved)"),
        "null_arm_for": ("the spm_train BUILD only. The fp32null retrain holds "
                         "the trainer code, corpus, base tokenizer and CLI "
                         "flags fixed against the fp64 arm and changes only the "
                         "binary."),
        "if_arm_a_fails": (
            "the fp32 -> patched fp64 spm_train change is not what makes the "
            "fp64 retrain differ from the transformed stored model. Arm (b) "
            "then measures how much the build DOES account for."),
        "extra_required_log_strings": {},
    },
    "fp32null_cap4192": {
        "model": SCRATCH / "wili_100k_500_fp32null_cap4192.unilid",
        "job_id": 3173500,
        "job_log": SCRATCH / "logs" / "wili_w-wili100k-fp32cap_3173500.out",
        "submit_script": "slurm_wili_train_fp32null_cap4192.sh",
        "out": OUT_DIR / "wili_fp32null_cap4192_verdict.json",
        "max_sentence_length": 4192,
        "trainer_code": (
            "UNILID 0.3.0 plus train.py --max-sentence-length "
            "(patches/unilid_max_sentence_length.patch). The flag defaults to "
            "the 1,000,000 that used to be hardcoded, so every other run's "
            "spm_train argv is unchanged and the trainer is otherwise identical "
            "to the fp64 arm."),
        "changes_vs_fp64": (
            "TWO things: the unpatched fp32 spm_train AND sentencepiece's "
            "upstream default max_sentence_length of 4192. Against the fp32null "
            "arm exactly ONE thing changes, the cap."),
        "arm_a_description": ("sentence-cap null arm: fp32 retrain at "
                             "max_sentence_length=4192 vs transformed stored "
                             "model"),
        "arm_b_description": ("build AND cap together: fp64 retrain vs the "
                              "fp32/cap-4192 retrain. NOT a pure build effect; "
                              "the cap alone is the fp32null-to-cap4192 "
                              "difference, not this one."),
        "null_arm_for": (
            "the upstream sentence-length cap, read against the fp32null arm. "
            "sentencepiece SKIPS a line longer than max_sentence_length rather "
            "than truncating it (src/trainer_interface.cc), so this arm trains "
            "on strictly fewer lines for the 65 languages that have one, and "
            "(by experiment B0) should land on a HIGHER unseen-token floor -- "
            "the direction the stored model sits."),
        "if_arm_a_fails": (
            "the cap does not close the gap either, on its own or together with "
            "the build. Compare this arm's failing set against the fp32null "
            "arm's: languages that fail there and pass here are the ones the "
            "cap explains."),
        # Printed by slurm_wili_train_fp32null_cap4192.sh's own post-run check,
        # which reads the value back out of training_summary.json. Without it
        # the log could belong to a run that never applied the cap.
        "extra_required_log_strings": {
            "cap_recorded_by_the_job": "method.max_sentence_length = 4192",
        },
    },
}

# --- constants defined by this driver -------------------------------------
# Tolerance on the additivity identity signed_mean_o == signed_mean_a +
# signed_mean_b, checked on the values the three ARMS reported rather than on
# freshly re-fetched rows. Re-fetching would only re-derive the algebraic
# identity (A-T) - ((B-T) + (A-B)) == 0 and would pass even if an arm had been
# handed the wrong matrix, which is the failure worth catching. Taken from the
# arm outputs the check has teeth: a swapped array or a mislabelled arm breaks
# the sum. The three means are float64 averages over 99,996 entries of magnitude
# up to ~30, so rounding is ~1e-13; 1e-9 is far above that and far below the
# smallest gated deviation (0.01 nats).
ADDITIVITY_TOL = 1e-9
# Cap on the characters read from a language's training file to classify its
# script. Files SHORTER than this are read whole (95 of the 235 are), which is
# fine: a whole file names a script at least as well as a prefix. The cap exists
# only to bound the work on the largest files, since the dominant script of a
# language does not change within a file.
SCRIPT_SAMPLE_CHARS = 200_000
# A language is called Latin-script when at least this share of its LETTERS
# carry "LATIN" in their Unicode character name. Real text mixes in digits,
# punctuation and loanwords, so the split is not at 1.0.
LATIN_LETTER_FRACTION = 0.5
# Half-width of the band around LATIN_LETTER_FRACTION within which a language's
# classification is reported as contestable. The claim that the threshold sits
# in an empty middle is only meaningful against a band centred ON the threshold,
# so this is the diagnostic that defends it; a count of languages far from the
# boundary would not.
LATIN_AMBIGUOUS_HALF_WIDTH = 0.2
# --------------------------------------------------------------------------


def latin_letter_fraction(lang: str) -> float:
    """Share of letters in this language's training file named LATIN.*.

    Measured, not looked up: WiLI labels are ISO 639-3 codes with no script tag,
    so the script has to come from the text the model was trained on.
    """
    path = CORPUS_DIR / f"{lang}_train.txt"
    if not path.is_file():
        die(f"corpus file missing for {lang}: {path}")
    text = path.read_text(encoding="utf-8", errors="strict")[:SCRIPT_SAMPLE_CHARS]
    letters = [c for c in text if c.isalpha()]
    if not letters:
        die(f"{path} has no alphabetic characters in its first "
            f"{SCRIPT_SAMPLE_CHARS:,} characters; cannot name its script")
    n_latin = 0
    for c in letters:
        try:
            if unicodedata.name(c).startswith("LATIN"):
                n_latin += 1
        except ValueError:      # unnamed codepoint: not Latin
            pass
    return n_latin / len(letters)


def arm(tag: str, desc: str, num_rows, num_langs, ref_rows, ref_langs,
        mask, langs_sorted, num_label, ref_label) -> dict:
    """One full 235-language comparison. `num` is row_metrics' first argument."""
    rows = []
    for lang in langs_sorted:
        n_row = np.asarray(num_rows[num_langs.index(lang)]).astype(np.float64)
        r_row = np.asarray(ref_rows[ref_langs.index(lang)]).astype(np.float64)
        n_mass = row_checks(n_row, mask, lang, num_label)
        r_mass = row_checks(r_row, mask, lang, ref_label)
        rec = {"lang": lang, **row_metrics(n_row[mask], r_row[mask]),
               "real_mass_numerator": n_mass, "real_mass_reference": r_mass,
               "n_identical_entries": int((n_row[mask] == r_row[mask]).sum())}
        rows.append(rec)
        del n_row, r_row

    failed = [r["lang"] for r in rows if not r["passed"]]

    def top(key, reverse=True, absolute=False):
        keyed = sorted(rows, key=lambda r: abs(r[key]) if absolute else r[key],
                       reverse=reverse)[:TOP_N]
        return [{"lang": r["lang"], key: r[key], "signed_mean": r["signed_mean"],
                 "correlation": r["correlation"],
                 "mass_weighted_diff": r["mass_weighted_diff"]} for r in keyed]

    return {
        "tag": tag,
        "comparison": desc,
        "delta_definition": f"{num_label} minus {ref_label}, over real tokens",
        "mass_weights_from": ref_label,
        "n_languages": len(rows),
        "n_failed": len(failed),
        "failed_languages": failed,
        "passed": not failed,
        "threshold_failures": {
            "signed_mean": [r["lang"] for r in rows if not r["passes_signed_mean"]],
            "mass_weighted_diff": [r["lang"] for r in rows
                                   if not r["passes_mass_weighted"]],
            "correlation": [r["lang"] for r in rows if not r["passes_correlation"]],
        },
        "n_rows_bit_identical": sum(1 for r in rows if r["max_abs_diff"] == 0.0),
        "worst_offenders": {
            "abs_signed_mean": top("signed_mean", absolute=True),
            "mass_weighted_diff": top("mass_weighted_diff"),
            "max_abs_diff": top("max_abs_diff"),
            "lowest_correlation": top("correlation", reverse=False),
        },
        "languages": rows,
    }


def verify_null_job_log(cfg: dict) -> dict:
    """Confirm the null arm's provenance strings against the job that wrote it.

    Everything this artifact says about the null arm's trainer -- which binary,
    which sha256, which base tokenizer, which cap -- is otherwise a literal
    typed into this script's registry. If the container were ever regenerated by
    a different job, those literals would go on being reported as fact. Grepping
    the job's own stdout makes them measured, and a missing line aborts.

    `packed_container` is the guard against the WRONG log being handed in: an
    arm's log names the container that arm packed, so pointing --job-log at
    another arm's log fails here rather than silently attributing one run's
    provenance to another's model.
    """
    job_log, model = cfg["job_log"], cfg["model"]
    if not job_log.is_file():
        die(f"{cfg['tag']} job log missing: {job_log}; the trainer provenance "
            f"recorded in this artifact would be unverifiable assertion")
    text = job_log.read_text(encoding="utf-8", errors="strict")
    required = {
        "spm_train_sha256": UNPATCHED_FP32_SHA256,
        "unpatched_confirmation":
            "spm_train confirmed UNPATCHED (fp32, pre-d0208d9/c5921a2).",
        "base_tokenizer_sha256": BASE_TOKENIZER_SHA256,
        "defect_absent": "DEFECT PRESENT: False",
        "packed_container": str(model),
        **cfg["extra_required_log_strings"],
    }
    missing = [k for k, v in required.items() if v not in text]
    if missing:
        die(f"{job_log} does not contain {missing}; the {cfg['tag']} provenance "
            f"block in this artifact cannot be confirmed against the job that "
            f"produced the model")
    if PATCHED_FP64_SHA256 in text:
        die(f"{job_log} mentions the PATCHED fp64 spm_train sha256 "
            f"{PATCHED_FP64_SHA256}; this arm was supposed to resolve the "
            f"unpatched build only")
    return {"job_log": str(job_log), "strings_found": sorted(required),
            "patched_sha256_absent": True, "verified": True}


def jaccard(left: set, right: set, left_name: str, right_name: str) -> dict:
    """Set overlap. The keys name the ARMS explicitly: this artifact already has
    arms called a, b and o, so generic n_a / only_b keys would be misread."""
    inter, union = left & right, left | right
    return {"left_arm": left_name, "right_arm": right_name,
            f"n_failing_{left_name}": len(left),
            f"n_failing_{right_name}": len(right),
            "n_intersection": len(inter), "n_union": len(union),
            "jaccard": (len(inter) / len(union)) if union else None,
            f"only_in_{left_name}": sorted(left - right),
            f"only_in_{right_name}": sorted(right - left)}


def resolve_arm(argv=None) -> dict:
    """Turn the command line into one fully specified arm, or abort.

    A registered --arm (or --null-model naming a registered container) supplies
    the job id, job log and output path that make the artifact traceable. An
    unregistered container has none of those, so they must be given explicitly:
    inventing a job id, or reusing another arm's, would put an unverified claim
    in the record.
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", default=None, choices=sorted(ARMS),
                    help="registered null arm (default: fp32null)")
    ap.add_argument("--null-model", default=None,
                    help="container packed by the null-arm training job. A "
                         "registered path selects that arm; any other path "
                         "needs --tag, --job-id, --job-log and --out too.")
    ap.add_argument("--tag", default=None,
                    help="short name for the arm; keys the per-arm blocks in "
                         "the output JSON")
    ap.add_argument("--job-id", type=int, default=None,
                    help="SLURM job id of the training run")
    ap.add_argument("--job-log", default=None,
                    help="that job's stdout log; its provenance strings are "
                         "checked against this arm")
    ap.add_argument("--submit-script", default=None,
                    help="submit script the training job ran, recorded as "
                         "provenance")
    ap.add_argument("--out", default=None, help="output artifact path")
    ap.add_argument("--work-dir", default=None,
                    help="directory for the temporary transformed container "
                         f"(~100 MB). Default: {DEFAULT_WORK_DIR}")
    args = ap.parse_args(argv)

    if args.arm and args.null_model:
        by_path = {str(v["model"]): k for k, v in ARMS.items()}
        named = by_path.get(str(Path(args.null_model)))
        if named != args.arm:
            die(f"--arm {args.arm} and --null-model {args.null_model} disagree: "
                f"that container belongs to arm {named!r}. Pass one or the "
                f"other.")
    if args.null_model and not args.arm:
        by_path = {str(v["model"]): k for k, v in ARMS.items()}
        args.arm = by_path.get(str(Path(args.null_model)))
    if args.arm:
        cfg = dict(ARMS[args.arm])
        cfg["tag"] = args.arm
    elif args.null_model:
        need = {"--tag": args.tag, "--job-id": args.job_id,
                "--job-log": args.job_log, "--out": args.out}
        absent = sorted(k for k, v in need.items() if v is None)
        if absent:
            die(f"--null-model {args.null_model} is not a registered arm "
                f"({sorted(ARMS)}), so {absent} must be given as well. Every "
                f"one of them is provenance this run cannot derive or guess.")
        cfg = {"model": Path(args.null_model), "tag": args.tag,
               "job_id": args.job_id, "job_log": Path(args.job_log),
               "submit_script": args.submit_script,
               "out": Path(args.out), "max_sentence_length": None,
               "trainer_code": None, "changes_vs_fp64": None,
               "arm_a_description": f"null arm {args.tag}: {args.tag} retrain "
                                    f"vs transformed stored model",
               "arm_b_description": f"fp64 retrain vs the {args.tag} retrain",
               "null_arm_for": None, "if_arm_a_fails": None,
               "extra_required_log_strings": {}}
    else:
        cfg = dict(ARMS["fp32null"])
        cfg["tag"] = "fp32null"

    # An explicit flag always wins over the registry, so a re-run of a known arm
    # against a moved log or a new output path stays possible without editing
    # the registry -- but the model itself is never silently re-pointed.
    for key, val in (("job_id", args.job_id), ("submit_script", args.submit_script)):
        if val is not None:
            cfg[key] = val
    for key, val in (("job_log", args.job_log), ("out", args.out)):
        if val is not None:
            cfg[key] = Path(val)
    if args.null_model is not None:
        cfg["model"] = Path(args.null_model)
    if not cfg["model"].is_file():
        die(f"null-arm container missing: {cfg['model']}")
    if not cfg["out"].parent.is_dir():
        die(f"output directory missing: {cfg['out'].parent}")
    cfg["work_dir"] = (Path(args.work_dir) if args.work_dir
                       else DEFAULT_WORK_DIR)
    if args.work_dir and not cfg["work_dir"].is_dir():
        die(f"--work-dir does not exist: {cfg['work_dir']}")
    return cfg


def main(argv=None):
    cfg = resolve_arm(argv)
    tag, null_path, out_path = cfg["tag"], cfg["model"], cfg["out"]
    work_dir = cfg["work_dir"]
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"null arm {tag}: {null_path}\n  job {cfg['job_id']} "
          f"({cfg['submit_script']}), artifact {out_path}", flush=True)
    corpus = corpus_manifest()
    print(f"corpus {corpus['directory']}: {corpus['n_files']} files, "
          f"{corpus['total_bytes']:,} bytes, manifest "
          f"{corpus['manifest_sha256'][:16]}...", flush=True)

    stored_rec = resolve_stored(NAME)
    stored_path = Path(stored_rec["model"])
    s_vocab, s_weights, s_langs, s_ver = load_container(stored_path, "stored model")
    del s_weights
    f64_vocab, f64_w, f64_langs, f64_ver = load_container(FP64, "fp64 retrain")
    f32_vocab, f32_w, f32_langs, f32_ver = load_container(null_path,
                                                          f"{tag} retrain")

    corpus_labels = read_label_set(WILI_LABELS)
    for lbl, langs in (("fp64", f64_langs), (tag, f32_langs)):
        require_same_label_set(set(s_langs), set(langs),
                               f"stored {stored_path.name}", lbl)
    require_same_label_set(set(s_langs), corpus_labels,
                           f"stored {stored_path.name}", f"WiLI y_train")
    vocab_cmp = {
        "stored_vs_fp64": compare_ordered_vocab(s_vocab, f64_vocab, "stored", "fp64"),
        f"stored_vs_{tag}": compare_ordered_vocab(s_vocab, f32_vocab,
                                                  "stored", tag),
    }
    mask = real_token_mask(s_vocab, f"{NAME} vocabulary")
    langs_sorted = sorted(corpus_labels)
    print(f"{len(langs_sorted)} languages, {len(s_vocab):,} vocab, "
          f"{int(mask.sum()):,} real tokens", flush=True)

    with tempfile.TemporaryDirectory(dir=work_dir) as tmp:
        tpath = Path(tmp) / f"{NAME}_transformed.unilid"
        print(f"transforming stored -> {tpath}", flush=True)
        transform_summary = correct(stored_path, tpath)
        t_vocab, t_w, t_langs, t_ver = load_container(tpath, "transformed stored")
        compare_ordered_vocab(s_vocab, t_vocab, "stored", "transformed stored")
        if t_langs != s_langs:
            die("the transformation changed the stored model's language list")

        print(f"arm (a) {tag} vs transformed stored ...", flush=True)
        a = arm("a", cfg["arm_a_description"],
                f32_w, f32_langs, t_w, t_langs, mask, langs_sorted,
                tag, "transformed stored")
        print("arm (o) fp64 vs transformed stored (gate run, recomputed) ...",
              flush=True)
        o = arm("o", "the original gate: fp64 retrain vs transformed stored model",
                f64_w, f64_langs, t_w, t_langs, mask, langs_sorted,
                "fp64", "transformed stored")

        del t_w

    print(f"arm (b) fp64 vs {tag} ...", flush=True)
    b = arm("b", cfg["arm_b_description"],
            f64_w, f64_langs, f32_w, f32_langs, mask, langs_sorted,
            "fp64", tag)
    del f64_w, f32_w

    # --- wiring checks, run on what the ARMS reported ----------------------
    # Both of these are algebraic identities IF each arm was handed the matrix
    # its labels name. Re-deriving them from freshly re-read rows would prove
    # nothing, because the identity holds for any three rows; taken from the arm
    # outputs they fail exactly when an arm was given the wrong matrix, which is
    # the mistake worth catching.
    sm = {k: {r["lang"]: r["signed_mean"] for r in res["languages"]}
          for k, res in (("a", a), ("b", b), ("o", o))}
    add_worst = max(abs(sm["o"][l] - (sm["a"][l] + sm["b"][l]))
                    for l in langs_sorted)
    if add_worst > ADDITIVITY_TOL:
        die(f"the arms' own signed means violate signed_mean_o == "
            f"signed_mean_a + signed_mean_b by {add_worst:g} > "
            f"{ADDITIVITY_TOL:g}. At least one arm was handed a matrix its "
            f"labels do not name; no comparison in this run is meaningful.")

    # The same three matrices appear in two arms each, so their per-row
    # real-token masses must agree across arms. This catches a swap that
    # happened to preserve the signed-mean sum.
    mass = {}
    for k, res in (("a", a), ("b", b), ("o", o)):
        for r in res["languages"]:
            mass[(k, r["lang"], "num")] = r["real_mass_numerator"]
            mass[(k, r["lang"], "ref")] = r["real_mass_reference"]
    for lang in langs_sorted:
        for lhs, rhs, who in ((("a", lang, "num"), ("b", lang, "ref"), tag),
                              (("o", lang, "num"), ("b", lang, "num"), "fp64"),
                              (("a", lang, "ref"), ("o", lang, "ref"),
                               "transformed stored")):
            if mass[lhs] != mass[rhs]:
                die(f"{who} shows real-token mass {mass[lhs]!r} in arm "
                    f"{lhs[0]} but {mass[rhs]!r} in arm {rhs[0]} for language "
                    f"{lang}. The two arms are not reading the same matrix.")

    # --- independent recomputation of the published gate result ------------
    if not GATE_JSON.is_file():
        die(f"published gate artifact missing: {GATE_JSON}; arm (o) has nothing "
            f"to be checked against")
    gate_prev = json.loads(GATE_JSON.read_text())
    if gate_prev.get("retrained") != str(FP64):
        die(f"{GATE_JSON} was produced against retrained model "
            f"{gate_prev.get('retrained')!r}, but this run compares {FP64}")
    prev_digest = gate_prev["training_corpus"]["manifest_sha256"]
    if prev_digest != corpus["manifest_sha256"]:
        die(f"the training corpus has changed since {GATE_JSON} was written: "
            f"manifest {prev_digest} then, {corpus['manifest_sha256']} now")
    prev_sm = {r["lang"]: r["signed_mean"] for r in gate_prev["languages"]}
    if set(prev_sm) != set(sm["o"]):
        die(f"the recomputed gate arm covers a different language set than "
            f"{GATE_JSON}")
    gate_repro_max = max(abs(prev_sm[l] - sm["o"][l]) for l in prev_sm)
    gate_repro_same_set = (set(gate_prev["failed_languages"])
                           == set(o["failed_languages"]))
    # Arm (o) re-runs gate_model's arithmetic on the same inputs, so the only
    # correct outcome is exact agreement. Anything else means an input moved
    # under the published number and this run must not paper over it.
    if gate_repro_max != 0.0 or not gate_repro_same_set:
        die(f"arm (o) did not reproduce {GATE_JSON} exactly: max signed-mean "
            f"difference {gate_repro_max:g}, same failing set "
            f"{gate_repro_same_set}. The fp64 container, the stored container "
            f"or the transformation has changed since that artifact was written.")

    # --- script classification, measured from the corpus -------------------
    latin_frac = {l: latin_letter_fraction(l) for l in langs_sorted}
    is_latin = {l: latin_frac[l] >= LATIN_LETTER_FRACTION for l in langs_sorted}
    lo = LATIN_LETTER_FRACTION - LATIN_AMBIGUOUS_HALF_WIDTH
    hi = LATIN_LETTER_FRACTION + LATIN_AMBIGUOUS_HALF_WIDTH
    contested = sorted(((v, l) for l, v in latin_frac.items() if lo < v < hi),
                       key=lambda x: abs(x[0] - LATIN_LETTER_FRACTION))
    margin = min(abs(v - LATIN_LETTER_FRACTION) for v in latin_frac.values())

    def script_split(res):
        """Per-script summary of BOTH gated magnitude statistics.

        signed_mean averages over all 99,996 real columns, most of which sit on
        the unseen-token floor, so on its own it cannot tell a floor shift from a
        disagreement about text the model actually saw. mass_weighted_diff
        weights each token by the reference row's probability and is the
        score-relevant one, so both are reported.
        """
        out = {}
        by = {r["lang"]: r for r in res["languages"]}
        failed = set(res["failed_languages"])
        for key, want in (("latin", True), ("non_latin", False)):
            langs = [l for l in langs_sorted if is_latin[l] == want]
            if not langs:
                die(f"script class {key!r} is empty; the Latin/non-Latin split "
                    f"produced no languages and its summary would be undefined")
            entry = {"n_languages": len(langs),
                     "n_failed": sum(1 for l in langs if l in failed)}
            for stat in ("signed_mean", "mass_weighted_diff", "correlation"):
                v = np.array([by[l][stat] for l in langs])
                entry[stat] = {"mean": float(v.mean()),
                               "median": float(np.median(v)),
                               "min": float(v.min()), "max": float(v.max())}
            out[key] = entry
        return out

    job_log_check = verify_null_job_log(cfg)

    sets = {k: set(r["failed_languages"]) for k, r in (("a", a), ("b", b), ("o", o))}
    sm_by_arm = {k: {r["lang"]: r["signed_mean"] for r in res["languages"]}
                 for k, res in (("a", a), ("b", b), ("o", o))}

    def corr(x_key, y_key):
        x = np.array([sm_by_arm[x_key][l] for l in langs_sorted])
        y = np.array([sm_by_arm[y_key][l] for l in langs_sorted])
        if x.std() == 0 or y.std() == 0:
            return None
        return float(np.corrcoef(x, y)[0, 1])

    result = {
        "question": (f"Does the wili_100k_500 stored release model reproduce "
                     f"under the {tag} training configuration? If it does, the "
                     f"107-language gate FAIL against the fp64 retrain is "
                     f"explained by what this arm changes rather than by a "
                     f"defect in the stored model. This arm changes, relative "
                     f"to the fp64 arm: {cfg['changes_vs_fp64']}"),
        "null_arm_tag": tag,
        "run_complete": True,
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {**provenance(stored_path, null_path),
                       "note": (f"provenance()'s 'retrained_*' keys describe "
                                f"the {tag} null arm; the fp64 container's size "
                                f"and mtime are under models.fp64")},
        "models": {
            "stored": {"path": str(stored_path), "container_version": s_ver,
                       "defect_present": stored_rec.get("defect_present"),
                       "st_size": stored_path.stat().st_size,
                       "st_mtime": stored_path.stat().st_mtime},
            "fp64": {"path": str(FP64), "container_version": f64_ver,
                     "st_size": FP64.stat().st_size,
                     "st_mtime": FP64.stat().st_mtime},
            tag: {"path": str(null_path), "container_version": f32_ver,
                  "st_size": null_path.stat().st_size,
                  "st_mtime": null_path.stat().st_mtime},
            "transformed_stored": {"container_version": t_ver,
                                   "produced_by": "analysis.correct_special_token_mass.correct"},
        },
        f"{tag}_provenance": {
            "job_id": cfg["job_id"],
            "submit_script": cfg["submit_script"],
            "spm_train": "/capstor/scratch/cscs/cmeister747/unilid_analysis/sp_fp32_env/bin/spm_train",
            "spm_train_sha256": UNPATCHED_FP32_SHA256,
            "spm_train_build": ("UNPATCHED fp32: cimeister/sentencepiece "
                                "2b7ec9b8e86a61f7772236471f948850872d8918, the "
                                "parent of d0208d9; c5921a2 also absent"),
            "confirmed_in_job_log": ("spm_train confirmed UNPATCHED (fp32, "
                                     "pre-d0208d9/c5921a2)"),
            "trainer_code": cfg["trainer_code"],
            "per_language_max_sentence_length": cfg["max_sentence_length"],
            "changes_vs_fp64_arm": cfg["changes_vs_fp64"],
            "base_tokenizer": (f"copied from results_wili_100k_500_fp64/"
                               f"tokenizers/langspec_base_tokenizer.json, "
                               f"sha256 {BASE_TOKENIZER_SHA256}"),
            "verified_against_job_log": job_log_check,
        },
        "training_corpus": corpus,
        "n_languages": len(langs_sorted),
        "vocab_size": len(s_vocab),
        "n_real_tokens": int(mask.sum()),
        "row_alignment": "by language name (langs.index), never by position",
        "token_inventory_comparison": vocab_cmp,
        "transform_summary": transform_summary,
        "thresholds": {"max_abs_signed_mean": MAX_ABS_SIGNED_MEAN,
                       "max_mass_weighted_diff": MAX_MASS_WEIGHTED_DIFF,
                       "min_correlation": MIN_CORRELATION},
        "threshold_provenance": {
            "source": "analysis/gate_correction.py (module constants)",
            "calibrated_on": ("the GlotLID-C base model, comparing retrains "
                              "produced by the SAME trainer build; the only "
                              "benign source of variance there was a different "
                              "subsampling draw of the same corpus"),
            "re_derived_here": False,
            "applied_to": ("all three arms identically, so the failing sets "
                           "compare like for like"),
            "benign_variance_of_arms_a_and_o": (
                "not measured. Both cross the pre-0.3.0 / 0.3.0 trainer-version "
                "change; only arm (b) is within a single trainer version."),
        },
        "what_arm_a_does_and_does_not_establish": {
            "null_arm_for": cfg["null_arm_for"],
            "if_arm_a_fails": cfg["if_arm_a_fails"],
            "what_it_still_does_not_establish": (
                f"that the difference lies in the stored model. The {tag} "
                f"retrain runs UNILID 0.3.0 while the stored model was produced "
                f"by pre-0.3.0 code, so arms (a) and (o) both still cross the "
                f"trainer-version change. Attributing the residual to the "
                f"stored model assumes "
                f"analysis/correct_special_token_mass.correct undoes the ONLY "
                f"pre-0.3.0/0.3.0 difference that reaches a row. This run does "
                f"not test that assumption."),
            "corpus_identity": (
                "see training_corpus.identity_with_published_training_corpus "
                "and, if present, corpus_identity_check."),
        },
        "non_gating_columns": {
            "columns": ["mass_weighted_diff_retrain_weighted",
                        "median_abs_diff", "max_abs_diff"],
            "note": ("mass_weighted_diff_retrain_weighted keeps the name it has "
                     "in analysis/gate_correction.py, where the numerator side "
                     "was always a retrain. Here it means the same sum weighted "
                     "by the NUMERATOR row's probabilities, whatever that side "
                     "is in the arm; in arm (b) neither side is a transform."),
        },
        "arms": {"a": a, "b": b, "o": o},
        "internal_checks": {
            "additivity_max_residual": add_worst,
            "additivity_tolerance": ADDITIVITY_TOL,
            "gate_recomputation_max_signed_mean_diff": gate_repro_max,
            "gate_recomputation_same_failing_set": gate_repro_same_set,
            "gate_artifact": str(GATE_JSON),
        },
        "failing_set_overlap": {
            "a_vs_o": jaccard(sets["a"], sets["o"], "a", "o"),
            "b_vs_o": jaccard(sets["b"], sets["o"], "b", "o"),
            "a_vs_b": jaccard(sets["a"], sets["b"], "a", "b"),
        },
        "signed_mean_correlation_across_arms": {
            "a_vs_o": corr("a", "o"), "b_vs_o": corr("b", "o"),
            "a_vs_b": corr("a", "b"),
            "these_are_not_independent": (
                "signed_mean_o == signed_mean_a + signed_mean_b exactly, so "
                "cov(a,o) = var(a) + cov(a,b). a_vs_o approaches 1 whenever "
                "var(a) dominates var(b), which it does here; the correlation "
                "is therefore a restatement of the magnitude comparison, not "
                "separate evidence for it."),
        },
        "script_breakdown": {
            "method": (f"share of letters in the language's own training file "
                       f"whose Unicode name starts with LATIN, over the first "
                       f"{SCRIPT_SAMPLE_CHARS:,} characters; Latin iff >= "
                       f"{LATIN_LETTER_FRACTION}"),
            "smallest_distance_to_threshold": margin,
            "languages_within_%.2f_of_the_threshold" % LATIN_AMBIGUOUS_HALF_WIDTH:
                [{"lang": l, "latin_letter_fraction": v,
                  "classified": "latin" if v >= LATIN_LETTER_FRACTION
                                else "non_latin"} for v, l in contested],
            "n_latin": sum(1 for v in is_latin.values() if v),
            "n_non_latin": sum(1 for v in is_latin.values() if not v),
            "per_arm": {k: script_split(r) for k, r in
                        (("a", a), ("b", b), ("o", o))},
            "latin_letter_fraction": latin_frac,
        },
    }
    result["produced_by"] = [
        f"analysis/wili_null_arm_verdict.py --arm {tag} "
        f"(the three arms and their cross-checks)"]
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out_path}")

    for k in ("a", "b", "o"):
        r = result["arms"][k]
        print(f"\n=== arm {k}: {r['comparison']}")
        print(f"  {'PASS' if r['passed'] else 'FAIL'}  "
              f"{r['n_languages'] - r['n_failed']}/{r['n_languages']} languages pass, "
              f"{r['n_failed']} fail; {r['n_rows_bit_identical']} rows bit-identical")
        print(f"  threshold failures: "
              + ", ".join(f"{t} {len(v)}" for t, v in r["threshold_failures"].items()))
        sp = result["script_breakdown"]["per_arm"][k]
        for cls in ("latin", "non_latin"):
            d = sp[cls]
            print(f"  {cls:10} n={d['n_languages']:3} failed={d['n_failed']:3}  "
                  f"signed mean median {d['signed_mean']['median']:+.4f} "
                  f"[{d['signed_mean']['min']:+.4f}, {d['signed_mean']['max']:+.4f}]"
                  f"   mass-weighted median {d['mass_weighted_diff']['median']:.4e} "
                  f"max {d['mass_weighted_diff']['max']:.4e}")
        print("  worst by mass-weighted diff:")
        for w in r["worst_offenders"]["mass_weighted_diff"][:TOP_N]:
            print(f"    {w['lang']:8} mw {w['mass_weighted_diff']:.4e}  "
                  f"signed mean {w['signed_mean']:+.4e}  corr {w['correlation']:.6f}")
    print("\n=== overlap ===")
    for k, v in result["failing_set_overlap"].items():
        li, ri = v["left_arm"], v["right_arm"]
        print(f"  {k}: |{li}|={v[f'n_failing_{li}']} |{ri}|={v[f'n_failing_{ri}']} "
              f"inter={v['n_intersection']} union={v['n_union']} "
              f"Jaccard={v['jaccard']:.4f}  "
              f"only in {li}: {v[f'only_in_{li}']}  only in {ri}: {v[f'only_in_{ri}']}")
    print("=== signed-mean correlation across arms ===")
    for k, v in result["signed_mean_correlation_across_arms"].items():
        print(f"  {k}: {v}")
    print("=== internal checks ===")
    print(f"  arms' signed means satisfy o == a + b to {add_worst:.3e} "
          f"(tol {ADDITIVITY_TOL:g}); real-token masses agree across arms")
    print(f"  {tag} job log {cfg['job_log'].name}: provenance strings verified")
    print(f"  gate recomputation: max |d signed_mean| {gate_repro_max:.3e}, "
          f"same failing set {gate_repro_same_set}")
    return 0


if __name__ == "__main__":
    # Exit codes follow analysis/wili_transform_gate.py: 0 the run completed,
    # 2 the run aborted. This driver never returns 1: it reports measured
    # deviations rather than gating on them.
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        print("FATAL: aborted on an unexpected exception (see traceback above)",
              file=sys.stderr, flush=True)
        sys.exit(2)
