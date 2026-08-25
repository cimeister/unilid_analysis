"""Add to a null-arm verdict artifact the things the three arms alone do not
answer.

Promoted 2026-08-24 from scratchpad/fp32null_augment.py, which augmented
outputs/rerelease/wili_fp32null_verdict.json in place. Reads the artifact
analysis/wili_null_arm_verdict.py wrote for the same arm and rewrites it with
four sections added; the arm is selected exactly as it is there.

1. CORPUS IDENTITY, verified rather than assumed. wili_transform_gate.py records
   the corpus as a hash manifest and states outright that its identity with the
   published training corpus is ASSUMED. This checks the half that is checkable:
   that every wili_corpus_shared/<lang>_train.txt is exactly the lines of the
   released WiLI-2018 x_train.txt whose y_train label is <lang>, in order. That
   still does not tie the released MODEL to these bytes, but it removes "the
   corpus directory was assembled wrong" from the candidate causes.

2. WHERE the deviation lives. Each row's real tokens split into the unseen-token
   floor (the repeated minimum a row assigns to tokens it never saw; measured by
   B0 on 2026-08-17 to be set by corpus size) and everything above it. If the
   deviation is carried by the floor, the two sides disagree about tokens neither
   observed; if it is carried by the tokens above the floor, they disagree about
   the text itself.

3. The SENTENCE-CAP SPLIT: whether having a training line longer than
   sentencepiece's upstream max_sentence_length separates the languages that
   fail arm (a) from those that pass. Descriptive on its own; the arm that
   measures the cap is fp32null_cap4192.

4. The WiLI-2018 test cells of all three models side by side, read from the
   wili_eval artifacts rather than retyped. An arm that has not been evaluated
   yet has no cells; pass --no-wili-test-cells to say so deliberately rather
   than have the section quietly go missing.

  python -m analysis.wili_null_arm_augment
  python -m analysis.wili_null_arm_augment --arm fp32null_cap4192
  python -m analysis.wili_null_arm_augment --arm fp32null_cap4192 --no-wili-test-cells
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from analysis.wili_transform_gate import (  # noqa: E402
    SCRATCH, WILI_LABELS, die, load_container, read_label_set, real_token_mask,
    resolve_stored,
)
from analysis.correct_special_token_mass import correct  # noqa: E402
# One registry, one set of arm paths: the augment stage must not be able to
# describe a different model than the verdict stage measured.
from analysis.wili_null_arm_verdict import ARMS, DEFAULT_WORK_DIR  # noqa: E402
# The byte-level alphabet the pipeline encodes every training line with. Taken
# from UNILID itself (wili_transform_gate put UNILID on sys.path) so this cannot
# drift from what the trainer does.
from unilid.constants import _BYTE_TO_UNI  # noqa: E402

WILI_DIR = SCRATCH / "wili_assets" / "wili-2018"
CORPUS_DIR = SCRATCH / "wili_corpus_shared"
FP64_MODEL = SCRATCH / "wili_100k_500_fp64.unilid"
# analysis/wili_eval.py names its artifact after the container it scored:
# <container stem>.unilid -> wili_eval_<container stem>.json. Derived rather
# than tabulated so a new arm needs no edit here, and checked for existence
# before it is read.
def eval_artifact_for(model: Path) -> Path:
    return REPO / "outputs" / "rerelease" / f"wili_eval_{model.stem}.json"
# The stored (defective) model's published cells, reproduced by
# analysis/wili_eval.py in gate mode and recorded in this artifact.
STORED_EVAL = REPO / "outputs" / "rerelease" / "wili_instrument_gate.json"

# --- constants defined by this script -------------------------------------
# A real token is "on the floor" when its log-probability is within this many
# nats of the row's minimum. The floor is one repeated value, so this guards
# float32 storage jitter only; it does not define a band.
FLOOR_EPS = 1e-6
# SentencePiece's UPSTREAM default max_sentence_length, in bytes
# (sentencepiece src/sentencepiece_model.proto: max_sentence_length = 18
# [default = 4192]). The UNILID pipeline overrides it to 1,000,000 (visible in
# any job log's spm_train command line), so a run at the default SKIPS training
# lines this long -- skips, not truncates: src/trainer_interface.cc counts the
# line in too_long_lines and `continue`s. Used here only to split the languages,
# never to reconfigure anything; the arm that actually sets the cap is
# fp32null_cap4192.
UPSTREAM_MAX_SENTENCE_BYTES = 4192
# A language's arm-(a) deviation is called material above this many nats of
# absolute signed mean. It is the gate's own signed-mean threshold
# (MAX_ABS_SIGNED_MEAN), reused so the "material" set is exactly the set the
# gate's signed-mean criterion rejects.
MATERIAL_SIGNED_MEAN = 0.01
# --------------------------------------------------------------------------


def verify_corpus_against_wili() -> dict:
    x_path, y_path = WILI_DIR / "x_train.txt", WILI_DIR / "y_train.txt"
    for p in (x_path, y_path):
        if not p.is_file():
            die(f"WiLI training split missing: {p}")
    x = x_path.read_text(encoding="utf-8").splitlines()
    y = y_path.read_text(encoding="utf-8").splitlines()
    if len(x) != len(y):
        die(f"{x_path} has {len(x):,} lines but {y_path} has {len(y):,}")
    by = defaultdict(list)
    for text, lab in zip(x, y):
        by[lab.strip()].append(text)
    missing, mismatched = [], []
    for lang in sorted(by):
        p = CORPUS_DIR / f"{lang}_train.txt"
        if not p.is_file():
            missing.append(lang)
            continue
        if p.read_text(encoding="utf-8").splitlines() != by[lang]:
            mismatched.append(lang)
    return {
        "claim": ("every wili_corpus_shared/<lang>_train.txt is exactly the "
                  "lines of the released WiLI-2018 x_train.txt carrying that "
                  "y_train label, in order"),
        "sources": {"x_train": str(x_path), "y_train": str(y_path),
                    "corpus_dir": str(CORPUS_DIR)},
        "n_labels": len(by), "n_lines": len(x),
        "corpus_files_missing": missing,
        "corpus_files_mismatched": mismatched,
        "verified": not missing and not mismatched,
        "what_this_does_not_establish": (
            "that the PUBLISHED model was trained on these bytes. It "
            "establishes only that the retrains' corpus is the released "
            "WiLI-2018 training split verbatim."),
    }


def floor_decomposition(null_model: Path, tag: str, work_dir: Path) -> dict:
    rec = resolve_stored("wili_100k_500")
    stored_path = Path(rec["model"])
    s_vocab, sw, _s_langs, _ = load_container(stored_path, "stored")
    del sw
    _, f64, f64_l, _ = load_container(FP64_MODEL, "fp64")
    _, f32, f32_l, _ = load_container(null_model, tag)
    mask = real_token_mask(s_vocab, "vocabulary")
    langs = sorted(read_label_set(WILI_LABELS))

    rows = []
    work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=work_dir) as tmp:
        tp = Path(tmp) / "t.unilid"
        correct(stored_path, tp)
        _, tw, t_l, _ = load_container(tp, "transformed stored")
        for lang in langs:
            t = np.asarray(tw[t_l.index(lang)]).astype(np.float64)[mask]
            a = np.asarray(f32[f32_l.index(lang)]).astype(np.float64)[mask]
            b = np.asarray(f64[f64_l.index(lang)]).astype(np.float64)[mask]
            above_t = t > t.min() + FLOOR_EPS
            above_a = a > a.min() + FLOOR_EPS
            above_b = b > b.min() + FLOOR_EPS
            both = above_t & above_a
            d = a - t
            n = d.size
            rows.append({
                "lang": lang,
                "floor_storedT": float(t.min()), "floor_null_arm": float(a.min()),
                "floor_fp64": float(b.min()),
                "n_above_floor_storedT": int(above_t.sum()),
                "n_above_floor_null_arm": int(above_a.sum()),
                "n_above_floor_fp64": int(above_b.sum()),
                "floor_mass_storedT": float(np.exp(t[~above_t]).sum()),
                "floor_mass_null_arm": float(np.exp(a[~above_a]).sum()),
                "floor_mass_fp64": float(np.exp(b[~above_b]).sum()),
                "above_floor_set_jaccard_storedT_null_arm":
                    float(both.sum() / (above_t | above_a).sum()),
                "signed_mean_a_total": float(d.mean()),
                "signed_mean_a_on_storedT_floor_tokens": float(d[~above_t].sum() / n),
                "signed_mean_a_on_other_tokens": float(d[above_t].sum() / n),
                "signed_mean_a_restricted_above_both_floors":
                    float((a[both] - t[both]).mean()),
                "max_abs_diff_a_restricted_above_both_floors":
                    float(np.abs(a[both] - t[both]).max()),
                "correlation_a_restricted_above_both_floors":
                    float(np.corrcoef(a[both], t[both])[0, 1]),
            })
        del tw
    return rows


def summarize(rows, failed_set) -> dict:
    def grp(sel):
        g = [r for r in rows if sel(r)]
        out = {"n_languages": len(g)}
        for key in ("above_floor_set_jaccard_storedT_null_arm",
                    "signed_mean_a_restricted_above_both_floors",
                    "correlation_a_restricted_above_both_floors",
                    "max_abs_diff_a_restricted_above_both_floors",
                    "floor_mass_storedT", "floor_mass_null_arm", "floor_mass_fp64"):
            v = np.array([r[key] for r in g])
            out[key] = {"median": float(np.median(v)), "min": float(v.min()),
                        "max": float(v.max())}
        return out

    material = [r for r in rows
                if abs(r["signed_mean_a_total"]) > MATERIAL_SIGNED_MEAN]
    share = np.array([r["signed_mean_a_on_storedT_floor_tokens"]
                      / r["signed_mean_a_total"] for r in material])
    # An arm can leave this set EMPTY: if no language deviates by more than
    # MATERIAL_SIGNED_MEAN there is no deviation to attribute to floor tokens,
    # and the quantiles below would be taken over a zero-length array. Recorded
    # as an explicit empty result rather than computed, because a median of
    # nothing would read as a measured share. (Hit 2026-08-25 by arm
    # fp32null_cap4192, whose arm (a) passes all 235 languages.)
    if material:
        share_summary = {"median": float(np.median(share)),
                         "min": float(share.min()), "max": float(share.max())}
    else:
        share_summary = {
            "n_languages": 0,
            "median": None, "min": None, "max": None,
            "note": (f"no language in this arm has |arm-(a) signed mean| above "
                     f"{MATERIAL_SIGNED_MEAN} nats, so there is no deviation to "
                     f"attribute to the stored model's floor tokens. The "
                     f"quantiles are omitted, not zero.")}
    return {
        "floor_definition": (f"a real token is on the floor when its log-prob is "
                             f"within {FLOOR_EPS:g} nats of the row's minimum"),
        "floor_is_set_by_corpus_size": (
            "measured by experiment B0 on 2026-08-17 "
            "(outputs/rerelease/plateau_vs_corpus_size.json): within a fixed "
            "language the floor falls about 2.19 nats per decade of training "
            "tokens, R-squared 0.999"),
        "n_languages_with_material_arm_a_signed_mean": len(material),
        "material_threshold_nats": MATERIAL_SIGNED_MEAN,
        "share_of_arm_a_signed_mean_carried_by_stored_floor_tokens":
            share_summary,
        "by_original_gate_verdict": {
            "failed": grp(lambda r: r["lang"] in failed_set),
            "passed": grp(lambda r: r["lang"] not in failed_set)},
        "languages": rows,
    }


def encoded_byte_length(line: bytes) -> int:
    """Length of `line` in the file the per-language spm_train actually reads.

    unilid.vocab_io.write_hf_bytelevel_corpus rewrites every corpus line through
    the GPT-2 byte-level alphabet before spm_train sees it, and sentencepiece
    measures max_sentence_length against THAT file. Every byte becomes one
    character, but the characters above U+007F cost two bytes in UTF-8, so a
    non-Latin line roughly doubles.

    Verified against the trainer rather than derived on paper: on a 50-line bod
    mini-corpus this function counts 20 lines over 4192, and spm_train at
    --max_sentence_length=4192 logged exactly "Skipped 20 too long sentences."
    (2026-08-24 smoke test for slurm_wili_train_fp32null_cap4192.sh). The raw
    byte count for the same 50 lines is 12, which would have been wrong.
    """
    return sum(1 if _BYTE_TO_UNI[b] < "\u0080" else 2 for b in line)


def over_cap_split(arm_a_rows) -> dict:
    """Does having a line longer than the upstream sentence cap separate the
    languages that fail arm (a) from those that pass?

    A candidate cause, not a measured one. The retrains pass
    max_sentence_length=1,000,000; upstream's default is 4,192 bytes. A run at
    the default skips the long lines, trains on fewer tokens, and (by B0) lands
    on a HIGHER unseen-token floor -- which is the direction the stored model
    sits relative to both retrains. This only measures whether the two groups
    separate; it does not establish that the published run used the default,
    and nothing here reconfigures a trainer.

    Counted in the unit sentencepiece measures: the line of the BYTE-LEVEL
    ENCODED file write_hf_bytelevel_corpus writes, not the raw corpus line (see
    encoded_byte_length). The distinction decides the answer rather than
    refining it -- over the WiLI training corpus, 367 lines in 65 languages
    exceed 4,192 RAW bytes, but 2,052 lines in 106 languages exceed it ENCODED.
    The 2026-08-24 run of this section counted raw bytes and concluded from it
    that "41 languages fail arm (a) with no over-cap line at all, so the cap
    cannot be the whole explanation"; in the encoded unit those 41 all have one.
    The raw count is still reported, under raw_byte_counts.
    """
    by_lang = {r["lang"]: r for r in arm_a_rows}
    groups = {"fail_with_over_cap_line": [], "fail_without": [],
              "pass_with_over_cap_line": [], "pass_without": []}
    counts, counts_raw = {}, {}
    for lang, r in by_lang.items():
        path = CORPUS_DIR / f"{lang}_train.txt"
        if not path.is_file():
            die(f"corpus file missing for {lang}: {path}")
        lines = [x for x in path.read_bytes().split(b"\n") if x]
        n_over = sum(1 for x in lines
                     if encoded_byte_length(x) > UPSTREAM_MAX_SENTENCE_BYTES)
        counts[lang] = n_over
        counts_raw[lang] = sum(1 for x in lines
                               if len(x) > UPSTREAM_MAX_SENTENCE_BYTES)
        key = ("fail" if not r["passed"] else "pass") + (
            "_with_over_cap_line" if n_over else "_without")
        groups[key].append(abs(r["signed_mean"]))
    out = {
        "cap_bytes": UPSTREAM_MAX_SENTENCE_BYTES,
        "pipeline_setting": ("the retrains pass max_sentence_length=1000000; "
                             "sentencepiece's upstream default is 4192"),
        "status": ("CANDIDATE CAUSE. No record states what max_sentence_length "
                   "the published run used. The retrain at the default ran as "
                   "job 3173500 (slurm_wili_train_fp32null_cap4192.sh, arm "
                   "fp32null_cap4192); THIS split is descriptive either way -- "
                   "it counts corpus lines, it does not measure a model. The "
                   "measurement is that arm's own three-arm comparison in "
                   "outputs/rerelease/wili_fp32null_cap4192_verdict.json."),
        "counted_over": ("the BYTE-LEVEL ENCODED line, the unit sentencepiece "
                         "applies the cap to: every byte becomes one character "
                         "and the ones above 0x7F cost two UTF-8 bytes. Counting "
                         "raw corpus bytes instead understates both the lines "
                         "and the languages affected; see raw_byte_counts."),
        "n_lines_over_cap_whole_corpus": sum(counts.values()),
        "n_languages_with_any_over_cap_line":
            sum(1 for v in counts.values() if v),
        "per_language_over_cap_line_count": counts,
        "raw_byte_counts": {
            "n_lines_over_cap_whole_corpus": sum(counts_raw.values()),
            "n_languages_with_any_over_cap_line":
                sum(1 for v in counts_raw.values() if v),
            "per_language_over_cap_line_count": counts_raw,
            "note": ("the same count measured on the RAW corpus line instead of "
                     "the encoded one. Kept because the 2026-08-24 run of this "
                     "section reported it, and because the two disagree: 367 "
                     "lines in 65 languages raw, 2,052 lines in 106 languages "
                     "encoded. The encoded figure is the one that governs which "
                     "lines a run at the default skips."),
        },
    }
    for key, vals in groups.items():
        out[key] = {"n_languages": len(vals)}
        if vals:
            out[key].update({
                "abs_signed_mean_median": float(np.median(vals)),
                "abs_signed_mean_max": float(max(vals))})
    pass_with = out["pass_with_over_cap_line"]["n_languages"]
    fail_with = out["fail_with_over_cap_line"]["n_languages"]
    fail_without = out["fail_without"]["n_languages"]
    # `exact` is a claim about a DISCRIMINATOR, and a discriminator needs two
    # groups to discriminate. When the arm has no failing languages at all
    # (fp32null_cap4192, 2026-08-25) there is nothing for the over-cap flag to
    # separate: the old wording would have reported "the split is not exact, so
    # the cap cannot be the whole explanation" off pass_with > 0 alone, which
    # states the opposite of what an all-pass arm shows. Report the split as
    # undefined instead, and leave the reading to the arm's own comparison.
    n_failing = fail_with + fail_without
    exact = (n_failing > 0 and pass_with == 0 and fail_without == 0)
    if n_failing == 0:
        separation = (
            f"this arm has NO failing languages, so the over-cap flag has "
            f"nothing to separate; {pass_with} of the 235 passing languages "
            f"have a line over the cap once encoded and {out['pass_without']['n_languages']} "
            f"do not. The split is undefined here, not negative: what the arm "
            f"shows is read off its own three-arm comparison, not off this "
            f"section.")
    else:
        separation = (
            f"{pass_with} of the languages that PASS arm (a) have a line over "
            f"the cap once encoded; {fail_with} of the failing ones do, and "
            f"{fail_without} fail with no over-cap line at all. "
            + ("The two groups separate exactly -- every failing language has "
               "an over-cap line and no passing one does. That is what the cap "
               "hypothesis predicts; arm fp32null_cap4192 is what tests it."
               if exact else
               "The split is not exact, so the cap cannot be the whole "
               "explanation."))
    out["separation"] = separation
    out["separates_exactly"] = exact
    out["n_failing_languages_in_this_arm"] = n_failing
    return out


def eval_cells(null_model: Path, tag: str) -> dict:
    cells = {}
    if not STORED_EVAL.is_file():
        die(f"stored model's eval artifact missing: {STORED_EVAL}")
    gate = json.loads(STORED_EVAL.read_text())
    cells["stored_defective"] = {
        "source": str(STORED_EVAL),
        "accuracy": gate["accuracy"], "macro_f1": gate["macro_f1"],
        "macro_fpr": gate["macro_fpr"]}
    for label, model in (("fp64", FP64_MODEL), (tag, null_model)):
        path = eval_artifact_for(model)
        if not path.is_file():
            die(f"{label} has no WiLI eval artifact at {path}. Run "
                f"analysis/wili_eval.py on {model} first, or pass "
                f"--no-wili-test-cells to record this artifact without the "
                f"test cells rather than with cells from another model.")
        d = json.loads(path.read_text())
        if d["model"] != str(model):
            die(f"{path} scored {d['model']!r}, not the {label} container "
                f"{model} this run describes")
        cells[label] = {"source": str(path), "model": d["model"],
                        "accuracy": d["accuracy"], "macro_f1": d["macro_f1"],
                        "macro_fpr": d["macro_fpr"]}
    # WiLI-2018 test is balanced at 500 items per language, so n - support_l is
    # the same 117,000 for every label and macro FPR collapses to
    # (total errors) / (n_labels * (n - support_l)). It therefore carries no
    # information beyond accuracy on this benchmark; recorded so that two models
    # sharing a macro FPR is not read as independent agreement.
    n, n_lab, sup = 117_500, 235, 500
    for tag, c in cells.items():
        errors = round(n * (1 - c["accuracy"]))
        c["n_errors_implied_by_accuracy"] = errors
        c["macro_fpr_predicted_from_accuracy"] = errors / (n_lab * (n - sup))
        c["macro_fpr_matches_prediction"] = bool(
            abs(c["macro_fpr_predicted_from_accuracy"] - c["macro_fpr"]) < 1e-12)
    cells["note_macro_fpr_is_not_independent"] = (
        "WiLI-2018 test is balanced (500 items per label), so every label has "
        "n - support = 117,000 and macro FPR = total errors / (235 * 117,000), "
        "a deterministic function of accuracy. Verified per row above.")
    return cells


def resolve(argv=None):
    """Select the arm whose verdict artifact is to be augmented.

    Mirrors analysis/wili_null_arm_verdict.py's rules and reads the same
    registry, so the two stages cannot end up describing different models.
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", default=None, choices=sorted(ARMS),
                    help="registered null arm (default: fp32null)")
    ap.add_argument("--null-model", default=None,
                    help="container packed by the null-arm training job; a "
                         "registered path selects that arm")
    ap.add_argument("--tag", default=None,
                    help="short name for an unregistered arm")
    ap.add_argument("--verdict", default=None,
                    help="verdict artifact to augment in place (default: the "
                         "registered arm's)")
    ap.add_argument("--work-dir", default=None,
                    help=f"directory for the temporary transformed container "
                         f"(~100 MB). Default: {DEFAULT_WORK_DIR}")
    ap.add_argument("--wili-test-cells", dest="wili_test_cells",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="record the WiLI-2018 test cells of the three models. "
                         "--no-wili-test-cells omits the section for an arm "
                         "that has not been evaluated yet; the omission is "
                         "recorded in the artifact.")
    args = ap.parse_args(argv)

    by_path = {str(v["model"]): k for k, v in ARMS.items()}
    if args.arm and args.null_model:
        named = by_path.get(str(Path(args.null_model)))
        if named != args.arm:
            die(f"--arm {args.arm} and --null-model {args.null_model} disagree: "
                f"that container belongs to arm {named!r}")
    arm = args.arm or (by_path.get(str(Path(args.null_model)))
                       if args.null_model else "fp32null")
    if arm is None:
        if not (args.tag and args.verdict):
            die(f"--null-model {args.null_model} is not a registered arm "
                f"({sorted(ARMS)}), so --tag and --verdict must be given too")
        model, tag, verdict = Path(args.null_model), args.tag, Path(args.verdict)
    else:
        cfg = ARMS[arm]
        model = Path(args.null_model) if args.null_model else cfg["model"]
        tag = args.tag or arm
        verdict = Path(args.verdict) if args.verdict else cfg["out"]
    if not model.is_file():
        die(f"null-arm container missing: {model}")
    work_dir = Path(args.work_dir) if args.work_dir else DEFAULT_WORK_DIR
    if args.work_dir and not work_dir.is_dir():
        die(f"--work-dir does not exist: {work_dir}")
    return model, tag, verdict, work_dir, args.wili_test_cells


def main(argv=None):
    null_model, tag, verdict_path, work_dir, want_cells = resolve(argv)
    if not verdict_path.is_file():
        die(f"verdict artifact missing: {verdict_path}; run "
            f"`python -m analysis.wili_null_arm_verdict --arm {tag}` first")
    result = json.loads(verdict_path.read_text())
    recorded = result.get("null_arm_tag")
    if recorded is not None and recorded != tag:
        die(f"{verdict_path} was written for arm {recorded!r}, but this run "
            f"describes {tag!r}; augmenting it would mix two arms in one "
            f"artifact")
    failed_set = set(result["arms"]["o"]["failed_languages"])
    print(f"augmenting {verdict_path} (arm {tag}, model {null_model.name})",
          flush=True)

    print("verifying corpus against the released WiLI-2018 training split ...",
          flush=True)
    corpus_check = verify_corpus_against_wili()
    print(f"  verified={corpus_check['verified']} "
          f"missing={len(corpus_check['corpus_files_missing'])} "
          f"mismatched={len(corpus_check['corpus_files_mismatched'])}", flush=True)

    print("decomposing the deviation into floor and above-floor ...", flush=True)
    rows = floor_decomposition(null_model, tag, work_dir)
    summary = summarize(rows, failed_set)

    print("splitting by the upstream sentence-length cap ...", flush=True)
    over_cap = over_cap_split(result["arms"]["a"]["languages"])
    print(f"  {over_cap['separation']}", flush=True)

    result["corpus_identity_check"] = corpus_check
    result["upstream_sentence_cap_split"] = over_cap
    result["floor_decomposition"] = summary
    if want_cells:
        result["wili_test_cells"] = eval_cells(null_model, tag)
    else:
        # Recorded, not omitted: a reader must be able to tell a section that
        # was deliberately skipped from one that was never produced.
        result["wili_test_cells"] = {
            "omitted": True,
            "reason": ("--no-wili-test-cells: this arm's container has not "
                       "been scored on WiLI-2018 test yet"),
            "expected_artifact": str(eval_artifact_for(null_model))}
    result["augmented_utc"] = datetime.now(timezone.utc).isoformat()
    result["produced_by"] = [
        f"analysis/wili_null_arm_verdict.py --arm {tag} (the three arms and "
        f"their cross-checks)",
        f"analysis/wili_null_arm_augment.py --arm {tag} (corpus check, floor "
        f"decomposition, upstream sentence-cap split, WiLI test cells)"]
    verdict_path.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {verdict_path}")

    s = summary
    n_material = s["n_languages_with_material_arm_a_signed_mean"]
    share_med = s["share_of_arm_a_signed_mean_carried_by_stored_floor_tokens"][
        "median"]
    if n_material:
        print(f"\nshare of the arm-(a) signed mean carried by tokens the stored "
              f"model put on its floor, over the {n_material} languages with "
              f"|signed mean| > {MATERIAL_SIGNED_MEAN}: median {share_med:.3f}")
    else:
        print(f"\nno language in this arm has |arm-(a) signed mean| > "
              f"{MATERIAL_SIGNED_MEAN}; the floor/above-floor attribution has "
              f"no deviation to split")
    # `verdict_group`, not `tag`: the arm tag is needed again below for the test
    # cells, and reusing the name silently looked the cells up under "passed".
    for verdict_group in ("failed", "passed"):
        g = s["by_original_gate_verdict"][verdict_group]
        print(f"\n{verdict_group} the original gate (n={g['n_languages']}), "
              f"stored vs {tag}:")
        for k in ("above_floor_set_jaccard_storedT_null_arm",
                  "signed_mean_a_restricted_above_both_floors",
                  "correlation_a_restricted_above_both_floors"):
            v = g[k]
            print(f"  {k:52} median {v['median']:+.6g}  "
                  f"[{v['min']:+.6g}, {v['max']:+.6g}]")
    if want_cells:
        print("\nWiLI-2018 test cells:")
        for label in ("stored_defective", tag, "fp64"):
            c = result["wili_test_cells"][label]
            print(f"  {label:18} macro F1 {c['macro_f1']:.6f}  "
                  f"macro FPR {c['macro_fpr']:.4e}  acc {c['accuracy']:.6f}  "
                  f"errors {c['n_errors_implied_by_accuracy']:,}  "
                  f"FPR==f(acc) {c['macro_fpr_matches_prediction']}")
    else:
        print(f"\nWiLI-2018 test cells omitted (--no-wili-test-cells); "
              f"expected artifact {eval_artifact_for(null_model)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
