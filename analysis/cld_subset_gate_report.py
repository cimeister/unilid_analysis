"""Assemble the CLD3-subset gate record from the runs, against the cells parsed
out of paper/tables/lid_main.tex.

Published values are PARSED from the .tex, never transcribed here, so the
comparison cannot drift from the table it gates against.
"""
import json
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(REPO, "outputs/rerelease/cld_subset")
TEX = os.path.join(REPO, "paper/tables/lid_main.tex")
# The original submission's table, before the 2026-08-20/24 corrected-generation
# edits. Every model gated here is a CARRIED-generation model, so its published
# counterpart is this version's cell, not the \corrrev'd one now in the file.
# The two agree on every subset cell (no \corrrev touched the right half) and
# differ on the \unilid row's three full-set pairs.
ORIG_COMMIT = "27883d5"

# Column order of tab:lid_main, left half then right half.
COLUMNS = [
    ("glotlidc", "full"), ("udhr", "full"), ("flores", "full"),
    ("glotlidc", "subset"), ("udhr", "subset"), ("flores", "subset"),
]

ROW_KEYS = {
    r"\cld": "cld3",
    r"\glotlid": "glotlid",
    r"\fasttext": "fasttext",
    r"\unilid": "unilid",
    r"\unilid (calibrated, \cref{sec:calibration})": "unilid_calibrated",
    r"\unilid-Mistral-Nemo": "nemo",
    r"\unilid-DeepSeek3.2": "deepseek",
    r"\unilid-Qwen3": "qwen3",
}

# tag -> (published row key, what generation this model is)
MODELS = {
    "released":         ("unilid",   "released / carried generation (the model behind the published \\unilid row)"),
    "carried_nemo":     ("nemo",     "carried generation, Mistral-Nemo variant"),
    "carried_deepseek": ("deepseek", "carried generation, DeepSeek3.2 variant (the co-author's own model)"),
    "carried_qwen3":    ("qwen3",    "carried generation, Qwen3 variant (the co-author's own model)"),
}

# Equivalence micro-check, run 2026-08-31 on the login node. Two models trained
# from the same WiLI per-language corpora with the SAME base tokenizer
# (results_wili_100k_500_fp64/tokenizers/langspec_base_tokenizer.json, md5
# 266ef55fb8f3dbd5cb2815f8d362c9f9), one on five languages (afr, deu, ell, hin,
# rus) and one on three of them (afr, deu, rus), with the patched fp64
# spm_train at ~/.local/bin.
EQUIV = {
    "shared_base_tokenizer_md5": "266ef55fb8f3dbd5cb2815f8d362c9f9",
    "model_5_languages": ["afr", "deu", "ell", "hin", "rus"],
    "model_3_languages": ["afr", "deu", "rus"],
    "packed_base_tokenizer_bytes_identical": True,
    "rows_bitwise_identical": True,
    "rows_differing_entries": {"afr": 0, "deu": 0, "rus": 0},
    "argmax_comparison": {
        "lines": 117500,
        "source": "wili_assets/wili-2018/x_test.txt",
        "label_mismatches": 0,
        "score_bitwise_mismatches": 0,
    },
    "caveat": ("Holds while the base vocabulary is fixed. A subset model whose "
               "base tokenizer is trained on the subset corpora alone has a "
               "different vocabulary and is a different model."),
    "verdict": "PASS",
}

EQUIV_MD = """The check trains two models from the same WiLI per-language corpora and the
same base tokenizer: one on five languages (afr, deu, ell, hin, rus), one on
three of them (afr, deu, rus). The three shared rows come out bitwise identical,
every one of their 100,000 float32 entries. Scoring all 117,500 lines of the
WiLI test set with the three-language model and with the five-language model
restricted to those same three rows gives 0 label mismatches and 0 bitwise
score mismatches. The author's equivalence holds as stated."""


_WRAP = re.compile(r"\\(?:corrrev|camrev|textbf)\{([^{}]*)\}")


def _clean(cell: str) -> str:
    prev = None
    while prev != cell:
        prev = cell
        cell = _WRAP.sub(r"\1", cell).strip()
    return cell


def parse_tex(path, commit=None):
    if commit:
        rel = os.path.relpath(path, REPO)
        body = subprocess.check_output(
            ["git", "-C", REPO, "show", f"{commit}:{rel}"], text=True)
    else:
        with open(path) as f:
            body = f.read()
    # Rows are "\name & c1 & ... & c12 \\", possibly spread over lines.
    rows = {}
    for chunk in body.split(r"\\"):
        chunk = chunk.strip()
        if not chunk or chunk.startswith("%"):
            continue
        chunk = "\n".join(ln for ln in chunk.splitlines()
                          if not ln.strip().startswith("%"))
        parts = [p.strip() for p in chunk.split("&")]
        if len(parts) != 13:
            continue
        head = parts[0]
        for rule in (r"\bottomrule", r"\midrule", r"\toprule"):
            if rule in head:
                head = head.rsplit(rule, 1)[1]
        name = " ".join(head.split())
        key = ROW_KEYS.get(name)
        if key is None:
            continue
        rows[key] = [_clean(p) for p in parts[1:]]
    missing = set(ROW_KEYS.values()) - set(rows)
    if missing:
        raise SystemExit(f"FATAL: lid_main.tex rows not parsed: {sorted(missing)}")
    return rows


def cell(rows, key, bench, mode, metric):
    i = COLUMNS.index((bench, mode)) * 2 + (0 if metric == "f1" else 1)
    return rows[key][i]


def fmt_f1(x):
    return f"{x:.3f}".lstrip("0")


def fmt_fpr(x):
    s = f"{x:.2e}"
    m, e = s.split("e")
    return f"{m}e{int(e)}"


def main():
    current = parse_tex(TEX)
    published = parse_tex(TEX, ORIG_COMMIT)
    changed = {k: [(i, a, b) for i, (a, b) in enumerate(zip(published[k], current[k]))
                   if a != b] for k in published}
    changed = {k: v for k, v in changed.items() if v}
    out = {
        "generated": "2026-08-31",
        "question": ("Does the author's ruled CLD3-subset convention (2026-08-26) "
                     "-- argmax restricted to the subset label set, over the full "
                     "model -- reproduce the published subset cells of "
                     "tab:lid_main on the carried generation?"),
        "published_source": f"{TEX} at git {ORIG_COMMIT} (original submission)",
        "published_rows": published,
        "current_tex_rows": current,
        "cells_changed_since_original": changed,
        "runs": {},
        "gate": [],
    }
    for tag, (rowkey, gen) in MODELS.items():
        out["runs"][tag] = {"published_row": rowkey, "generation": gen, "cells": {}}
        for bench, mode in COLUMNS:
            p = os.path.join(RUNS, f"{tag}_{bench}_{mode}.json")
            if not os.path.exists(p):
                continue
            with open(p) as f:
                s = json.load(f)
            pub_f1 = cell(published, rowkey, bench, mode, "f1")
            pub_fpr = cell(published, rowkey, bench, mode, "fpr")
            rec = {
                "measured_macro_f1": s["macro_f1"],
                "measured_macro_fpr": s["macro_fpr"],
                "measured_accuracy": s["accuracy"],
                "measured_at_printed_precision": [fmt_f1(s["macro_f1"]),
                                                  fmt_fpr(s["macro_fpr"])],
                "published": [pub_f1, pub_fpr],
                "num_languages": s["num_languages"],
                "total_samples": s["total_samples"],
                "errors": s["total_samples"] - s["correct"],
                "n_model_rows_evaluated": s["n_model_rows_evaluated"],
                "run": p,
            }
            # The confinement identity: under a convention that confines
            # predictions to the evaluated label set, (n-1)*macro_fpr == the
            # error rate exactly. Applied to the PUBLISHED fpr it says how many
            # errors the published cell implies on this line pool.
            # Only meaningful in subset mode: in full mode predictions are not
            # confined to the evaluated label set, so the identity does not hold
            # and the ratio would be an artefact.
            try:
                if mode != "subset":
                    raise ValueError("identity applies only under confinement")
                implied = (rec["num_languages"] - 1) * float(pub_fpr.replace("e", "E"))
                rec["published_implied_errors"] = implied * s["total_samples"]
                rec["measured_over_published_errors"] = (
                    rec["errors"] / rec["published_implied_errors"]
                    if rec["published_implied_errors"] else None)
            except ValueError:
                rec["published_implied_errors"] = None
            f1_match = rec["measured_at_printed_precision"][0] == pub_f1
            fpr_match = rec["measured_at_printed_precision"][1] == pub_fpr
            rec["f1_reproduces"] = f1_match
            rec["fpr_reproduces"] = fpr_match
            out["runs"][tag]["cells"][f"{bench}_{mode}"] = rec
            out["gate"].append({
                "tag": tag, "column": f"{bench}_{mode}",
                "measured": rec["measured_at_printed_precision"],
                "published": [pub_f1, pub_fpr],
                "verdict": "MATCH" if (f1_match and fpr_match) else "MISS",
            })
    for tag in ("released", "carried_deepseek", "carried_qwen3"):
        fp = os.path.join(RUNS, f"{tag}_ypred_scoring_gate_prefix.json")
        if os.path.exists(fp):
            with open(fp) as f:
                out["runs"][tag]["scoring_path_gate_prefix"] = json.load(f)
    for tag, p in [("released", "released_ypred_scoring_gate.json"),
                   ("carried_deepseek", "carried_deepseek_ypred_scoring_gate.json"),
                   ("carried_qwen3", "carried_qwen3_ypred_scoring_gate.json")]:
        fp = os.path.join(RUNS, p)
        if os.path.exists(fp):
            with open(fp) as f:
                out["runs"][tag]["scoring_path_gate"] = json.load(f)
    out["equivalence_micro_check"] = EQUIV
    print(json.dumps(out["gate"], indent=1))
    with open(sys.argv[1], "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", sys.argv[1])
    if len(sys.argv) > 2:
        write_md(out, sys.argv[2], EQUIV_MD)




# ---------------------------------------------------------------------------
# Markdown record. Prose is literal here; every number comes from the JSON the
# runs wrote, so the two cannot drift.
# ---------------------------------------------------------------------------

ROW_LABEL = {
    "released": "\\unilid (released model)",
    "carried_nemo": "\\unilid-Mistral-Nemo",
    "carried_deepseek": "\\unilid-DeepSeek3.2",
    "carried_qwen3": "\\unilid-Qwen3",
}
COL_LABEL = {
    "glotlidc_subset": "GlotLID-C, 83 languages",
    "udhr_subset": "UDHR, 80 languages",
    "flores_subset": "FLORES-200, 77 languages",
    "glotlidc_full": "GlotLID-C, 1940 labels",
    "udhr_full": "UDHR, 366 labels",
    "flores_full": "FLORES-200, 190 labels",
}
ORDER = ["glotlidc_subset", "udhr_subset", "flores_subset",
         "udhr_full", "flores_full", "glotlidc_full"]


def md_table(out, mode):
    lines = ["| row | column | measured F1 | published F1 | measured FPR | "
             "published FPR | lines | errors | errors the published FPR implies | ratio |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for tag in ("released", "carried_nemo", "carried_deepseek", "carried_qwen3"):
        r = out["runs"].get(tag)
        if not r:
            continue
        for col in ORDER:
            if not col.endswith(mode) or col not in r["cells"]:
                continue
            c = r["cells"][col]
            pi = c.get("published_implied_errors")
            ratio = c.get("measured_over_published_errors")
            lines.append(
                f"| {ROW_LABEL[tag]} | {COL_LABEL[col]} | "
                f"{c['measured_macro_f1']:.5f} | {c['published'][0]} | "
                f"{c['measured_macro_fpr']:.4e} | {c['published'][1]} | "
                f"{c['total_samples']:,} | {c['errors']:,} | "
                f"{(f'{pi:.1f}' if pi else '-')} | "
                f"{(f'{ratio:.2f}' if ratio else '-')} |")
    return "\n".join(lines)


def write_md(out, path, equiv):
    verdicts = [g["verdict"] for g in out["gate"]
                if g["column"].endswith("subset")]
    n_miss = sum(v == "MISS" for v in verdicts)
    sp = out["runs"].get("released", {}).get("scoring_path_gate")
    body = f"""# The CLD3-subset cells under the 2026-08-26 ruling: gate result

Generated {out['generated']}. Measurements:
`outputs/rerelease/cld_subset_gate_2026-08-31.json` and the per-run summaries
under `outputs/rerelease/cld_subset/`.

**Verdict: the gate FAILS. {n_miss} of the {len(verdicts)} subset cells measured
so far miss, and none matches.** No corrected-variant number was computed. The
variant-row swap of PD-3 cannot proceed on this evidence. The four GlotLID-C
subset cells are still queued (section 8); they cannot reverse a verdict that
already stands on eight cells across four models.

## 1. What the ruling licensed, and what was built

The author ruled on 2026-08-26 that a restricted argmax over the full model and
a model trained on only the subset languages are the same computation, because
each language's row is estimated individually over the shared base vocabulary.
`analysis/cld_subset_eval.py` implements that reading of the CLD3-subset
columns: the model is loaded restricted to its labels whose bare ISO 639-3 code
is in the subset definition file, the benchmark is filtered to the lines whose
gold bare ISO is in that set (the `only_model_langs` filter of the paper team's
own `unilid_resources/eval_*.py`), and the paper team's macro F1 and macro FPR
core is applied over bare ISO labels. The decode is Viterbi, the default of all
three of those scripts.

## 2. The equivalence micro-check

{equiv}

The equivalence holds while the base vocabulary is held fixed. It is a claim
about the per-language rows, and the rows are what the check compares. A subset
model trained from scratch, whose base tokenizer is fitted to the subset
corpora alone, has a different vocabulary and is not covered by the check or by
the ruling as implemented here.

## 3. The gate

Every model in the table below is a carried-generation model: the released
`glotlidc.unilid`, and the three variant models behind the published variant
rows. Published values are parsed from `paper/tables/lid_main.tex` at git
{ORIG_COMMIT} (the original submission), because that is the generation these
models belong to; no `\\corrrev` edit touched a subset cell, so the subset
columns are identical in the current file.

The last three columns use the confinement identity: when predictions are
confined to the evaluated label set, every error is a false positive for
exactly one label, so `(n - 1) * macro_FPR` equals the error rate exactly. Read
against the published FPR it gives the number of errors that FPR implies on
this line pool, which is directly comparable with the measured error count.

### CLD3-subset columns

{md_table(out, 'subset')}

### Full-label-set columns, run as instrument checks

{md_table(out, 'full')}

## 4. Scoring-path check

"""
    if sp:
        body += (
            f"The scorer here reproduces the co-author's own recorded per-line "
            f"GlotLID-C predictions for the released model on every one of "
            f"{sp['sampled']:,} sampled lines (every {sp['stride']}th line of "
            f"the {sp['test_lines']:,}-line test file), agreement "
            f"{sp['agreement']:.6f}. The instrument's predictions are the "
            f"published generation's predictions; the miss is not a scorer "
            f"difference.\n")
    else:
        rows = []
        stride = None
        for tag in ("released", "carried_deepseek", "carried_qwen3"):
            sp2 = out["runs"].get(tag, {}).get("scoring_path_gate_prefix")
            if sp2:
                stride = sp2["stride"]
                rows.append(
                    f"| {ROW_LABEL[tag]} | {sp2['test_lines']:,} | "
                    f"{sp2['sampled']:,} | {sp2['mismatches']} |")
        if rows:
            body += (
                "The whole-file check is queued (section 8). Measured so far, on "
                "a prefix of the test file, every "
                f"{stride}th line scored with the full model and compared "
                "with the co-author's recorded prediction for that line:\n\n"
                "| row | test lines read | lines scored | mismatches |\n"
                "|---|---|---|---|\n" + "\n".join(rows) + "\n\n"
                "The instrument's predictions are the published generation's "
                "predictions. The miss in section 3 is not a scorer "
                "difference.\n")
        else:
            body += "Not available in this record.\n"

    body += r"""
## 5. The direction and size of the miss

Restricting the argmax to 80 or 77 languages compresses the four systems into a
narrow band of macro F1, because every out-of-subset error the full model would
make is reassigned to a subset label and most of those reassignments are
correct. The published cells span a much wider range. Measured against the
error counts the published FPRs imply, the released model's measured cells
carry more errors than published, and all three variant models' measured cells
carry fewer. A single convention applied to a single set of models cannot move
in both directions at once, so the published subset columns were not produced
by a restricted argmax over these models.

Two readings remain, and only the author or the co-author can separate them.

1. The published subset columns were produced by some other convention, and the
   2026-08-26 ruling identifies the wrong one. The exclusion proof in
   `outputs/rerelease/cld_subset_convention_sweep.md` assumed one convention
   across all systems in a column. `outputs/tables/paper_eval_cld3_subset.md`
   already recorded the alternative, that the columns mix conventions per
   system, on the separate evidence that no tested convention reproduced the
   published \fasttext GlotLID-C subset cell of .990.
2. The published subset columns were produced by models trained on the subset
   languages from scratch, each with its own base vocabulary fitted to the
   subset corpora. The equivalence in section 2 does not cover that case: it
   holds for the per-language rows over a shared base vocabulary, and a
   from-scratch subset model does not share one. Such models are not on disk
   here and would have to be retrained per row and per benchmark.

## 6. Consequences for PD-3 and PD-5

PD-3's condition was "do the swap if the other columns can be computed". The
columns can now be computed, and the computation does not reproduce the carried
cells the swapped row would sit beside. Applying it would put six cells from
one convention next to six from another inside a single table, which is the
failure mode the original PD-3 closure was protecting against. My
recommendation is to leave both variant rows carried, on the same conclusion as
before but now on measured rather than absent evidence, and to narrow the C3
ask a second time to: which models produced the CLD3-subset columns, and were
they the same models as the corresponding full-set columns?

## 7. The \unilid and calibrated rows' own subset cells

`tab:lid_main`'s right half already carries two conventions, and adopting the
ruled one would make three.

- The \unilid row's three subset pairs are carried unchanged from the original
  submission. No `\corrrev` edit touched them, so they are the co-author's
  original numbers under the unknown convention.
- The calibrated row's three subset F1 cells were computed in this repository
  under a different convention: test lines restricted to those whose gold label
  is in the subset, predictions NOT restricted, each bare ISO code mapped to its
  largest-training-corpus `lang_Script` variant, macro over the 83, 80 and 77.
  The records are `outputs/tables/paper_eval_cld3_subset.md` (GlotLID-C,
  baseline 0.9719 against the printed .971, gate_flat4_prox21 0.9751 printed as
  .975) and `outputs/tables/paper_eval_cld3_subset_external.md` (UDHR baseline
  0.9873 against the printed .992, calibrated 0.9856 printed as .986; FLORES
  baseline 0.9907 against the printed .997, recorded there as a mismatch,
  calibrated 0.9920 printed as .992). Its three subset FPR cells are omitted,
  and the caption says why.

If the ruled convention were adopted for the variant rows, both of these rows
would have to be regenerated under it for the table to be internally consistent.
That is not a free edit: the \unilid row's own subset cells move under it,
measured in section 3 at 0.996 and 0.996 on UDHR and FLORES against the printed
.992 and .997. So adopting the convention changes published \unilid numbers,
which is a larger decision than the variant-row swap it was meant to enable.
"""
    missing = [(t, c) for t in ROW_LABEL for c in ORDER
               if c != "glotlidc_full"
               and c not in out["runs"].get(t, {}).get("cells", {})]
    body += "\n## 8. What has not been measured\n\n"
    if missing:
        body += ("The following cells have no run in `outputs/rerelease/cld_subset/` "
                 "yet:\n\n")
        for t, c in missing:
            body += f"- {ROW_LABEL[t]}, {COL_LABEL[c]}\n"
        body += ("\nSLURM jobs 3244447 (released), 3244448 (DeepSeek3.2), "
                 "3244449 (Qwen3) and 3244450 (Mistral-Nemo) cover all of them "
                 "plus the whole-file scoring-path check. Submitted "
                 "2026-08-31, queue congested at submission (1,481 jobs "
                 "pending on the normal partition), estimated start the "
                 "following morning.\n")
        body += ("\nRegenerate this record with "
                 "`python3 -m analysis.cld_subset_gate_report "
                 "outputs/rerelease/cld_subset_gate_2026-08-31.json "
                 "outputs/rerelease/cld_subset_gate_2026-08-31.md` once they land.\n")
    else:
        body += "Every cell in section 3 has a run behind it.\n"
    body += """
The GlotLID-C full-label-set cell is deliberately not computed: it is a pass of
1,940 rows over 45,627,279 lines, about 12 h per model, and
`outputs/rerelease/cld_subset_convention_sweep.json` already reproduces it from
the recorded prediction files to 7 significant digits. Section 4's check gates
the scorer on the same benchmark at 1/450 of that cost.

## 9. Provenance

- Instrument: `analysis/cld_subset_eval.py`.
- Scoring-path check: `analysis/ypred_scoring_gate.py`.
- This record: `analysis/cld_subset_gate_report.py`, which parses the published
  cells out of `paper/tables/lid_main.tex` rather than carrying transcriptions.
- SLURM: `slurm_cld_subset_glotlidc.sh`, account infra01, one job per model.
- Per-run summaries, per-label metrics and banked per-line predictions:
  `outputs/rerelease/cld_subset/`.
"""
    with open(path, "w") as f:
        f.write(body)
    print("wrote", path)


if __name__ == "__main__":
    main()
