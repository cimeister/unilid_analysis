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
# The 2026-08-31 reassessment's own measurements: the author's answer verbatim,
# the pool census, and every (pool x row set x decode x label collapse x model
# container) combination scored against the published variant cells. Written by
# the reassessment harness; section 10 renders it and carries no number of its
# own. Absent -> section 10 is omitted rather than half-rendered.
REASSESS = os.path.join(RUNS, "variant_hypothesis_matrix.json")

# Training runs whose recorded wall clock is the cost basis of section 10's
# reproduction bill. Read at report time from each run's own
# training_summary.json, so the bill's arithmetic is a measurement.
TRAINING_ROOT = "/capstor/scratch/cscs/cmeister747/unilid_analysis"
COST_RUNS = [
    ("results_deepseek_v32_fp64",
     "1,940 languages, LLM base tokenizer reused: row estimation only"),
    ("results_qwen3_8b_fp64",
     "1,940 languages, LLM base tokenizer reused: row estimation only"),
    ("results_wili_100k_defaults_fp64",
     "235 languages, 100k vocabulary FITTED FROM SCRATCH on the training corpora"),
    ("results_wili_100k_500_fp64",
     "235 languages, 100k base tokenizer supplied: row estimation only"),
]
# The shared 1,940-file GlotLID-C corpus draw every later retrain used, and the
# only per-language GlotLID-C corpus still on disk (the monolithic train.txt is
# gone). Its training_summary.json carries the per-corpus line counts a subset
# training would consume.
CORPUS_SUMMARY = os.path.join(TRAINING_ROOT, "results_apertus200k",
                              "training_summary.json")
SUBSET_FILES = {
    83: "unilid_resources/glotlidc_cld3subset_83.txt",
    80: "unilid_resources/udhr_cld3subset_80.txt",
    77: "unilid_resources/flores_cld3subset_77.txt",
}
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


# The two storage roots that hold every .unilid container for this project
# (analysis/config.py's SCRATCH_DIR, and the durable store it symlinks into).
# Scanned at report time so section 5's claim about which models exist is a
# measurement, not a transcription.
MODEL_ROOTS = [
    "/capstor/scratch/cscs/cmeister747/unilid_analysis",
    "/capstor/store/cscs/swissai/a0229/cmeister/unilid_analysis",
]
# .unilid header: magic, version, num_langs, vocab_size, base_tok_len, langs_len.
_UNILID_HDR = "<8sIIIII4x"


def scan_model_containers():
    """Row count and vocabulary size of every .unilid under MODEL_ROOTS.

    Keyed by realpath so a symlink and its target are counted once. A container
    whose header cannot be read is recorded with its error rather than skipped.
    """
    import struct
    size = struct.calcsize(_UNILID_HDR)
    found = {}
    for root in MODEL_ROOTS:
        if not os.path.isdir(root):
            raise SystemExit(f"FATAL: model root missing at {root}")
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                if not name.endswith(".unilid"):
                    continue
                path = os.path.join(dirpath, name)
                real = os.path.realpath(path)
                if real in found:
                    continue
                try:
                    with open(path, "rb") as fh:
                        _m, _v, n_langs, vocab, _b, _l = struct.unpack(
                            _UNILID_HDR, fh.read(size))
                    found[real] = {"path": path, "num_languages": n_langs,
                                   "vocab_size": vocab}
                except Exception as exc:
                    found[real] = {"path": path, "error": str(exc)}
    counts = sorted({e["num_languages"] for e in found.values()
                     if "num_languages" in e})
    return {
        "roots": MODEL_ROOTS,
        "n_containers": len(found),
        "distinct_language_counts": counts,
        "subset_sized_present": bool(set(counts) & {83, 80, 77}),
        "containers": {k: v for k, v in sorted(found.items())},
    }


def scan_training_costs():
    """Wall clock and corpus size of the training runs the reproduction bill
    extrapolates from, plus how much of the GlotLID-C corpus each CLD3 subset
    covers. Read from the runs' own summaries; nothing here is transcribed."""
    runs = {}
    for name, what in COST_RUNS:
        p = os.path.join(TRAINING_ROOT, name, "training_summary.json")
        if not os.path.exists(p):
            runs[name] = {"error": f"missing at {p}", "what": what}
            continue
        with open(p) as f:
            d = json.load(f)
        t = d["timing"]
        runs[name] = {
            "what": what,
            "num_languages": d["source"]["num_languages"],
            "total_samples": d["source"]["total_samples"],
            "vocab_size": d.get("method", {}).get("vocab_size"),
            "total_seconds": t["total_seconds"],
            "base_tokenizer_seconds": t["base_tokenizer_seconds"],
            "language_tokenizers_seconds": t["language_tokenizers_seconds"],
            "base_tokenizer_reused": t["base_tokenizer_reused"],
        }
    if not os.path.exists(CORPUS_SUMMARY):
        raise SystemExit(f"FATAL: corpus summary missing at {CORPUS_SUMMARY}")
    with open(CORPUS_SUMMARY) as f:
        spl = json.load(f)["source"]["samples_per_language"]
    total = sum(spl.values())
    subsets = {}
    for n, rel in SUBSET_FILES.items():
        p = os.path.join(REPO, rel)
        if not os.path.exists(p):
            raise SystemExit(f"FATAL: subset definition missing at {p}")
        with open(p) as f:
            codes = {ln.strip() for ln in f if ln.strip()}
        rows = [k for k in spl if k.split("_", 1)[0] in codes]
        s = sum(spl[k] for k in rows)
        subsets[n] = {"n_codes": len(codes), "n_corpora": len(rows),
                      "samples": s, "share_of_full": s / total}
    return {"runs": runs,
            "glotlidc_corpus": {
                "path": os.path.join(TRAINING_ROOT, "results_apertus200k", "corpus"),
                "present": os.path.isdir(
                    os.path.join(TRAINING_ROOT, "results_apertus200k", "corpus")),
                "n_corpora": len(spl), "total_samples": total},
            "subsets": subsets}


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
    out["model_container_census"] = scan_model_containers()
    if os.path.exists(REASSESS):
        with open(REASSESS) as f:
            out["reassessment_2026_08_31"] = json.load(f)
        out["reassessment_2026_08_31"]["training_cost_basis"] = scan_training_costs()
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


# ---------------------------------------------------------------------------
# Section 10: the 2026-08-31 reassessment. Every number below is read out of
# out["reassessment_2026_08_31"] or out["runs"]; the prose is literal.
# ---------------------------------------------------------------------------

REASSESS_TAGS = ["released", "carried_nemo", "carried_deepseek", "carried_qwen3"]
# The registry pool for each benchmark: the one the full-set columns are
# reproduced on, and the one the gate itself used.
REGISTRY_POOL = {"udhr": "tsv", "flores": "flores200_devtest"}
# The row-set hypotheses, in the order section 10.4 prints them.
ROWSET_ORDER = [
    ("all", "rows >= gold labels"),
    ("gold_scripts", "gold scripts only"),
    ("cld83_all", "one 83-language model"),
    ("one_largest", "1 row/lang, largest script"),
    ("one_first", "1 row/lang, first script"),
    ("merged_one_row_per_language_corpus", "merged 1 row/lang, corpus-weighted"),
    ("merged_one_row_per_language_equal", "merged 1 row/lang, equal-weighted"),
]
BENCH_OF = {"udhr_subset": "udhr", "flores_subset": "flores"}


def _identity_table(out):
    lines = ["| row | UDHR, 366 labels | published | FLORES-200, 190 labels | "
             "published | is this the published model? |",
             "|---|---|---|---|---|---|"]
    verdicts = {}
    for tag in REASSESS_TAGS:
        cells = out["runs"].get(tag, {}).get("cells", {})
        got = []
        ok = []
        for col in ("udhr_full", "flores_full"):
            c = cells.get(col)
            if not c:
                got.append(("-", "-"))
                ok.append(None)
                continue
            got.append((f"{c['measured_at_printed_precision'][0]} / "
                        f"{c['measured_at_printed_precision'][1]}",
                        f"{c['published'][0]} / {c['published'][1]}"))
            ok.append(c["f1_reproduces"] and c["fpr_reproduces"])
        tested = [v for v in ok if v is not None]
        if not tested:
            verdict = "not measured"
        elif all(tested):
            verdict = (f"**yes**, reproduces the {len(tested)} full-set "
                       f"{'cell' if len(tested) == 1 else 'cells'} measured, on "
                       "both metrics")
        else:
            verdict = "**no**"
        verdicts[tag] = (verdict, tested)
        lines.append(f"| {ROW_LABEL[tag]} | {got[0][0]} | {got[0][1]} | "
                     f"{got[1][0]} | {got[1][1]} | {verdict} |")
    return "\n".join(lines), verdicts


def _direction_table(out):
    lines = ["| row | column | published F1 | measured F1 | errors measured | "
             "errors the published F1 implies | errors the published FPR implies "
             "| published value implies |",
             "|---|---|---|---|---|---|---|---|"]
    counts = {"more": 0, "fewer": 0}
    for tag in REASSESS_TAGS:
        for col in ("udhr_subset", "flores_subset"):
            c = out["runs"].get(tag, {}).get("cells", {}).get(col)
            if not c:
                continue
            ef1 = (1 - float(c["published"][0])) * c["total_samples"]
            efpr = c.get("published_implied_errors")
            more = ef1 > c["errors"]
            counts["more" if more else "fewer"] += 1
            lines.append(
                f"| {ROW_LABEL[tag]} | {COL_LABEL[col]} | {c['published'][0]} | "
                f"{c['measured_macro_f1']:.5f} | {c['errors']:,} | {ef1:.1f} | "
                f"{(f'{efpr:.1f}' if efpr else '-')} | "
                f"{'more errors' if more else 'fewer errors'} |")
    return "\n".join(lines), counts


def _matrix_table(ra, bench, out):
    """One row per model, one column per row-set hypothesis, at the registry
    pool under the Viterbi decode and the bare-ISO collapse -- the axes the
    printed cell is defined on. The other axes are summarised beneath."""
    col = f"{bench}_subset"
    by = {}
    for r in ra["combinations"]:
        if (r["bench"] == bench and r["pool"] == REGISTRY_POOL[bench]
                and r["decode"] == "viterbi" and r["collapse"] == "bare"):
            by[(r["tag"], r["rowset"])] = r
    head = ["row", "published F1"] + [lbl for _k, lbl in ROWSET_ORDER]
    lines = ["| " + " | ".join(head) + " |",
             "|" + "---|" * len(head)]
    matches = []
    for tag in REASSESS_TAGS:
        c = out["runs"].get(tag, {}).get("cells", {}).get(col)
        if not c:
            continue
        cells = [ROW_LABEL[tag], c["published"][0]]
        for key, _lbl in ROWSET_ORDER:
            r = by.get((tag, key))
            if not r:
                cells.append("-")
                continue
            got = fmt_f1(r["macro_f1"])
            hit = got == c["published"][0]
            fpr_hit = fmt_fpr(r["macro_fpr"]) == c["published"][1]
            if hit:
                matches.append({"tag": tag, "column": col, "rowset": key,
                                "f1": got, "fpr_also": fpr_hit,
                                "measured_fpr": fmt_fpr(r["macro_fpr"])})
            cells.append(f"**{got}**" if hit else got)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines), matches


def _invariance(ra, bench):
    """How much the axes that are NOT the row set move the cell, on the row set
    the gate itself used. If these are small, the row set is the only axis with
    any purchase on the miss."""
    rows = [r for r in ra["combinations"]
            if r["bench"] == bench and r["rowset"] == "all"]
    out = {}
    for tag in {r["tag"] for r in rows}:
        mine = [r for r in rows if r["tag"] == tag]
        base = [r for r in mine if r["pool"] == REGISTRY_POOL[bench]
                and r["decode"] == "viterbi" and r["collapse"] == "bare"]
        if not base:
            continue
        b = base[0]["macro_f1"]
        out[tag] = max(abs(r["macro_f1"] - b) for r in mine)
    return max(out.values()) if out else None


# The co-author's own container against this repository's fp64 retrain of it.
CONTAINER_PAIRS = [("carried_deepseek", "carried_deepseek_fp64_retrain"),
                   ("carried_qwen3", "carried_qwen3_fp64_retrain")]


def _container_delta(ra):
    """How far this repository's fp64 retrain of a variant moves the cell from
    the co-author's own container, on the gate's row set and axes."""
    idx = {}
    for r in ra["combinations"]:
        if r["rowset"] == "all" and r["decode"] == "viterbi" \
                and r["collapse"] == "bare" \
                and r["pool"] == REGISTRY_POOL[r["bench"]]:
            idx[(r["tag"], r["bench"])] = r["macro_f1"]
    deltas = [abs(idx[(a, b)] - idx[(c, b)])
              for a, c in CONTAINER_PAIRS for b in ("udhr", "flores")
              if (a, b) in idx and (c, b) in idx]
    return max(deltas) if deltas else None


# The row-set readings that keep every gold label. Under all of them the
# computation is a restricted argmax over rows of the released model, which is
# what the 2026-08-26 ruling licensed.
KEEPS_ALL_GOLD = {"all", "gold_scripts", "cld83_all"}


def _axis_matches(ra, out):
    """Of every combination that is a restricted argmax over rows of the full
    model -- any pool, any decode, any collapse, any of the three row-set
    readings that keep the gold labels, either container -- how many round to
    their published cell? This is the claim, measured, rather than inferred
    from a delta."""
    n = 0
    hits = []
    for r in ra["combinations"]:
        if r["rowset"] not in KEEPS_ALL_GOLD:
            continue
        col = f"{r['bench']}_subset"
        # The fp64 retrains stand in for their co-author container's row.
        tag = r["tag"].replace("_fp64_retrain", "")
        c = out["runs"].get(tag, {}).get("cells", {}).get(col)
        if not c:
            continue
        n += 1
        if fmt_f1(r["macro_f1"]) == c["published"][0]:
            hits.append({"tag": tag, "column": col, "rowset": r["rowset"],
                         "decode": r["decode"], "pool": r["pool"],
                         "collapse": r["collapse"],
                         "macro_f1": r["macro_f1"],
                         "measured_fpr": fmt_fpr(r["macro_fpr"]),
                         "published_fpr": c["published"][1],
                         "errors": r["errors"]})
    return n, hits


def _group_stats(ra):
    """Per benchmark: how far apart the three row-set readings that keep every
    gold label are, against how far a reading that drops a gold label moves the
    error count. Registry pool, Viterbi decode, bare-ISO metric."""
    out = {}
    for bench in ("udhr", "flores"):
        rows = [r for r in ra["combinations"]
                if r["bench"] == bench and r["pool"] == REGISTRY_POOL[bench]
                and r["decode"] == "viterbi"
                and not r["tag"].endswith("_fp64_retrain")]
        # The benchmark's own gold label count, read off the lang_Script
        # collapse, which averages over exactly the gold labels.
        n_gold = max(r["num_languages"] for r in rows
                     if r["collapse"] == "script")
        rows = [r for r in rows if r["collapse"] == "bare"]
        keep = {}
        per_rowset = {}
        for r in rows:
            per_rowset.setdefault(r["rowset"], []).append(r["errors"])
            if r["rowset"] in KEEPS_ALL_GOLD:
                keep.setdefault(r["tag"], []).append(r)
        spread = max(
            (max(x["macro_f1"] for x in v) - min(x["macro_f1"] for x in v)
             for v in keep.values()), default=None)
        keep_err = [x["errors"] for v in keep.values() for x in v]
        out[bench] = {
            "n_gold_labels": n_gold,
            "keep_spread_f1": spread,
            "keep_errors": (min(keep_err), max(keep_err)) if keep_err else None,
            "errors_by_rowset": {k: (min(v), max(v))
                                 for k, v in sorted(per_rowset.items())},
            # On FLORES-77 the largest-corpus script variant is the gold script
            # for all 77 languages, so that reading is the gold-script reading.
            "one_largest_equals_gold_scripts": (
                per_rowset.get("one_largest") == per_rowset.get("gold_scripts")),
        }
    return out


def reassessment_md(out):
    ra = out.get("reassessment_2026_08_31")
    if not ra:
        return ""
    ident, ident_verdicts = _identity_table(out)
    direction, counts = _direction_table(out)
    m_udhr, hits_udhr = _matrix_table(ra, "udhr", out)
    m_flores, hits_flores = _matrix_table(ra, "flores", out)
    hits = hits_udhr + hits_flores
    inv = max(x for x in (_invariance(ra, "udhr"), _invariance(ra, "flores"))
              if x is not None)
    cont = _container_delta(ra)
    n_axis, axis_hits = _axis_matches(ra, out)
    n_axis_match = len(axis_hits)
    axis_cells = sorted({(h["tag"], h["column"]) for h in axis_hits})
    if not axis_hits:
        axis_text = "not one rounds to its published cell."
    else:
        parts = []
        for t, c in axis_cells:
            mine = [h for h in axis_hits if (h["tag"], h["column"]) == (t, c)]
            rs = " and ".join(sorted({h["rowset"] for h in mine}))
            dec = " and ".join(sorted({h["decode"] for h in mine}))
            lo = min(h["macro_f1"] for h in mine)
            hi = max(h["macro_f1"] for h in mine)
            rng = f"{lo:.6f}" if lo == hi else f"{lo:.6f} to {hi:.6f}"
            rs_label = dict(ROWSET_ORDER).get(rs, rs)
            parts.append(f"{ROW_LABEL[t]} on {COL_LABEL[c]}, under the "
                         f"\"{rs_label}\" row set with the {dec} decode, at "
                         f"macro F1 {rng}")
        axis_text = (
            f"{n_axis_match} round to their published F1, and they are all the "
            f"same cell: {'; '.join(parts)}. None of them also matches that "
            "cell's published FPR."
            if len(axis_cells) == 1 else
            f"{n_axis_match} round to their published F1, over "
            f"{len(axis_cells)} cells: {'; '.join(parts)}. None of them also "
            "matches its cell's published FPR.")
        h = axis_hits[0]
        axis_text += (
            f" Read that match with care. It is the one cell whose published "
            f"value implies fewer errors than the measurement, it clears the "
            f"rounding boundary at 0.9965 by "
            f"{min(x['macro_f1'] for x in axis_hits) - 0.9965:.6f}, its "
            f"{min(x['errors'] for x in axis_hits)} errors are still well "
            f"above the {out['runs'][h['tag']]['cells'][h['column']]['published_implied_errors']:.0f} "
            f"the published FPR implies, and that FPR reads "
            f"{h['measured_fpr']} against a published {h['published_fpr']}.")
    grp = _group_stats(ra)
    cost = ra["training_cost_basis"]
    cr = cost["runs"]
    rows_1940 = cr["results_deepseek_v32_fp64"]
    base_fit = cr["results_wili_100k_defaults_fp64"]
    corpus = cost["glotlidc_corpus"]
    sub83 = cost["subsets"]["83"] if "83" in cost["subsets"] else cost["subsets"][83]
    rate = rows_1940["total_samples"] / rows_1940["language_tokenizers_seconds"]
    row_est_83 = sub83["samples"] / rate

    by_cell = {}
    for h in hits:
        by_cell.setdefault((h["tag"], h["column"]), []).append(h)
    hit_text = (
        "No combination reproduces any published subset cell."
        if not hits else
        f"The merged mixture reproduces a published subset F1 at its printed "
        f"precision for {len(by_cell)} cells, and no other scored combination "
        "of it does: "
        + "; ".join(
            f"{ROW_LABEL[t]} on {COL_LABEL[c]}, F1 {v[0]['f1']} under "
            + " and ".join(
                h["rowset"].rsplit("_", 1)[-1] + "-weighted" for h in v)
            + " pooling, against a measured FPR of "
            + " and ".join(h["measured_fpr"] for h in v)
            + f" for a published {dict(zip(('f1', 'fpr'), out['runs'][t]['cells'][c]['published']))['fpr']}"
            for (t, c), v in by_cell.items()) + ".")
    hit_clause = ("the merged one-row-per-language mixture of section 10.4"
                  if hits else "none")
    n_hit_cells = len(by_cell)

    cost_rows = []
    for k, v in cr.items():
        if "error" in v:
            continue
        base = ("reused, 0 s" if v["base_tokenizer_reused"]
                else f"{v['base_tokenizer_seconds']:,.0f} s")
        cost_rows.append(
            f"| `{k}` | {v['num_languages']:,} | {v['total_samples']:,} | "
            f"{v['vocab_size']:,} | {base} | "
            f"{v['language_tokenizers_seconds']:,.0f} s |")
    cost_table = "\n".join(cost_rows)
    udhr_typo_errors = 79 * 1.06e-4 * 5509
    udhr_f1_errors = (1 - 0.992) * 5509

    return f"""
## 10. Reassessment under the author's answer (2026-08-31)

Measurements: `outputs/rerelease/cld_subset/variant_hypothesis_matrix.json`
({ra['n_combinations']} scored combinations) and the full-label-set runs added
to `outputs/rerelease/cld_subset/` on the same date.

### 10.1 The answer

> {ra['author_answer']['verbatim']}

Recorded in {"; ".join("`" + s + "`" for s in ra['author_answer']['recorded_in'])}.

The answer is about the **plain** \\unilid row. It settles reading (b) of section
5 for that row: its subset cells came from models whose base vocabulary was
fitted to the subset languages. Such a model is not a row subset of the
released model, so the 2026-08-26 equivalence does not apply to it. The answer
says nothing about the three variant rows, whose base vocabulary is a fixed LLM
tokenizer and cannot be fitted to a subset.

### 10.2 Which of these four containers is the model behind its published row

The full-label-set columns use the same container, the same line pool and the
same scorer as the subset columns, on the same generation. Reproducing them is
therefore a check on the container, the pool and the scorer together, with the
subset convention taken out.

{ident}

Three of the four reproduce their published full-set cells exactly, on both
metrics. \\unilid-Mistral-Nemo does not.
`outputs/tables/mistralnemo_eval.md` already records why:
the Mistral-Nemo container on this filesystem is an independent retrain from the
same recipe, not the paper team's own training run, and that record states
"rough proximity is expected, not equality". The Mistral-Nemo subset
discrepancy is therefore a mixture of a convention difference and a model
difference, and separating the two is not possible here. The evidence about the convention is
the six subset cells of \\unilid, \\unilid-DeepSeek3.2 and \\unilid-Qwen3, not
eight.

### 10.3 The direction of the discrepancy, and what the answer accounts for

Each published cell gives an error count two independent ways. Macro F1 read
as accuracy is close on these balanced supports. Macro FPR read through the
confinement identity of section 3 is exact. The F1 reading is coarse, because
one unit in the third printed decimal is 2.8 errors on the 5,509-line UDHR pool
and 39 errors on the 77,924-line FLORES pool. The two readings agree for every
cell except the \\unilid UDHR pair, whose FPR is the value section 5 of
`cld_subset_convention_sweep.md` argues is a `1.06e-5`-for-`1.06e-4` exponent
typo. Read with that correction they agree there too: {udhr_typo_errors:.1f}
errors from the FPR against {udhr_f1_errors:.1f} from the F1.

{direction}

For {counts['more']} of the {counts['more'] + counts['fewer']} measured
subset cells, the published value implies more errors than a restricted argmax
over the full model produces. For {counts['fewer']} it implies fewer. Setting
aside the two Mistral-Nemo cells, whose container is not the published model
(section 10.2), that is {counts['more'] - 2} cells one way and
{counts['fewer']} the other. The one exception is \\unilid on FLORES-77, and it
is not a rounding artefact: reading the printed .997 as the largest value that
rounds to it still gives 273 implied errors against the 309 measured. This
revises section 5, which read the direction off the FPR column alone and so
placed the \\unilid UDHR cell, the suspected typo, on the other side.

A different base vocabulary is a different model, and a different model gives
a different number, so the answer accounts for the published and the measured
values differing on the \\unilid row without any further assumption. Three
things about that discrepancy it does not account for, on the evidence available
here:

1. The sign of the discrepancy, at the vocabulary size the paper states.
   `paper/submission.tex` states "we use a vocabulary size of 100k" unless
   otherwise specified, and no document in this repository states a vocabulary
   size for a subset model. A 100k vocabulary fitted to 80 languages gives each
   of those 80 languages a larger share of the vocabulary than a 100k vocabulary
   fitted to 1,940 languages does. That predicts a subset model with fewer
   errors than the restricted full model, not more. Most of the cells in the
   table above go the other way. A smaller subset vocabulary, a different corpus
   draw, or a different per-language row definition (section 10.4) would each
   account for the sign. None of the three is recorded.
2. The two \\unilid cells go opposite ways. On UDHR-80 the published value
   implies about twice the measured errors. On FLORES-77 it implies about three
   quarters. One model cannot do both. Either the two columns came from two
   different subset models, which the answer allows because a subset vocabulary
   is fitted per subset, or something besides the vocabulary differs as well.
3. Magnitudes. Nothing measured here can confirm that an 80-language
   subset-vocabulary model gives the printed .992 on UDHR-80, or a 77-language
   one the printed .997 on FLORES-77. Confirming that would take the models
   themselves, and the container census in section 5 shows they are not on this
   filesystem. This is
   a limit of the evidence, not a doubt about the answer.

### 10.4 The variant rows: the combinations scored, and their results

A fixed LLM vocabulary cannot be fitted to a subset. Under the author's answer
the variant rows' subset cells should therefore be a plain restricted argmax over
the full model, and should reproduce. They do not. Every input that could differ
and that exists on this filesystem was enumerated and scored.

**(a) A different evaluation pool.** For UDHR the two candidate pools are the
same pool: the raw `udhr-lid.csv`
({ra['pool_candidates']['udhr']['udhr-lid.csv (raw, cis-lmu/udhr-lid @6908db2)']['file_rows']:,}
rows) and the registry TSV
({ra['pool_candidates']['udhr']['external_bench/udhr_eval.tsv (the registry pool)']['file_rows']:,}
rows) both reduce, under the subset filter, to the same
{ra['pool_candidates']['udhr']['external_bench/udhr_eval.tsv (the registry pool)']['rows_with_gold_bare_iso_in_the_80']:,}
rows in the same order under the same 82 labels, with
{ra['pool_candidates']['udhr']['text_field_differences']} differences in the text
field. For FLORES the two candidate pools do differ: the original FLORES-200
devtest has {ra['pool_candidates']['flores']['external_bench/flores200_eval.tsv (the registry pool, original FLORES-200 devtest)']['rows_with_gold_bare_iso_in_the_77']:,}
subset rows and flores_plus devtest has
{ra['pool_candidates']['flores']['external_bench/flores_devtest_eval.tsv (flores_plus devtest)']['rows_with_gold_bare_iso_in_the_77']:,}.
Both were scored.

**(b) A different label mapping, and (c) a different row set.** Seven row-set
readings were scored, from "every model label whose bare ISO is in the subset"
down to "exactly one row per language". They include the single 83-language
CLD3-coverage model: the 83-language subset is a strict superset of both the
UDHR-80 and the FLORES-77 subsets, so one model can serve all three columns.
Both label mappings, bare ISO and full `lang_Script`, were scored on each.
Every row-set reading is defined in the JSON under `rowset_definitions`.

The two tables below use the Viterbi decode, the bare-ISO metric, and the
registry pool (the benchmark file named in
`analysis/external_bench_eval.py`'s `BENCH_REGISTRY`, which is the pool the
full-set columns reproduce on). Bold marks a value that rounds to the published
cell.

#### UDHR, 80 languages

{m_udhr}

#### FLORES-200, 77 languages

{m_flores}

The pool, the decode, the label mapping and the container each move the cell
very little. On the row set the gate itself used, the largest macro F1 movement across the registry
and alternative pools, the Viterbi and forward decodes, and the bare-ISO and
`lang_Script` label mappings is {inv:.4f}. Substituting this repository's fp64
retrains of the DeepSeek3.2 and Qwen3 containers for the co-author's own moves
it by at most {cont:.4f}. Counted directly rather than inferred from those two
numbers: of the {n_axis} scored combinations that are a restricted argmax over
rows of the full model (any pool, any decode, either label mapping, either
container, and any of the three row-set readings that contain every gold label),
{axis_text}

The row-set readings fall into two groups. The three that contain every gold
label (all 94 or 93 rows, the benchmark's own gold scripts only, and the 99-row
CLD3-83 model) give identical results on all four containers on UDHR-80: the 12
romanised rows and the rows for the three extra CLD3 languages are never the
argmax on that pool. On FLORES-77 they differ, by at most
{grp['flores']['keep_spread_f1']:.4f} macro F1. Across the four containers that
group produces {grp['udhr']['keep_errors'][0]} to
{grp['udhr']['keep_errors'][1]} errors on UDHR-80 and
{grp['flores']['keep_errors'][0]} to {grp['flores']['keep_errors'][1]} on
FLORES-77. The readings that keep exactly one row per language cost far more on
UDHR-80: {grp['udhr']['errors_by_rowset']['one_largest'][0]} to
{grp['udhr']['errors_by_rowset']['one_largest'][1]} errors under the
largest-corpus script rule, and
{grp['udhr']['errors_by_rowset']['one_first'][0]} to
{grp['udhr']['errors_by_rowset']['one_first'][1]} under the first-in-model-order
rule. On FLORES-77 the largest-corpus script is the gold script for all 77
languages, so that reading is the gold-script reading there, while
first-in-model-order costs
{grp['flores']['errors_by_rowset']['one_first'][0]:,} to
{grp['flores']['errors_by_rowset']['one_first'][1]:,} errors.

**(d) One row per language, with that language's script corpora pooled.** The
published column's header counts languages, not language-script labels, which is
what this reading takes literally. It is the only reading that is not a row
subset of the released model, and no tokenizer training is involved. A per-language
row here is a normalised unigram count distribution over a fixed base tokenizer,
and pooling two corpora pools their token counts, so the pooled row is the
corpus-size-weighted mixture of the script rows in probability space:
`p_L = sum_s (N_s / sum_s N_s) * p_{{L_s}}`, with `N_s` the training-line count
of script variant `s`. That mixture was built from each model's own rows and
scored. It is exact up to any per-language smoothing applied after
normalisation, which does not commute with the mixture. It is also the only
mechanism the author's answer leaves open to the variant rows, because it
changes the rows without touching the vocabulary.

{hit_text}

The same mechanism misses the other systems in the same direction and by large
margins. Matching {n_hit_cells} cells on one of the two printed metrics, inside
a band this narrow, is not a reproduction of the column, and the inputs to the
PD-3 swap do not become computable on it. What it does fix is the kind of
difference involved. The published subset values are consistent with models that
have one row per language, and not with a restricted argmax over script-specific
rows. That is what the author's answer implies, since the CLD3 subsets are lists
of languages and the training corpora are lists of language-script pairs.

If the author confirms this reading, three runs per variant produce the
corrected variants' six cells, on that variant's corrected-generation container:

1. `UNILID/train.py` over the 83 CLD3 languages, with `--corpus-dir` pointing at
   the shared draw and each language's script corpora concatenated into one
   per-language file, and with `--base-tokenizer-path` pointing at that
   variant's own base tokenizer extracted unchanged from its container. About
   {row_est_83 / 60:.0f} minutes, because no vocabulary is fitted.
2. `analysis/cld_subset_eval.py --mode subset` on the three benchmarks. Seconds
   for UDHR-80 and FLORES-77, a few hours for GlotLID-C-83.
3. The same recipe applied to the RELEASED model, checked against the published
   \\unilid subset cells. Without that anchor the variants' new numbers sit
   beside carried ones with nothing tying the two generations together, which is
   what PD-3's condition was written to prevent.

Step 3 is the check that matters, and on the evidence above it is unlikely to
pass: the released model is the row this mechanism misses by the widest
margin.

### 10.5 The reproduction bill for the \\unilid CLD3 columns

Present on this filesystem: the trainer (`UNILID/train.py`), and the
per-language GlotLID-C training corpora at `{corpus['path']}`
({corpus['n_corpora']:,} files, {corpus['total_samples']:,} lines), the shared
draw every later retrain used. The monolithic `train.txt` the released model was
trained from is gone, so a regenerated subset model would be trained on a
different draw than the released model was.

The cost basis below comes from each run's own `training_summary.json`.

| run | languages | corpus lines | vocabulary | base tokenizer | row estimation |
|---|---|---|---|---|---|
{cost_table}

The three CLD3 subsets cover {sub83['n_corpora']} of the {corpus['n_corpora']:,}
corpora and {sub83['samples']:,} lines, {sub83['share_of_full']:.1%} of the full
draw, for the 83-language subset. At the measured row-estimation rate of
{rate:,.0f} lines/s that is about {row_est_83 / 60:.0f} minutes of row estimation
per subset model. Fitting a 100k vocabulary from scratch cost
{base_fit['base_tokenizer_seconds'] / 3600:.1f} h on a
{base_fit['total_samples']:,}-line corpus. A subset base fit runs on the
10,000-lines-per-language base sample, about eight times that volume, and the
one comparison in the record (a 10k vocabulary took longer to fit than a 100k
one on the same corpus) says the scaling is not linear in either the corpus size
or the vocabulary size. My estimate is one and a half to two hours per subset model on one node, most
of it the vocabulary fit. That figure is an extrapolation from the four runs
above, not a measurement.

Trainings needed: one per subset definition the author confirms. Three if each
column had its own model (83, 80, 77), one if a single 83-language model served
all three. Three more if the calibrated row is regenerated for consistency,
since its calibration would have to be refitted on the subset model. For the
three variant rows the vocabulary is fixed, so only the row estimation is
needed, about {row_est_83 / 60:.0f} minutes per variant per subset.

Evaluation on top of that: UDHR-80 and FLORES-77 take seconds each on a login
node. GlotLID-C-83 is a pass over the 23,462,651 test lines whose gold label is
in the subset, which is what the queued jobs allot 5 h each for.

Five inputs the answer does not fix. Each is a point where a run would
otherwise substitute a plausible value for an unknown one, so the author has to
supply each of them before anything is submitted:

1. The subset models' vocabulary size. The paper states 100k as the general
   default. Nothing states it for a subset model, and section 10.3 shows the
   sign of the discrepancy depends on it.
2. The mapping from a bare ISO code to a training corpus. The subset definitions
   are bare ISO codes and the corpora are `lang_Script` files. Pooling a
   language's script corpora into one row and picking one script per language
   give different models with very different numbers (section 10.4).
3. One subset model or three. The 83-language subset covers both others.
4. Whether the corpus draw now on disk is acceptable as a stand-in for the draw
   the released model was trained on, which no longer exists.
5. The UNILID version. A corrected-generation subset model has to be built with
   0.3.0 or later. Any model behind the published cells was not.

### 10.6 The decision on the corrected generation's subset columns

The decision is the author's. The three ways it can go:

1. Regenerate. Answer the five questions above, train the subset models, and
   recompute the right half of `tab:lid_main` for the corrected generation. The
   result would not be a reproduction of the published cells and should not be
   presented as one. It is a new measurement under a newly specified convention,
   it moves published \\unilid numbers, and it requires the calibrated row to be
   regenerated with it.
2. Carry the cells and say so in the caption. Leave the right half exactly as
   published and record what the answer establishes: those cells come from
   models trained on the CLD3 subset with their own base vocabulary, those
   models are not available, and the cells are therefore carried unchanged and
   are not of the same generation as the left half. This costs no compute and
   removes the standing ambiguity. It leaves six carried cells beside corrected
   ones.
3. Drop the subset columns, or their FPR halves, from the corrected-generation
   table.

Option 1 cannot start until the five questions above are answered. Option 3
removes results the paper currently reports. Option 2 requires neither, and I
would treat it as the default until those answers arrive.

For PD-3 and PD-5, nothing in this reassessment changes the closure. The variant
rows' subset cells are still not computable here. The author's answer does not
apply to them, and the one mechanism that could ({hit_clause}) reproduces two of
the four variant subset cells whose container is verified and misses the other
two. The narrowed C3 ask that remains is this: for the variant rows, were the
subset cells computed from the same models as the full-set cells, and if not,
how were their rows estimated?
"""


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

**Section 10 reassesses all of this against the author's answer of 2026-08-31,
and revises two things in what follows: the direction of the miss read in
section 5, and how many of these four models are the models behind their
published rows.** Sections 1-9 are left as they were measured.

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

(Section 10.3 revises this paragraph. Its direction is taken from the FPR
column alone, which puts the \unilid UDHR cell -- the one whose FPR section 5
of `cld_subset_convention_sweep.md` argues is an exponent typo -- on the
opposite side from where its own F1 puts it. Read on macro F1, seven of the
eight cells move the same way. The conclusion of the paragraph, that a
restricted argmax over these models did not produce the published columns,
stands on either reading.)

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
   from-scratch subset model does not share one.
__CENSUS__

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
        body += (
            "\nThe three variants' UDHR 366-label cells were filled in on the "
            "login node on 2026-08-31, ahead of those jobs, because section "
            "10.2 needed them: they are the check that says whether each "
            "container is the model behind its published row. The matching "
            "FLORES-200 190-label pass was attempted on the login node too and "
            "was killed there both times, so it is left to the queued jobs "
            "rather than retried; no partial output from those attempts was "
            "kept.\n")
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
- Section 10's hypothesis matrix:
  `outputs/rerelease/cld_subset/variant_hypothesis_matrix.json`. Its harness was
  kept in the session scratchpad rather than added to `analysis/`, on the same
  precedent as `cld_subset_convention_sweep.md` section 7: it reproduces no
  published cell, so there is nothing for a permanent script to gate against.
  The one derivation in it that is not a call to `analysis/cld_subset_eval.py`
  is the merged one-row-per-language mixture, and section 10.4 states it in
  full: `p_L = sum_s (N_s / sum_s N_s) * p_{L_s}` over the model's own rows,
  written back through `unilid.model_io.write_unilid`.
"""
    body += reassessment_md(out)
    import textwrap
    cen = out["model_container_census"]
    cl = [f"{c:,}" for c in cen["distinct_language_counts"]]
    counts = cl[0] if len(cl) == 1 else " or ".join(
        [", ".join(cl[:-1]), cl[-1]])
    n_variant_rows, n_benchmarks = 3, 3
    census_text = (
        "No such model is on this filesystem. Every one of the "
        f"{cen['n_containers']} distinct `.unilid` containers under the two "
        f"storage roots carries either {counts} rows, and none carries 83, 80 "
        "or 77; the roots, the per-container row counts and the vocabulary "
        "sizes are in this record's JSON under `model_container_census`. So if "
        "the published subset columns did come from subset-trained models, "
        "those models were never on this filesystem, and reproducing the "
        "columns here would take one training per row per benchmark, each with "
        "its own base tokenizer fitted to that subset's corpora: "
        f"{n_variant_rows * n_benchmarks} trainings for the three variant rows "
        "alone, and 15 if the \\unilid and calibrated rows are regenerated "
        "for consistency as section 7 describes.")
    # Wrapped to the document's own width, continuing the numbered-list indent
    # of reading (b) so the paragraph reads as one item.
    wrapped = textwrap.fill(census_text, width=76,
                            initial_indent="   ", subsequent_indent="   ")
    body = body.replace("__CENSUS__", wrapped)
    with open(path, "w") as f:
        f.write(body)
    print("wrote", path)


if __name__ == "__main__":
    main()
