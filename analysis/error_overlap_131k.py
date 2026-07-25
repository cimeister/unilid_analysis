"""Error overlap analysis: 131k baseline vs 100k baseline (Exp 30, plan follow-up).

Question (user, 2026-07-24): does the 131k multilingual-base model repeat exactly the
same errors as the 100k baseline, or does it trade different errors? Both prediction
sets cover the same 45,377,279 kept lines (Exp 16 / Exp 29 memmaps), so the overlap
is computed line-exactly.

Reported:
1. Error-set overlap: shared errors, errors the 131k fixes, errors it introduces;
   among shared errors, how often the two models agree on the same wrong label.
2. Where the fixed and introduced errors live (stratum of the true language; script
   groups relevant to the tokenizer's documented strengths: Indic scripts, Han/CJK).
3. Per-language F1 deltas: the largest improvements and regressions under 131k.
4. Tail FP structure under 131k: top receiving tail languages and their sources,
   against the Exp 24 baseline list.

Gates: recomputed accuracies must match the recorded values (0.9608 / 0.9612 at 4dp);
line counts must match the recorded totals.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from analysis.config import TOTAL_LINES
from analysis.full_test_eval import SCRATCH_DIR as SCRATCH_100K
from analysis.full_test_eval_131k import SCRATCH_DIR as SCRATCH_131K

PRF_100K = "outputs/diagnostic/full_test_per_lang_prf.csv"
PRF_131K = "outputs/diagnostic/full_test_131k_per_lang_prf.csv"
DIAG_CSV = "outputs/diagnostic/lang_diagnostic.csv"
OUT_MD = "outputs/tables/error_overlap_131k.md"
# Indic scripts + Han/CJK: the groups where the 131k tokenizer's compression is
# documented as strongest (apertus-tokenizer-development README)
INDIC_SCRIPTS = {"Deva", "Beng", "Taml", "Telu", "Knda", "Mlym", "Gujr", "Guru",
                 "Orya", "Sinh"}
CJK_SCRIPTS = {"Hani", "Hans", "Hant", "Jpan", "Hira", "Kana", "Hang"}
TOP_N = 12


def run(out_md: str = OUT_MD) -> str:
    prf = pd.read_csv(PRF_100K)
    prf131 = pd.read_csv(PRF_131K)
    if list(prf.lang) != list(prf131.lang):
        raise RuntimeError("language order differs between the two PRF CSVs")
    langs = prf.lang.tolist()
    n_lang = len(langs)
    N = prf.N.values
    diag = pd.read_csv(DIAG_CSV)
    if list(diag.lang) != langs:
        raise RuntimeError("lang_diagnostic order differs from the PRF CSVs")
    script = diag.script.values

    y = np.asarray(np.lib.format.open_memmap(
        os.path.join(SCRATCH_100K, "y_true.npy"), mode="r"))
    p100 = np.asarray(np.lib.format.open_memmap(
        os.path.join(SCRATCH_100K, "pred_baseline.npy"), mode="r"))
    p131 = np.asarray(np.lib.format.open_memmap(
        os.path.join(SCRATCH_131K, "pred_baseline131k.npy"), mode="r"))
    kept = y >= 0
    yk = y[kept]
    a, b = p100[kept], p131[kept]
    n = len(yk)
    if n != 45_377_279:
        raise RuntimeError(f"kept lines {n:,} != recorded 45,377,279")
    acc100, acc131 = float((a == yk).mean()), float((b == yk).mean())
    if round(acc100, 4) != 0.9608 or round(acc131, 4) != 0.9612:
        raise RuntimeError(f"accuracy gate failed: {acc100:.4f}/{acc131:.4f} vs "
                           "recorded 0.9608/0.9612")

    e100 = a != yk
    e131 = b != yk
    shared = e100 & e131
    fixed = e100 & ~e131
    intro = ~e100 & e131
    same_wrong = shared & (a == b)

    strata = {"tail": N < 1_000, "lowmid": (N >= 1_000) & (N < 18_000),
              "head": N >= 18_000}

    # per-line stratum of the TRUE language
    line_stratum = np.full(n, "lowmid", dtype=object)
    line_stratum[np.isin(yk, np.where(strata["tail"])[0])] = "tail"
    line_stratum[np.isin(yk, np.where(strata["head"])[0])] = "head"
    line_script = script[yk]
    is_indic = np.isin(line_script, list(INDIC_SCRIPTS))
    is_cjk = np.isin(line_script, list(CJK_SCRIPTS))

    L = ["# Error overlap: 131k baseline vs 100k baseline (Exp 30)\n",
         f"{n:,} kept lines; accuracies 100k {acc100:.4f}, 131k {acc131:.4f} "
         "(gates passed).\n",
         "## Error-set overlap\n",
         f"- 100k errors: {int(e100.sum()):,}; 131k errors: {int(e131.sum()):,}.",
         f"- Shared (both wrong): {int(shared.sum()):,} = "
         f"{shared.sum() / e100.sum():.1%} of 100k errors; of the shared errors, "
         f"{int(same_wrong.sum()):,} ({same_wrong.sum() / shared.sum():.1%}) pick "
         "the SAME wrong label.",
         f"- Fixed by 131k (100k wrong, 131k right): {int(fixed.sum()):,} "
         f"({fixed.sum() / e100.sum():.1%} of 100k errors).",
         f"- Introduced by 131k (100k right, 131k wrong): {int(intro.sum()):,}.",
         f"- Net: {int(intro.sum() - fixed.sum()):+,} errors.\n",
         "## Where the fixed and introduced errors live (true language)\n",
         "| set | tail | lowmid | head | Indic scripts | Han/CJK |",
         "|---|---|---|---|---|---|"]
    for name, mask in [("fixed", fixed), ("introduced", intro),
                       ("shared", shared)]:
        row = [f"{int((mask & (line_stratum == s)).sum()):,}"
               for s in ["tail", "lowmid", "head"]]
        row += [f"{int((mask & is_indic).sum()):,}", f"{int((mask & is_cjk).sum()):,}"]
        L.append(f"| {name} | " + " | ".join(row) + " |")
    tot_indic = int(is_indic.sum())
    tot_cjk = int(is_cjk.sum())
    fixed_rate_indic = (fixed & is_indic).sum() / max((e100 & is_indic).sum(), 1)
    intro_rate_indic = (intro & is_indic).sum() / max((~e100 & is_indic).sum(), 1)
    L.append(f"\nIndic-script lines total {tot_indic:,}; 131k fixes "
             f"{fixed_rate_indic:.1%} of the 100k errors there and introduces new "
             f"errors on {intro_rate_indic:.2%} of previously-correct lines. "
             f"Han/CJK lines total {tot_cjk:,}.")

    d = prf131.f1_131k.values - prf.f1_baseline.values
    order_up = np.argsort(-d)[:TOP_N]
    order_dn = np.argsort(d)[:TOP_N]
    L += ["\n## Largest per-language global-F1 changes under 131k\n",
          "Improvements: " + ", ".join(
              f"{langs[i]} +{d[i]:.3f}" for i in order_up if d[i] > 0.01),
          "Regressions: " + ", ".join(
              f"{langs[i]} {d[i]:.3f}" for i in order_dn if d[i] < -0.01),
          f"Languages improved by more than 0.01: {int((d > 0.01).sum())}; "
          f"regressed by more than 0.01: {int((d < -0.01).sum())}."]

    tail_idx = np.where(strata["tail"])[0]
    fpm = np.isin(b, tail_idx) & (b != yk)
    pairs = pd.DataFrame({"pred": b[fpm], "true": yk[fpm]}).value_counts()
    L += [f"\n## FPs into tail labels under 131k ({int(fpm.sum()):,} vs 22,522 "
          "under 100k)\n", "Top (pred <- true) pairs:"]
    for (pi, ti), c in pairs.head(TOP_N).items():
        L.append(f"- {langs[pi]} <- {langs[ti]}: {int(c):,}")

    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    return out_md


if __name__ == "__main__":
    run()
