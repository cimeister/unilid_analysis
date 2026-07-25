"""Non-content token tying across languages (parameter tying, added 2026-07-18).

Motivation (Exp 10): 86.4% of the score margin toward wrong languages comes from short
non-content tokens (<=3-char subwords 51%, punctuation 20%), driven by resource-tied
differences in the tiny probabilities languages assign to tokens they rarely see. Tokens
that carry no language identity should not contribute to score DIFFERENCES at all; tying
their probabilities across languages enforces that invariance exactly, while segmentation
and all content-token evidence stay untouched.

Tied sets (a token is classified on its byte-decoded text; tokens that do not decode as
UTF-8 are never tied):
  digits_ws      : text is only digits/whitespace
  nonalpha_ascii : text is ASCII with no alphabetic character (digits, ASCII punctuation,
                   whitespace); script-specific punctuation stays untied
  nonalpha_all   : no alphabetic character in any script (adds unicode punctuation,
                   emoji, symbols); may tie language-informative marks

Construction: PURE tying, no renormalization. Each tied column is set to the log of the
resource-weighted mean (w = min(N, RES_CAP), over all languages) of the per-language
probabilities; every other entry, including the special tokens, stays bit-identical. The
rows are then no longer normalized distributions, which the scorer never required (it
sums raw log-weights along the Viterbi path). Renormalizing instead would rescale every
untied token of language L by -log Z_L, a per-language per-token offset of up to 0.36
nats/token concentrated on the flat confusers (derived and measured in the 2026-07-18
pre-run review), conflating the tying with a second mechanism. With pure tying, tied
tokens contribute an identical amount to every language's score for the same tokenized
span, so they cancel from all score differences.

Special-token note (peak-probability finding, 2026-07-18): every language row gives
exactly p=0.2 to each of <s> </s> <pad> <unk>; these are asserted and never tied.
Selection on the val half via the all-strata guard; test half scored once. The Rust
weight cache is ~776 MB; peak memory is a few GB -> SLURM for the CPU count.
"""
from __future__ import annotations

import json
import os
import struct

import numpy as np

from analysis.config import DEFAULT_SAMPLE_SIZE
from analysis.transfer_sweep import (
    UNILID_MODEL_PATH, _load_model_data, _load_train_counts, _load_unilid_model,
    _stream_sampled_texts, predict_all,
)
from analysis.sample_data import load_sample
from analysis.hierarchical_pool import (
    RES_CAP, _strata, _macro_f1, _bootstrap_delta, passes_guard,
    GUARD_STRATA, GUARD_TOL, VAL_MASK, DIAG_CSV,
)

SPECIAL_TOKENS = ("<s>", "</s>", "<pad>", "<unk>")
SPECIAL_P = 0.2                      # asserted per-row probability of each special token
SWEEP = ["digits_ws", "nonalpha_ascii", "nonalpha_all"]
OUT_DIR = "outputs"

# Refined tied set (2026-07-19, user direction after the Exp 18 negative): digits and
# linguistically NEUTRAL punctuation only. Never whitespace or newlines (their
# frequencies encode spacing conventions, e.g. spaced vs unspaced scripts; including
# them in Exp 18 was a design error). Explicit exclusions, with reasons:
#   apostrophes  ' ’ ‘      contraction/elision/glottal marks (lexical)
#   hyphens      - ‐ – —   compounding orthography (lexical)
#   ampersand    &                    abbreviates a word
#   currency     $ and all non-ASCII  country/language correlated
#   inverted     ¿ ¡        Spanish-specific
#   typographic quotes «»„“”‹›   quote conventions
#                                     are language-specific within a script
#   all non-ASCII punctuation         script-specific (danda, Arabic marks, CJK marks)
#                                     by construction of the ASCII-only alphabet
# Tying is applied within SCRIPT groups (dp_script): each script's languages share that
# script's resource-weighted mean, so one writing system's usage conventions never leak
# into another (this also keeps the Greek use of ';' as a question mark coherent inside
# Grek). dp_global (one mean over all languages) is run alongside for comparison.
TIED_DIGITS = set("0123456789")
NEUTRAL_PUNCT = set('.,:;!?()[]{}/\\|@#*+=<>~`_"%^')
DP_ALPHABET = TIED_DIGITS | NEUTRAL_PUNCT
DP_SWEEP = ["dp_global", "dp_script"]


def _load_vocab(model_path: str = UNILID_MODEL_PATH) -> list[str]:
    """Token strings from the base tokenizer JSON embedded in the .unilid file."""
    header_fmt = "<8sIIIII4x"
    with open(model_path, "rb") as f:
        header = f.read(struct.calcsize(header_fmt))
        _, _, _n, vocab_size, base_tok_len, _l = struct.unpack(header_fmt, header)
        base = json.loads(f.read(base_tok_len).decode("utf-8"))
    vocab = [t for t, _s in base["model"]["vocab"]]
    if len(vocab) != vocab_size:
        raise RuntimeError(f"vocab length {len(vocab)} != header vocab_size {vocab_size}")
    return vocab


def _gpt2_byte_decoder() -> dict:
    """Inverse of the GPT-2 bytes_to_unicode map used by the byte-level vocab."""
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1))
          + list(range(ord("\xae"), ord("\xff") + 1)))
    cs = bs[:]
    m = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + m)
            m += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def _token_classes(token: str, decoder: dict) -> set[str]:
    """Which tied sets the token belongs to, based on its decoded text."""
    try:
        text = bytes(decoder[ch] for ch in token).decode("utf-8")
    except (KeyError, UnicodeDecodeError):
        return set()                 # not byte-decodable -> never tied
    if not text or any(ch.isalpha() for ch in text):
        return set()
    classes = {"nonalpha_all"}
    if all(ord(ch) < 128 for ch in text):
        classes.add("nonalpha_ascii")
    if all(ch.isdigit() or ch.isspace() for ch in text):
        classes.add("digits_ws")
    if all(ch in DP_ALPHABET for ch in text):
        classes.add("dp")            # digits + neutral punctuation, no whitespace
    return classes


def build_tied_weights(W: np.ndarray, N: np.ndarray, tie_ids: np.ndarray) -> np.ndarray:
    """Set the tied columns to the shared log resource-weighted mean probability; leave
    every other column bit-identical. No renormalization (see module docstring)."""
    w = np.minimum(N, float(RES_CAP))
    w = w / w.sum()
    tied_cols = np.exp(W[:, tie_ids].astype(np.float64))
    tied_mean = (w[:, None] * tied_cols).sum(axis=0)
    out = np.array(W, dtype=np.float32)
    out[:, tie_ids] = np.log(tied_mean)[None, :].astype(np.float32)
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite weights after tying; zero probabilities introduced")
    return out


def build_tied_weights_by_script(W: np.ndarray, N: np.ndarray, scripts: np.ndarray,
                                 tie_ids: np.ndarray) -> np.ndarray:
    """Tie the columns within each SCRIPT group: languages of script s share that
    script's resource-weighted mean over the tied columns. Single-language scripts are
    unchanged by construction (the mean equals the language's own values). Scripts with
    zero total resource weight are left unmodified and counted. No renormalization."""
    out = np.array(W, dtype=np.float32)
    tie_ids = np.asarray(tie_ids)
    tied_cols = np.exp(W[:, tie_ids].astype(np.float64))
    n_skipped = 0
    for s in np.unique(scripts):
        m = scripts == s
        w = np.minimum(N[m], float(RES_CAP))
        tot = w.sum()
        if tot <= 0:
            n_skipped += int(m.sum())
            continue
        mean_s = ((w / tot)[:, None] * tied_cols[m]).sum(axis=0)
        out[np.ix_(np.where(m)[0], tie_ids)] = \
            np.log(mean_s)[None, :].astype(np.float32)
    if n_skipped:
        print(f"    {n_skipped} languages in zero-weight scripts left unmodified")
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite weights after script tying")
    return out


def run(sample_size: int = DEFAULT_SAMPLE_SIZE, out_dir: str = OUT_DIR,
        sweep: tuple = tuple(SWEEP), tag: str = ""):
    import pandas as pd
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)
    weights, langs, lang_to_idx = _load_model_data()
    train_counts = _load_train_counts()
    N = np.array([train_counts.get(l, 0) for l in langs], dtype=np.float64)
    scripts = np.array([l.rsplit("_", 1)[-1] if "_" in l else "Unknown" for l in langs])
    vocab = _load_vocab()
    W = np.array(weights, dtype=np.float32)
    n_lang, V = W.shape
    diag = pd.read_csv(DIAG_CSV)

    special_ids = np.array([vocab.index(t) for t in SPECIAL_TOKENS])
    sp = np.exp(W[:, special_ids].astype(np.float64))
    if not np.allclose(sp, SPECIAL_P, atol=1e-4):
        raise RuntimeError(f"special-token probabilities are not uniformly {SPECIAL_P}: "
                           f"range [{sp.min():.6f}, {sp.max():.6f}]")
    real_ids = np.setdiff1d(np.arange(V), special_ids)

    decoder = _gpt2_byte_decoder()
    classes = [_token_classes(vocab[i], decoder) for i in real_ids]
    def _class_key(cfg):
        return "dp" if cfg in DP_SWEEP else cfg
    tie_pos_by_cfg = {cfg: np.array([k for k, cl in enumerate(classes)
                                     if _class_key(cfg) in cl])
                      for cfg in sweep}
    for cfg in sweep:
        pos = tie_pos_by_cfg[cfg]
        if len(pos) == 0:
            raise RuntimeError(f"tied set {cfg} is empty; classification is broken")
        sample_toks = [vocab[real_ids[k]] for k in pos[:12]]
        print(f"{cfg}: {len(pos)} tokens tied, e.g. {sample_toks}")

    texts = _stream_sampled_texts(sample_size)
    data = load_sample(sample_size)
    y_true = np.array(data["y_true"])
    if len(texts) != len(y_true):
        raise RuntimeError(f"text/y_true mismatch {len(texts)} vs {len(y_true)}")
    val = np.load(VAL_MASK)
    test = ~val
    strata = _strata(y_true, langs, lang_to_idx, N, diag)

    model = _load_unilid_model()
    print("Baseline prediction...")
    base_pred = np.array(predict_all(texts, model))
    agree = (base_pred == np.array(data["pred_UniLID"])).mean()
    print(f"  baseline agreement with recorded UniLID preds: {agree:.4f}")
    if agree < 0.99:
        raise RuntimeError(f"baseline agreement {agree:.4f} < 0.99; model/scorer changed")

    def strat_row(pred, mask_split):
        return {st: _macro_f1(y_true[mask_split], pred[mask_split], strata[st][mask_split])
                for st in strata}

    val_base = strat_row(base_pred, val)
    rows = [{"config": "baseline", "n_tied": 0,
             **{f"val_{k}": v for k, v in val_base.items()}}]
    preds_by_cfg = {"baseline": base_pred}
    for cfg in sweep:
        print(f"Config {cfg}...")
        tie_ids = real_ids[tie_pos_by_cfg[cfg]]
        if cfg == "dp_script":
            w_new = build_tied_weights_by_script(W, N, scripts, tie_ids)
        else:
            w_new = build_tied_weights(W, N, tie_ids)
        model.model.set_weight_sets(w_new.tolist())
        del w_new
        pred = np.array(predict_all(texts, model))
        preds_by_cfg[cfg] = pred
        vr = strat_row(pred, val)
        rows.append({"config": cfg, "n_tied": len(tie_pos_by_cfg[cfg]),
                     **{f"val_{k}": v for k, v in vr.items()}})
        print("    val macro-F1:", {k: round(v, 4) for k, v in vr.items()})

    eligible = [r for r in rows if r["config"] != "baseline"
                and passes_guard(r, rows[0], prefix="val_")]
    best = max(eligible, key=lambda r: r["val_overall"])["config"] if eligible else "baseline"
    print(f"\nBest config (val, guarded): {best}")

    best_pred = preds_by_cfg[best]
    lines = ["# Non-content token tying — TEST evaluation\n",
             f"Best config selected on val: **{best}** (baseline means nothing passed the "
             f"guard). Baseline agreement {agree:.4f}.",
             f"Selection guard: val overall must improve and no stratum "
             f"({'/'.join(GUARD_STRATA)}) may drop more than {GUARD_TOL} vs baseline.\n",
             "| stratum | base macroF1 | tied macroF1 | delta | 95% CI |",
             "|---|---|---|---|---|"]
    for st in ["overall", "magnets", "tail", "twins", "head"]:
        b = _macro_f1(y_true[test], base_pred[test], strata[st][test])
        nw = _macro_f1(y_true[test], best_pred[test], strata[st][test])
        d, lo, hi = _bootstrap_delta(y_true[test], base_pred[test], best_pred[test],
                                     strata[st][test])
        lines.append(f"| {st} | {b:.4f} | {nw:.4f} | {d:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    acc_b = (base_pred[test] == y_true[test]).mean()
    acc_n = (best_pred[test] == y_true[test]).mean()
    lines.append(f"\nOverall accuracy: {acc_b:.4f} -> {acc_n:.4f} ({acc_n-acc_b:+.4f}).")

    md = pd.DataFrame(rows).round(4).to_markdown(index=False) + "\n\n" + "\n".join(lines)
    out_path = os.path.join(out_dir, "tables", f"token_tying{tag}.md")
    with open(out_path, "w") as f:
        f.write(md)
    print("\n" + "\n".join(lines))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    run()
