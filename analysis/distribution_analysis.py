"""Analysis of per-language distributions vs base and vs related languages.

Investigates how per-language EM re-estimation differs from the base distribution,
whether divergence correlates with training data size, and whether noise in the
EM process is visible in closely related language pairs.
"""

from __future__ import annotations

import json
import os
import struct

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.config import CONFUSION_CLUSTERS, SCRATCH_DIR
from analysis.format_utils import to_markdown, to_latex


UNILID_MODEL_PATH = f"{SCRATCH_DIR}/glotlidc.unilid"

# Related language pairs for pairwise analysis.
# Grouped by expected similarity — within-family pairs where EM noise
# may dominate genuine distributional differences.
RELATED_PAIRS = [
    # (lang_a, lang_b, description)
    ("ind_Latn", "zsm_Latn", "Indonesian / Malay"),
    ("hin_Deva", "anp_Deva", "Hindi / Angika"),
    ("dan_Latn", "nob_Latn", "Danish / Bokmal"),
    ("eng_Latn", "sco_Latn", "English / Scots"),
    ("arb_Arab", "arz_Arab", "MSA / Egyptian Arabic"),
    ("arb_Arab", "ary_Arab", "MSA / Moroccan Arabic"),
    ("fas_Arab", "glk_Arab", "Persian / Gilaki"),
    ("cmn_Hani", "wuu_Hani", "Mandarin / Wu"),
    ("cmn_Hani", "yue_Hani", "Mandarin / Cantonese"),
    ("kin_Latn", "run_Latn", "Kinyarwanda / Kirundi"),
    ("nob_Latn", "nno_Latn", "Bokmal / Nynorsk"),
    ("heb_Hebr", "hbo_Hebr", "Hebrew / Biblical Hebrew"),
    ("rus_Cyrl", "ukr_Cyrl", "Russian / Ukrainian"),
    ("spa_Latn", "por_Latn", "Spanish / Portuguese"),
    ("slk_Latn", "ces_Latn", "Slovak / Czech"),
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_model_data(model_path: str = UNILID_MODEL_PATH):
    """Load base tokenizer scores, per-language weights, language list, and token strings."""
    HEADER_FMT = "<8sIIIII4x"
    HEADER_SIZE = struct.calcsize(HEADER_FMT)

    with open(model_path, "rb") as f:
        header = f.read(HEADER_SIZE)
        _, _, num_langs, vocab_size, base_tok_len, langs_len = struct.unpack(HEADER_FMT, header)
        base_tok_json = f.read(base_tok_len)
        langs = json.loads(f.read(langs_len).decode("utf-8"))
        weights_offset = f.tell()

    from tokenizers import Tokenizer
    base_tok = Tokenizer.from_str(base_tok_json.decode("utf-8"))
    state = json.loads(base_tok.model.__getstate__().decode("utf-8"))
    tokens = [t for t, _ in state["vocab"]]
    base_scores = np.array([s for _, s in state["vocab"]], dtype=np.float64)

    weights = np.memmap(
        model_path, dtype=np.float32, mode="r",
        offset=weights_offset, shape=(num_langs, vocab_size),
    )
    lang_to_idx = {l: i for i, l in enumerate(langs)}

    return base_scores, weights, langs, lang_to_idx, tokens


def _load_train_counts():
    from analysis.config import DATA_DIR
    with open(os.path.join(DATA_DIR, "glotlid_train_counts.json")) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _to_prob(log_scores: np.ndarray) -> np.ndarray:
    """Convert log-scores to a proper probability distribution via softmax-style normalization."""
    # Shift for numerical stability, exponentiate, normalize
    shifted = log_scores - log_scores.max()
    probs = np.exp(shifted)
    probs /= probs.sum()
    return probs


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(P || Q) with smoothing to avoid log(0)."""
    eps = 1e-30
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    return float(np.sum(p * np.log(p / q)))


def _symmetric_kl(p: np.ndarray, q: np.ndarray) -> float:
    """Symmetric KL: (KL(P||Q) + KL(Q||P)) / 2."""
    return (_kl_divergence(p, q) + _kl_divergence(q, p)) / 2


def _top_divergent_tokens(
    scores_a: np.ndarray, scores_b: np.ndarray,
    tokens: list[str], n: int = 20,
) -> list[dict]:
    """Find the n tokens with largest KL contribution from A relative to B.

    KL contribution of token i = p_a(i) * log(p_a(i) / p_b(i)).
    This highlights tokens that are both probable in A and much more probable
    in A than in B — the tokens driving the distributional divergence.
    """
    probs_a = _to_prob(scores_a)
    probs_b = _to_prob(scores_b)
    eps = 1e-30
    kl_contrib = probs_a * np.log(np.clip(probs_a, eps, None) / np.clip(probs_b, eps, None))
    top_idx = np.argsort(np.abs(kl_contrib))[-n:][::-1]
    return [
        {"token": tokens[i], "score_a": float(scores_a[i]),
         "score_b": float(scores_b[i]), "delta": float(scores_a[i] - scores_b[i]),
         "kl_contrib": float(kl_contrib[i])}
        for i in top_idx
    ]


# ---------------------------------------------------------------------------
# Analysis: vs base
# ---------------------------------------------------------------------------

def analyze_vs_base(
    base_scores, weights, langs, lang_to_idx, tokens, train_counts,
):
    """For each language, compute KL from base and correlation with base."""
    base_probs = _to_prob(base_scores)
    results = []

    for lang in langs:
        idx = lang_to_idx[lang]
        lang_scores = np.array(weights[idx], dtype=np.float64)
        lang_probs = _to_prob(lang_scores)

        kl = _kl_divergence(lang_probs, base_probs)
        corr = float(np.corrcoef(lang_scores, base_scores)[0, 1])
        mad = float(np.abs(lang_scores - base_scores).mean())
        tc = train_counts.get(lang, 0)
        script = lang.rsplit("_", 1)[-1]

        results.append({
            "lang": lang, "script": script, "train_count": tc,
            "kl_from_base": kl, "corr_with_base": corr, "mad_from_base": mad,
        })

    return results


# Languages to show top divergent tokens vs base
VS_BASE_TOKEN_LANGS = [
    "eng_Latn", "fra_Latn", "cmn_Hani", "arb_Arab", "hin_Deva",
    "rus_Cyrl", "spa_Latn", "ind_Latn",
    # High divergence from base
    "lis_Lisu", "ydd_Hebr", "tam_Taml",
    # Low divergence (low-resource)
    "otw_Latn", "ldn_Latn",
]


def analyze_vs_base_tokens(
    base_scores, weights, lang_to_idx, tokens,
    langs_to_show=VS_BASE_TOKEN_LANGS, n: int = 20,
):
    """For selected languages, find top tokens diverging from base by KL contribution."""
    results = {}
    for lang in langs_to_show:
        if lang not in lang_to_idx:
            continue
        idx = lang_to_idx[lang]
        lang_scores = np.array(weights[idx], dtype=np.float64)
        top = _top_divergent_tokens(lang_scores, base_scores, tokens, n=n)
        results[lang] = top
    return results


# ---------------------------------------------------------------------------
# Analysis: pairwise
# ---------------------------------------------------------------------------

def analyze_pairwise(
    weights, lang_to_idx, tokens, train_counts, pairs=RELATED_PAIRS,
):
    """For each related pair, compute symmetric KL and top divergent tokens."""
    results = []

    for lang_a, lang_b, desc in pairs:
        if lang_a not in lang_to_idx or lang_b not in lang_to_idx:
            continue
        scores_a = np.array(weights[lang_to_idx[lang_a]], dtype=np.float64)
        scores_b = np.array(weights[lang_to_idx[lang_b]], dtype=np.float64)

        probs_a = _to_prob(scores_a)
        probs_b = _to_prob(scores_b)

        sym_kl = _symmetric_kl(probs_a, probs_b)
        corr = float(np.corrcoef(scores_a, scores_b)[0, 1])
        mad = float(np.abs(scores_a - scores_b).mean())
        top_tokens = _top_divergent_tokens(scores_a, scores_b, tokens, n=20)
        tc_a = train_counts.get(lang_a, 0)
        tc_b = train_counts.get(lang_b, 0)

        results.append({
            "lang_a": lang_a, "lang_b": lang_b, "desc": desc,
            "sym_kl": sym_kl, "corr": corr, "mad": mad,
            "train_count_a": tc_a, "train_count_b": tc_b,
            "min_train_count": min(tc_a, tc_b),
            "top_tokens": top_tokens,
        })

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def generate_distribution_analysis(output_dir: str = "outputs"):
    """Run the full distribution analysis and write outputs."""
    print("  Loading model data...")
    base_scores, weights, langs, lang_to_idx, tokens = _load_model_data()
    train_counts = _load_train_counts()

    tables_dir = os.path.join(output_dir, "tables")
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # =================================================================
    # Part 1: All languages vs base
    # =================================================================
    print("  Analyzing all languages vs base...")
    vs_base = analyze_vs_base(base_scores, weights, langs, lang_to_idx, tokens, train_counts)

    # Table: summary by resource level
    vs_base_sorted = sorted(vs_base, key=lambda x: x["kl_from_base"], reverse=True)

    headers_vb = ["Language", "Script", "Train N", "KL from base", "Corr w/ base", "MAD"]
    rows_top = []
    for r in vs_base_sorted[:20]:
        rows_top.append([
            r["lang"], r["script"], f"{r['train_count']:,}",
            f"{r['kl_from_base']:.2f}", f"{r['corr_with_base']:.4f}", f"{r['mad_from_base']:.2f}",
        ])
    md_top_kl = to_markdown(rows_top, headers_vb, caption="Top 20 languages most divergent from base (by KL)")

    rows_bottom = []
    for r in vs_base_sorted[-20:]:
        rows_bottom.append([
            r["lang"], r["script"], f"{r['train_count']:,}",
            f"{r['kl_from_base']:.2f}", f"{r['corr_with_base']:.4f}", f"{r['mad_from_base']:.2f}",
        ])
    md_bottom_kl = to_markdown(rows_bottom, headers_vb, caption="Top 20 languages closest to base (by KL)")

    # Figure: KL vs training count
    kls = np.array([r["kl_from_base"] for r in vs_base])
    tcs = np.array([r["train_count"] for r in vs_base])
    corrs = np.array([r["corr_with_base"] for r in vs_base])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.scatter(tcs, kls, alpha=0.3, s=8)
    ax.set_xscale("log")
    ax.set_xlabel("Training samples (log scale)")
    ax.set_ylabel("KL divergence from base")
    ax.set_title("KL(lang || base) vs training data size")
    # Add correlation
    log_tcs = np.log10(np.clip(tcs, 1, None))
    r_val = np.corrcoef(log_tcs, kls)[0, 1]
    ax.text(0.05, 0.95, f"r(log N, KL) = {r_val:.3f}", transform=ax.transAxes, va="top")

    ax = axes[1]
    ax.scatter(tcs, corrs, alpha=0.3, s=8)
    ax.set_xscale("log")
    ax.set_xlabel("Training samples (log scale)")
    ax.set_ylabel("Pearson correlation with base")
    ax.set_title("Correlation with base vs training data size")
    r_val2 = np.corrcoef(log_tcs, corrs)[0, 1]
    ax.text(0.05, 0.95, f"r(log N, corr) = {r_val2:.3f}", transform=ax.transAxes, va="top")

    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "kl_vs_training_size.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Summary stats by resource bin
    from analysis.config import RESOURCE_BINS, RESOURCE_LABELS
    headers_bin = ["Resource bin", "# Langs", "Mean KL", "Median KL", "Mean corr", "Mean MAD"]
    rows_bin = []
    for i in range(len(RESOURCE_LABELS)):
        lo, hi = RESOURCE_BINS[i], RESOURCE_BINS[i + 1]
        mask = [(lo <= r["train_count"] < hi) for r in vs_base]
        subset = [r for r, m in zip(vs_base, mask) if m]
        if not subset:
            continue
        kls_bin = [r["kl_from_base"] for r in subset]
        corrs_bin = [r["corr_with_base"] for r in subset]
        mads_bin = [r["mad_from_base"] for r in subset]
        rows_bin.append([
            RESOURCE_LABELS[i], f"{len(subset)}", f"{np.mean(kls_bin):.2f}",
            f"{np.median(kls_bin):.2f}", f"{np.mean(corrs_bin):.4f}", f"{np.mean(mads_bin):.2f}",
        ])
    md_resource_kl = to_markdown(rows_bin, headers_bin, caption="KL from base by resource level")

    # Top divergent tokens vs base for selected languages
    print("  Computing top divergent tokens vs base...")
    vs_base_tokens = analyze_vs_base_tokens(base_scores, weights, lang_to_idx, tokens)
    md_base_tokens_parts = []
    for lang, top_toks in vs_base_tokens.items():
        tc = train_counts.get(lang, 0)
        headers_bt = ["Token", f"Score ({lang})", "Score (base)", "Delta", "KL contrib"]
        rows_bt = []
        for t in top_toks:
            rows_bt.append([
                repr(t["token"]),
                f"{t['score_a']:.2f}", f"{t['score_b']:.2f}",
                f"{t['delta']:+.2f}", f"{t['kl_contrib']:.4f}",
            ])
        md_base_tokens_parts.append(
            to_markdown(rows_bt, headers_bt,
                        caption=f"Top divergent tokens vs base: {lang} (train N={tc:,})")
        )

    # =================================================================
    # Part 2: Pairwise related language analysis
    # =================================================================
    print("  Analyzing related language pairs...")
    pairwise = analyze_pairwise(weights, lang_to_idx, tokens, train_counts)

    # Summary table
    headers_pw = ["Pair", "Sym KL", "Corr", "MAD", "Train A", "Train B"]
    rows_pw = []
    for r in pairwise:
        rows_pw.append([
            r["desc"],
            f"{r['sym_kl']:.2f}", f"{r['corr']:.4f}", f"{r['mad']:.3f}",
            f"{r['train_count_a']:,}", f"{r['train_count_b']:,}",
        ])
    md_pairs = to_markdown(rows_pw, headers_pw, caption="Related language pair divergence")

    # Top divergent tokens per pair
    md_tokens_parts = []
    for r in pairwise:
        headers_tok = ["Token", f"Score ({r['lang_a']})", f"Score ({r['lang_b']})", "Delta", "KL contrib"]
        rows_tok = []
        for t in r["top_tokens"]:
            rows_tok.append([
                repr(t["token"]),
                f"{t['score_a']:.2f}", f"{t['score_b']:.2f}", f"{t['delta']:+.2f}",
                f"{t['kl_contrib']:.4f}",
            ])
        md_tokens_parts.append(
            to_markdown(rows_tok, headers_tok,
                        caption=f"Top divergent tokens: {r['desc']} ({r['lang_a']} vs {r['lang_b']})")
        )

    # Figure: pairwise log-prob scatter for a few key pairs
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    plot_pairs = [
        ("ind_Latn", "zsm_Latn"), ("hin_Deva", "anp_Deva"), ("arb_Arab", "arz_Arab"),
        ("eng_Latn", "sco_Latn"), ("cmn_Hani", "wuu_Hani"), ("fas_Arab", "glk_Arab"),
    ]
    for ax, (la, lb) in zip(axes.flat, plot_pairs):
        if la not in lang_to_idx or lb not in lang_to_idx:
            continue
        sa = np.array(weights[lang_to_idx[la]], dtype=np.float64)
        sb = np.array(weights[lang_to_idx[lb]], dtype=np.float64)
        ax.scatter(sa, sb, alpha=0.05, s=2)
        lims = [min(sa.min(), sb.min()), max(sa.max(), sb.max())]
        ax.plot(lims, lims, "r--", linewidth=0.5)
        corr = np.corrcoef(sa, sb)[0, 1]
        ax.set_xlabel(la)
        ax.set_ylabel(lb)
        ax.set_title(f"r={corr:.3f}")

    fig.suptitle("Log-probability scatter: related language pairs", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "pairwise_logprob_scatter.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # =================================================================
    # Part 3: Correlation of divergence with training size
    # =================================================================
    print("  Analyzing divergence vs training size for pairs...")

    # For all pairs, check if lower training count correlates with higher divergence
    # Use a broader set: all confusion cluster pairs
    all_pair_results = []
    for cluster_key, cluster_info in CONFUSION_CLUSTERS.items():
        cluster_langs = cluster_info["languages"]
        for i, la in enumerate(cluster_langs):
            for lb in cluster_langs[i + 1:]:
                if la not in lang_to_idx or lb not in lang_to_idx:
                    continue
                sa = np.array(weights[lang_to_idx[la]], dtype=np.float64)
                sb = np.array(weights[lang_to_idx[lb]], dtype=np.float64)
                pa, pb = _to_prob(sa), _to_prob(sb)
                sym_kl = _symmetric_kl(pa, pb)
                tc_min = min(train_counts.get(la, 0), train_counts.get(lb, 0))
                all_pair_results.append({
                    "lang_a": la, "lang_b": lb, "cluster": cluster_key,
                    "sym_kl": sym_kl, "min_train_count": tc_min,
                })

    if all_pair_results:
        pair_kls = np.array([r["sym_kl"] for r in all_pair_results])
        pair_tcs = np.array([r["min_train_count"] for r in all_pair_results])
        log_pair_tcs = np.log10(np.clip(pair_tcs, 1, None))
        r_pair = np.corrcoef(log_pair_tcs, pair_kls)[0, 1]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(pair_tcs, pair_kls, alpha=0.5, s=20)
        ax.set_xscale("log")
        ax.set_xlabel("Min training samples in pair (log scale)")
        ax.set_ylabel("Symmetric KL divergence")
        ax.set_title(f"Pairwise KL vs min training size (r = {r_pair:.3f})")
        for r in all_pair_results:
            if r["sym_kl"] > np.percentile(pair_kls, 95):
                ax.annotate(f"{r['lang_a']}/{r['lang_b']}", (r["min_train_count"], r["sym_kl"]),
                            fontsize=6, alpha=0.7)
        fig.tight_layout()
        fig.savefig(os.path.join(figures_dir, "pairwise_kl_vs_training.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # =================================================================
    # Write outputs
    # =================================================================
    full_md = "\n".join([
        md_top_kl, md_bottom_kl, md_resource_kl,
        *md_base_tokens_parts,
        md_pairs,
        *md_tokens_parts,
    ])
    with open(os.path.join(tables_dir, "distribution_analysis.md"), "w") as f:
        f.write(full_md)

    # LaTeX
    latex_parts = [
        to_latex([[r[0]] + r[1:] for r in rows_top], headers_vb,
                 caption="Top 20 languages most divergent from base.", label="tab:kl-top"),
        to_latex(rows_bin, headers_bin,
                 caption="KL from base by resource level.", label="tab:kl-resource"),
        to_latex(rows_pw, headers_pw,
                 caption="Related language pair divergence.", label="tab:pairwise-kl"),
    ]
    with open(os.path.join(tables_dir, "distribution_analysis.tex"), "w") as f:
        f.write("\n\n".join(latex_parts))

    # Print summary
    print(f"  --- Languages vs base ---")
    print(f"  Mean KL from base: {np.mean(kls):.2f} (std={np.std(kls):.2f})")
    print(f"  Correlation log(train_count) vs KL: {np.corrcoef(log_tcs, kls)[0, 1]:.3f}")
    print(f"  Correlation log(train_count) vs corr_with_base: {np.corrcoef(log_tcs, corrs)[0, 1]:.3f}")
    print(f"  --- Related pairs ---")
    for r in pairwise:
        print(f"    {r['desc']:>30s}: sym_KL={r['sym_kl']:.2f}, corr={r['corr']:.4f}, MAD={r['mad']:.3f}")
    if all_pair_results:
        print(f"  --- Cluster pairs: r(log min_train, KL) = {r_pair:.3f} ---")
    print(f"  Outputs: {tables_dir}/distribution_analysis.{{md,tex}}")
    print(f"           {figures_dir}/kl_vs_training_size.png")
    print(f"           {figures_dir}/pairwise_logprob_scatter.png")
    if all_pair_results:
        print(f"           {figures_dir}/pairwise_kl_vs_training.png")
