"""Classify divergent tokens between confused language pairs.

Categorizes the top KL-contributing tokens into linguistic categories
(function words, content words, morphological affixes, domain terms,
punctuation, encoding artifacts, etc.) and aggregates statistics.
"""

from __future__ import annotations

import os
import re
import unicodedata

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.config import CONFUSION_CLUSTERS, SCRATCH_DIR
from analysis.distribution_analysis import (
    _load_model_data,
    _load_train_counts,
    _top_divergent_tokens,
    analyze_pairwise,
    RELATED_PAIRS,
)
from analysis.format_utils import to_markdown, to_latex


# ---------------------------------------------------------------------------
# Token categories
# ---------------------------------------------------------------------------

CATEGORIES = [
    "function_word",
    "content_word",
    "morphological_affix",
    "domain_religious",
    "punctuation",
    "script_encoding",
    "multi_word_unit",
    "character_phonotactic",
]

CATEGORY_LABELS = {
    "function_word": "Function word",
    "content_word": "Content word",
    "morphological_affix": "Morph. affix",
    "domain_religious": "Domain/religious",
    "punctuation": "Punctuation",
    "script_encoding": "Script/encoding",
    "multi_word_unit": "Multi-word unit",
    "character_phonotactic": "Char/phonotactic",
}

# ---------------------------------------------------------------------------
# Function word lists (common across many languages in each script family)
# ---------------------------------------------------------------------------

LATIN_FUNCTION_WORDS = {
    # English
    "the", "of", "and", "in", "to", "is", "for", "that", "with", "was",
    "on", "are", "be", "by", "it", "as", "at", "or", "an", "not",
    "from", "but", "have", "this", "had", "has", "his", "her", "its",
    "they", "their", "who", "which", "will", "would", "been", "were",
    "one", "all", "also", "can", "more", "than", "other", "into",
    "said", "about", "over", "after", "when", "where", "how", "each",
    # Romance
    "de", "la", "le", "les", "du", "des", "un", "une", "en", "et",
    "est", "que", "qui", "dans", "par", "sur", "au", "aux", "ou",
    "el", "los", "las", "del", "con", "por", "se", "su", "es",
    "da", "di", "il", "lo", "gli", "nel", "dei", "al", "che", "si",
    "do", "dos", "das", "no", "na", "nos", "ao", "aos", "em", "com",
    # Germanic
    "der", "die", "das", "den", "dem", "ein", "eine", "und", "ist",
    "von", "zu", "auf", "mit", "im", "an", "er", "sie", "es",
    "og", "er", "i", "en", "et", "af", "til", "med", "for", "har",
    "av", "som", "var", "det", "att", "inte", "kan", "om",
    # Slavic
    "na", "je", "se", "ze", "ve", "ke", "do", "za", "po",
    # Malay/Indonesian
    "yang", "dan", "di", "ini", "itu", "dari", "ke", "untuk",
    "dengan", "pada", "ada", "akan", "juga", "oleh", "karena",
    "bahwa", "saya", "kita", "mereka", "ia", "nya",
    # Bantu
    "na", "ya", "wa", "ni", "kwa", "mu",
}

DOMAIN_RELIGIOUS_TERMS = {
    "yehuwa", "jehov", "jehovah", "allah", "biblia", "bible",
    "psalm", "jesus", "christ", "gospel", "scripture", "verse",
    "church", "pray", "lord", "god", "prophet", "apostle",
    "kina", "mose", "moses", "israel", "jerusalem",
}


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def _strip_space_marker(token: str) -> str:
    """Remove the Ġ (space) prefix if present."""
    if token.startswith("Ġ"):
        return token[1:]
    return token


def _is_valid_unicode_in_script(token: str) -> bool:
    """Check if the token's bytes form valid Unicode characters (not encoding artifacts)."""
    stripped = _strip_space_marker(token)
    if not stripped:
        return False
    try:
        for ch in stripped:
            cat = unicodedata.category(ch)
            if cat == "Cc":
                return False
        return True
    except (ValueError, TypeError):
        return False


def _count_space_markers(token: str) -> int:
    """Count how many Ġ (space) markers appear in the token."""
    return token.count("Ġ")


def classify_token(token: str) -> str:
    """Classify a single token into a linguistic category.

    Categories: function_word, content_word, morphological_affix,
    domain_religious, punctuation, script_encoding, multi_word_unit,
    character_phonotactic.
    """
    stripped = _strip_space_marker(token)
    has_space_prefix = token.startswith("Ġ")

    # Multi-word unit: multiple space markers
    if _count_space_markers(token) >= 2:
        return "multi_word_unit"

    # Punctuation: mostly non-alphanumeric (after stripping space marker)
    if stripped and not any(c.isalpha() for c in stripped):
        return "punctuation"

    # Mixed punct+alpha (e.g., ',"Ġ', 'Ġ-M')
    alpha_ratio = sum(1 for c in stripped if c.isalpha()) / max(len(stripped), 1)
    if alpha_ratio < 0.5 and len(stripped) <= 4:
        return "punctuation"

    # Script/encoding: check for replacement characters or encoding artifacts
    if not _is_valid_unicode_in_script(token):
        return "script_encoding"

    # Domain/religious terms
    if stripped.lower() in DOMAIN_RELIGIOUS_TERMS:
        return "domain_religious"
    for term in DOMAIN_RELIGIOUS_TERMS:
        if term in stripped.lower() and len(stripped) < len(term) + 5:
            return "domain_religious"

    # Function word: starts with space + matches function word list
    if has_space_prefix and stripped.lower() in LATIN_FUNCTION_WORDS:
        return "function_word"

    # Character/phonotactic: very short (1-2 chars), alphabetic
    if len(stripped) <= 2 and stripped.isalpha():
        return "character_phonotactic"

    # Morphological affix: no space prefix (subword), short
    if not has_space_prefix and len(stripped) <= 5 and stripped.isalpha():
        return "morphological_affix"

    # Content word: has space prefix, not matched above
    if has_space_prefix:
        return "content_word"

    # Longer subword without space prefix
    if not has_space_prefix and len(stripped) > 5:
        return "content_word"

    # Fallback: morphological affix for remaining short subwords
    return "morphological_affix"


# ---------------------------------------------------------------------------
# Pair-level classification
# ---------------------------------------------------------------------------

def classify_pair_tokens(top_tokens: list[dict]) -> list[dict]:
    """Classify all top tokens for a pair, adding a 'category' field."""
    results = []
    for t in top_tokens:
        cat = classify_token(t["token"])
        results.append({**t, "category": cat})
    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_classifications(
    all_classified: list[dict],
) -> dict[str, dict]:
    """Aggregate token classifications across all pairs.

    all_classified is a list of dicts, each with:
      pair_desc, lang_a, lang_b, tokens (list of dicts with category, kl_contrib, delta)
    """
    # Per-category aggregation
    cat_stats = {cat: {"count": 0, "sum_kl": 0.0, "sum_abs_delta": 0.0} for cat in CATEGORIES}
    total_tokens = 0
    total_kl = 0.0

    for pair in all_classified:
        for t in pair["tokens"]:
            cat = t["category"]
            cat_stats[cat]["count"] += 1
            cat_stats[cat]["sum_kl"] += abs(t["kl_contrib"])
            cat_stats[cat]["sum_abs_delta"] += abs(t["delta"])
            total_tokens += 1
            total_kl += abs(t["kl_contrib"])

    return {
        "per_category": cat_stats,
        "total_tokens": total_tokens,
        "total_kl": total_kl,
    }


def aggregate_by_resource_level(all_classified: list[dict], train_counts: dict) -> dict:
    """Split pairs into high-resource (both >50k) vs mixed-resource (one <10k)."""
    high_res = {"tokens": [], "pairs": []}
    mixed_res = {"tokens": [], "pairs": []}

    for pair in all_classified:
        tc_a = train_counts.get(pair["lang_a"], 0)
        tc_b = train_counts.get(pair["lang_b"], 0)
        min_tc = min(tc_a, tc_b)

        if min_tc >= 50000:
            high_res["tokens"].extend(pair["tokens"])
            high_res["pairs"].append(pair["pair_desc"])
        elif min_tc < 10000:
            mixed_res["tokens"].extend(pair["tokens"])
            mixed_res["pairs"].append(pair["pair_desc"])

    return {"high_resource": high_res, "mixed_resource": mixed_res}


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_token_classification_report(output_dir: str = "outputs"):
    """Run the full token classification analysis."""
    print("  Loading model data...")
    base_scores, weights, langs, lang_to_idx, tokens = _load_model_data()
    train_counts = _load_train_counts()

    tables_dir = os.path.join(output_dir, "tables")
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # ===================================================================
    # Compute pairwise divergent tokens and classify them
    # ===================================================================
    print("  Computing pairwise token divergences...")
    pairwise = analyze_pairwise(weights, lang_to_idx, tokens, train_counts)

    all_classified = []
    for r in pairwise:
        classified_tokens = classify_pair_tokens(r["top_tokens"])
        all_classified.append({
            "pair_desc": r["desc"],
            "lang_a": r["lang_a"],
            "lang_b": r["lang_b"],
            "train_count_a": r["train_count_a"],
            "train_count_b": r["train_count_b"],
            "tokens": classified_tokens,
        })

    # ===================================================================
    # Per-pair tables with classifications
    # ===================================================================
    print("  Generating per-pair tables...")
    md_pair_parts = []
    for pair in all_classified:
        headers = ["Token", "Category", f"Score ({pair['lang_a']})",
                   f"Score ({pair['lang_b']})", "Delta", "KL contrib"]
        rows = []
        for t in pair["tokens"]:
            rows.append([
                repr(t["token"]),
                CATEGORY_LABELS[t["category"]],
                f"{t['score_a']:.2f}",
                f"{t['score_b']:.2f}",
                f"{t['delta']:+.2f}",
                f"{t['kl_contrib']:.4f}",
            ])
        md_pair_parts.append(
            to_markdown(rows, headers,
                        caption=f"Classified tokens: {pair['pair_desc']} ({pair['lang_a']} vs {pair['lang_b']})")
        )

    # ===================================================================
    # Aggregate table
    # ===================================================================
    print("  Computing aggregate statistics...")
    agg = aggregate_classifications(all_classified)

    headers_agg = ["Category", "# Tokens", "% Tokens", "Mean |delta|", "Sum |KL|", "% of KL"]
    rows_agg_md = []
    rows_agg_latex = []
    for cat in CATEGORIES:
        s = agg["per_category"][cat]
        n = s["count"]
        if n == 0:
            continue
        pct_tokens = n / agg["total_tokens"] * 100
        mean_delta = s["sum_abs_delta"] / n
        pct_kl = s["sum_kl"] / agg["total_kl"] * 100 if agg["total_kl"] > 0 else 0
        rows_agg_md.append([
            CATEGORY_LABELS[cat], f"{n}", f"{pct_tokens:.1f}",
            f"{mean_delta:.2f}", f"{s['sum_kl']:.4f}", f"{pct_kl:.1f}",
        ])
        rows_agg_latex.append([
            CATEGORY_LABELS[cat], n, pct_tokens, mean_delta, s["sum_kl"], pct_kl,
        ])

    md_agg = to_markdown(rows_agg_md, headers_agg, caption="Aggregate token classification across all pairs")

    # ===================================================================
    # By resource level
    # ===================================================================
    by_res = aggregate_by_resource_level(all_classified, train_counts)

    md_res_parts = []
    for level_name, level_data in [("High-resource pairs (both >50k)", by_res["high_resource"]),
                                     ("Mixed-resource pairs (one <10k)", by_res["mixed_resource"])]:
        if not level_data["tokens"]:
            continue
        cat_counts = {}
        total = len(level_data["tokens"])
        for t in level_data["tokens"]:
            cat = t["category"]
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        headers_res = ["Category", "# Tokens", "% Tokens"]
        rows_res = []
        for cat in CATEGORIES:
            n = cat_counts.get(cat, 0)
            if n == 0:
                continue
            rows_res.append([CATEGORY_LABELS[cat], f"{n}", f"{n / total * 100:.1f}"])

        pairs_str = ", ".join(level_data["pairs"][:5])
        if len(level_data["pairs"]) > 5:
            pairs_str += f" (+{len(level_data['pairs']) - 5} more)"
        md_res_parts.append(
            to_markdown(rows_res, headers_res,
                        caption=f"{level_name} ({len(level_data['pairs'])} pairs: {pairs_str})")
        )

    # ===================================================================
    # Per-pair category breakdown (for stacked bar chart)
    # ===================================================================
    pair_descs = [p["pair_desc"] for p in all_classified]
    cat_fractions = {cat: [] for cat in CATEGORIES}
    for pair in all_classified:
        total = len(pair["tokens"])
        for cat in CATEGORIES:
            n = sum(1 for t in pair["tokens"] if t["category"] == cat)
            cat_fractions[cat].append(n / total if total > 0 else 0)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(pair_descs))
    bottom = np.zeros(len(pair_descs))
    colors = plt.cm.Set3(np.linspace(0, 1, len(CATEGORIES)))

    for cat, color in zip(CATEGORIES, colors):
        vals = cat_fractions[cat]
        ax.bar(x, vals, bottom=bottom, label=CATEGORY_LABELS[cat], color=color, width=0.8)
        bottom += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels(pair_descs, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Fraction of top-20 tokens")
    ax.set_title("Token category distribution per confused language pair")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "token_categories_stacked.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ===================================================================
    # Domain dependency analysis
    # ===================================================================
    domain_dependent_pairs = []
    for pair in all_classified:
        total_kl = sum(abs(t["kl_contrib"]) for t in pair["tokens"])
        domain_kl = sum(abs(t["kl_contrib"]) for t in pair["tokens"] if t["category"] == "domain_religious")
        pct = domain_kl / total_kl * 100 if total_kl > 0 else 0
        if pct > 5:
            domain_dependent_pairs.append((pair["pair_desc"], pair["lang_a"], pair["lang_b"], pct))

    md_domain = ""
    if domain_dependent_pairs:
        headers_dom = ["Pair", "Lang A", "Lang B", "% KL from domain terms"]
        rows_dom = [[desc, la, lb, f"{pct:.1f}"] for desc, la, lb, pct in domain_dependent_pairs]
        md_domain = to_markdown(rows_dom, headers_dom,
                                 caption="Pairs with >5% of KL from domain/religious tokens")

    # ===================================================================
    # Write outputs
    # ===================================================================
    parts = [md_agg] + md_res_parts
    if md_domain:
        parts.append(md_domain)
    parts.extend(md_pair_parts)

    with open(os.path.join(tables_dir, "token_classification.md"), "w") as f:
        f.write("\n".join(parts))

    latex_parts = [
        to_latex(rows_agg_latex, headers_agg,
                 caption="Aggregate token classification across all confused pairs.",
                 label="tab:token-class-agg",
                 col_formats=["str", "int", "metric", "metric", "metric", "metric"]),
    ]
    with open(os.path.join(tables_dir, "token_classification.tex"), "w") as f:
        f.write("\n\n".join(latex_parts))

    # Print summary
    print(f"\n  === Token Classification Summary ===")
    print(f"  Total tokens classified: {agg['total_tokens']}")
    for cat in CATEGORIES:
        s = agg["per_category"][cat]
        if s["count"] > 0:
            pct = s["count"] / agg["total_tokens"] * 100
            kl_pct = s["sum_kl"] / agg["total_kl"] * 100 if agg["total_kl"] > 0 else 0
            print(f"    {CATEGORY_LABELS[cat]:>20s}: {s['count']:>3d} ({pct:>5.1f}%), {kl_pct:>5.1f}% of KL")
    if domain_dependent_pairs:
        print(f"  Domain-dependent pairs (>5% KL from religious terms):")
        for desc, _, _, pct in domain_dependent_pairs:
            print(f"    {desc}: {pct:.1f}%")
    print(f"  Outputs: {tables_dir}/token_classification.{{md,tex}}")
    print(f"           {figures_dir}/token_categories_stacked.png")
