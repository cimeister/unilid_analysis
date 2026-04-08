"""Experiment 7: Training data analysis.

Single streaming pass over train.txt collecting per-language statistics:
  7.1 Domain distribution (religious/Bible, Wikipedia, other)
  7.4 Per-language corpus quality (text length, char entropy, vocab size)
  7.5 Script verification (detect script mismatches)
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.config import (
    CONFUSION_CLUSTERS,
    DATA_DIR,
    RESOURCE_BINS,
    RESOURCE_LABELS,
    TRAIN_FILE,
    TRAIN_LINES,
)
from analysis.format_utils import to_markdown, to_latex


# ---------------------------------------------------------------------------
# Domain classification heuristics
# ---------------------------------------------------------------------------

# Compiled regex patterns for domain detection
_VERSE_RE = re.compile(r'\b\d{1,3}:\d{1,3}\b')
_WIKI_SECTION_RE = re.compile(r'={2,}[^=]+=+')
_WIKI_LINK_RE = re.compile(r'\[\[')
_WIKI_CITE_RE = re.compile(r'\[\d+\]')

# Religious/Bible markers (case-insensitive substrings)
_RELIGIOUS_TERMS = {
    'yehuwa', 'jehova', 'jehovah', 'watchtower', 'jw.org',
    'genesis', 'exodus', 'leviticus', 'deuteronomy',
    'psalms', 'proverbs', 'isaiah', 'jeremiah', 'ezekiel',
    'matthew', 'matayo', 'yohana', 'revelation',
    'bible', 'scripture', 'gospel', 'apostle',
    'psalm', 'salmo', 'biblia', 'evangel',
}

_RELIGIOUS_RE = re.compile(
    '|'.join(re.escape(t) for t in _RELIGIOUS_TERMS),
    re.IGNORECASE
)


def classify_domain(text: str) -> str:
    """Classify a text line into a domain category."""
    # Religious/Bible: verse references or religious terms
    if _VERSE_RE.search(text) or _RELIGIOUS_RE.search(text):
        return "religious"
    # Wikipedia: section headers, wikilinks, citations
    if _WIKI_SECTION_RE.search(text) or _WIKI_LINK_RE.search(text) or _WIKI_CITE_RE.search(text):
        return "wikipedia"
    return "other"


# ---------------------------------------------------------------------------
# Script detection via Unicode block ranges
# ---------------------------------------------------------------------------

# Major script ranges (start, end, name)
_SCRIPT_RANGES = [
    (0x0041, 0x024F, "Latin"),       # Basic Latin + Extended
    (0x1E00, 0x1EFF, "Latin"),       # Latin Extended Additional
    (0x0400, 0x04FF, "Cyrillic"),
    (0x0500, 0x052F, "Cyrillic"),    # Cyrillic Supplement
    (0x0600, 0x06FF, "Arabic"),
    (0x0750, 0x077F, "Arabic"),      # Arabic Supplement
    (0xFB50, 0xFDFF, "Arabic"),      # Arabic Presentation Forms
    (0x0900, 0x097F, "Devanagari"),
    (0x0980, 0x09FF, "Bengali"),
    (0x0A80, 0x0AFF, "Gujarati"),
    (0x0B00, 0x0B7F, "Oriya"),
    (0x0B80, 0x0BFF, "Tamil"),
    (0x0C00, 0x0C7F, "Telugu"),
    (0x0C80, 0x0CFF, "Kannada"),
    (0x0D00, 0x0D7F, "Malayalam"),
    (0x0D80, 0x0DFF, "Sinhala"),
    (0x0E00, 0x0E7F, "Thai"),
    (0x0E80, 0x0EFF, "Lao"),
    (0x0F00, 0x0FFF, "Tibetan"),
    (0x1000, 0x109F, "Myanmar"),
    (0x10A0, 0x10FF, "Georgian"),
    (0x1100, 0x11FF, "Hangul"),
    (0x3040, 0x309F, "Hiragana"),
    (0x30A0, 0x30FF, "Katakana"),
    (0x3100, 0x312F, "Bopomofo"),
    (0x3400, 0x4DBF, "Han"),
    (0x4E00, 0x9FFF, "Han"),
    (0x20000, 0x2A6DF, "Han"),
    (0xAC00, 0xD7AF, "Hangul"),
    (0x0530, 0x058F, "Armenian"),
    (0x0590, 0x05FF, "Hebrew"),
    (0x1200, 0x137F, "Ethiopic"),
    (0x0370, 0x03FF, "Greek"),
    (0xA980, 0xA9DF, "Javanese"),
    (0x1780, 0x17FF, "Khmer"),
    (0xA800, 0xA82F, "Syloti_Nagri"),
    (0xA4D0, 0xA4FF, "Lisu"),
    (0x2D30, 0x2D7F, "Tifinagh"),
    (0x0780, 0x07BF, "Thaana"),
]

# Build a lookup for fast script detection
_SCRIPT_LOOKUP = {}
for _start, _end, _name in _SCRIPT_RANGES:
    for _cp in range(_start, _end + 1):
        _SCRIPT_LOOKUP[_cp] = _name

# Map ISO 15924 script codes to Unicode script names
SCRIPT_CODE_MAP = {
    "Latn": "Latin", "Cyrl": "Cyrillic", "Arab": "Arabic",
    "Deva": "Devanagari", "Beng": "Bengali", "Gujr": "Gujarati",
    "Orya": "Oriya", "Taml": "Tamil", "Telu": "Telugu",
    "Knda": "Kannada", "Mlym": "Malayalam", "Sinh": "Sinhala",
    "Thai": "Thai", "Laoo": "Lao", "Tibt": "Tibetan",
    "Mymr": "Myanmar", "Geor": "Georgian", "Hang": "Hangul",
    "Hira": "Hiragana", "Kana": "Katakana", "Hani": "Han",
    "Armn": "Armenian", "Hebr": "Hebrew", "Ethi": "Ethiopic",
    "Grek": "Greek", "Jpan": "Han", "Khmr": "Khmer",
    "Cans": "Canadian_Aboriginal", "Lisu": "Lisu", "Tfng": "Tifinagh",
    "Thaa": "Thaana",
}


def detect_script(text: str) -> Counter:
    """Count characters per script in text. Ignores Common (digits, punct, space)."""
    counts = Counter()
    for ch in text:
        cp = ord(ch)
        script = _SCRIPT_LOOKUP.get(cp)
        if script:
            counts[script] += 1
    return counts


# ---------------------------------------------------------------------------
# HyperLogLog for vocabulary size estimation
# ---------------------------------------------------------------------------

import hashlib

class HyperLogLog:
    """Minimal HyperLogLog sketch for cardinality estimation."""

    def __init__(self, p=14):
        self.p = p
        self.m = 1 << p
        self.registers = bytearray(self.m)

    def add(self, item: str):
        h = int(hashlib.md5(item.encode('utf-8', errors='replace')).hexdigest(), 16)
        idx = h & (self.m - 1)
        remaining = h >> self.p
        self.registers[idx] = max(self.registers[idx], self._leading_zeros(remaining) + 1)

    @staticmethod
    def _leading_zeros(value):
        if value == 0:
            return 64
        count = 0
        for i in range(63, -1, -1):
            if value & (1 << i):
                break
            count += 1
        return count

    def estimate(self) -> int:
        alpha = 0.7213 / (1 + 1.079 / self.m)
        raw = alpha * self.m ** 2 / sum(2.0 ** (-r) for r in self.registers)
        if raw <= 2.5 * self.m:
            zeros = self.registers.count(0)
            if zeros > 0:
                return int(self.m * math.log(self.m / zeros))
        return int(raw)


# ---------------------------------------------------------------------------
# Per-language accumulators
# ---------------------------------------------------------------------------

class LangStats:
    """Accumulates statistics for a single language."""
    __slots__ = [
        'n_samples', 'len_sum', 'len_sum_sq',
        'char_counts', 'domain_counts', 'script_counts',
        'hll',
    ]

    def __init__(self):
        self.n_samples = 0
        self.len_sum = 0
        self.len_sum_sq = 0
        self.char_counts = Counter()
        self.domain_counts = Counter()
        self.script_counts = Counter()
        self.hll = HyperLogLog(p=12)

    def update(self, text: str, domain: str):
        self.n_samples += 1
        tlen = len(text)
        self.len_sum += tlen
        self.len_sum_sq += tlen * tlen

        self.domain_counts[domain] += 1

        for ch in text:
            self.char_counts[ch] += 1
            cp = ord(ch)
            script = _SCRIPT_LOOKUP.get(cp)
            if script:
                self.script_counts[script] += 1

        # Vocabulary: add whitespace-delimited tokens to HLL
        for word in text.split():
            self.hll.add(word)

    @property
    def mean_length(self):
        return self.len_sum / self.n_samples if self.n_samples else 0

    @property
    def std_length(self):
        if self.n_samples < 2:
            return 0
        var = (self.len_sum_sq / self.n_samples) - (self.mean_length ** 2)
        return max(0, var) ** 0.5

    @property
    def char_entropy(self):
        total = sum(self.char_counts.values())
        if total == 0:
            return 0
        entropy = 0.0
        for count in self.char_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    @property
    def vocab_size(self):
        return self.hll.estimate()


# ---------------------------------------------------------------------------
# Main streaming analysis
# ---------------------------------------------------------------------------

def run_streaming_analysis(train_path: str = TRAIN_FILE) -> dict[str, LangStats]:
    """Single-pass streaming analysis of the training file."""
    stats = defaultdict(LangStats)

    with open(train_path, 'r', errors='replace') as f:
        for i, line in enumerate(f):
            if not line.startswith('__label__'):
                continue
            space_idx = line.index(' ', 9) if ' ' in line[9:] else len(line)
            lang = line[9:space_idx]
            text = line[space_idx + 1:].rstrip('\n')

            domain = classify_domain(text)
            stats[lang].update(text, domain)

            if i % 5_000_000 == 0:
                print(f"  Processed {i:,} / {TRAIN_LINES:,} lines ({len(stats)} languages)", flush=True)

    print(f"  Done: {i + 1:,} lines, {len(stats)} languages")
    return dict(stats)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _load_train_counts():
    with open(os.path.join(DATA_DIR, "glotlid_train_counts.json")) as f:
        return json.load(f)


def _get_resource_bin(n: int) -> str:
    for i in range(len(RESOURCE_BINS) - 1):
        if RESOURCE_BINS[i] <= n < RESOURCE_BINS[i + 1]:
            return RESOURCE_LABELS[i]
    return RESOURCE_LABELS[-1]


def generate_train_analysis(output_dir: str = "outputs"):
    """Run all training data analyses and generate reports."""
    tables_dir = os.path.join(output_dir, "tables")
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("  Streaming training data...")
    stats = run_streaming_analysis()

    train_counts = _load_train_counts()
    all_md = []

    # =======================================================================
    # 7.1 Domain distribution
    # =======================================================================
    print("  Generating domain analysis...")

    # Overall domain counts
    total_domain = Counter()
    for lang, s in stats.items():
        for domain, cnt in s.domain_counts.items():
            total_domain[domain] += cnt
    total_samples = sum(total_domain.values())

    headers_dom = ["Domain", "Samples", "% of total"]
    rows_dom = []
    for domain in ["religious", "wikipedia", "other"]:
        cnt = total_domain.get(domain, 0)
        rows_dom.append([domain, f"{cnt:,}", f"{cnt / total_samples * 100:.1f}"])
    all_md.append(to_markdown(rows_dom, headers_dom, caption="Overall domain distribution"))

    # Domain by resource bin
    bin_domain = defaultdict(Counter)
    for lang, s in stats.items():
        tc = train_counts.get(lang, s.n_samples)
        rbin = _get_resource_bin(tc)
        for domain, cnt in s.domain_counts.items():
            bin_domain[rbin][domain] += cnt

    headers_rdom = ["Resource bin", "Religious %", "Wikipedia %", "Other %", "N samples"]
    rows_rdom = []
    for rbin in RESOURCE_LABELS:
        dc = bin_domain[rbin]
        t = sum(dc.values())
        if t == 0:
            continue
        rows_rdom.append([
            rbin,
            f"{dc['religious'] / t * 100:.1f}",
            f"{dc['wikipedia'] / t * 100:.1f}",
            f"{dc['other'] / t * 100:.1f}",
            f"{t:,}",
        ])
    all_md.append(to_markdown(rows_rdom, headers_rdom, caption="Domain distribution by resource level"))

    # Domain for confusion cluster languages
    cluster_langs = set()
    for cinfo in CONFUSION_CLUSTERS.values():
        cluster_langs.update(cinfo["languages"])

    headers_cdom = ["Language", "N", "Religious %", "Wikipedia %", "Other %"]
    rows_cdom = []
    for ckey, cinfo in CONFUSION_CLUSTERS.items():
        for lang in cinfo["languages"]:
            s = stats.get(lang)
            if not s:
                continue
            t = s.n_samples
            rows_cdom.append([
                lang, f"{t:,}",
                f"{s.domain_counts['religious'] / t * 100:.1f}" if t else "--",
                f"{s.domain_counts['wikipedia'] / t * 100:.1f}" if t else "--",
                f"{s.domain_counts['other'] / t * 100:.1f}" if t else "--",
            ])
    all_md.append(to_markdown(rows_cdom, headers_cdom, caption="Domain distribution for confusion cluster languages"))

    # Domain stacked bar chart
    fig, ax = plt.subplots(figsize=(14, 6))
    cluster_lang_list = []
    for cinfo in CONFUSION_CLUSTERS.values():
        cluster_lang_list.extend(cinfo["languages"])
    rel_pcts, wiki_pcts, other_pcts = [], [], []
    labels = []
    for lang in cluster_lang_list:
        s = stats.get(lang)
        if not s or s.n_samples == 0:
            continue
        labels.append(lang)
        t = s.n_samples
        rel_pcts.append(s.domain_counts["religious"] / t * 100)
        wiki_pcts.append(s.domain_counts["wikipedia"] / t * 100)
        other_pcts.append(s.domain_counts["other"] / t * 100)

    x = np.arange(len(labels))
    ax.bar(x, rel_pcts, label="Religious/Bible", color="#e74c3c")
    ax.bar(x, wiki_pcts, bottom=rel_pcts, label="Wikipedia", color="#3498db")
    ax.bar(x, other_pcts, bottom=[r + w for r, w in zip(rel_pcts, wiki_pcts)], label="Other", color="#95a5a6")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("% of training samples")
    ax.set_title("Domain distribution for confusion cluster languages")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "train_domain_stacked.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # =======================================================================
    # 7.4 Corpus quality
    # =======================================================================
    print("  Generating corpus quality analysis...")

    # Per-resource-bin quality summary
    bin_stats = defaultdict(lambda: {"lengths": [], "entropies": [], "vocabs": [], "n_langs": 0})
    for lang, s in stats.items():
        tc = train_counts.get(lang, s.n_samples)
        rbin = _get_resource_bin(tc)
        bin_stats[rbin]["lengths"].append(s.mean_length)
        bin_stats[rbin]["entropies"].append(s.char_entropy)
        bin_stats[rbin]["vocabs"].append(s.vocab_size)
        bin_stats[rbin]["n_langs"] += 1

    headers_qual = ["Resource bin", "# Langs", "Mean text len", "Mean char entropy", "Mean vocab size"]
    rows_qual = []
    for rbin in RESOURCE_LABELS:
        bs = bin_stats[rbin]
        if bs["n_langs"] == 0:
            continue
        rows_qual.append([
            rbin, str(bs["n_langs"]),
            f"{np.mean(bs['lengths']):.1f}",
            f"{np.mean(bs['entropies']):.2f}",
            f"{np.mean(bs['vocabs']):,.0f}",
        ])
    all_md.append(to_markdown(rows_qual, headers_qual, caption="Corpus quality by resource level"))

    # Scatter: char entropy vs training count
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    train_ns = []
    entropies = []
    for lang, s in stats.items():
        tc = train_counts.get(lang, s.n_samples)
        train_ns.append(tc)
        entropies.append(s.char_entropy)

    axes[0].scatter(train_ns, entropies, s=3, alpha=0.3)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Training samples")
    axes[0].set_ylabel("Character entropy (bits)")
    axes[0].set_title("Character entropy vs training size")

    vocab_sizes = [stats[l].vocab_size for l in stats]
    axes[1].scatter(train_ns, vocab_sizes, s=3, alpha=0.3)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Training samples")
    axes[1].set_ylabel("Vocabulary size (HLL estimate)")
    axes[1].set_title("Vocabulary size vs training size")

    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "train_quality_scatter.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # =======================================================================
    # 7.5 Script verification
    # =======================================================================
    print("  Generating script verification...")

    script_issues = []
    for lang, s in stats.items():
        expected_code = lang.rsplit("_", 1)[-1] if "_" in lang else None
        expected_script = SCRIPT_CODE_MAP.get(expected_code)
        if not expected_script or not s.script_counts:
            continue

        total_script_chars = sum(s.script_counts.values())
        if total_script_chars == 0:
            continue
        expected_frac = s.script_counts.get(expected_script, 0) / total_script_chars
        if expected_frac < 0.95:
            # Find dominant script
            dominant = s.script_counts.most_common(1)[0]
            script_issues.append({
                "lang": lang,
                "expected": expected_script,
                "expected_frac": expected_frac,
                "dominant": dominant[0],
                "dominant_frac": dominant[1] / total_script_chars,
                "n_samples": s.n_samples,
            })

    script_issues.sort(key=lambda x: x["expected_frac"])

    headers_script = ["Language", "N", "Expected script", "Expected %", "Dominant script", "Dominant %"]
    rows_script = []
    for issue in script_issues[:50]:
        rows_script.append([
            issue["lang"], f"{issue['n_samples']:,}",
            issue["expected"], f"{issue['expected_frac'] * 100:.1f}",
            issue["dominant"], f"{issue['dominant_frac'] * 100:.1f}",
        ])
    all_md.append(to_markdown(rows_script, headers_script,
                              caption=f"Script verification: {len(script_issues)} languages with >5% unexpected script (showing top 50)"))

    # Script purity histogram
    purities = []
    for lang, s in stats.items():
        expected_code = lang.rsplit("_", 1)[-1] if "_" in lang else None
        expected_script = SCRIPT_CODE_MAP.get(expected_code)
        if not expected_script or not s.script_counts:
            continue
        total_script_chars = sum(s.script_counts.values())
        if total_script_chars > 0:
            purities.append(s.script_counts.get(expected_script, 0) / total_script_chars)

    if purities:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(purities, bins=50, edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Fraction of characters in expected script")
        ax.set_ylabel("Number of languages")
        ax.set_title(f"Script purity distribution ({len(purities)} languages)")
        ax.axvline(0.95, color="red", linestyle="--", label="95% threshold")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(figures_dir, "train_script_purity.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # =======================================================================
    # Write combined output
    # =======================================================================
    full_md = "\n\n".join(all_md)
    with open(os.path.join(tables_dir, "train_data_analysis.md"), "w") as f:
        f.write(full_md)

    # Verification: check line counts match
    print("\n  Verification: line counts vs glotlid_train_counts.json")
    mismatches = 0
    for lang, s in stats.items():
        expected = train_counts.get(lang, None)
        if expected is not None and s.n_samples != expected:
            if mismatches < 5:
                print(f"    MISMATCH: {lang}: streamed {s.n_samples}, expected {expected}")
            mismatches += 1
    if mismatches:
        print(f"    Total mismatches: {mismatches}")
    else:
        print(f"    All {len(stats)} languages match")

    print(f"\n  Outputs: {tables_dir}/train_data_analysis.md, {figures_dir}/train_*.png")
    return stats
