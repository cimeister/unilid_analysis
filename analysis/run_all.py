#!/usr/bin/env python3
"""Entry point: generate all tables, comparisons, and confusion matrices."""

from __future__ import annotations

import argparse
import os
import sys

# Allow running from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.sample_data import load_sample, sample_data
from analysis.tables import generate_all_tables
from analysis.comparison import generate_all_comparisons
from analysis.confusion import generate_confusion_matrices


def main():
    parser = argparse.ArgumentParser(description="UniLID analysis pipeline")
    parser.add_argument("--format", choices=["markdown", "latex", "both"], default="both",
                        help="Output format for tables (default: both)")
    parser.add_argument("--sample-size", type=int, default=500_000,
                        help="Number of samples to use (default: 500000)")
    parser.add_argument("--output-dir", default="outputs",
                        help="Output directory (default: outputs)")
    parser.add_argument("--resample", action="store_true",
                        help="Force re-sampling even if pickle exists")
    parser.add_argument("--length-bias", action="store_true",
                        help="Run tokenization length bias analysis (requires glotlidc.unilid)")
    parser.add_argument("--normalized", action="store_true",
                        help="Run normalized vs raw scoring comparison (requires glotlidc.unilid)")
    parser.add_argument("--distribution", action="store_true",
                        help="Run per-language distribution analysis vs base and related pairs")
    parser.add_argument("--token-classification", action="store_true",
                        help="Classify divergent tokens between confused language pairs")
    parser.add_argument("--floor-sweep", action="store_true",
                        help="Sweep log-prob floor parameter (requires glotlidc.unilid)")
    parser.add_argument("--train-analysis", action="store_true",
                        help="Run training data analysis (domain, quality, script verification)")
    args = parser.parse_args()

    tables_dir = os.path.join(args.output_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    # --- Load or generate sample ---
    print("=== Loading data ===")
    if args.resample:
        data = sample_data(sample_size=args.sample_size)
    else:
        data = load_sample(sample_size=args.sample_size)
    print(f"  Loaded {len(data['y_true']):,} samples\n")

    # --- Tables 1-4 ---
    print("=== Generating tables 1-4 (overall, length, resource, script) ===")
    tables = generate_all_tables(data)
    for name, (md, latex) in tables.items():
        _write_outputs(tables_dir, name, md, latex, args.format)
        print(f"  {name}: done")

    # --- Tables 5-7 ---
    print("\n=== Generating tables 5-7 (cross-system comparison) ===")
    comparisons = generate_all_comparisons(data)
    for name, (md, latex) in comparisons.items():
        _write_outputs(tables_dir, name, md, latex, args.format)
        print(f"  {name}: done")

    # --- Confusion matrices ---
    print("\n=== Generating confusion matrices ===")
    cm_results = generate_confusion_matrices(data, output_dir=args.output_dir)
    for cluster_key, info in cm_results.items():
        latex_path = os.path.join(tables_dir, f"cm_{cluster_key}.tex")
        with open(latex_path, "w") as f:
            f.write(info["latex"])
        n_figs = len(info["figures"])
        print(f"  {cluster_key}: {info['n_samples']:,} samples, "
              f"{info['n_languages']} langs, {n_figs} figures")

    # --- Length bias analysis (optional) ---
    if args.length_bias:
        from analysis.length_bias import generate_length_bias_analysis
        print("\n=== Length bias analysis ===")
        generate_length_bias_analysis(output_dir=args.output_dir)

    # --- Normalized scoring comparison (optional) ---
    if args.normalized:
        from analysis.normalized_predict import generate_normalized_analysis
        print("\n=== Normalized scoring comparison ===")
        generate_normalized_analysis(
            sample_size=args.sample_size,
            output_dir=args.output_dir,
        )

    # --- Distribution analysis (optional) ---
    if args.distribution:
        from analysis.distribution_analysis import generate_distribution_analysis
        print("\n=== Distribution analysis ===")
        generate_distribution_analysis(output_dir=args.output_dir)

    # --- Token classification (optional) ---
    if args.token_classification:
        from analysis.token_classification import generate_token_classification_report
        print("\n=== Token classification ===")
        generate_token_classification_report(output_dir=args.output_dir)

    # --- Training data analysis (optional) ---
    if args.train_analysis:
        from analysis.train_data_analysis import generate_train_analysis
        print("\n=== Training data analysis ===")
        generate_train_analysis(output_dir=args.output_dir)

    # --- Floor sweep (optional) ---
    if args.floor_sweep:
        from analysis.floor_sweep import generate_floor_sweep
        print("\n=== Floor sweep analysis ===")
        generate_floor_sweep(
            sample_size=args.sample_size,
            output_dir=args.output_dir,
        )

    print(f"\n=== Done. Outputs in {args.output_dir}/ ===")


def _write_outputs(tables_dir: str, name: str, md: str, latex: str, fmt: str):
    if fmt in ("markdown", "both"):
        with open(os.path.join(tables_dir, f"{name}.md"), "w") as f:
            f.write(md)
    if fmt in ("latex", "both"):
        with open(os.path.join(tables_dir, f"{name}.tex"), "w") as f:
            f.write(latex)


if __name__ == "__main__":
    main()
