"""Evaluate .unilid models on WiLI-2018 test set."""
import argparse
import json
import time
from pathlib import Path
from collections import Counter

from tqdm import tqdm
from unilid.model_io import UnilidModel


def evaluate(model_path, wili_dir, out_dir=None, forward=False, x_test=None):
    wili_dir = Path(wili_dir)
    x_path = Path(x_test) if x_test else wili_dir / "x_test.txt"
    y_path = wili_dir / "y_test.txt"

    texts = x_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    labels = [l.strip() for l in y_path.read_text(encoding="utf-8", errors="ignore").splitlines()]
    assert len(texts) == len(labels), f"Mismatched: {len(texts)} vs {len(labels)}"
    print(f"Loaded {len(texts)} test samples")

    model = UnilidModel(model_path)

    # Extract iso639-3 from model labels (e.g., "eng_Latn" -> "eng")
    def extract_iso639(label):
        return label.split("_", 1)[0]

    confusion = Counter()
    batch_size = 10000
    total_valid = 0

    start = time.perf_counter()
    for i in tqdm(range(0, len(texts), batch_size), desc="Inference"):
        batch_texts = texts[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]
        results = model.predict_batch(batch_texts, forward=forward)
        for (pred_full, tokens, score), gold in zip(results, batch_labels):
            if pred_full is None:
                continue
            total_valid += 1
            pred = extract_iso639(pred_full)
            confusion[(gold, pred)] += 1

    elapsed = time.perf_counter() - start
    print(f"Inference: {elapsed:.2f}s ({len(texts)/elapsed:.0f} samples/s)")

    # Compute metrics
    all_langs = sorted(set(g for g, _ in confusion) | set(p for _, p in confusion))
    correct = sum(v for (g, p), v in confusion.items() if g == p)
    total = sum(confusion.values())
    accuracy = correct / total if total else 0

    # Per-language precision/recall/F1/FPR
    per_lang = {}
    for lang in all_langs:
        tp = confusion.get((lang, lang), 0)
        fn = sum(v for (g, p), v in confusion.items() if g == lang and p != lang)
        fp = sum(v for (g, p), v in confusion.items() if p == lang and g != lang)
        tn = total - tp - fn - fp
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        fpr = fp / (fp + tn) if (fp + tn) else 0
        per_lang[lang] = {"precision": prec, "recall": rec, "f1": f1, "fpr": fpr, "support": tp + fn}

    macro_f1 = sum(v["f1"] for v in per_lang.values()) / len(per_lang) if per_lang else 0
    macro_fpr = sum(v["fpr"] for v in per_lang.values()) / len(per_lang) if per_lang else 0

    summary = {
        "model": str(model_path),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "macro_fpr": macro_fpr,
        "total_samples": total,
        "correct": correct,
        "num_languages": len(all_langs),
        "inference_time_s": elapsed,
        "samples_per_sec": len(texts) / elapsed,
    }

    print(f"\nAccuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Macro FPR: {macro_fpr:.4f}")
    print(f"Languages: {len(all_langs)}")

    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        model_name = Path(model_path).stem

        with open(out_dir / f"{model_name}_metrics.json", "w") as f:
            json.dump(summary, f, indent=2)

        with open(out_dir / f"{model_name}_per_language.json", "w") as f:
            json.dump(per_lang, f, indent=2)

        print(f"Saved to {out_dir}")

    return summary, per_lang


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help=".unilid model path")
    parser.add_argument("--wili-dir", required=True, help="WiLI-2018 directory")
    parser.add_argument("--out-dir", default=None, help="Output directory for results")
    parser.add_argument("--forward", action="store_true", help="Use forward algorithm (marginalize over segmentations)")
    parser.add_argument("--x-test", default=None, help="Override x_test.txt path")
    args = parser.parse_args()
    evaluate(args.model, args.wili_dir, args.out_dir, forward=args.forward, x_test=args.x_test)
