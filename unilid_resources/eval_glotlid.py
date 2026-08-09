"""Evaluate .unilid models on GlotLID test set (fastText format)."""
import argparse
import json
import re
import time
from pathlib import Path
from collections import Counter

from tqdm import tqdm
from unilid.model_io import UnilidModel


def evaluate(model_path, test_path, out_dir=None, lang_only=False, only_model_langs=True, forward=False):
    test_path = Path(test_path)
    label_re = re.compile(r"^__label__(\S+)")

    # Load test data
    texts, labels = [], []
    with open(test_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = label_re.match(line)
            if not m:
                continue
            label = m.group(1)
            parts = line.split(" ", 1)
            text = parts[1] if len(parts) == 2 else ""
            labels.append(label)
            texts.append(text)

    print(f"Loaded {len(texts)} samples from {test_path}")

    model = UnilidModel(model_path)

    def extract_lang(label):
        return label.split("_", 1)[0] if lang_only else label

    model_labels = set(model.langs)
    if lang_only:
        model_labels_check = {extract_lang(l) for l in model.langs}
    else:
        model_labels_check = model_labels

    # Filter to model languages if requested
    if only_model_langs:
        filtered_texts, filtered_labels = [], []
        for t, l in tqdm(zip(texts, labels), total=len(texts), desc="Filtering"):
            if extract_lang(l) in model_labels_check:
                filtered_texts.append(t)
                filtered_labels.append(l)
        print(f"Filtered to {len(filtered_texts)} samples in model languages (from {len(texts)} total)")
        texts, labels = filtered_texts, filtered_labels

    # Predict
    confusion = Counter()
    predictions = []
    batch_size = 10000
    num_chunks = (len(texts) + batch_size - 1) // batch_size

    start = time.perf_counter()
    for i in tqdm(range(0, len(texts), batch_size), total=num_chunks, desc="Inference"):
        batch_texts = texts[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]
        results = model.predict_batch(batch_texts, forward=forward)
        for (pred_full, tokens, score), gold in zip(results, batch_labels):
            if pred_full is None:
                pred_full = "NONE"
            predictions.append(pred_full)
            gold_lang = extract_lang(gold)
            pred_lang = extract_lang(pred_full)
            confusion[(gold_lang, pred_lang)] += 1

    elapsed = time.perf_counter() - start
    print(f"Inference: {elapsed:.2f}s ({len(texts)/elapsed:.0f} samples/s)")

    # Compute metrics
    all_langs = sorted(set(g for g, _ in confusion) | set(p for _, p in confusion))
    correct = sum(v for (g, p), v in confusion.items() if g == p)
    total = sum(confusion.values())
    accuracy = correct / total if total else 0

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
        "lang_only": lang_only,
        "only_model_langs": only_model_langs,
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

        # Save predictions (one per line, matching input order)
        y_pred_path = out_dir / f"{model_name}_y_pred.txt"
        with open(y_pred_path, "w", encoding="utf-8") as f:
            for pred in predictions:
                f.write(pred + "\n")

        print(f"Saved to {out_dir}")
        print(f"  y_pred: {y_pred_path}")

    return summary, per_lang


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help=".unilid model path")
    parser.add_argument("--test", required=True, help="GlotLID test file (fastText format)")
    parser.add_argument("--out-dir", default=None, help="Output directory")
    parser.add_argument("--lang-only", action="store_true", help="Compare language codes only (ignore script)")
    parser.add_argument("--all-langs", action="store_true", help="Evaluate on all languages, not just model languages")
    parser.add_argument("--forward", action="store_true", help="Use forward algorithm (marginalize over segmentations)")
    args = parser.parse_args()
    evaluate(args.model, args.test, args.out_dir, lang_only=args.lang_only, only_model_langs=not args.all_langs, forward=args.forward)
