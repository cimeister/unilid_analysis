import pickle
import numpy as np
from collections import defaultdict, Counter

DATA_DIR = "/users/cmeister747/unilid_analysis/glotlid_unilid"

with open(f"{DATA_DIR}/sample_500k.pkl", "rb") as f:
    data = pickle.load(f)

y_true = np.array(data["y_true"])
y_pred = np.array(data["y_pred"])
text_lengths = np.array(data["text_lengths"])
train_counts = data["train_counts"]

def compute_metrics(y_true, y_pred):
    """Compute accuracy and macro/weighted F1."""
    correct = y_true == y_pred
    accuracy = correct.mean()
    
    # Per-class precision, recall, F1
    labels = np.unique(np.concatenate([y_true, y_pred]))
    precisions = []
    recalls = []
    f1s = []
    supports = []
    
    for lab in labels:
        tp = ((y_pred == lab) & (y_true == lab)).sum()
        fp = ((y_pred == lab) & (y_true != lab)).sum()
        fn = ((y_pred != lab) & (y_true == lab)).sum()
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        supports.append((y_true == lab).sum())
    
    supports = np.array(supports, dtype=float)
    f1s = np.array(f1s)
    
    macro_f1 = f1s.mean()
    weighted_f1 = (f1s * supports).sum() / supports.sum() if supports.sum() > 0 else 0
    
    return accuracy, macro_f1, weighted_f1

# ============================================================
# Overall
# ============================================================
acc, macro_f1, weighted_f1 = compute_metrics(y_true, y_pred)
print("=" * 65)
print("OVERALL METRICS")
print("=" * 65)
print(f"  N = {len(y_true):,}")
print(f"  Accuracy:    {acc:.4f}")
print(f"  Macro F1:    {macro_f1:.4f}")
print(f"  Weighted F1: {weighted_f1:.4f}")

# ============================================================
# 1. By text length
# ============================================================
# Bins based on distribution: short / medium / long / very long
length_bins = [0, 30, 50, 75, 100, 150, 200, 300, 500, np.inf]
length_labels = ["<30", "30-50", "50-75", "75-100", "100-150", "150-200", "200-300", "300-500", "500+"]

print("\n" + "=" * 65)
print("ACCURACY BY TEXT LENGTH (characters)")
print("=" * 65)
print(f"{'Bin':<12} {'N':>8} {'% of data':>9} {'Accuracy':>10} {'Macro F1':>10} {'Wt. F1':>10}")
print("-" * 65)

for i in range(len(length_labels)):
    mask = (text_lengths >= length_bins[i]) & (text_lengths < length_bins[i+1])
    n = mask.sum()
    if n == 0:
        continue
    pct = n / len(y_true) * 100
    a, mf1, wf1 = compute_metrics(y_true[mask], y_pred[mask])
    print(f"{length_labels[i]:<12} {n:>8,} {pct:>8.1f}% {a:>10.4f} {mf1:>10.4f} {wf1:>10.4f}")

# ============================================================
# 2. By resource level (training count)
# ============================================================
# Map each true label to its training count
true_train_counts = np.array([train_counts.get(lab, 0) for lab in y_true])

# Bins based on distribution
resource_bins = [0, 500, 1000, 5000, 10000, 20000, 50000, 100000, np.inf]
resource_labels = ["<500", "500-1k", "1k-5k", "5k-10k", "10k-20k", "20k-50k", "50k-100k", "100k"]

print("\n" + "=" * 65)
print("ACCURACY BY TRAINING RESOURCE LEVEL")
print("=" * 65)
print(f"{'Bin':<12} {'# langs':>8} {'N':>8} {'% of data':>9} {'Accuracy':>10} {'Macro F1':>10} {'Wt. F1':>10}")
print("-" * 73)

for i in range(len(resource_labels)):
    lo, hi = resource_bins[i], resource_bins[i+1]
    mask = (true_train_counts >= lo) & (true_train_counts < hi)
    # For the last bin, include exactly 100000
    if i == len(resource_labels) - 1:
        mask = true_train_counts >= lo
    n = mask.sum()
    if n == 0:
        continue
    # Count distinct languages in this bin
    langs_in_bin = len(set(y_true[mask]))
    pct = n / len(y_true) * 100
    a, mf1, wf1 = compute_metrics(y_true[mask], y_pred[mask])
    print(f"{resource_labels[i]:<12} {langs_in_bin:>8} {n:>8,} {pct:>8.1f}% {a:>10.4f} {mf1:>10.4f} {wf1:>10.4f}")

# ============================================================
# 3. By script
# ============================================================
scripts = np.array([lab.rsplit("_", 1)[-1] for lab in y_true])
unique_scripts = sorted(set(scripts))

print("\n" + "=" * 65)
print("ACCURACY BY SCRIPT")
print("=" * 65)
print(f"{'Script':<12} {'# langs':>8} {'N':>8} {'% of data':>9} {'Accuracy':>10} {'Macro F1':>10} {'Wt. F1':>10}")
print("-" * 73)

# Sort by N descending
script_stats = []
for s in unique_scripts:
    mask = scripts == s
    n = mask.sum()
    langs_in_script = len(set(y_true[mask]))
    pct = n / len(y_true) * 100
    a, mf1, wf1 = compute_metrics(y_true[mask], y_pred[mask])
    script_stats.append((s, langs_in_script, n, pct, a, mf1, wf1))

script_stats.sort(key=lambda x: -x[2])

for s, langs, n, pct, a, mf1, wf1 in script_stats:
    print(f"{s:<12} {langs:>8} {n:>8,} {pct:>8.1f}% {a:>10.4f} {mf1:>10.4f} {wf1:>10.4f}")

