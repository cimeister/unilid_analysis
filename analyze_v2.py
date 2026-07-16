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
    """Compute accuracy, micro F1, macro F1, weighted F1."""
    correct = y_true == y_pred
    accuracy = correct.mean()
    
    labels = np.unique(np.concatenate([y_true, y_pred]))
    
    # Micro: aggregate TP/FP/FN across all classes
    micro_tp = 0
    micro_fp = 0
    micro_fn = 0
    
    f1s = []
    supports = []
    
    for lab in labels:
        tp = ((y_pred == lab) & (y_true == lab)).sum()
        fp = ((y_pred == lab) & (y_true != lab)).sum()
        fn = ((y_pred != lab) & (y_true == lab)).sum()
        
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        
        f1s.append(f1)
        supports.append((y_true == lab).sum())
    
    micro_prec = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0
    micro_rec = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0
    micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0
    
    macro_f1 = np.mean(f1s)
    
    return accuracy, micro_f1, macro_f1

# ============================================================
# Overall
# ============================================================
acc, micro_f1, macro_f1 = compute_metrics(y_true, y_pred)
print("=" * 70)
print("OVERALL METRICS")
print("=" * 70)
print(f"  N = {len(y_true):,}  |  # languages = {len(set(y_true)):,}")
print(f"  Accuracy:  {acc:.4f}")
print(f"  Micro F1:  {micro_f1:.4f}")
print(f"  Macro F1:  {macro_f1:.4f}")

HDR = f"{'Bin':<12} {'# langs':>8} {'N':>8} {'% data':>7} {'Acc':>8} {'Mi-F1':>8} {'Ma-F1':>8}"
SEP = "-" * 70

# ============================================================
# 1. By text length
# ============================================================
length_bins = [0, 30, 50, 75, 100, 150, 200, 300, 500, np.inf]
length_labels = ["<30", "30-50", "50-75", "75-100", "100-150", "150-200", "200-300", "300-500", "500+"]

print(f"\n{'='*70}")
print("ACCURACY BY TEXT LENGTH (characters)")
print(f"{'='*70}")
print(HDR)
print(SEP)

for i in range(len(length_labels)):
    mask = (text_lengths >= length_bins[i]) & (text_lengths < length_bins[i+1])
    n = mask.sum()
    if n == 0: continue
    n_langs = len(set(y_true[mask]))
    pct = n / len(y_true) * 100
    a, mi, ma = compute_metrics(y_true[mask], y_pred[mask])
    print(f"{length_labels[i]:<12} {n_langs:>8} {n:>8,} {pct:>6.1f}% {a:>8.4f} {mi:>8.4f} {ma:>8.4f}")

# ============================================================
# 2. By resource level (quintiles by # languages)
# ============================================================
resource_bins = [0, 9429, 17441, 23117, 48036, np.inf]
resource_labels = ["<9.4k", "9.4k-17.4k", "17.4k-23.1k", "23.1k-48k", "48k+"]

true_train_counts = np.array([train_counts.get(lab, 0) for lab in y_true])

print(f"\n{'='*70}")
print("ACCURACY BY TRAINING RESOURCE LEVEL (quintiles by # languages)")
print(f"{'='*70}")
print(HDR)
print(SEP)

for i in range(len(resource_labels)):
    lo, hi = resource_bins[i], resource_bins[i+1]
    mask = (true_train_counts >= lo) & (true_train_counts < hi)
    if i == len(resource_labels) - 1:
        mask = true_train_counts >= lo
    n = mask.sum()
    if n == 0: continue
    n_langs = len(set(y_true[mask]))
    pct = n / len(y_true) * 100
    a, mi, ma = compute_metrics(y_true[mask], y_pred[mask])
    print(f"{resource_labels[i]:<12} {n_langs:>8} {n:>8,} {pct:>6.1f}% {a:>8.4f} {mi:>8.4f} {ma:>8.4f}")

# ============================================================
# 3. By script
# ============================================================
scripts = np.array([lab.rsplit("_", 1)[-1] for lab in y_true])
unique_scripts = sorted(set(scripts))

print(f"\n{'='*70}")
print("ACCURACY BY SCRIPT")
print(f"{'='*70}")
print(HDR)
print(SEP)

script_stats = []
for s in unique_scripts:
    mask = scripts == s
    n = mask.sum()
    n_langs = len(set(y_true[mask]))
    pct = n / len(y_true) * 100
    a, mi, ma = compute_metrics(y_true[mask], y_pred[mask])
    script_stats.append((s, n_langs, n, pct, a, mi, ma))

script_stats.sort(key=lambda x: -x[2])
for s, nl, n, pct, a, mi, ma in script_stats:
    print(f"{s:<12} {nl:>8} {n:>8,} {pct:>6.1f}% {a:>8.4f} {mi:>8.4f} {ma:>8.4f}")

# ============================================================
# 4. Confusion analysis for high-resource & low-accuracy
# ============================================================
print(f"\n{'='*70}")
print("TOP CONFUSION PAIRS (true_label -> predicted_label)")
print(f"{'='*70}")

# Find all misclassifications
wrong_mask = y_true != y_pred
confusion_pairs = Counter(zip(y_true[wrong_mask], y_pred[wrong_mask]))

print(f"\nTotal misclassifications: {wrong_mask.sum():,} / {len(y_true):,} ({wrong_mask.mean()*100:.1f}%)")
print(f"\nTop 30 confusion pairs:")
print(f"{'True':<18} {'Predicted':<18} {'Count':>6} {'% of errors':>11} {'Train(true)':>12} {'Train(pred)':>12}")
print("-" * 85)

total_errors = wrong_mask.sum()
for (true_lab, pred_lab), count in confusion_pairs.most_common(30):
    tc_true = train_counts.get(true_lab, 0)
    tc_pred = train_counts.get(pred_lab, 0)
    print(f"{true_lab:<18} {pred_lab:<18} {count:>6} {count/total_errors*100:>10.2f}% {tc_true:>12,} {tc_pred:>12,}")

# ============================================================
# 5. Per-script confusion breakdown for problem scripts
# ============================================================
problem_scripts = ["Hani", "Arab", "Deva", "Latn", "Cyrl"]
for script in problem_scripts:
    mask = scripts == script
    wrong_in_script = mask & wrong_mask
    if wrong_in_script.sum() == 0:
        continue
    
    pairs = Counter(zip(y_true[wrong_in_script], y_pred[wrong_in_script]))
    total_in_script = mask.sum()
    errors_in_script = wrong_in_script.sum()
    
    print(f"\n--- {script} script: {errors_in_script:,} errors / {total_in_script:,} samples ({errors_in_script/total_in_script*100:.1f}%) ---")
    print(f"{'True':<18} {'Predicted':<18} {'Count':>6} {'% of script err':>15}")
    print("-" * 65)
    for (true_lab, pred_lab), count in pairs.most_common(10):
        print(f"{true_lab:<18} {pred_lab:<18} {count:>6} {count/errors_in_script*100:>14.1f}%")

