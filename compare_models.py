import pickle
import numpy as np
from collections import Counter

DATA_DIR = "/users/cmeister747/unilid_analysis/glotlid_unilid"

with open(f"{DATA_DIR}/sample_500k_all.pkl", "rb") as f:
    data = pickle.load(f)

y_true = np.array(data["y_true"])
pred = {
    "UniLID": np.array(data["pred_unilid"]),
    "UniLID-DeepSeek": np.array(data["pred_deepseek"]),
    "UniLID-Qwen": np.array(data["pred_qwen"]),
}
text_lengths = np.array(data["text_lengths"])
train_counts = data["train_counts"]
true_train_cts = np.array([train_counts.get(lab, 0) for lab in y_true])
scripts = np.array([lab.rsplit("_", 1)[-1] for lab in y_true])

MODELS = ["UniLID", "UniLID-DeepSeek", "UniLID-Qwen"]

def acc_and_f1(y_true, y_pred):
    """Returns (accuracy, micro_f1, macro_f1)."""
    correct = y_true == y_pred
    acc = correct.mean()
    labels = np.unique(np.concatenate([y_true, y_pred]))
    micro_tp = micro_fp = micro_fn = 0
    f1s = []
    for lab in labels:
        tp = ((y_pred == lab) & (y_true == lab)).sum()
        fp = ((y_pred == lab) & (y_true != lab)).sum()
        fn = ((y_pred != lab) & (y_true == lab)).sum()
        micro_tp += tp; micro_fp += fp; micro_fn += fn
        p = tp/(tp+fp) if tp+fp else 0
        r = tp/(tp+fn) if tp+fn else 0
        f1s.append(2*p*r/(p+r) if p+r else 0)
    mp = micro_tp/(micro_tp+micro_fp) if micro_tp+micro_fp else 0
    mr = micro_tp/(micro_tp+micro_fn) if micro_tp+micro_fn else 0
    micro_f1 = 2*mp*mr/(mp+mr) if mp+mr else 0
    macro_f1 = np.mean(f1s)
    return acc, micro_f1, macro_f1

# ============================================================
# Overall
# ============================================================
print("=" * 75)
print("OVERALL COMPARISON")
print("=" * 75)
print(f"{'Model':<20} {'N':>8} {'Acc':>8} {'Mi-F1':>8} {'Ma-F1':>8}")
print("-" * 55)
for m in MODELS:
    a, mi, ma = acc_and_f1(y_true, pred[m])
    print(f"{m:<20} {len(y_true):>8,} {a:>8.4f} {mi:>8.4f} {ma:>8.4f}")

# ============================================================
# Error overlap
# ============================================================
print(f"\n{'='*75}")
print("ERROR OVERLAP")
print(f"{'='*75}")
errs = {m: (y_true != pred[m]) for m in MODELS}
for m in MODELS:
    print(f"  {m}: {errs[m].sum():,} errors ({errs[m].mean()*100:.2f}%)")

all_wrong = errs["UniLID"] & errs["UniLID-DeepSeek"] & errs["UniLID-Qwen"]
any_wrong = errs["UniLID"] | errs["UniLID-DeepSeek"] | errs["UniLID-Qwen"]
print(f"\n  All 3 wrong:       {all_wrong.sum():>7,} ({all_wrong.sum()/any_wrong.sum()*100:.1f}% of any-wrong)")
print(f"  Any wrong:         {any_wrong.sum():>7,}")
print(f"  Only UniLID wrong: {(errs['UniLID'] & ~errs['UniLID-DeepSeek'] & ~errs['UniLID-Qwen']).sum():>7,}")
print(f"  Only DeepSeek wrong: {(~errs['UniLID'] & errs['UniLID-DeepSeek'] & ~errs['UniLID-Qwen']).sum():>7,}")
print(f"  Only Qwen wrong:  {(~errs['UniLID'] & ~errs['UniLID-DeepSeek'] & errs['UniLID-Qwen']).sum():>7,}")

# When all 3 are wrong, do they agree on the wrong answer?
all_wrong_mask = all_wrong
if all_wrong_mask.sum() > 0:
    agree_on_wrong = (pred["UniLID"][all_wrong_mask] == pred["UniLID-DeepSeek"][all_wrong_mask]) & \
                     (pred["UniLID"][all_wrong_mask] == pred["UniLID-Qwen"][all_wrong_mask])
    print(f"  All 3 wrong AND same pred: {agree_on_wrong.sum():>5,} ({agree_on_wrong.sum()/all_wrong_mask.sum()*100:.1f}% of all-3-wrong)")

# ============================================================
# By text length
# ============================================================
length_bins = [0, 30, 75, 150, 300, np.inf]
length_labels = ["<30", "30-75", "75-150", "150-300", "300+"]

print(f"\n{'='*75}")
print("BY TEXT LENGTH")
print(f"{'='*75}")

for metric_name, metric_idx in [("Accuracy", 0), ("Macro F1", 2)]:
    print(f"\n--- {metric_name} ---")
    print(f"{'Bin':<10} {'N':>8}", end="")
    for m in MODELS:
        print(f" {m:>16}", end="")
    print()
    print("-" * 65)
    
    for i in range(len(length_labels)):
        mask = (text_lengths >= length_bins[i]) & (text_lengths < length_bins[i+1])
        n = mask.sum()
        if n == 0: continue
        print(f"{length_labels[i]:<10} {n:>8,}", end="")
        for m in MODELS:
            vals = acc_and_f1(y_true[mask], pred[m][mask])
            print(f" {vals[metric_idx]:>16.4f}", end="")
        print()

# ============================================================
# By resource level
# ============================================================
resource_bins = [0, 1000, 12192, 17821, 35264, np.inf]
resource_labels = ["<1k", "1k-12k", "12k-18k", "18k-35k", "35k+"]

print(f"\n{'='*75}")
print("BY RESOURCE LEVEL")
print(f"{'='*75}")

for metric_name, metric_idx in [("Accuracy", 0), ("Macro F1", 2)]:
    print(f"\n--- {metric_name} ---")
    print(f"{'Bin':<10} {'# lang':>7} {'N':>8}", end="")
    for m in MODELS:
        print(f" {m:>16}", end="")
    print()
    print("-" * 75)
    
    for i in range(len(resource_labels)):
        lo, hi = resource_bins[i], resource_bins[i+1]
        mask = (true_train_cts >= lo) & (true_train_cts < hi)
        if i == len(resource_labels) - 1:
            mask = true_train_cts >= lo
        n = mask.sum()
        if n == 0: continue
        nl = len(set(y_true[mask]))
        print(f"{resource_labels[i]:<10} {nl:>7} {n:>8,}", end="")
        for m in MODELS:
            vals = acc_and_f1(y_true[mask], pred[m][mask])
            print(f" {vals[metric_idx]:>16.4f}", end="")
        print()

# ============================================================
# By script (top 10 + Other)
# ============================================================
script_counts = Counter(scripts)
top_scripts = [s for s, _ in script_counts.most_common(10)]

print(f"\n{'='*75}")
print("BY SCRIPT (top 10 by sample count)")
print(f"{'='*75}")

for metric_name, metric_idx in [("Accuracy", 0), ("Macro F1", 2)]:
    print(f"\n--- {metric_name} ---")
    print(f"{'Script':<8} {'# lang':>7} {'N':>8}", end="")
    for m in MODELS:
        print(f" {m:>16}", end="")
    print()
    print("-" * 75)
    
    for s in top_scripts:
        mask = scripts == s
        n = mask.sum()
        nl = len(set(y_true[mask]))
        print(f"{s:<8} {nl:>7} {n:>8,}", end="")
        for m in MODELS:
            vals = acc_and_f1(y_true[mask], pred[m][mask])
            print(f" {vals[metric_idx]:>16.4f}", end="")
        print()
    
    # Other
    other_mask = ~np.isin(scripts, top_scripts)
    if other_mask.sum() > 0:
        n = other_mask.sum()
        nl = len(set(y_true[other_mask]))
        print(f"{'Other':<8} {nl:>7} {n:>8,}", end="")
        for m in MODELS:
            vals = acc_and_f1(y_true[other_mask], pred[m][other_mask])
            print(f" {vals[metric_idx]:>16.4f}", end="")
        print()

# ============================================================
# Correlation of per-language error rates
# ============================================================
print(f"\n{'='*75}")
print("PER-LANGUAGE ERROR RATE CORRELATION")
print(f"{'='*75}")

# Compute per-language error rate for each model
lang_list = sorted(set(y_true))
lang_err_rates = {m: [] for m in MODELS}
lang_ns = []

for lang in lang_list:
    mask = y_true == lang
    n = mask.sum()
    if n < 10:  # skip tiny langs in sample
        continue
    lang_ns.append(n)
    for m in MODELS:
        err_rate = (pred[m][mask] != y_true[mask]).mean()
        lang_err_rates[m].append(err_rate)

for m in MODELS:
    lang_err_rates[m] = np.array(lang_err_rates[m])

print(f"\nPearson correlation of per-language error rates (N={len(lang_ns)} langs with >=10 samples):")
pairs = [("UniLID", "UniLID-DeepSeek"), ("UniLID", "UniLID-Qwen"), ("UniLID-DeepSeek", "UniLID-Qwen")]
for m1, m2 in pairs:
    r = np.corrcoef(lang_err_rates[m1], lang_err_rates[m2])[0, 1]
    print(f"  {m1} vs {m2}: r = {r:.4f}")

# Which languages show the biggest divergence?
print(f"\nLanguages with largest accuracy gap (UniLID vs best of other two):")
print(f"{'Language':<16} {'N':>6} {'UniLID':>8} {'DeepSeek':>8} {'Qwen':>8} {'Gap':>8}")
print("-" * 60)

gaps = []
for idx, lang in enumerate([l for l in lang_list if (y_true == l).sum() >= 10]):
    mask = y_true == lang
    n = mask.sum()
    u = (pred["UniLID"][mask] != y_true[mask]).mean()
    d = (pred["UniLID-DeepSeek"][mask] != y_true[mask]).mean()
    q = (pred["UniLID-Qwen"][mask] != y_true[mask]).mean()
    best_other = min(d, q)
    gap = u - best_other  # positive = UniLID worse
    gaps.append((lang, n, u, d, q, gap))

# Show where UniLID is most worse
gaps.sort(key=lambda x: -x[5])
print("\nUniLID worse than best alternative:")
for lang, n, u, d, q, gap in gaps[:15]:
    print(f"{lang:<16} {n:>6} {u:>8.3f} {d:>8.3f} {q:>8.3f} {gap:>+8.3f}")

print("\nUniLID better than best alternative:")
gaps.sort(key=lambda x: x[5])
for lang, n, u, d, q, gap in gaps[:15]:
    print(f"{lang:<16} {n:>6} {u:>8.3f} {d:>8.3f} {q:>8.3f} {gap:>+8.3f}")

