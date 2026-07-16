import json
import random
import numpy as np
import pickle

random.seed(42)
np.random.seed(42)

DATA_DIR = "/users/cmeister747/unilid_analysis/glotlid_unilid"
TOTAL_LINES = 45_627_279
SAMPLE_SIZE = 500_000

sample_indices = set(sorted(random.sample(range(TOTAL_LINES), SAMPLE_SIZE)))

with open(f"{DATA_DIR}/glotlid_train_counts.json") as f:
    train_counts = json.load(f)

y_true_labels = []
pred_unilid = []
pred_deepseek = []
pred_qwen = []
text_lengths = []

f_test = open(f"{DATA_DIR}/glotlid_correct_test.txt", "r")
f_uni = open(f"{DATA_DIR}/glotlidc_y_pred.txt", "r")
f_ds = open(f"{DATA_DIR}/deepseek_v3.2_glotlid_y_pred.txt", "r")
f_qw = open(f"{DATA_DIR}/qwen3_8b_glotlid_y_pred.txt", "r")

for i in range(TOTAL_LINES):
    test_line = f_test.readline()
    uni_line = f_uni.readline().strip()
    ds_line = f_ds.readline().strip()
    qw_line = f_qw.readline().strip()

    if i in sample_indices:
        parts = test_line.split(" ", 1)
        label = parts[0].replace("__label__", "")
        text = parts[1].rstrip("\n") if len(parts) > 1 else ""

        y_true_labels.append(label)
        pred_unilid.append(uni_line)
        pred_deepseek.append(ds_line)
        pred_qwen.append(qw_line)
        text_lengths.append(len(text))

    if i % 10_000_000 == 0:
        print(f"  {i:,} / {TOTAL_LINES:,}", flush=True)

for f in [f_test, f_uni, f_ds, f_qw]:
    f.close()

with open(f"{DATA_DIR}/sample_500k_all.pkl", "wb") as f:
    pickle.dump({
        "y_true": y_true_labels,
        "pred_unilid": pred_unilid,
        "pred_deepseek": pred_deepseek,
        "pred_qwen": pred_qwen,
        "text_lengths": text_lengths,
        "train_counts": train_counts,
    }, f)

print(f"Saved {len(y_true_labels):,} samples")
