import json
import random
import numpy as np

random.seed(42)
np.random.seed(42)

DATA_DIR = "/users/cmeister747/unilid_analysis/glotlid_unilid"
TOTAL_LINES = 45_627_279
SAMPLE_SIZE = 500_000

# Uniform sample indices
sample_indices = set(sorted(random.sample(range(TOTAL_LINES), SAMPLE_SIZE)))

# Read train counts
with open(f"{DATA_DIR}/glotlid_train_counts.json") as f:
    train_counts = json.load(f)

print(f"Number of languages in train_counts: {len(train_counts)}")
counts_arr = np.array(list(train_counts.values()))
print(f"\nTraining count distribution:")
for p in [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]:
    print(f"  p{p:3d}: {np.percentile(counts_arr, p):>10.0f}")

# Parse sample
y_true_labels = []
y_pred_labels = []
text_lengths = []

pred_file = open(f"{DATA_DIR}/glotlidc_y_pred.txt", "r")
test_file = open(f"{DATA_DIR}/glotlid_correct_test.txt", "r")

for i in range(TOTAL_LINES):
    test_line = test_file.readline()
    pred_line = pred_file.readline().strip()
    
    if i in sample_indices:
        # Parse fasttext format: __label__XXX text...
        parts = test_line.split(" ", 1)
        label = parts[0].replace("__label__", "")
        text = parts[1].rstrip("\n") if len(parts) > 1 else ""
        
        y_true_labels.append(label)
        y_pred_labels.append(pred_line)
        text_lengths.append(len(text))

    if i % 5_000_000 == 0:
        print(f"  processed {i:,} / {TOTAL_LINES:,} lines", flush=True)

test_file.close()
pred_file.close()

text_lengths = np.array(text_lengths)

print(f"\nSampled {len(text_lengths):,} lines")
print(f"\nText length (chars) distribution:")
for p in [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]:
    print(f"  p{p:3d}: {np.percentile(text_lengths, p):>10.0f}")

# Save sample for further analysis
import pickle
with open(f"{DATA_DIR}/sample_500k.pkl", "wb") as f:
    pickle.dump({
        "y_true": y_true_labels,
        "y_pred": y_pred_labels,
        "text_lengths": text_lengths.tolist(),
        "train_counts": train_counts,
    }, f)

print("\nSample saved to sample_500k.pkl")
