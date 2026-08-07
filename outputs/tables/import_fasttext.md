# Import external prediction file: fasttext

Source file: /capstor/scratch/cscs/cmeister747/unilid_analysis/glotlid_unilid/fasttext_y_pred.txt
Size: 410,645,511 bytes
sha256: fb977d13facfc7d8337a9b834b629853c748077fc57121e217fbcf4e9ff0dbc8
Output memmap: /capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval/pred_fasttext.npy (int16, shape (45,627,279,))
Git commit: 02a346e74b1a907cd30eeb0deb8d185190bc4be4

## Gates passed

- Coverage: all 45,627,279 lines written, zero UNSEEN entries remain.
- Blocking seed-42 sample gate: 100% agreement (500,000 lines) between the imported memmap and 'pred_fastText' in sample_500k_all.pkl.

## Comparability measurements (non-blocking)

- fastText macro F1 on the 45,377,279-line kept pool: 0.944339. The paper reports 0.944, computed on all 45,627,279 lines (paper team metrics JSON, total_samples field), so the two instruments differ by the 250,000 retired validation lines. Absolute difference 0.0003 (flag threshold 0.005).
