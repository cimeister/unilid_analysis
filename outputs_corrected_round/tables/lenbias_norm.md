# tab:lenbias-norm: length-normalized scoring by input length

Model: `/capstor/scratch/cscs/cmeister747/unilid_analysis/corrected/glotlidc_corrected.unilid`. Sample: 250,000 lines, the test half of the seed-42 500,000-line draw (the golden subset; the validation half is excluded).
Raw rescore is alpha=0.0, Normalized is alpha=1.0 (score divided by segmentation length).

Raw rescore reproduces the plain scorer exactly (1.000000 agreement), which is the implementation check.

| Length (chars) | N | Original | Raw rescore | Normalized |
|---|---|---|---|---|
| <30 | 13,708 | 0.795 | 0.795 | 0.494 |
| 30--75 | 88,503 | 0.951 | 0.951 | 0.776 |
| 75--150 | 97,861 | 0.977 | 0.977 | 0.883 |
| 150--300 | 43,566 | 0.988 | 0.988 | 0.946 |
| 300+ | 6,362 | 0.994 | 0.994 | 0.986 |
| Overall | 250,000 | 0.960 | 0.960 | 0.837 |
