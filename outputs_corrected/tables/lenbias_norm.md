# tab:lenbias-norm: length-normalized scoring by input length

Model: `/capstor/scratch/cscs/cmeister747/unilid_analysis/corrected/glotlidc_corrected.unilid`. Sample: 500,000 lines (seed-42 draw).
Raw rescore is alpha=0.0, Normalized is alpha=1.0 (score divided by segmentation length).

The Original column is omitted: no recorded prediction column exists for this model.

| Length (chars) | N | Raw rescore | Normalized |
|---|---|---|---|
| <30 | 27,328 | 0.792 | 0.493 |
| 30--75 | 177,256 | 0.952 | 0.777 |
| 75--150 | 195,267 | 0.978 | 0.883 |
| 150--300 | 87,096 | 0.987 | 0.946 |
| 300+ | 13,053 | 0.995 | 0.987 |
| Overall | 500,000 | 0.961 | 0.838 |
