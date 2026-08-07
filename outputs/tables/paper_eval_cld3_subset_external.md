# CLD3-subset cells, external benchmarks (restricted-lines convention, the one that reproduces the paper's UniLID cells; commit a952831b28658ba566634c9481f08e4e4efc6d97)

Bare codes mapped to the largest-training-corpus variant inside each benchmark's label set.
Baseline verification targets: the paper's UDHR CLD3 UniLID cell .992 and FLORES CLD3 UniLID cell .997.

- udhr: 80 subset labels, 5,388 restricted lines; baseline 0.9873 (paper 0.992, REPRODUCES); calibrated 0.9856
- flores: 77 subset labels, 77,924 restricted lines; baseline 0.9907 (paper 0.997, MISMATCH); calibrated 0.9920
