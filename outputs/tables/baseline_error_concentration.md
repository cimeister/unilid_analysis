# Baseline false-positive concentration into the two calibration groups (F2)

Instrument: 45,377,279-line scored pool; pred_baseline.npy vs y_true.npy.
Gates G1-G6 passed (support and fp columns match the E1 CSV exactly; fp total
equals the error count; group A matches tau_floor21_gate.csv).

- total baseline false positives: 1,779,499
- into the under-18,000 group (1080 languages): 475,356 (0.2671)
- into the high-entropy four (arg_Latn, bjn_Latn, sco_Latn, vls_Latn): 102,914 (0.0578)
- union (1084 languages): 578,270 (0.3250)
- union support on the scored pool (gold lines): 1,739,187 (0.0383 of the pool)
- concentration ratio (fp share / support share): 8.48
