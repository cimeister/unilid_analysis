# Session Status

## Ongoing experiments
- Directions 1-3 done (Exp 47: in the pool, collapse fail at 0.9534; Exp 48: eligible 0.9486; Exp 49: eligible 0.9498, the strongest eligible configuration). Direction 4 (Exp 50) next: unseen-token values set to a shared constant plus the token's log frequency in the pooled training data (the base tokenizer distribution in the model file), constant fixed so the mean plateau level equals -21. Needs one full scoring pass (~2 h SLURM, full_test_floor21-style script), then judge-part evaluation against floor21 solo (the like-for-like gate-less comparison); if positive, the Exp 49 gate rule goes on top with fresh candidate arrays under the new matrix.
- After Exp 50: the promotion proposal over the whole direction set goes to the user (current best eligible: gate_flat4_prox21 at 0.9498).

## Open decisions
- None until the Exp 50 verdict and the promotion proposal.
