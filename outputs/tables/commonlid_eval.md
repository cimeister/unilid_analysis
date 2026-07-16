# CommonLID (web-domain) evaluation of UniLID

Lines: 373,230. Tags: 109.
Macro-aware accuracy (baseline): **0.8452** (vs GlotLID-test 0.9615).
Macro-aware accuracy (+ frequency prior gamma=0.5): **0.8518** (+0.0067).
Macro-aware accuracy (+ learned bias): **0.8879** (+0.0427).
Tag-level macro-F1 (baseline): 0.7228.

## Trend checks (do GlotLID-test findings hold on web data?)
- Resource asymmetry: among errors, predicted language is RARER than the true language 0.616 of the time (GlotLID-test: ~0.86).
- Flat-magnet activity: diagnosed flat_magnets account for 0.277 of error predictions and 0.0528 of all predictions.

## Top confusions (true_tag -> pred_iso)
- arb -> ars: 5980
- ind -> zsm: 3354
- arb -> ary: 3054
- eng -> sco: 2612
- arb -> acm: 827
- fas -> mzn: 795
- ind -> bew: 757
- ind -> bjn: 735
- deu -> gsw: 691
- arb -> arz: 594
- arz -> ars: 541
- msa -> abs: 535
- uzb -> tly: 528
- msa -> bew: 425
- swh -> swc: 417
- uzb -> vol: 406
- uzb -> crh: 354
- eng -> pcm: 353
- fas -> glk: 346
- kan -> tcy: 328
- ind -> abs: 313
- uzb -> ido: 302
- eng -> nov: 239
- spa -> ast: 238
- rus -> rue: 231