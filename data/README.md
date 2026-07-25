# Data provenance

- `wals_languages.csv`: WALS (World Atlas of Language Structures) language export,
  copied 2026-07-18 from `~/tokenizer-lm/data/wals_languages.csv` (the tokenizer-lm
  project's typology data). Columns used here: `ISO639P3code` and `ISO_codes` (ISO 639-3
  keys), `Family`, `Genus`. Used by `analysis/family_backoff.py` to build genealogical
  back-off groups (family/genus within script). Covers 1,159 of the 1,940 UniLID
  languages; the remainder fall back to the script grouping.
