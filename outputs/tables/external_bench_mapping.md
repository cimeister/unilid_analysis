# External LID benchmark acquisition: UDHR and FLORES-plus devtest

Prepared 2026-08-07. Rebuilds the two external evaluation sets the paper used (UDHR, FLORES-200)
from current public sources, since the paper's own preprocessed files are unavailable. No scoring
was run. Nothing downstream of this report runs until the mapping decisions below are approved.

## Sources and access

| Dataset | HF repo | Revision (commit sha) | Files used | Access |
|---|---|---|---|---|
| UDHR | `cis-lmu/udhr-lid` | `6908db2a27c296158da7e69782d15df911652184` | `udhr-lid.csv` (single file) | Public, not gated. |
| FLORES-plus | `openlanguagedata/flores_plus` | `5fec6c13f9e5a4db2f745d4ec0d7c9721ddc4f06` | `devtest/*.jsonl`, 221 files | Gated with automatic approval. The `HF_TOKEN` already present in the environment downloaded both metadata and file content without any manual approval step or error. |

`HF_HOME` was already set in the shell environment to
`/capstor/store/cscs/swissai/a0229/cmeister/huggingface` before this task started; I did not set it
myself. Both datasets' blobs and snapshots resolve under that path, confirmed from the download
paths returned by `huggingface_hub`. No files landed in the home directory or in `/tmp`.

Downloads used `huggingface_hub.hf_hub_download` / `snapshot_download` directly rather than
`datasets.load_dataset`, to inspect the raw file schema before any dataset-script transformation and
to avoid a pandas NA-parsing hazard described below.

Python: `/users/cmeister747/.pyenv/versions/3.11.5/bin/python3`, `datasets==4.6.1`,
`huggingface_hub==0.36.2`, `pandas==3.0.1`.

## Canonical label set

1,940 `lang_Script` labels, read from the `lang` column of
`/users/cmeister747/unilid_analysis/outputs/diagnostic/full_test_per_lang_prf.csv`. Confirmed 1,940
rows, all distinct.

## UDHR: label field used

`udhr-lid.csv` has columns `id, sentence, iso639-3, iso15924, language`. The `id` column already
holds the `lang_Script` label directly (for example `tir_Ethi`), so `id` is the field used, unmodified.

Inspection found a pandas CSV-parsing hazard worth recording: when the file is read with pandas'
default NA handling, the `iso639-3` column value `nan` (the literal ISO 639-3 code for Min Nan
Chinese) is parsed as a missing value, not the string `"nan"`, for 58 rows. The `id` column is
unaffected because pandas' NA-token matching is exact-string, and `id` holds `nan_Hani`, not the bare
token `nan`. Reading with `id` as the label source sidesteps this hazard entirely; reconstructing the
label from `iso639-3 + "_" + iso15924` would require `keep_default_na=False`, which I verified also
reproduces `id` exactly (0 mismatches) once applied.

Total rows: 27,757. Distinct labels: 430.

## FLORES-plus devtest: label construction

Each `devtest/*.jsonl` row has fields `id, iso_639_3, iso_15924, glottocode, variant, text, url,
domain, topic, has_image, has_hyperlink, last_updated, split`. Column names verified directly from
the file content, matching the task's expectation. Label built as `f"{iso_639_3}_{iso_15924}"`. Every
row's `split` field reads `devtest` (checked for all 223,652 rows); no other split was mixed in.

221 files were read, giving 215 distinct `(iso_639_3, iso_15924)` labels, not 221, because 6 labels
are each backed by two separate files that share the same `iso_639_3`/`iso_15924` pair but differ in
`variant`/`glottocode` (dialect or orthography variants of the same language-script pair):

| Label | Source files | Combined rows |
|---|---|---|
| `apc_Arab` | `apc_Arab_nort3139`, `apc_Arab_sout3123` | 2,024 |
| `cat_Latn` | `cat_Latn`, `cat_Latn_vale1252` | 2,024 |
| `lld_Latn` | `lld_Latn`, `lld_Latn_gard1241` | 2,024 |
| `nob_Latn` | `nob_Latn`, `nob_Latn_radical` | 2,024 |
| `oci_Latn` | `oci_Latn`, `oci_Latn_aran1260` | 2,024 |
| `twi_Latn` | `twi_Latn_akua1239`, `twi_Latn_asan1239` | 2,024 |

All 6 of these labels are in the 1,940-label intersection, so `flores_devtest_eval.tsv` carries 2,024
rows under each of these 6 labels instead of the usual 1,012. This is a direct, documented consequence
of building the label from `iso_639_3` and `iso_15924` only, as instructed (no `variant` or
`glottocode` component). Flagged here as a data-quality note, not resolved.

Total rows: 223,652. Distinct labels: 215.

## Intersection analysis

| Metric | UDHR | FLORES-plus devtest |
|---|---|---|
| (a) Total rows | 27,757 | 223,652 |
| (b) Distinct labels | 430 | 215 |
| (c) Labels in the 1,940-label intersection | 366 | 205 |
| (c) Rows covered by those labels | 24,115 | 213,532 |
| (d) Labels not in the 1,940-label set | 64 | 10 |
| (d) Rows covered by those labels | 3,642 | 10,120 |
| (e) Paper's reported label count | 366 | 190 |
| (e) Difference (intersection minus paper) | 0 | +15 |

UDHR's intersection count matches the paper's 366 exactly.

FLORES-plus devtest's intersection count is 205, 15 above the paper's 190. `flores_plus` is an
actively maintained successor to the original FLORES-200 release, not a frozen copy: 4 of the 8
needs-decision labels below (`cmn_Hans`, `cmn_Hant`, `yue_Hant`, `khk_Mong`) ship their own per-language
`dataset_cards/*.md` entries in the repository, consistent with these being added or re-split after
the original 200-language release the paper used. I have not cross-checked the exact original
FLORES-200 language list against this repository, so the 15-label gap is not fully explained, only
consistent with dataset growth.

## Top out-of-set labels

Full lists are in the diagnostic CSVs (see Output files below). Top 30 by row count, UDHR:

| label | n_rows | label | n_rows | label | n_rows |
|---|---|---|---|---|---|
| lot_Latn | 140 | pbu_Arab | 59 | bax_Latn | 58 |
| ibb_Latn | 67 | bos_Latn | 59 | nan_Hani | 58 |
| ccp_Cakm | 62 | ppl_Latn | 59 | mzi_Latn | 58 |
| vai_Vaii | 61 | oki_Latn | 59 | hea_Latn | 58 |
| tgl_Latn | 60 | hms_Latn | 59 | fuf_Adlm | 58 |
| cic_Latn | 60 | gld_Cyrl | 59 | slr_Latn | 58 |
| src_Latn | 59 | mxi_Latn | 58 | ... (34 more, see CSV) | |
| orh_Latn | 59 | cjy_Hani | 58 | | |
| cnr_Latn | 59 | hrv_Latn | 58 | | |
| bos_Cyrl | 59 | que_Latn | 58 | | |
| | | gan_Hani | 58 | | |
| | | emk_Latn | 58 | | |
| | | rgn_Latn | 58 | | |
| | | ztu_Latn | 58 | | |

All 10 FLORES-plus out-of-set labels (fewer than 30, listed in full):

| label | n_rows |
|---|---|
| apd_Arab | 1,012 |
| pes_Arab | 1,012 |
| prs_Arab | 1,012 |
| acq_Arab | 1,012 |
| cmn_Hant | 1,012 |
| yue_Hant | 1,012 |
| hrv_Latn | 1,012 |
| cmn_Hans | 1,012 |
| khk_Mong | 1,012 |
| bos_Latn | 1,012 |

## Needs-decision table

These are the out-of-set labels with a plausible near-miss against the 1,940-label set: same
3-letter code under a different script, a macrolanguage code present in the canonical set while the
dataset uses an individual-language code (or vice versa), or a closely related distinct code. No
remapping has been applied. `udhr_eval.tsv` and `flores_devtest_eval.tsv` contain none of these rows;
they were excluded by the exact-match intersection rule, not by any judgment call made here.

| Out-of-set label(s) | Rows (UDHR / FLORES) | Canonical set has instead | Relationship |
|---|---|---|---|
| `pes_Arab`, `prs_Arab` | 58+56 / 1,012+1,012 | `fas_Arab` | `fas` (Persian) is the ISO 639-3 macrolanguage code; `pes` (Iranian Persian) and `prs` (Dari) are individual members. Canon has only the macro code. |
| `bos_Latn`, `bos_Cyrl`, `hrv_Latn`, `cnr_Latn` | 59+59+58+59 / 1,012+1,012 (`hrv_Latn`, `bos_Latn` only) | `hbs_Latn` (macro) and `srp_Latn`/`srp_Cyrl` (individual) | `hbs` (Serbo-Croatian) is the macro code; Bosnian, Croatian, Montenegrin are individual members without their own canon entries, while sibling member Serbian (`srp`) does have canon entries at both scripts. |
| `zlm_Latn`, `zlm_Arab` | 58+24 / 0 | `zsm_Latn`, `zsm_Arab` | `zlm` (Malay, individual) and `zsm` (Standard Malay, individual) are both members of macro `msa`; canon has `zsm` only. `msa` itself is absent from canon. |
| `que_Latn` | 58 / 0 | `qub_Latn`, `quy_Latn`, `quz_Latn`, `qvw_Latn` (and others) | `que` (Quechua) is the macro code; canon carries several individual Quechua varieties but not the bare macro code. |
| `cmn_Hans`, `cmn_Hant` | 0 / 1,012+1,012 | `cmn_Hani` | Canon uses the unified Han script code `Hani` for Mandarin; FLORES-plus splits Simplified/Traditional as `Hans`/`Hant`. |
| `yue_Hant` | 0 / 1,012 | `yue_Hani` | Same `Hani` vs `Hant` script-naming split, for Cantonese. |
| `khk_Mong` | 0 / 1,012 | `khk_Cyrl` | Same language code, FLORES-plus devtest text is in traditional Mongolian script (`Mong`); canon has Cyrillic (`Cyrl`) only. |
| `ccp_Cakm` | 62 / 0 | `ccp_Latn` | Same language code (Chakma), different script; canon has a Latin-transliterated form only. |
| `nan_Hani` | 58 / 0 | `nan_Latn` | Same language code (Min Nan Chinese), canon has a Latin-transliterated form only. |
| `fuf_Adlm` | 58 / 0 | `fuf_Latn` | Same language code (Pular/Fulfulde), canon has the Latin form only, not Adlam. |
| `guk_Latn` | 55 / 0 | `guk_Ethi` | Same language code (Gumuz), canon has the Ethiopic form only. |
| `blt_Tavt` | 55 / 0 | `blt_Latn` | Same language code (Tai Dam), canon has the Latin form only, not Tai Viet. |
| `san_Gran` | 51 / 0 | `san_Deva`, `san_Latn` | Same language code (Sanskrit), canon has Devanagari and Latin, not Grantha. |
| `pbu_Arab` | 59 / 0 | `pbt_Arab` | `pbu` (Northern Pashto) vs `pbt` (Southern Pashto): different individual-language codes, canon has only the latter. |
| `tgl_Latn`, `tgl_Tglg` | 60+1 / 0 | `fil_Latn` | Tagalog (`tgl`) vs Filipino (`fil`): closely related, distinct ISO 639-3 codes. Canon has only `fil_Latn`. |

Rows accounted for by this table: UDHR 948 of 3,642 out-of-set rows (18 of 64 labels); FLORES-plus
8,096 of 10,120 out-of-set rows (8 of 10 labels). The remaining UDHR labels (46, 2,694 rows) and
FLORES-plus labels (`apd_Arab`, `acq_Arab`; 2,024 rows) had no plausible near-miss found against the
canonical set and are treated as genuinely outside model coverage, not flagged for a mapping decision.

## Other data-quality notes

Two rows carry a literal tab character embedded inside the source text itself, not introduced by this
processing: UDHR `cof_Latn` (one row) and FLORES-plus `mhr_Cyrl` (one row). Both labels are in the
1,940-label intersection, so both rows are present in the eval TSVs. The text was preserved verbatim;
each TSV line was written with exactly one tab separating the label and text fields, so the two-column
structure holds even though these two lines contain an extra tab byte inside the text field. A
consumer that parses with "split on first tab only" will read both fields correctly; a consumer that
splits on every tab will see 3 fields for these 2 of 237,647 total eval rows.

## Output files written

Eval TSVs (`label<TAB>text`, UTF-8, no header, rows restricted to the exact 1,940-label intersection,
no remapping applied):
- `/capstor/scratch/cscs/cmeister747/unilid_analysis/external_bench/udhr_eval.tsv` (24,115 rows, 366 labels)
- `/capstor/scratch/cscs/cmeister747/unilid_analysis/external_bench/flores_devtest_eval.tsv` (213,532 rows, 205 labels)

Full unfiltered label histograms:
- `/users/cmeister747/unilid_analysis/outputs/diagnostic/external_bench/udhr_label_counts.csv` (430 labels, with `in_1940_set` flag)
- `/users/cmeister747/unilid_analysis/outputs/diagnostic/external_bench/flores_label_counts.csv` (215 labels, with `in_1940_set` flag)

Out-of-set label lists in full:
- `/users/cmeister747/unilid_analysis/outputs/diagnostic/external_bench/udhr_labels_not_in_1940.csv` (64 labels)
- `/users/cmeister747/unilid_analysis/outputs/diagnostic/external_bench/flores_labels_not_in_1940.csv` (10 labels)

This report:
- `/users/cmeister747/unilid_analysis/outputs/tables/external_bench_mapping.md`

No existing repository file was modified. No scoring or SLURM job was run.

## Addendum (2026-08-07): FLORES basis resolved to the original FLORES-200 release

The paper's 190-label FLORES count is reproduced exactly by the original FLORES-200
release, not by flores_plus. Measured: the official FLORES-200 language list (204 codes,
from the `facebookresearch/flores` repository README) intersected with the 1,940-label
set gives exactly 190 labels; the flores_plus devtest intersection gives 205. The
original devtest was therefore downloaded and used to build the FLORES eval file:

- Source: `https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz` (the target of
  the download link in the official README), sha256
  `b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6`, 204 devtest files,
  1,012 lines each.
- Built: `/capstor/scratch/cscs/cmeister747/unilid_analysis/external_bench/flores200_eval.tsv`
  (192,280 rows = 190 labels x 1,012 lines, `label<TAB>text`, exact-match labels, no
  remapping). The 14 excluded labels: acq_Arab, ajp_Arab, aka_Latn, bos_Latn, est_Latn,
  grn_Latn, hrv_Latn, kon_Latn, pes_Arab, prs_Arab, tgl_Latn, yue_Hant, zho_Hans,
  zho_Hant.
- The flores_plus reconstruction (`flores_devtest_eval.tsv`, 205 labels) is retained as
  a diagnostic only and is not proposed for scoring.

Recommendation for the approval checkpoint: score `udhr_eval.tsv` (366 labels, matching
the paper's count exactly) and `flores200_eval.tsv` (190 labels, matching exactly), both
pure exact-match intersections with no remapping decisions. The needs-decision table
above then requires no decision for the acceptance gate; it documents excluded rows.
