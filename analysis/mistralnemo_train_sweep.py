"""Resumable, per-language fault-isolated training driver for the Mistral-Nemo
retrain (task 3, "per-language training driver").

Why this exists instead of calling UNILID/train.py directly (as the Apertus
131k/200k fp64 retrains did): train.py's own per-language loop batches
languages (--lang-batch-size, 20 for both Apertus retrains) and wraps each
batch's spm_train calls in a try/except that logs and RE-RAISES
(UNILID/train.py, the `except subprocess.CalledProcessError` block). A single
language's spm_train failure inside a batch therefore aborts the whole
process: every language after the failing one, in that batch and every later
batch, never gets attempted, and re-running only resumes with
--skip-existing-langs, hitting the same failing language again. Under the
fp64 trainer this matters more than it used to: commit c5921a2 turned the
non-finite-expected-counts guard into a hard failure instead of a silent
zero-fill, so a numerical edge case that used to silently corrupt a row (the
azj_Latn collapse, Exp 41) now crashes spm_train instead. This driver isolates
each language's training so one crash does not stop the other 1,939.

Design: rather than shelling out to `train.py` once per language (1,940
separate Python process starts, each re-importing torch/numpy/tokenizers),
this driver stays in a single long-running process and calls
`LanguageSpecificUnigramLMTokenizer.train()` directly, once per language, with
a one-entry corpus_info dict. This is the exact same per-language code path
train.py's batched loop uses (train_with_sentencepiece_direct, which shells
out to spm_train exactly as it already did per-language inside a batch); the
only behavioral difference from train.py is that a fresh
LanguageSpecificUnigramLMTokenizer object is constructed per language instead
of per batch-of-20, and each language's exception is caught, logged, and does
not stop the sweep.

Resumability: before training, every language whose expected output file
already exists and is non-empty is skipped (same semantics as train.py's
--skip-existing-langs, checked here at per-language granularity so a
resumed run after a partial failure does not redo completed languages).

Failure handling: a language that raises is logged (driver log + a dedicated
per-language log file under LOG_DIR), recorded in the failed list, and the
sweep continues. At the end, if any language failed, the failure list is
written to FAILED_LANGS_PATH and the process exits with status 1 so a SLURM
job (or the caller of this script) sees a nonzero exit code and does not
proceed to packing an incomplete model.

Prerequisite: mistralnemo_export_vocab.py must have already produced
BASE_TOKENIZER_PATH. This script aborts loudly if it has not.
"""
from __future__ import annotations

import gc
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/users/cmeister747/unilid_analysis/UNILID")

from analysis.mistralnemo_constants import (
    BASE_TOKENIZER_PATH,
    CORPUS_DIR,
    EXPECTED_LANGS,
    EXPECTED_TOKENIZER_SHA256,
    EXPECTED_VOCAB,
    FAILED_LANGS_PATH,
    LOG_DIR,
    PER_LANG_TOKENIZER_TAG,
    SNAPSHOT_HASH,
    SWEEP_SUMMARY_PATH,
    TOKENIZER_JSON,
    TOKENIZERS_DIR,
    ensure_results_dirs,
)
from unilid.trainers.language_specific_trainer import LanguageSpecificUnigramLMTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("mistralnemo_train_sweep")


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError) as e:
        raise RuntimeError(f"git rev-parse HEAD failed: {e}") from e


def _expected_output_path(lang: str) -> str:
    return os.path.join(
        TOKENIZERS_DIR, f"langspec_{PER_LANG_TOKENIZER_TAG}_{lang}.tokenizer.json"
    )


def _discover_languages() -> list[str]:
    if not os.path.isdir(CORPUS_DIR):
        raise FileNotFoundError(f"corpus split missing: {CORPUS_DIR}")
    files = sorted(f for f in os.listdir(CORPUS_DIR) if f.endswith("_train.txt"))
    if len(files) != EXPECTED_LANGS:
        raise RuntimeError(
            f"corpus split has {len(files)} language files under {CORPUS_DIR}, "
            f"expected {EXPECTED_LANGS}"
        )
    return [f[: -len("_train.txt")] for f in files]


def _per_lang_log_handler(lang: str) -> logging.FileHandler:
    path = os.path.join(LOG_DIR, f"{lang}.log")
    handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    )
    handler.setLevel(logging.INFO)
    return handler


def main() -> None:
    if not os.path.exists(BASE_TOKENIZER_PATH):
        raise FileNotFoundError(
            f"base tokenizer not found at {BASE_TOKENIZER_PATH}; run "
            f"mistralnemo_export_vocab.py first"
        )
    ensure_results_dirs()

    languages = _discover_languages()
    corpus_files = {l: os.path.join(CORPUS_DIR, f"{l}_train.txt") for l in languages}

    n_total = len(languages)
    already_done = [l for l in languages
                     if os.path.isfile(_expected_output_path(l))
                     and os.path.getsize(_expected_output_path(l)) > 0]
    to_train = [l for l in languages if l not in set(already_done)]

    logger.info(
        "%d languages total; %d already trained (resuming), %d to train",
        n_total, len(already_done), len(to_train),
    )

    root_logger = logging.getLogger()
    trained_this_run: list[str] = []
    failed: list[dict] = []

    for i, lang in enumerate(to_train, start=1):
        t0 = time.time()
        handler = _per_lang_log_handler(lang)
        root_logger.addHandler(handler)
        try:
            logger.info("[%d/%d] training %s", i, len(to_train), lang)
            tok = LanguageSpecificUnigramLMTokenizer(
                vocab_size=EXPECTED_VOCAB,
                reestimation_em_mode=PER_LANG_TOKENIZER_TAG,
                byte_level=True,
            )
            tok.train(
                corpus_info={lang: corpus_files[lang]},
                base_tokenizer_path=BASE_TOKENIZER_PATH,
                output_dir=TOKENIZERS_DIR,
                tokenizer_path_format=None,
                load_base_if_exists=True,
                load_tokenizers_if_exists=True,
                use_sentencepiece=True,
            )
            out_path = _expected_output_path(lang)
            if not (os.path.isfile(out_path) and os.path.getsize(out_path) > 0):
                raise RuntimeError(
                    f"train() returned without writing a non-empty output at "
                    f"{out_path}"
                )
            elapsed = time.time() - t0
            logger.info("[%d/%d] %s OK in %.1fs", i, len(to_train), lang, elapsed)
            trained_this_run.append(lang)
            del tok
        except Exception as e:  # noqa: BLE001 -- deliberately broad: any failure
                                 # for this language must not stop the sweep
            elapsed = time.time() - t0
            logger.error(
                "[%d/%d] %s FAILED after %.1fs: %s",
                i, len(to_train), lang, elapsed, e,
            )
            logger.error(traceback.format_exc())
            failed.append({
                "lang": lang,
                "error": str(e),
                "elapsed_seconds": round(elapsed, 1),
            })
        finally:
            root_logger.removeHandler(handler)
            handler.close()
            gc.collect()

    n_trained = len(trained_this_run)
    n_failed = len(failed)
    logger.info(
        "SWEEP DONE: %d total, %d already done before this run, %d trained "
        "this run, %d failed",
        n_total, len(already_done), n_trained, n_failed,
    )

    summary = {
        "step": "mistralnemo_train_sweep",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "source_tokenizer": {
            "hf_snapshot_hash": SNAPSHOT_HASH,
            "tokenizer_json_path": TOKENIZER_JSON,
            "tokenizer_json_sha256": EXPECTED_TOKENIZER_SHA256,
        },
        "n_languages_total": n_total,
        "n_already_done_before_this_run": len(already_done),
        "n_trained_this_run": n_trained,
        "n_failed_this_run": n_failed,
        "trained_this_run": trained_this_run,
        "failed": failed,
    }
    with open(SWEEP_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("wrote sweep summary: %s", SWEEP_SUMMARY_PATH)

    if failed:
        with open(FAILED_LANGS_PATH, "w", encoding="utf-8") as f:
            for item in failed:
                f.write(f"{item['lang']}\t{item['error']}\n")
        logger.error(
            "%d language(s) failed: %s. See %s and per-language logs under %s",
            n_failed, ", ".join(x["lang"] for x in failed), FAILED_LANGS_PATH,
            LOG_DIR,
        )
        sys.exit(1)

    remaining = n_total - (len(already_done) + n_trained)
    if remaining != 0:
        raise RuntimeError(
            f"accounting mismatch: {remaining} languages neither already-done, "
            f"trained, nor failed; this is a bug in the sweep driver"
        )

    print("\nALL LANGUAGES TRAINED")


if __name__ == "__main__":
    main()
