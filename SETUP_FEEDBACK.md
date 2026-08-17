# UNILID setup: second pass, on PR #3

Context: a follow-up to an earlier onboarding pass on the `release` branch,
this time against `cimeister/calibration-release` (PR
[#3](https://github.com/Ahmetcanyvz/UNILID/pull/3)), which is a maintainer
response to that earlier feedback. Same environment as before (clean macOS
checkout, fresh venv), same task (install, add a language, test it), run
independently rather than by trusting the PR description. All five prior
issues were re-tested from scratch; four are fixed outright, one is fixed as
documentation. One new, minor, cosmetic issue turned up.

## Previously reported issues: verified status

1. **Submodules silently absent.** `doctor.py` (new in this PR) now checks
   `sentencepiece`/`tokenizers` checkout state explicitly and prints
   `git submodule update --init --recursive` as the fix before any build is
   attempted, rather than letting a missing submodule surface later as an
   unrelated Cargo error. Verified: running `python doctor.py` against a repo
   with only the top-level clone (no `.venv`, submodules present since `gh pr
   checkout` fetches them, but before Rust/maturin were on PATH) correctly
   named each missing piece one at a time as it was resolved. **Fixed.**

2. **No minimum Rust version / misleading nightly-flag error.**
   `tokenizers/tokenizers/rust-toolchain` and
   `tokenizers/bindings/python/rust-toolchain` now pin `1.93.1`, so `rustup`
   fetches a known-good toolchain automatically instead of using whatever
   `stable` happens to be installed (which is what produced the confusing
   `-Z weak-dep-features` error last time). `doctor.py` also reports the
   resolved `rustc` version directly. **Fixed.**

3. **No upper bound on Python, tested only to 3.12.** `pyproject.toml`
   classifiers now list 3.9 through 3.14. Installed directly onto the system
   Python here (3.14.5, no venv pin needed this time) with
   `pip install -e ".[dev]"`: no warnings, no errors. **Fixed** (confirmed by
   actually using 3.14 rather than sidestepping it as before).

4. **`pytest` failing out of the box on the minimal `[dev]` install.** Ran
   `python -m pytest tests/` right after `pip install -e ".[dev]"` and
   `maturin develop --release`, no SentencePiece anywhere: **102 passed, 2
   skipped**, zero errors. The two `spm_train`-dependent tests now skip with
   an explicit reason instead of erroring. Also checked the specific failure
   mode from before — running `unilid-add-language ... --method sp` without
   `spm_train` on PATH now raises a `RuntimeError` naming the missing binary
   and the fix (`method='sp' ... needs the spm_train executable ... build it
   (see README) or pass method='em'`), rather than a bare
   `FileNotFoundError: spm_train`. The PR description attributes this to two
   underlying bugs (a `spm is None` check that couldn't distinguish "package
   missing" from "package shadowed by the sentencepiece submodule directory
   as a namespace package" at the repo root, and a base/per-language training
   method conflation causing `spm_train` to be reached even under
   `--method em`); I didn't inspect the diff, but the observable behavior
   matches what the fix claims. **Fixed.**

5. **Add-language toy vocabulary caveat undocumented.** Both `README.md`
   ("Add your own language") and `examples/add_language/README.md` now state
   up front that the 300-token toy vocabulary doesn't cover real-world text
   and point at the released 1,940-language model as the realistic starting
   point. **Fixed** (as a documentation caveat, which was the requested fix).

## Re-running the concrete test: adding a real language

Repeated the same experiment as before — extracting real Python source lines
from this repository (300 train / 50 held-out test lines, deterministic
split) and adding them to the toy calibrated model as a new language
`python_Code` via `unilid-add-language ... --method em` — end to end on the
PR #3 code, without reusing any prior artifacts.

Result: **held-out accuracy 0.98** (up from 0.68 on the same experiment
against the pre-PR code), with the three base toy languages unaffected
(1.00). The EM trainer still reports a ~21.5% UNK rate against the 300-token
vocabulary (consistent with the now-documented caveat that a toy vocabulary
doesn't cover real text), but overall accuracy improved substantially. This
is very likely a side effect of the PR's `base_em_mode`/`use_sp_seed_vocab`
split: `run_example.sh` no longer trains the *base* vocabulary with the same
narrow soft-EM path used for the toy per-language distributions, and now
defaults to HuggingFace's `UnigramTrainer` for the base step (documented in
the updated `examples/add_language/README.md`, step 2). One live sanity
check still misfires as expected given the narrow vocabulary: a plain English
sentence ("The quick brown fox...") is classified as `python_Code` rather
than as any base language, which is consistent with the documented caveat
that out-of-vocabulary text tends to fall toward whatever language's
distribution is least peaked, not a new problem.

## New finding: the default `--method sp` path degrades much further than documented when the base vocabulary is narrow and the new language is real text

`unilid-add-language` defaults to `--method sp` (`unilid/add_language.py:299`,
`default="sp"`), and the code's own error message calls it "the
release-verified training path"; `--method em` is the explicitly-unverified
fallback used only when `spm_train` is unavailable. Everything reported above
used `--method em`, so it exercised the non-default, non-release-verified
path only. Verifying the actual default required building the optional
SentencePiece CLI, which needs `cmake` (not present in the base environment;
only `g++`/`clang++` were). Exact steps, run from the repo root with the
venv already active from the base install above:

```bash
brew install cmake
git submodule update --init sentencepiece
cd sentencepiece && mkdir -p build && cd build
cmake ..
make -j"$(sysctl -n hw.ncpu)"
cd ../..
export PATH="$(pwd)/sentencepiece/build/src:$PATH"   # spm_train on PATH, no sudo install
pip install -e ".[train]"                             # sentencepiece pip package
python doctor.py                                       # now reports "ready" fully green,
                                                         # sentencepiece: "binary and Python
                                                         # package both present"
python -m pytest tests/ -q                             # 104 passed (the 2 that skip
                                                         # without spm_train now run)
```

With that in place, the same `python_Code` addition (300 real Python source
lines from this repository, same held-out 50-line test split) was re-run
with the default method:

```bash
cd examples/add_language
unilid-add-language work/toy_calibrated.unilid python_Code \
    work/python_Code_train.txt -o work/toy_extended_python_sp.unilid
    # no --method flag: this is the documented top-level usage from the
    # README's "Add your own language" section, and resolves to sp
```

Result, compared to the `em` run already reported above:

| method | held-out accuracy on `python_Code` | calibration outcome |
|---|---|---|
| `em` (non-default, "not verified against the release's end-to-end chain") | 0.98 | normal (281/300 own-won calibration lines) |
| `sp` (**default**, "the release-verified training path") | **0.14** | **excluded**: only 39/300 own-won calibration lines, `cause=low_calibration`, printed as `'python_Code' will never be re-examined` |

The three base toy languages (`aaa_Latn`, `bbb_Latn`, `ccc_Latn`) stayed at
1.00 accuracy under both methods — existing languages are not put at risk
either way. The gap is specific to the newly added language's own quality.

`examples/add_language/README.md` already documents that `sp` and `em`
diverge at toy data sizes (0.60 vs 0.98, for the constructed `ddd_Latn`
case), and separately documents that the toy vocabulary doesn't generalize
to real text. What isn't documented is that the two problems compound: with
a real-text language over the same narrow toy vocabulary, the default `sp`
method didn't just score lower (0.60-ish, as the existing caveat might
imply) — it scored 0.14 and got permanently excluded from re-examination.
That's a materially different failure mode than "somewhat less accurate,"
and it's the one a user gets by default, silently, by following the
README's own top-level example command (which passes no `--method` flag and
therefore doesn't route through `em` at all). Because `sp` is also the
release-verified path, this experiment doesn't validate whether the same
compounding happens against the full 1,940-language released vocabulary
(where real-world byte coverage should be much better) — the toy vocabulary
here is a known confound, and I don't have the released model on hand to
retest against it.

**Suggested fix:** extend the existing "two training methods differ at toy
data sizes" caveat in `examples/add_language/README.md` (and/or the
`add_language()` docstring / README section) to explicitly call out that the
combination of a narrow/mismatched vocabulary and the default `sp` method
can produce a permanently-excluded (`low_calibration`) language rather than
just lower accuracy, since that's a stronger and more surprising outcome
than "flatter distribution." If practical, it would also be worth stating
whether this compounding is toy-vocabulary-specific or something that can
also occur against the released model with a poorly-matched new language,
since that's the scenario an actual user of `unilid-add-language` is in.

## One new issue found (minor, cosmetic)

**`examples/add_language/run_example.sh` prints a repeated `RuntimeWarning`
that doesn't appear when using the package normally.** The script sets
`PYTHONPATH` to the repo root (with a comment noting this is "redundant"
after `pip install -e .`) and then runs `python -m unilid.add_language`.
When the package actually is pip-installed — i.e. exactly the state the
README's install steps leave you in — this produces the following on every
invocation, repeated many times over the run (once per EM iteration, so 5+
times per language added):

```
<frozen runpy>:130: RuntimeWarning: 'unilid.add_language' found in
sys.modules after import of package 'unilid', but prior to execution of
'unilid.add_language'; this may result in unpredictable behaviour
```

It's cosmetic — the run completes correctly and results are unaffected (I
confirmed this by calling the installed `unilid-add-language` console script
directly on the same inputs: zero warnings, identical output) — but it's
noisy stderr output on the officially documented "run this to see the
feature work" script, which could read as a red flag to a new user even
though nothing is actually wrong.

**Suggested fix:** in `run_example.sh`, only set `PYTHONPATH` when `unilid`
isn't already importable (e.g. `python -c "import unilid" 2>/dev/null ||
export PYTHONPATH=...`), or switch the script to invoke the installed
`unilid-add-language` console script directly instead of
`python -m unilid.add_language`, consistent with the comment already present
in the script noting the console-script form as the normal usage.

## Summary

Four of five previously reported issues are fixed in code; the fifth is
fixed as a documentation caveat, which was the appropriate fix for that one.
The new `doctor.py` script and the CI workflow (`.github/workflows/ci.yml`,
which deliberately runs the documented minimal install across Python 3.9–3.14
without building SentencePiece) look like they should prevent regressions on
all of these going forward.

Two new things turned up in this pass: one cosmetic (the `run_example.sh`
warning spam), and one substantive — the default `--method sp` add-language
path, run against a real-text language over a narrow vocabulary, produced a
permanently-excluded, 0.14-accuracy result versus 0.98 for the non-default,
non-release-verified `em` method on identical data. That's a documentation
gap (the existing caveat undersells how bad the default path's failure mode
can be) rather than a code defect, but it's the one a user hits by default
and without any indication something went wrong beyond a quieter log line.

