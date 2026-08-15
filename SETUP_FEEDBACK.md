# UNILID setup: friction points for new users

Context: followed the README's Installation and "Add your own language" sections
on a clean macOS checkout (fresh clone, fresh Python venv, system-installed but
old Rust toolchain) to get the package running and try `unilid-add-language`.
The install ultimately succeeded and the add-language walkthrough and test
suite both work, but five points cost significant back-and-forth versus what
the README implies is a straight-line setup. Listed roughly in the order a new
user would hit them, with a suggested fix for each.

## 1. Submodules silently absent, and the failure downstream doesn't say why

`git submodule status` on a fresh clone showed both `sentencepiece` and
`tokenizers` prefixed with `-`, meaning uninitialized — even though the repo
was cloned in a way that should have pulled them. Nothing at clone time warns
about this; the first sign of trouble is `maturin develop --release` failing
much later with a Cargo manifest parse error that has nothing to do with
submodules on its face.

**Suggested fix:** have `pip install -e .` or the maturin build step check
whether `tokenizers/tokenizers/Cargo.toml` (or similar) exists before
attempting to build, and fail fast with "submodules not initialized — run
`git submodule update --init --recursive`" instead of letting Cargo produce an
unrelated-looking error.

## 2. No minimum Rust version stated, and the failure mode when it's too old is misleading

The README says a Rust toolchain is required ("install via rustup") but gives
no minimum version. With whatever `stable` a system happened to have installed
previously (in this case rustc 1.51.0, from 2021), the build fails with:

```
optional dependency features with `?` syntax are only allowed on the nightly
channel and requires the `-Z weak-dep-features` flag on the command line
```

That error reads like it wants a nightly toolchain flag, not "your stable
toolchain is four years old." The actual fix was `rustup update stable`
(→ 1.97.1), which resolved it immediately.

**Suggested fix:** state a minimum Rust version in the README (or better, ship
a `rust-toolchain.toml` in `tokenizers/` pinning a known-good version so
`rustup` fetches it automatically), and/or catch this class of Cargo error in
the install instructions with a one-line "if you see a weak-dep-features
error, run `rustup update stable`."

## 3. No upper bound on supported Python, so pip installs happily onto untested versions

`pyproject.toml` sets `requires-python = ">=3.9"` with no ceiling, but the
`Programming Language :: Python :: 3.x` classifiers stop at 3.12. The system
default Python here was 3.14.5; pip installed the package onto it without any
warning. It's unclear whether 3.13/3.14 are actually supported — I sidestepped
the question by installing into a 3.12 venv instead, but a user who doesn't
think to check classifiers against their interpreter version has no signal
that they're outside the tested range until something breaks non-obviously.

**Suggested fix:** either extend CI/testing to current Python versions and
update the classifiers, or cap `requires-python` (e.g. `>=3.9,<3.13`) so an
unsupported interpreter fails at install time with a clear message rather than
silently.

## 4. `pytest` fails out of the box on the documented minimal dev install

Following the README exactly — `pip install -e ".[dev]"` (no `[train]`, no
SentencePiece CLI build, since both are documented as optional) — and then
running `python -m pytest tests/` produces two failures:

```
tests/test_add_language_integration.py::test_add_language_em_end_to_end
tests/test_add_language_integration.py::test_add_language_em_provenance_and_input_untouched
FileNotFoundError: [Errno 2] No such file or directory: 'spm_train'
```

Both tests have `_em_` in their names, which reads as "tests the EM path" (the
one that needs no SentencePiece binary per the README), but they apparently
also exercise or depend on the `sp` path internally and error rather than
skip. A new user running the test suite right after a stock install has no way
to tell these are expected failures from missing an optional component versus
real bugs — the README's "Optional extras" section doesn't mention that
`pytest` needs the compiled `spm_train` binary for a clean pass.

**Suggested fix:** mark the SentencePiece-dependent assertions/tests with
`pytest.mark.skipif(not shutil.which("spm_train"), reason="needs the compiled "
"SentencePiece CLI, see Installation")` so `pytest` passes cleanly on the
documented minimal install, and note in the README that a full-green test run
needs the `sp` binary built.

## 5. The add-language worked example's toy vocabulary is too narrow to show realistic behavior on non-toy text

Not a bug, but worth a documentation caveat: `examples/add_language/`'s base
model has a vocabulary of only 300 byte-level tokens, learned from three
constructed languages whose alphabets cover roughly two dozen byte values.
That's fine for the example's own held-out data (0.98 accuracy adding a
fourth constructed language), but if a user follows the same recipe to add a
language built from real-world text (e.g. natural-language sentences or
source code, both of which use a much wider byte range), EM training reports
a large UNK rate (~23% in my test) and held-out accuracy drops sharply (0.68
in my test), with some real text misclassified because neither it nor the toy
alphabet is well represented in the tiny shared vocabulary.

This is expected given the toy example's scale, but the README/example docs
don't currently flag that the demo's accuracy numbers don't generalize to
non-toy vocabularies — a user adapting the example to their own (non-toy)
first language could reasonably read the 0.98 figure as representative and be
surprised.

**Suggested fix:** add a line to `examples/add_language/README.md` noting that
the toy base vocabulary is intentionally tiny and narrow for speed, and that
accuracy when adding a language built from real-world (non-toy-alphabet) text
will be substantially lower unless the base model's vocabulary already covers
that text's byte/character range — e.g. point users toward the released
1,940-language model as the realistic starting point for real languages.

## Summary of concrete asks

- Fail fast (with a clear message) if submodules aren't initialized, rather
  than surfacing a Cargo error later.
- Document a minimum Rust version, or pin one via `rust-toolchain.toml`.
- Either support and test current Python versions, or cap `requires-python`
  so installing on an unsupported interpreter is an explicit, early error.
- Make `pytest` pass cleanly on the documented minimal (`[dev]`-only) install
  by skipping SentencePiece-dependent tests when `spm_train` isn't present.
- Add a short caveat to the add-language example about its toy vocabulary not
  generalizing to real-world text.

