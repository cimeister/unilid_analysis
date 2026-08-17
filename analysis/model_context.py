"""Which model a scoring run uses, and where it is allowed to write.

Every script in the evaluation chain resolved its model through
`analysis.transfer_sweep.UNILID_MODEL_PATH` and its output paths through
`analysis.full_test_eval.SCRATCH_DIR`, both module constants with no override.
Pointing any of them at a second model therefore meant editing the constant, and
the directory those constants name is not an ordinary scratch directory: its
entries are symlinks into `/capstor/store/cscs/swissai`, the durable copies that
back every published number. A run against corrected weights would have
overwritten the reference arrays for both release gates and the provenance chain
for every GlotLID-C figure in the paper.

This module is the single place that resolves the pair and refuses the unsafe
combinations. The rule it enforces:

    a model that is not the default may not write into the default output root,
    nor into any root that is backed by the durable store.

Failing loudly is the point. A run that silently scores the wrong weights, or
writes a second model's predictions over the first's, produces numbers that look
valid and answer a different question, which is worse than a run that refuses to
start.

Use it as:

    ctx = resolve(model_path=args.model, scratch_dir=args.scratch_dir,
                  purpose="floor-21 full-pool scoring")
    W = load(ctx.model_path)
    out = os.path.join(ctx.scratch_dir, "pred_floor21.npy")
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import List, Optional

# The durable store. Anything whose real path is under this is a published
# artifact, not scratch, regardless of the path used to reach it.
STORE_ROOT = "/capstor/store/cscs/swissai"

# The released model and the output root holding its memmaps. Imported lazily in
# _defaults() so that importing this module does not pull in the whole analysis
# package (transfer_sweep loads config, which reads the dataset paths).
_DEFAULT_SCRATCH = "/capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval"


class UnsafeModelContext(RuntimeError):
    """A model and an output root that must not be combined."""


def _defaults():
    from analysis.transfer_sweep import UNILID_MODEL_PATH
    return UNILID_MODEL_PATH, _DEFAULT_SCRATCH


def default_model_path() -> str:
    return _defaults()[0]


def default_scratch_dir() -> str:
    return _DEFAULT_SCRATCH


def model_sha256(model_path: str) -> str:
    """Hash of the model file a run's outputs were produced from."""
    h = hashlib.sha256()
    with open(model_path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def store_backed_entries(directory: str) -> List[str]:
    """Entries of ``directory`` whose real path lands in the durable store, plus
    the directory itself when it is a symlink into the store (which is how
    `analysis/mistralnemo_eval.py`'s root is arranged)."""
    backed = []
    if not os.path.exists(directory):
        return backed
    if os.path.realpath(directory).startswith(STORE_ROOT):
        backed.append(".")
    try:
        names = sorted(os.listdir(directory))
    except NotADirectoryError:
        return backed
    for name in names:
        if os.path.realpath(os.path.join(directory, name)).startswith(STORE_ROOT):
            backed.append(name)
    return backed


@dataclass(frozen=True)
class ModelContext:
    """A resolved (model, output root) pair that has passed the safety check."""

    model_path: str
    scratch_dir: str
    is_default_model: bool
    default_model_path: str
    default_scratch_dir: str

    def sha256(self) -> str:
        return model_sha256(self.model_path)

    def describe(self) -> str:
        which = "the released model" if self.is_default_model else "a NON-DEFAULT model"
        return (f"{which}\n  model   {self.model_path}\n"
                f"  outputs {self.scratch_dir}")


def resolve(model_path: Optional[str] = None,
            scratch_dir: Optional[str] = None,
            *,
            default_scratch: Optional[str] = None,
            purpose: str = "") -> ModelContext:
    """Resolve the model and output root, refusing the unsafe combinations.

    ``default_scratch`` lets a script that owns a different output root (the
    Mistral-Nemo chain) keep its own default while getting the same guard.
    """
    default_model, module_default_scratch = _defaults()
    default_scratch = default_scratch or module_default_scratch

    model_path = os.path.abspath(model_path or default_model)
    scratch_dir = os.path.abspath(scratch_dir or default_scratch)
    is_default_model = model_path == os.path.abspath(default_model)

    if not os.path.exists(model_path):
        raise UnsafeModelContext(
            f"model file does not exist: {model_path}"
            + (f" (for {purpose})" if purpose else ""))

    if not is_default_model:
        where = f" while {purpose}" if purpose else ""
        if scratch_dir == os.path.abspath(default_scratch):
            raise UnsafeModelContext(
                f"refusing to write {model_path} results into the default output "
                f"root{where}.\n"
                f"  root: {scratch_dir}\n"
                f"That root holds the released model's memmaps, and its entries are "
                f"symlinks into {STORE_ROOT}, which backs every published number and "
                f"both release gates' reference arrays. Pass an explicit "
                f"scratch_dir/--scratch-dir pointing at a fresh directory.")
        backed = store_backed_entries(scratch_dir)
        if backed:
            shown = ", ".join(backed[:6]) + ("..." if len(backed) > 6 else "")
            raise UnsafeModelContext(
                f"refusing to write {model_path} results into a store-backed output "
                f"root{where}.\n"
                f"  root: {scratch_dir}\n"
                f"  store-backed entries: {shown}\n"
                f"Writing there would overwrite published artifacts through the "
                f"symlinks. Pass a fresh directory instead.")

    return ModelContext(model_path=model_path, scratch_dir=scratch_dir,
                        is_default_model=is_default_model,
                        default_model_path=os.path.abspath(default_model),
                        default_scratch_dir=os.path.abspath(default_scratch))


def require_default_model(ctx: ModelContext, what: str) -> None:
    """Guard for a script that reads an artifact recorded from the released model
    and has no way to regenerate it for another one."""
    if not ctx.is_default_model:
        raise UnsafeModelContext(
            f"{what} was recorded from the released model and cannot be paired "
            f"with {ctx.model_path}. Regenerate it for that model first, or run "
            f"this script against the released model.")


def add_arguments(parser, *, default_scratch: Optional[str] = None):
    """The two flags, spelled the same way everywhere in the chain."""
    parser.add_argument(
        "--model", dest="model_path", default=None,
        help="path to the .unilid model (default: the released model)")
    parser.add_argument(
        "--scratch-dir", dest="scratch_dir", default=default_scratch,
        help="output root; required when --model is not the released model")
    return parser
