"""
Processor import shim — skip the Qwen3.5 / Qwen3-VL video sub-processor so
we don't have to install torch + torchvision just to load the tokenizer and
image processor.

Why this exists
---------------
`transformers.AutoProcessor.from_pretrained("mlx-community/Qwen3.5-*")`
internally calls `ProcessorMixin._get_arguments_from_pretrained`, which
iterates over the class' `get_attributes()` list. For Qwen3-VL-style
processors that list is `["image_processor", "tokenizer", "video_processor"]`.
Loading the primary `video_processor` ends up calling
`AutoVideoProcessor.from_pretrained`, which resolves
`Qwen3VLVideoProcessor` — a class whose `__getattribute__` is guarded by
`requires_backends(("torch", "torchvision"))`. Merely touching
`.from_pretrained` on it raises::

    ImportError: Qwen3VLVideoProcessor requires the Torchvision library ...

Our inference stack is 100% MLX; torch is intentionally NOT in the
dependency tree (see `pyproject.toml`). The video processor is only used
for video inputs, which this server does not accept — image + text only.

What the shim does
------------------
1. Patches `ProcessorMixin.get_attributes` to strip `video_processor` from
   the attribute list that `_get_arguments_from_pretrained` iterates over.
   Effect: the video sub-processor is never loaded, and `image_processor`
   + `tokenizer` sit at indices 0 and 1 of the returned args tuple, which
   matches the first two positional params of `Qwen3VLProcessor.__init__`.

2. Patches `ProcessorMixin.from_args_and_dict` to drop `video_processor`
   from `processor_dict` before the normal flow. Without this,
   `validate_init_kwargs` would see the raw `video_processor` sub-config
   dict stored inside `processor_config.json` and forward it to the
   constructor as `video_processor=<dict>`, giving us a broken stub.

Both patches are idempotent and safe to call multiple times.

Call `install_no_video_processor_shim()` exactly once **before**
`omlx.compat.vlm.load(...)` runs (which transitively calls
`AutoProcessor.from_pretrained`).
"""

from __future__ import annotations

_INSTALLED = False


def install_no_video_processor_shim() -> None:
    """Install the ProcessorMixin patches described in the module docstring."""
    global _INSTALLED
    if _INSTALLED:
        return

    from transformers.processing_utils import ProcessorMixin

    # ---- Patch 1: get_attributes ----
    _orig_get_attrs_fn = ProcessorMixin.get_attributes.__func__  # underlying function

    @classmethod
    def _patched_get_attrs(cls):
        return [a for a in _orig_get_attrs_fn(cls) if a != "video_processor"]

    ProcessorMixin.get_attributes = _patched_get_attrs

    # ---- Patch 2: from_args_and_dict ----
    _orig_from_args_and_dict_fn = ProcessorMixin.from_args_and_dict.__func__

    @classmethod
    def _patched_from_args_and_dict(cls, args, processor_dict, **kwargs):
        if isinstance(processor_dict, dict) and "video_processor" in processor_dict:
            processor_dict = {
                k: v for k, v in processor_dict.items() if k != "video_processor"
            }
        return _orig_from_args_and_dict_fn(cls, args, processor_dict, **kwargs)

    ProcessorMixin.from_args_and_dict = _patched_from_args_and_dict

    _INSTALLED = True
