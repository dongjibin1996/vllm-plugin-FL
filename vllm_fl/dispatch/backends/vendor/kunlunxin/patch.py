# Copyright (c) 2026 BAAI. All rights reserved.

# Environment variables:
#   VLLM_KUNLUNXIN_BYPASS_GDN
#       Set to "1" to bypass the GDN fused post-conv Triton kernel on
#       Kunlunxin XPU. Only the faulting GDN kernel is routed to a PyTorch
#       fallback; all other Triton operators are left untouched.
#       Set to "0"/"false"/"no" to disable the bypass.
#   VLLM_FL_PLATFORM=kunlunxin
#       Compatible fallback marker. If VLLM_KUNLUNXIN_BYPASS_GDN is not set,
#       this value is used to recognize the Kunlunxin platform and apply the
#       same bypass. Explicitly setting VLLM_KUNLUNXIN_BYPASS_GDN is preferred
#       for auditing purposes.
#
# Usage:
#   export VLLM_KUNLUNXIN_BYPASS_GDN=1
#   vllm serve /path/to/model --enforce-eager --port 8199

import logging

logger = logging.getLogger(__name__)
_patches_applied = False


def _kunlunxin_bypass_enabled() -> bool:
    """Whether the Kunlunxin GDN bypass is active.

    Explicit ``VLLM_KUNLUNXIN_BYPASS_GDN`` wins; otherwise fall back to the
    platform marker ``VLLM_FL_PLATFORM=kunlunxin``.
    """
    import os

    value = os.environ.get("VLLM_KUNLUNXIN_BYPASS_GDN")
    if value is None:
        return os.environ.get("VLLM_FL_PLATFORM", "").strip().lower() == "kunlunxin"
    return value.strip().lower() not in ("0", "false", "no")


def apply_kunlunxin_patches():
    """Apply all Kunlunxin-specific patches."""
    global _patches_applied
    if _patches_applied:
        return
    if not _kunlunxin_bypass_enabled():
        logger.debug(
            "VLLM_KUNLUNXIN_BYPASS_GDN not enabled; skipping Kunlunxin GDN "
            "bypass patches"
        )
        return
    _patches_applied = True
    patch_gdn_warmup()
    patch_gdn_triton_ops()


def patch_gdn_warmup():
    """Disable GDN prefill kernel warmup on Kunlunxin XPU.

    ``GatedDeltaNetAttention._warmup_prefill_kernels`` profiles the GDN
    prefill Triton kernels (fused_post_conv_prep / chunk_gated_delta_rule).
    On FlagTree 3.6 + Kunlunxin XPU this compilation can hang at
    TritonXPULegalizePass / sortOpTreeBwd, so the warmup is skipped entirely.
    """
    try:
        import vllm.model_executor.layers.mamba.gdn_linear_attn as gdn_lib

        def _noop_warmup(self, mixed_qkv):
            logger.info(
                "Kunlunxin: bypassed GDN prefill Triton warmup for %s",
                getattr(self, "prefix", ""),
            )

        gdn_lib.GatedDeltaNetAttention._warmup_prefill_kernels = _noop_warmup
        logger.info("Kunlunxin: disabled GDN prefill kernel warmup")
    except Exception as e:
        logger.warning("Kunlunxin: failed to patch GDN warmup: %s", e)


def patch_gdn_triton_ops():
    """Replace the GDN fused post-conv Triton kernel with a PyTorch fallback.

    ``fused_post_conv_prep`` is the entry point that triggers compilation of
    ``_fused_post_conv_kernel`` (@triton.jit), which hangs on Kunlunxin XPU.
    Rebinding it to the pure-PyTorch implementation keeps the operator
    semantics identical without invoking the Triton compiler.
    """
    try:
        from .impl.fla.gdn_torch_ops import fused_post_conv_prep_torch

        import vllm.model_executor.layers.fla.ops.fused_gdn_prefill_post_conv as _fpc_lib
        import vllm.model_executor.layers.fla.ops as _fla_ops_lib
        import vllm.model_executor.layers.mamba.gdn_linear_attn as _gdn_lib

        _fpc_lib.fused_post_conv_prep = fused_post_conv_prep_torch
        _fla_ops_lib.fused_post_conv_prep = fused_post_conv_prep_torch
        _gdn_lib.fused_post_conv_prep = fused_post_conv_prep_torch

        logger.info(
            "Kunlunxin: using PyTorch fallback for fused_post_conv_prep; "
            "_fused_post_conv_kernel will not be compiled"
        )
    except Exception as e:
        logger.warning("Kunlunxin: failed to patch GDN Triton ops: %s", e)
