"""Exercise the packaged CUDA acceleration kernels on the current GPU."""

from __future__ import annotations

import json
import os
from importlib import metadata

import torch
import triton
import triton.language as tl
from flash_attn import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache


@triton.jit
def _add_kernel(x_ptr, y_ptr, output_ptr, n_elements, block_size: tl.constexpr):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < n_elements
    tl.store(
        output_ptr + offsets,
        tl.load(x_ptr + offsets, mask=mask) + tl.load(y_ptr + offsets, mask=mask),
        mask=mask,
    )


def _triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(x.numel(), meta["block_size"]),)
    _add_kernel[grid](x, y, output, x.numel(), block_size=256)
    return output


def main() -> int:
    if not torch.cuda.is_available():
        print(json.dumps({"skipped": True, "reason": "CUDA is not available"}, indent=2))
        return 0
    torch.manual_seed(1234)
    x = torch.randn(4096, device="cuda")
    y = torch.randn_like(x)
    triton_result = _triton_add(x, y)
    torch.testing.assert_close(triton_result, x + y)

    q = torch.randn((1, 32, 4, 64), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    dense = flash_attn_func(q, k, v, causal=True)
    cu_seqlens = torch.tensor([0, q.shape[1]], device="cuda", dtype=torch.int32)
    varlen = flash_attn_varlen_func(
        q[0], k[0], v[0], cu_seqlens, cu_seqlens, q.shape[1], q.shape[1], causal=True
    )
    cache_q = torch.randn((1, 1, 4, 64), device="cuda", dtype=torch.bfloat16)
    k_cache = torch.randn((1, 32, 4, 64), device="cuda", dtype=torch.bfloat16)
    cache = flash_attn_with_kvcache(
        cache_q,
        k_cache,
        torch.randn_like(k_cache),
        cache_seqlens=torch.tensor([32], device="cuda", dtype=torch.int32),
        causal=True,
    )
    for output in (dense, varlen, cache):
        assert torch.isfinite(output).all()

    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from liger_kernel.transformers import LigerRMSNorm

    fla_q = torch.randn((1, 64, 4, 32), device="cuda", dtype=torch.bfloat16)
    fla_k = torch.nn.functional.normalize(torch.randn_like(fla_q), p=2, dim=-1)
    fla_v = torch.randn_like(fla_q)
    fla_g = torch.nn.functional.logsigmoid(
        torch.randn((1, 64, 4), device="cuda", dtype=torch.bfloat16)
    )
    fla_beta = torch.sigmoid(torch.randn_like(fla_g))
    fla_output, _ = chunk_gated_delta_rule(fla_q, fla_k, fla_v, fla_g, fla_beta)
    assert torch.isfinite(fla_output).all()

    liger_norm = LigerRMSNorm(32, eps=1e-6).to("cuda", dtype=torch.bfloat16)
    liger_output = liger_norm(torch.randn((2, 16, 32), device="cuda", dtype=torch.bfloat16))
    assert torch.isfinite(liger_output).all()

    os.environ.setdefault("DS_BUILD_OPS", "0")
    import deepspeed

    ds_model = torch.nn.Linear(64, 64, bias=False).to("cuda", dtype=torch.bfloat16).eval()
    ds_engine = deepspeed.init_inference(
        ds_model, config={"dtype": torch.bfloat16, "replace_with_kernel_inject": False}
    )
    ds_output = ds_engine(torch.randn((2, 64), device="cuda", dtype=torch.bfloat16))
    assert torch.isfinite(ds_output).all()

    report = {
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "flash_attn": metadata.version("flash-attn"),
        "fla": metadata.version("flash-linear-attention"),
        "liger": metadata.version("liger-kernel"),
        "deepspeed": metadata.version("deepspeed"),
        "triton_max_error": float((triton_result - (x + y)).abs().max()),
        "flash_dense_shape": list(dense.shape),
        "flash_varlen_shape": list(varlen.shape),
        "flash_kvcache_shape": list(cache.shape),
        "fla_shape": list(fla_output.shape),
        "liger_shape": list(liger_output.shape),
        "deepspeed_engine": type(ds_engine).__name__,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
