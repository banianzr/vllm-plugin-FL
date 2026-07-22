# Copyright (c) 2026 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ascend MoE gating (softmax + top-k) via the NPU native operator.

Wraps ``torch.ops._C_ascend.moe_gating_top_k`` — the same operator
vllm-ascend uses (``vllm_ascend/device/device_op.py::moe_gating_top_k``).
On profiling (2026-07-21, Qwen3.6-35B-A3B) the FlagGems
``topk_gating_softmax_kernel_2`` this replaces accounted for ~26% of FL
compute and was ~60x slower than the native ``MoeGatingTopK``.
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch

logger = logging.getLogger(__name__)


def topk_softmax_ascend(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    token_expert_indices: torch.Tensor,  # unused; kept for call_op signature parity
    gating_output: torch.Tensor,
    renormalize: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """MoE gating: softmax + top-k using the Ascend NPU native operator.

    Drop-in replacement for the FlagGems ``topk_softmax`` on Ascend.
    ``topk_weights`` / ``topk_ids`` are pre-allocated by ``fused_topk`` and
    filled in place, matching the call_op contract used by
    ``vllm_fl/ops/fused_moe/fused_moe.py::fused_topk``.

    Configuration mirrors vllm-ascend's non-grouped softmax path
    (``group_count=1``, ``k_group=1``, ``norm_type=0``); see
    ``check_npu_moe_gating_top_k`` for the supported envelope. If the NPU
    operator rejects the configuration, fall back to a PyTorch reference so
    correctness is never compromised.
    """
    k = topk_ids.size(1)

    try:
        topk_w, topk_i, _ = torch.ops._C_ascend.moe_gating_top_k(
            gating_output,
            k=k,
            k_group=1,
            group_count=1,
            group_select_mode=1,
            renorm=int(bool(renormalize)),
            norm_type=0,  # 0: softmax; 1: sigmoid
            out_flag=False,
            routed_scaling_factor=1.0,
            eps=1e-20,
            bias_opt=None,
        )
        topk_weights.copy_(topk_w)
        topk_ids.copy_(topk_i.to(torch.int32))
    except Exception as e:  # noqa: BLE001
        # NPU op unavailable or config rejected (e.g. unsupported num_experts,
        # grouped routing) — keep correctness, log for follow-up.
        logger.warning(
            "moe_gating_top_k failed (num_experts=%d, k=%d, renorm=%s); "
            "falling back to PyTorch reference: %s",
            gating_output.shape[-1], k, renormalize, e,
        )
        softmax_scores = gating_output.softmax(dim=-1)
        ref_weights, ref_ids = softmax_scores.topk(k, dim=-1)
        topk_weights.copy_(ref_weights)
        topk_ids.copy_(ref_ids.to(torch.int32))
        if renormalize:
            topk_weights.div_(topk_weights.sum(dim=-1, keepdim=True))

    return topk_weights, topk_ids
