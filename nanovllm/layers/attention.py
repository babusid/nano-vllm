import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    offsets = block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offsets < D
    key_offsets = idx * key_stride + offsets
    value_offsets = idx * value_stride + offsets
    key = tl.load(key_ptr + key_offsets, mask=mask, other=0.0)
    value = tl.load(value_ptr + value_offsets, mask=mask, other=0.0)
    cache_offsets = slot * D + offsets
    tl.store(k_cache_ptr + cache_offsets, key, mask=mask)
    tl.store(v_cache_ptr + cache_offsets, value, mask=mask)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    BLOCK_D = 256
    grid = (N, triton.cdiv(D, BLOCK_D))
    store_kvcache_kernel[grid](
        key,
        key.stride(0),
        value,
        value.stride(0),
        k_cache,
        v_cache,
        slot_mapping,
        D,
        BLOCK_D,
    )


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:  # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(
                q,
                k,
                v,
                max_seqlen_q=context.max_seqlen_q,
                cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k,
                cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale,
                causal=True,
                block_table=context.block_tables,
            )
        elif context.is_medusa_tree_decode:
            o = self._medusa_tree_attn(q, k, v, k_cache, v_cache, context)
        else:  # decode (single-query) or verify (multi-query)
            # seqlen_q = 1 for regular decode, spec_len+1 for verify.
            # All seqs in the batch must share the same seqlen_q (enforced by
            # prepare_decode / prepare_verify) so the (bs, seqlen_q, ...)
            # reshape is valid and causal=True handles intra-query masking.
            bs = context.context_lens.size(0)
            seqlen_q = q.size(0) // bs
            q_paged = q.view(bs, seqlen_q, self.num_heads, self.head_dim)
            o = flash_attn_with_kvcache(
                q_paged,
                k_cache,
                v_cache,
                cache_seqlens=context.context_lens,
                block_table=context.block_tables,
                softmax_scale=self.scale,
                causal=True,
            )
            o = o.view(bs * seqlen_q, self.num_heads, self.head_dim)
        return o

    def _medusa_tree_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        context,
    ) -> torch.Tensor:
        """Split-attention for MEDUSA tree decode (batch size = 1).

        The KV of the tree candidates has already been written to the paged
        cache by store_kvcache above.  We compute attention in two parts and
        combine them with the online log-sum-exp trick so that the result is
        mathematically equivalent to a single softmax over the full KV:

          Part 1 – prefix attention (FlashAttention):
            All medusa_len query tokens attend to the committed prefix KV
            (context.prefix_lens slots), with causal=False because every tree
            node should see the full context.  We ask FA to return the LSE so
            we can later recombine.

          Part 2 – tree-to-tree attention (dense matmul):
            The medusa_len queries also attend to each other, but only to their
            ancestor nodes.  The precomputed additive mask (0 / -inf) enforces
            the tree structure.  The tree is small (≤ 64 nodes), so the dense
            matmul cost is negligible.

          Combination:
            out = softmax_weight(lse_prefix) * out_prefix
                + softmax_weight(lse_tree)   * out_tree
            where the weights come from the log-sum-exp of each part.
        """
        # k, v: [medusa_len, num_kv_heads, head_dim]  — the tree candidate K/V
        medusa_len = k.size(0)

        # --- Part 1: prefix attention via FlashAttention ---
        # prefix_lens holds the committed-only lengths, so FA reads exactly
        # the prefix slots and ignores the tree slots we just stored.
        q_4d = q.view(1, medusa_len, self.num_heads, self.head_dim)
        out_p, lse_p = flash_attn_with_kvcache(
            q_4d,
            k_cache,
            v_cache,
            cache_seqlens=context.prefix_lens,
            block_table=context.block_tables,
            softmax_scale=self.scale,
            causal=False,
            return_softmax_lse=True,
        )
        # out_p: [1, medusa_len, num_heads, head_dim]
        # lse_p: [1, num_heads, medusa_len]
        out_p = out_p.squeeze(0)             # [medusa_len, num_heads, head_dim]
        lse_p = lse_p.squeeze(0)             # [num_heads, medusa_len]

        # --- Part 2: tree-to-tree attention via dense matmul ---
        # Reshape for [num_heads, medusa_len, head_dim] matmuls.
        # Use num_heads for q but num_kv_heads for k/v; expand kv if GQA.
        num_kv = self.num_kv_heads
        num_q  = self.num_heads
        kv_groups = num_q // num_kv

        # q_h: [num_heads, medusa_len, head_dim]
        q_h = q.permute(1, 0, 2)
        # k_h: [num_kv_heads, head_dim, medusa_len] → expand to [num_heads, ...]
        k_h = k.permute(1, 2, 0)
        if kv_groups > 1:
            k_h = k_h.repeat_interleave(kv_groups, dim=0)
        # v_h: [num_heads, medusa_len, head_dim]
        v_h = v.permute(1, 0, 2)
        if kv_groups > 1:
            v_h = v_h.repeat_interleave(kv_groups, dim=0)

        # Attention scores [num_heads, medusa_len, medusa_len] with tree mask
        scores = torch.matmul(q_h, k_h) * self.scale
        # medusa_tree_mask: [1, 1, medusa_len, medusa_len] additive bias
        scores = scores + context.medusa_tree_mask.squeeze(0)  # [1, Lq, Lq] broadcast

        # LSE and weighted output for tree part
        lse_t = torch.logsumexp(scores.float(), dim=-1)         # [num_heads, medusa_len]
        attn_w = torch.softmax(scores.float(), dim=-1).to(v_h.dtype)
        out_t = torch.matmul(attn_w, v_h)                       # [num_heads, medusa_len, head_dim]

        # --- Combine via online log-sum-exp ---
        lse_c = torch.logaddexp(lse_p.float(), lse_t)           # [num_heads, medusa_len]
        w_p = (lse_p.float() - lse_c).exp().unsqueeze(-1)       # [num_heads, medusa_len, 1]
        w_t = (lse_t          - lse_c).exp().unsqueeze(-1)

        # out_p is [medusa_len, num_heads, head_dim]; permute for [num_heads, ...]
        out_p_h = out_p.permute(1, 0, 2)                        # [num_heads, medusa_len, head_dim]
        out_combined = w_p * out_p_h + w_t * out_t              # [num_heads, medusa_len, head_dim]

        # Return [medusa_len, num_heads, head_dim]
        return out_combined.permute(1, 0, 2).contiguous()
