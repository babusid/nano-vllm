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
        """Split-attention for MEDUSA tree decode (any batch size).

        The KV of the tree candidates has already been written to the paged
        cache by store_kvcache above.  We compute attention in two parts and
        combine them with the online log-sum-exp trick so that the result is
        mathematically equivalent to a single softmax over the full KV:

          Part 1 – prefix attention (FlashAttention):
            All medusa_len query tokens per seq attend to that seq's committed
            prefix KV (context.prefix_lens[b] slots), with causal=False because
            every tree node should see the full context.  FA returns the LSE so
            we can later recombine.

          Part 2 – tree-to-tree attention (dense matmul):
            The medusa_len queries also attend to each other, but only to their
            ancestor nodes.  The precomputed additive mask (0 / -inf) enforces
            the tree structure.  The tree is small (≤ 64 nodes), so the dense
            matmul cost is negligible even batched.

          Combination:
            out = softmax_weight(lse_prefix) * out_prefix
                + softmax_weight(lse_tree)   * out_tree
            where the weights come from the log-sum-exp of each part.
        """
        # q, k, v: [B * medusa_len, num_{q|kv}_heads, head_dim]
        B = context.prefix_lens.size(0)
        N = q.size(0)
        medusa_len = N // B
        H = self.num_heads
        Hkv = self.num_kv_heads
        D = self.head_dim
        kv_groups = H // Hkv

        # --- Part 1: prefix attention via FlashAttention ---
        # prefix_lens holds per-seq committed lengths, so FA reads exactly each
        # seq's prefix slots and ignores the tree slots we just stored.
        q_4d = q.view(B, medusa_len, H, D)
        out_p, lse_p = flash_attn_with_kvcache(
            q_4d,
            k_cache,
            v_cache,
            cache_seqlens=context.prefix_lens,       # [B]
            block_table=context.block_tables,        # [B, max_bt]
            softmax_scale=self.scale,
            causal=False,
            return_softmax_lse=True,
        )
        # out_p: [B, medusa_len, H, D]      lse_p: [B, H, medusa_len]

        # --- Part 2: tree-to-tree attention via dense matmul ---
        # Reshape for [B, H, medusa_len, D] matmuls.  Expand kv heads for GQA.
        q_h = q.view(B, medusa_len, H, D).permute(0, 2, 1, 3)                  # [B, H, L, D]
        k_h = k.view(B, medusa_len, Hkv, D).permute(0, 2, 3, 1)                # [B, Hkv, D, L]
        v_h = v.view(B, medusa_len, Hkv, D).permute(0, 2, 1, 3)                # [B, Hkv, L, D]
        if kv_groups > 1:
            k_h = k_h.repeat_interleave(kv_groups, dim=1)                      # [B, H, D, L]
            v_h = v_h.repeat_interleave(kv_groups, dim=1)                      # [B, H, L, D]

        # Attention scores [B, H, L, L] with tree mask (broadcasts over B and H)
        scores = torch.matmul(q_h, k_h) * self.scale
        scores = scores + context.medusa_tree_mask                             # [1,1,L,L]

        # LSE and weighted output for tree part
        lse_t = torch.logsumexp(scores.float(), dim=-1)                        # [B, H, L]
        attn_w = torch.softmax(scores.float(), dim=-1).to(v_h.dtype)
        out_t = torch.matmul(attn_w, v_h)                                       # [B, H, L, D]

        # --- Combine via online log-sum-exp ---
        lse_c = torch.logaddexp(lse_p.float(), lse_t)                          # [B, H, L]
        w_p = (lse_p.float() - lse_c).exp().unsqueeze(-1)                      # [B, H, L, 1]
        w_t = (lse_t          - lse_c).exp().unsqueeze(-1)

        out_p_h = out_p.permute(0, 2, 1, 3)                                    # [B, H, L, D]
        out_combined = w_p * out_p_h + w_t * out_t                              # [B, H, L, D]

        # Cast back to the original query dtype (e.g. float16) before returning.
        # w_p / w_t are computed in float32 for numerical stability, which
        # promotes the combined output; without this cast, o_proj (float16
        # weights) raises a dtype mismatch.
        return (
            out_combined.to(q.dtype)
            .permute(0, 2, 1, 3)
            .reshape(B * medusa_len, H, D)
            .contiguous()
        )
