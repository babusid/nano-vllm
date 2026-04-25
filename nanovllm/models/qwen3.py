import torch
from torch import nn
import torch.distributed as dist
from transformers import LlamaConfig, Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import (
    QKVParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from nanovllm.layers.rotary_embedding import RotaryEmbedding, get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.models.vicuna import (
    VicunaAttention,
    VicunaMLP,
    VicunaRMSNorm,
    VicunaRotaryEmbedding,
)


class Qwen3RMSNorm(RMSNorm):
    def rms_forward_2d(self, x):
        return RMSNorm.rms_forward_2d(self, x)

    def rms_forward_3d(self, x):
        return RMSNorm.rms_forward_3d(self, x)

    def add_rms_forward_2d(self, x, residual):
        return RMSNorm.add_rms_forward_2d(self, x, residual)

    def add_rms_forward_3d(self, x, residual):
        return RMSNorm.add_rms_forward_3d(self, x, residual)


class Qwen3SiluAndMul(SiluAndMul):
    @torch.compile
    def forward(self, x):
        return SiluAndMul.forward(self, x)


class Qwen3RotaryEmbedding(RotaryEmbedding):
    @torch.compile
    def forward(self, positions, query, key):
        return RotaryEmbedding.forward(self, positions, query, key)


class Qwen3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.qkv_bias = qkv_bias

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
            cls=Qwen3RotaryEmbedding,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        if not self.qkv_bias:
            self.q_norm = Qwen3RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = Qwen3RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output


class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = Qwen3SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", True),
            head_dim=getattr(config, "head_dim", None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # EAGLE-3 hidden-state capture: when set, forward() stores post-layer
        # residual-stream hiddens at the listed indices into self._captured.
        # Read by ModelRunner via captured_hidden() / clear_captures(). The
        # capture is best-effort eager-only — CUDA-graphed forwards skip it,
        # so EAGLE eager runs work today and we can revisit graph capture
        # once the chain path is validated end-to-end.
        self._capture_layer_ids: tuple[int, ...] | None = None
        self._captured: list[torch.Tensor] = []

    def set_capture_layer_ids(self, ids: list[int] | None) -> None:
        """Configure which post-layer hiddens to stash on each forward.

        Indices are layer indices in self.layers (0-based). Pass None to
        disable capture.
        """
        self._capture_layer_ids = tuple(ids) if ids is not None else None
        self._captured = []

    def captured_hidden(self) -> torch.Tensor:
        """Return the captured hiddens concatenated along the feature dim:
        shape [N, len(ids) * hidden_size]. Errors if capture wasn't set up
        or the layer count doesn't match the configured indices.
        """
        assert self._capture_layer_ids is not None, "capture layers not set"
        assert len(self._captured) == len(self._capture_layer_ids), (
            f"expected {len(self._capture_layer_ids)} captures, "
            f"got {len(self._captured)}"
        )
        return torch.cat(self._captured, dim=-1)

    def clear_captures(self) -> None:
        self._captured = []

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        capture_ids = self._capture_layer_ids
        if capture_ids is not None:
            self._captured = []
        for i, layer in enumerate(self.layers):
            hidden_states, residual = layer(positions, hidden_states, residual)
            if capture_ids is not None and i in capture_ids:
                # The decoder layer returns (delta, residual); the actual
                # residual-stream value flowing INTO the next layer's input
                # norm is `hidden_states + residual`. That's what we want to
                # snapshot. Detach to keep it out of any autograd graph.
                self._captured.append((hidden_states + residual).detach())
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)


# ---------------------------------------------------------------------------
# EAGLE-3 draft head for the Qwen3 family
# ---------------------------------------------------------------------------
#
# Reference checkpoint: AngelSlim/Qwen3-32B_eagle3
#   - architectures: ["LlamaForCausalLMEagle3"]
#   - model_type:    "llama"  (Llama-style attention: no q/k norm, no qkv bias)
#   - hidden_size:   5120     (same as Qwen3-32B target)
#   - head_dim:      80       (custom — different from Qwen3-32B's 128)
#   - num_attn_heads:64,  num_key_value_heads: 8     (GQA)
#   - num_hidden_layers: 1   (single decoder layer = the "midlayer")
#   - vocab_size:    151936  (target vocab — for embed_tokens input)
#   - draft_vocab_size: 32000  (lm_head output — head's own restricted vocab)
#
# Inference structure:
#   embed_tokens(target_vocab, H)         # consumes committed target tokens
#   fc(3H -> H)                           # fuses 3-layer target hidden
#   hidden_norm(3H)                       # applied to fused target hidden pre-fc
#   input_norm(H)                         # applied to token embedding
#   midlayer = one Llama-style decoder layer (with its own paged KV cache)
#   norm(H)                               # final
#   lm_head(H -> draft_vocab)             # restricted-vocab logits
#   d2t [draft_vocab] (long)              # target_id = draft_id + d2t[draft_id]
#   t2d [target_vocab] (bool)             # True where target token is reachable
#
# Forward contract: the head's forward always takes a [N, H] "prev_hidden"
# input. The runner is responsible for projecting a fresh fused target hidden
# (3H -> H) via project_target_hidden() at the seeded step, then reusing the
# head's own returned hidden for subsequent recurrent draft steps. Keeping
# this contract uniform makes the chain-draft path CUDA-graphable.


class Qwen3Eagle3DecoderLayer(nn.Module):
    """Single Llama-style decoder layer used as the EAGLE-3 head's midlayer.

    Reuses VicunaAttention/MLP because the eagle config is Llama-shaped (no
    q/k norm, no qkv bias) — the only Qwen3-specific modules in the base
    model that we'd otherwise pull in here.
    """

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3Eagle3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            rope_theta=getattr(config, "rope_theta", 10000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = VicunaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.hidden_norm = VicunaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = VicunaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = VicunaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        token_hidden: torch.Tensor,
        prev_hidden: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            raise NotImplementedError(
                "EAGLE-3 midlayer currently expects a single-layer residual path"
            )
        # The qkv path consumes the normalized token branch and recurrent
        # branch concatenated to 2H, but the residual stream remains in H.
        # Keep both sources in the residual by summing the unnormalized
        # branches before the post-attention RMSNorm.
        residual = token_hidden + prev_hidden
        token_hidden = self.input_layernorm(token_hidden)
        prev_hidden = self.hidden_norm(prev_hidden)
        attn_input = torch.cat([token_hidden, prev_hidden], dim=-1)
        hidden_states = self.self_attn(positions, attn_input)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Eagle3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        # The published Qwen3 Eagle-3 head stores q/k/v projections with
        # in_features = 2 * hidden_size, which matches concatenating the
        # normalized token embedding and the recurrent hidden state.
        self.qkv_proj = QKVParallelLinear(
            hidden_size * 2,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
            cls=VicunaRotaryEmbedding,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        return self.o_proj(o.flatten(1, -1))


class Qwen3Eagle3ForCausalLM(nn.Module):
    """EAGLE-3 draft head for Qwen3 targets.

    See module docstring for the architectural details. The exact module names
    here (fc, hidden_norm, input_norm, midlayer, norm) follow the published
    SafeAILab/AngelSlim checkpoint convention; on first integration, do a
    strict=False load and print missing/unexpected keys to confirm.
    """

    # Same packed-modules mapping as base Qwen3/Vicuna so that q/k/v_proj
    # weights merge into qkv_proj and gate/up_proj into gate_up_proj at load.
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_size
        target_vocab = config.vocab_size
        self.draft_vocab_size = getattr(config, "draft_vocab_size", target_vocab)

        self.embed_tokens = VocabParallelEmbedding(target_vocab, h)
        self.fc = nn.Linear(h * 3, h, bias=False)
        self.midlayer = Qwen3Eagle3DecoderLayer(config)
        self.norm = VicunaRMSNorm(h, eps=config.rms_norm_eps)
        # lm_head outputs over the smaller draft vocabulary; use a plain
        # Linear (not ParallelLMHead) since draft_vocab is small and the
        # eagle runner always runs at TP=1.
        self.lm_head = nn.Linear(h, self.draft_vocab_size, bias=False)
        # Vocabulary mapping buffers (loaded from checkpoint).
        # d2t: target_id = draft_id + d2t[draft_id]  (additive offset table)
        # t2d: True where a target token is also in the draft vocab
        self.register_buffer(
            "d2t",
            torch.zeros(self.draft_vocab_size, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "t2d",
            torch.zeros(target_vocab, dtype=torch.bool),
            persistent=False,
        )

    def project_target_hidden(self, fused_hidden: torch.Tensor) -> torch.Tensor:
        """Project a fresh 3-layer concatenated target hidden into the head's
        hidden dim. Called once per seeded position; the result is what
        forward() expects as `prev_hidden`.
        """
        return self.fc(fused_hidden)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        prev_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """One head step.

        Args:
            input_ids:   [N] target-vocab token ids (the committed target token
                         at the seeded step, or the previous step's drafted
                         token translated to target vocab via d2t).
            positions:   [N] absolute positions for RoPE / KV slot mapping.
            prev_hidden: [N, H] either project_target_hidden(fused_hidden) at
                         the seeded step, or the head's own hidden returned
                         from the previous draft step.

        Returns:
            hidden: [N, H] post-norm hidden state, suitable for compute_logits
                    and as input to the next recurrent step.
        """
        e = self.embed_tokens(input_ids)
        residual = None
        x, residual = self.midlayer(positions, e, prev_hidden, residual)
        x, _ = self.norm(x, residual)
        return x

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
