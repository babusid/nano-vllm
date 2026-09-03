from ntpath import expanduser
import os
import pickle
import torch
import torch.nn as nn
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.medusa_utils import MedusaBlock, remap_medusa_lm_head_state_dict
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.models.vicuna import VicunaForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class Qwen3Sampler(Sampler):
    @torch.compile
    def forward(self, logits, temperatures):
        return Sampler.forward(self, logits, temperatures)


class VicunaSampler(Sampler):
    @torch.compile
    def forward(self, logits, temperatures):
        return Sampler.forward(self, logits, temperatures)


class ModelRunner:
    def __init__(
        self,
        config: Config,
        rank: int,
        event: Event | list[Event],
        block_managers: list[BlockManager] | None = None,
        model_runner_idx: int = 0,
        verify_seqlen_q: int | None = None,
        # MEDUSA-specific params — all None/0 for non-MEDUSA runners
        medusa_model_path: str | None = None,
        medusa_num_heads: int = 0,
        medusa_num_layers: int = 1,
        medusa_buffers: dict | None = None,
    ):
        self.config = config
        self.block_managers = block_managers if block_managers is not None else []
        self.block_table_idx = model_runner_idx
        self.model_runner_idx = model_runner_idx
        # When set, capture a second family of CUDA graphs sized for
        # bs * verify_seqlen_q query tokens so verify() can replay instead of
        # running eagerly. Only the verifier runner needs this.
        self.verify_seqlen_q = verify_seqlen_q
        # MEDUSA state (None when not in MEDUSA mode)
        self.medusa_buffers = medusa_buffers
        self.medusa_tree_mask = (
            medusa_buffers["medusa_attn_mask"] if medusa_buffers else None
        )
        self.medusa_len = medusa_buffers["medusa_len"] if medusa_buffers else 0
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self._owns_process_group = False
        model_dtype = getattr(hf_config, "dtype", None)
        if model_dtype is None:
            model_dtype = hf_config.torch_dtype
        # FlashAttention requires fp16 or bf16.  If the checkpoint config
        # declares float32 (common for small/older models like llama-68m that
        # were originally trained in fp32), fall back to fp16 so that all
        # forward passes are compatible with FlashAttention.
        if model_dtype not in (torch.float16, torch.bfloat16):
            model_dtype = torch.float16
        self.model_dtype = model_dtype

        if not dist.is_initialized():
            dist.init_process_group(
                "nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank
            )
            self._owns_process_group = True

        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self.model_dtype)
        torch.set_default_device("cuda")
        model_type = getattr(hf_config, "model_type", None)
        if model_type == "qwen3":
            self.model = Qwen3ForCausalLM(hf_config)
        elif model_type == "llama":
            self.model = VicunaForCausalLM(hf_config)
        else:
            raise NotImplementedError(
                f"Unsupported model_type {model_type!r}; "
                "nano-vllm supports qwen3 and llama (e.g. Vicuna) checkpoints."
            )
        load_model(self.model, config.model)

        # Load MEDUSA heads alongside the base model weights when in MEDUSA mode.
        # Each head is medusa_num_layers ResBlocks + vocab Linear (see MedusaBlock).
        # Weights are read from medusa_lm_head.pt in the checkpoint.
        if medusa_model_path:
            hidden_size = hf_config.hidden_size
            vocab_size = hf_config.vocab_size
            self.medusa_heads = nn.ModuleList(
                [
                    MedusaBlock(hidden_size, vocab_size, medusa_num_layers)
                    for _ in range(medusa_num_heads)
                ]
            )
            head_ckpt = os.path.join(medusa_model_path, "medusa_lm_head.pt")
            state = remap_medusa_lm_head_state_dict(
                torch.load(head_ckpt, map_location="cuda")
            )
            our_keys = set(self.medusa_heads.state_dict().keys())
            ckpt_keys = set(state.keys())
            matched = our_keys & ckpt_keys
            print(
                f"[MEDUSA] checkpoint keys: {len(ckpt_keys)}, "
                f"module keys: {len(our_keys)}, "
                f"matched: {len(matched)}"
            )
            if not matched:
                print(
                    f"[MEDUSA] WARNING: no keys matched! "
                    f"Sample ckpt keys: {list(ckpt_keys)[:4]} | "
                    f"Sample module keys: {list(our_keys)[:4]}"
                )
            result = self.medusa_heads.load_state_dict(state, strict=False)
            if result.unexpected_keys:
                print(f"[MEDUSA] unexpected keys in ckpt: {result.unexpected_keys[:4]}")
            if result.missing_keys:
                print(f"[MEDUSA] missing keys in module: {result.missing_keys[:4]}")
            self.medusa_heads = self.medusa_heads.to(
                dtype=next(self.model.parameters()).dtype
            )
            self.medusa_heads.eval()

        if model_type == "qwen3":
            self.sampler = Qwen3Sampler()
        elif model_type == "llama":
            self.sampler = VicunaSampler()
        else:
            self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            if not medusa_model_path:
                # Standard decode + optional verify graphs (none/naive modes)
                self.capture_cudagraph()
                if self.verify_seqlen_q is not None and self.verify_seqlen_q > 1:
                    self.capture_verify_cudagraph()
            else:
                # MEDUSA: capture regular decode graph family (reused for seed
                # and bonus passes in _medusa_step) AND the tree-decode graph
                # family (bucketed by batch size) for tree verification.
                self.capture_cudagraph()
                self.capture_medusa_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            if hasattr(self, "graphs"):
                del self.graphs, self.graph_pool
            if hasattr(self, "verify_graphs"):
                del self.verify_graphs
            if hasattr(self, "medusa_graphs"):
                del self.medusa_graphs, self.medusa_graph_vars
        torch.cuda.synchronize()
        if self._owns_process_group and dist.is_initialized():
            dist.destroy_process_group()

    def loop(self):
        # used for non-rank 0 processes to run the methods
        # specified in the shared memory by the rank 0 process
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        # non-rank0 processes read the method name and arguments
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4 : n + 4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        # rank0 process writes the method name and arguments
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4 : n + 4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        # reflection based call so that we can use
        # method name in shmem instead of direct calls
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = (
            self.config.max_num_batched_tokens,
            self.config.max_model_len,
        )
        num_seqs = min(
            max_num_batched_tokens // max_model_len, self.config.max_num_seqs
        )
        seqs = [
            Sequence(
                token_ids=[0] * max_model_len,
                num_block_tables=self.block_table_idx + 1,
            )
            for _ in range(num_seqs)
        ]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def _block_table(self, seq: Sequence) -> list[int]:
        # get paired block table
        return seq.block_tables[self.block_table_idx]

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(
            hf_config,
            "head_dim",
            hf_config.hidden_size // hf_config.num_attention_heads,
        )
        block_bytes = (
            2
            * hf_config.num_hidden_layers
            * self.block_size
            * num_kv_heads
            * head_dim
            * torch.tensor([], dtype=self.model_dtype).element_size()
        )
        available_bytes = int(free * config.gpu_memory_utilization)
        config.num_kvcache_blocks = available_bytes // block_bytes
        if config.num_kvcache_blocks <= 0:
            free_gib = free / (1024**3)
            block_mib = block_bytes / (1024**2)
            raise ValueError(
                "Insufficient GPU memory for KV cache allocation: "
                f"free={free_gib:.2f}GiB, util={config.gpu_memory_utilization:.3f}, "
                f"block_size={block_mib:.2f}MiB"
            )
        self.kv_cache = torch.empty(
            2,
            hf_config.num_hidden_layers,
            config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            head_dim,
        )
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        # pads block tables to max length and stacks them
        # together in a tensor to give a uniform tensor of block
        # tables
        max_len = max(len(self._block_table(seq)) for seq in seqs)
        block_tables = [
            self._block_table(seq) + [-1] * (max_len - len(self._block_table(seq)))
            for seq in seqs
        ]
        block_tables = torch.tensor(
            block_tables, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        return block_tables

    def prepare_verify(self, seqs: list[Sequence], draft_model_idx: int = 0):
        # Decode-style context: verify is multi-query paged attention
        # (seqlen_q = spec_len + 1 per seq, all seqs uniform) so it can share
        # flash_attn_with_kvcache and the decode CUDA graph buffer layout.
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []

        for seq in seqs:
            block_table = self._block_table(seq)
            committed_len = len(seq)
            draft_token_list = seq.draft_token_ids[draft_model_idx]
            if not draft_token_list:
                # seqs with no drafts contribute no query tokens; upstream
                # split logic will produce an empty slice for them.
                context_lens.append(0)
                continue
            input_ids.append(seq.last_token)
            input_ids.extend(draft_token_list)
            positions.extend(
                list(range(committed_len - 1, committed_len + len(draft_token_list)))
            )
            # cache covers committed tokens + newly-stored draft K/V
            context_lens.append(committed_len + len(draft_token_list))
            if not block_table:  # warmup — no cache slots yet
                continue

            last_idx = committed_len - 1
            last_block = block_table[last_idx // self.block_size]
            slot_mapping.append(
                last_block * self.block_size + last_idx % self.block_size
            )

            for draft_idx in range(len(draft_token_list)):
                token_idx = committed_len + draft_idx
                block_idx = token_idx // self.block_size
                assert block_idx < len(
                    block_table
                )  # if reservation worked this shouldn't fire
                block_offset = token_idx % self.block_size
                slot_mapping.append(
                    block_table[block_idx] * self.block_size + block_offset
                )

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        context_lens = torch.tensor(
            context_lens, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)

        set_context(
            False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )
        return input_ids, positions

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]  # prefix array of sequence lengths without num_cached_tokens
        cu_seqlens_k = [0]  # prefix array of sequence lengths
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None

        for seq in seqs:
            block_table = self._block_table(seq)
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens :])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not block_table:  # warmup
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens
                slot_mapping.extend(list(range(start, end)))

        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:  # prefix cache
            block_tables = self.prepare_block_tables(seqs)

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )

        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )

        cu_seqlens_q = torch.tensor(
            cu_seqlens_q, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)

        cu_seqlens_k = torch.tensor(
            cu_seqlens_k, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)

        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)

        set_context(
            True,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            None,
            block_tables,
        )
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []  # list of single query tokens for this batch
        positions = []  # positions of those tensors in the sequence are needed for rope
        slot_mapping = []
        context_lens = []  # the total length of each sequence
        for seq in seqs:
            block_table = self._block_table(seq)
            draft_token_list = seq.draft_token_ids[self.model_runner_idx]
            # token idx is the committed tokens + the draft tokens from this model runner
            token_idx = len(seq) + len(draft_token_list) - 1
            # last committed token is the query token
            # if there are no draft tokens
            input_ids.append(
                seq.last_token if len(draft_token_list) == 0 else draft_token_list[-1]
            )

            positions.append(token_idx)  # index of the current last token for RoPE
            context_lens.append(len(seq) + len(draft_token_list))
            block_idx = token_idx // self.block_size
            assert block_idx < len(
                block_table
            )  # if reservation worked this shouldn't fire
            block_offset = token_idx % self.block_size
            slot_mapping.append(  # where to store the K/V of the query token
                block_table[block_idx] * self.block_size + block_offset
            )

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )  # stack query tokens into a tensor
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        context_lens = torch.tensor(
            context_lens, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(
            False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(
            temperatures, dtype=torch.float32, pin_memory=True
        ).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = (
            self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        )
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = (
            self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        )
        reset_context()
        return token_ids, logits

    def sample(self, logits, temperatures):
        return self.sampler(logits, temperatures).tolist() if self.rank == 0 else None

    @torch.inference_mode()
    def verify(
        self, seqs: list[Sequence], draft_model_idx: int = 0
    ) -> tuple[list[list[int]] | None, list[torch.Tensor] | None]:
        if any(len(seq.draft_token_ids) > 2 for seq in seqs):
            # 2 because drafter and verifier both have tables
            raise NotImplementedError("Multi draft verification not supported yet")
        input_ids, positions = self.prepare_verify(seqs, draft_model_idx)
        seq_temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        # prepare_verify sets is_prefill=False, so ParallelLMHead keeps every
        # query-token's logits (no last-token truncation), and run_verify_model
        # replays a captured graph when possible.
        hidden_states = self.run_verify_model(input_ids, positions)
        logits = self.model.compute_logits(hidden_states)
        reset_context()
        dlens = [len(seq.draft_token_ids[draft_model_idx]) for seq in seqs]
        if self.rank == 0:
            repeats = torch.tensor(
                [dlen + 1 if dlen > 0 else 0 for dlen in dlens],
                dtype=torch.int64,
                device=seq_temperatures.device,
            )
            expanded_temperatures = seq_temperatures.repeat_interleave(repeats)
            assert expanded_temperatures.size(0) == logits.size(0)
            flat_token_ids = self.sample(logits, expanded_temperatures)
            split_logits = []
            split_token_ids = []
            start = 0
            for dlen in dlens:
                end = start + (dlen + 1 if dlen > 0 else 0)
                split_logits.append(logits[start:end])
                split_token_ids.append(flat_token_ids[start:end])
                start = end
            return split_token_ids, split_logits
        return None, None

    @torch.inference_mode()
    def run_verify_model(
        self, input_ids: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        # Mirrors run_model's graph-replay path but sized for multi-query verify
        # (bs * seqlen_q tokens). Falls back to eager when graphs aren't
        # available or the batch overflows the captured buffer.
        if self.enforce_eager or not hasattr(self, "verify_graphs"):
            return self.model(input_ids, positions)
        seqlen_q = self.verify_seqlen_q
        N = input_ids.size(0)
        assert (
            N % seqlen_q == 0
        ), f"verify expects uniform seqlen_q={seqlen_q} per seq, got {N} tokens"
        bs = N // seqlen_q
        graph_bs = next((x for x in self.verify_graph_bs if x >= bs), None)
        if graph_bs is None:
            return self.model(input_ids, positions)
        context = get_context()
        graph = self.verify_graphs[graph_bs]
        gv = self.verify_graph_vars
        gv["input_ids"][:N] = input_ids
        gv["positions"][:N] = positions
        gv["slot_mapping"].fill_(-1)
        gv["slot_mapping"][: context.slot_mapping.size(0)] = context.slot_mapping
        gv["context_lens"].zero_()
        gv["context_lens"][:bs] = context.context_lens
        gv["block_tables"][:bs, : context.block_tables.size(1)] = context.block_tables
        graph.replay()
        return gv["outputs"][:N]

    @torch.inference_mode()
    def run_model(
        self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool
    ):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][
                :bs, : context.block_tables.size(1)
            ] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(
                False,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
            )
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )

    @torch.inference_mode()
    def capture_verify_cudagraph(self):
        # Verify graphs share the decode attention kernel but are sized for
        # bs * seqlen_q query tokens (seqlen_q = spec_len + 1). Reuses the
        # decode graph_pool so buffers live in the same mempool.
        config = self.config
        hf_config = config.hf_config
        seqlen_q = self.verify_seqlen_q
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        max_N = max_bs * seqlen_q
        input_ids = torch.zeros(max_N, dtype=torch.int64)
        positions = torch.zeros(max_N, dtype=torch.int64)
        slot_mapping = torch.zeros(max_N, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_N, hf_config.hidden_size)
        self.verify_graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.verify_graphs = {}
        # start from decode pool so allocations can be shared
        pool = self.graph_pool

        for bs in reversed(self.verify_graph_bs):
            N = bs * seqlen_q
            graph = torch.cuda.CUDAGraph()
            set_context(
                False,
                slot_mapping=slot_mapping[:N],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
            )
            outputs[:N] = self.model(input_ids[:N], positions[:N])  # warmup
            with torch.cuda.graph(graph, pool):
                outputs[:N] = self.model(input_ids[:N], positions[:N])  # capture
            if pool is None:
                pool = graph.pool()
            self.verify_graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.verify_graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )

    # ------------------------------------------------------------------
    # MEDUSA tree-decode methods
    # ------------------------------------------------------------------

    def prepare_medusa_tree(
        self, seqs: list, tree_candidates: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build input tensors for one MEDUSA tree-decode step (any batch size).

        The scheduler has pre-reserved medusa_len extra KV slots per sequence,
        so we can safely write tree-candidate K/V there.  prefix_lens is set
        to each seq's committed length so the prefix FA call ignores those
        reserved tree slots.

        Args:
            seqs: list of B sequences to verify this step.
            tree_candidates: [B, medusa_len] CUDA tensor of candidate token ids,
                in flat tree layout (produced by generate_candidates).
        """
        B = len(seqs)
        medusa_len = self.medusa_len
        medusa_position_ids = self.medusa_buffers["medusa_position_ids"]  # [medusa_len]

        slot_list = []
        prefix_list = []
        bt_list = []

        for seq in seqs:
            block_table = self._block_table(seq)
            committed_len = len(seq)
            for i in range(medusa_len):
                token_idx = committed_len + i
                slot_list.append(
                    block_table[token_idx // self.block_size] * self.block_size
                    + token_idx % self.block_size
                )
            prefix_list.append(committed_len)
            bt_list.append(block_table)

        tree_tokens = tree_candidates.reshape(B * medusa_len)  # already on CUDA
        slot_mapping = torch.tensor(slot_list, dtype=torch.int32, pin_memory=True).cuda(
            non_blocking=True
        )
        prefix_lens = torch.tensor(
            prefix_list, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        prefix_lens_i64 = prefix_lens.to(torch.int64)
        positions = (
            prefix_lens_i64.unsqueeze(1) + medusa_position_ids.unsqueeze(0)
        ).reshape(B * medusa_len)
        max_bt_len = max((len(bt) for bt in bt_list), default=1)
        block_tables = torch.tensor(
            [bt + [-1] * (max_bt_len - len(bt)) for bt in bt_list],
            dtype=torch.int32,
            pin_memory=True,
        ).cuda(non_blocking=True)
        self._last_medusa_block_tables = block_tables
        self._last_medusa_prefix_lens = prefix_lens_i64

        set_context(
            False,
            slot_mapping=slot_mapping,
            context_lens=prefix_lens,  # not used in tree path but kept for consistency
            block_tables=block_tables,
            is_medusa_tree_decode=True,
            medusa_tree_mask=self.medusa_tree_mask,
            prefix_lens=prefix_lens,
        )
        return tree_tokens, positions

    @torch.inference_mode()
    def run_medusa_tree(
        self, seqs: list, tree_candidates: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one MEDUSA tree-decode step for a batch of B sequences.

        Args:
            seqs: list of B sequences.
            tree_candidates: [B, medusa_len] CUDA tensor of candidate token ids.

        Returns:
            lm_logits:     [B * medusa_len, vocab_size]
            medusa_logits: [num_medusa_heads, B * medusa_len, vocab_size]
        """
        input_ids, positions = self.prepare_medusa_tree(seqs, tree_candidates)
        hidden = self._run_medusa_graph(input_ids, positions)
        lm_logits = self.model.compute_logits(hidden)
        medusa_logits = torch.stack([head(hidden) for head in self.medusa_heads], dim=0)
        reset_context()
        return lm_logits, medusa_logits

    def copy_accepted_kv_slots_batched(
        self,
        src_slots: list[int],
        dst_slots: list[int],
    ) -> None:
        """Batched K/V copy from tree-unique slots to sequential slots.

        During tree decode, tree node i was written at the KV slot for logical
        position old_committed+i (unique per node).  After acceptance, sequential
        positions old_committed, old_committed+1, … must hold the accepted path's
        correct K/V so the next step's prefix attention is accurate.  The root
        (node 0) is always at the right slot and is skipped upstream.

        src_slots / dst_slots are flat lists of global KV slot indices
        (block_id * block_size + offset) across all seqs in the batch.
        """
        if not src_slots:
            return
        src = torch.tensor(src_slots, dtype=torch.int64, device="cuda")
        dst = torch.tensor(dst_slots, dtype=torch.int64, device="cuda")
        # kv_cache: [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        # Collapse (num_blocks, block_size) into a single "slot" dim so we can
        # scatter over it with a 1-D index tensor.
        nl = self.kv_cache.size(1)
        nkv = self.kv_cache.size(4)
        hd = self.kv_cache.size(5)
        flat = self.kv_cache.view(2, nl, -1, nkv, hd)
        flat[:, :, dst] = flat[:, :, src]

    def copy_accepted_kv_slots_medusa(
        self,
        seqs: list,
        chosen_nodes: torch.Tensor,
        accept_length: torch.Tensor,
    ) -> None:
        """Copy accepted MEDUSA tree KV slots using GPU-side slot tensors.

        chosen_nodes contains the tree-node indices for the selected candidate
        path per sequence. Accepted tree nodes are written during tree decode at
        logical positions old_len + tree_idx, but future decode expects the
        accepted path packed sequentially at old_len + step.
        """
        B, depth = chosen_nodes.shape
        if B == 0 or depth == 0:
            return

        device = chosen_nodes.device
        nodes = chosen_nodes
        if nodes.dtype is not torch.int64:
            nodes = nodes.to(torch.int64)
        if accept_length.dtype is not torch.int64:
            accept_length = accept_length.to(torch.int64)

        committed = getattr(self, "_last_medusa_prefix_lens", None)
        block_tables = getattr(self, "_last_medusa_block_tables", None)
        if committed is None or block_tables is None or committed.size(0) < B:
            committed = torch.tensor(
                [len(seq) for seq in seqs], dtype=torch.int64, device=device
            )
            block_tables = self.prepare_block_tables(seqs)
        committed = committed[:B]
        block_tables = block_tables[:B]

        nl = self.kv_cache.size(1)
        nkv = self.kv_cache.size(4)
        hd = self.kv_cache.size(5)
        flat = self.kv_cache.view(2, nl, -1, nkv, hd)

        for step in range(depth):
            node = nodes[:, step]
            dst_lp = committed + step
            # Avoid boolean indexing here: tensor[bool_mask] lowers through
            # aten::nonzero, which synchronizes. Non-copy lanes become
            # harmless self-copies at the sequential destination slot.
            should_copy = (step <= accept_length) & (node != step)
            src_lp = committed + torch.where(
                should_copy, node, node.new_full((B,), step)
            )
            src_block = torch.gather(
                block_tables, 1, (src_lp // self.block_size).unsqueeze(1)
            ).squeeze(1)
            dst_block = torch.gather(
                block_tables, 1, (dst_lp // self.block_size).unsqueeze(1)
            ).squeeze(1)
            src_slots = src_block * self.block_size + src_lp % self.block_size
            dst_slots = dst_block * self.block_size + dst_lp % self.block_size
            flat[:, :, dst_slots] = flat[:, :, src_slots]

    @torch.inference_mode()
    def run_medusa_bonus_batch(
        self,
        seqs: list,
        bonus_tokens: torch.Tensor,
        accept_length: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the MEDUSA bonus-token seed pass from CUDA decisions."""
        B = bonus_tokens.size(0)
        device = bonus_tokens.device
        committed = getattr(self, "_last_medusa_prefix_lens", None)
        block_tables = getattr(self, "_last_medusa_block_tables", None)
        if committed is None or block_tables is None or committed.size(0) < B:
            committed = torch.tensor(
                [len(seq) for seq in seqs], dtype=torch.int64, device=device
            )
            block_tables = self.prepare_block_tables(seqs)
        committed = committed[:B]
        block_tables = block_tables[:B]

        if accept_length.dtype is not torch.int64:
            accept_length = accept_length.to(torch.int64)
        positions = committed + accept_length + 1
        context_lens = (positions + 1).to(torch.int32)
        block_idx = (positions // self.block_size).unsqueeze(1)
        block = torch.gather(block_tables, 1, block_idx).squeeze(1)
        slot_mapping = (block * self.block_size + positions % self.block_size).to(
            torch.int32
        )

        set_context(
            False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )
        hidden = self._run_decode_hidden(bonus_tokens, positions)
        reset_context()

        lm_logits = self.model.compute_logits(hidden)
        medusa_logits = torch.stack([head(hidden) for head in self.medusa_heads], dim=0)
        return lm_logits, medusa_logits

    @torch.inference_mode()
    def _run_decode_hidden(
        self, input_ids: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Run one decode forward pass and return the hidden states.

        Mirrors run_model's decode-graph replay path but stops before the
        lm_head so callers (MEDUSA seed/bonus) can feed the hidden into both
        the LM head and the MEDUSA heads without duplicate work.
        """
        if self.enforce_eager or not hasattr(self, "graphs"):
            return self.model(input_ids, positions)
        bs = input_ids.size(0)
        if bs > 512:
            return self.model(input_ids, positions)
        context = get_context()
        graph_bs = next((x for x in self.graph_bs if x >= bs), None)
        if graph_bs is None:
            return self.model(input_ids, positions)
        graph = self.graphs[graph_bs]
        gv = self.graph_vars
        gv["input_ids"][:bs] = input_ids
        gv["positions"][:bs] = positions
        gv["slot_mapping"].fill_(-1)
        gv["slot_mapping"][:bs] = context.slot_mapping
        gv["context_lens"].zero_()
        gv["context_lens"][:bs] = context.context_lens
        gv["block_tables"][:bs, : context.block_tables.size(1)] = context.block_tables
        graph.replay()
        return gv["outputs"][:bs]

    @torch.inference_mode()
    def run_medusa_decode_batch(
        self,
        seqs: list,
        input_tokens: list[int],
        positions: list[int],
        context_lens: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched 1-token-per-seq forward used for MEDUSA seed and bonus passes.

        Reuses the regular decode CUDA graph family (same one used by NONE mode),
        so there is no separate bonus graph to maintain and any batch size that
        fits in the captured buckets gets graph-accelerated.

        Args:
            seqs:         list of B sequences.
            input_tokens: length-B list of token ids to commit to KV.
            positions:    length-B list of logical positions (committed_len for
                          a seed pass, committed_len + accept_length + 1 for a
                          bonus pass).
            context_lens: length-B list of KV lengths to attend over
                          (position + 1 for the typical case).

        Returns:
            lm_logits:     [B, vocab_size]
            medusa_logits: [num_medusa_heads, B, vocab_size]
        """
        B = len(seqs)
        input_ids = torch.tensor(input_tokens, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        pos_t = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        ctx_t = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(
            non_blocking=True
        )

        slot_list = []
        for seq, p in zip(seqs, positions):
            bt = self._block_table(seq)
            block_idx = p // self.block_size
            if block_idx >= len(bt):
                raise RuntimeError(
                    "Insufficient KV reservation for MEDUSA decode batch: "
                    f"position={p}, required_block_idx={block_idx}, "
                    f"allocated_blocks={len(bt)}"
                )
            slot_list.append(bt[block_idx] * self.block_size + (p % self.block_size))
        slot_t = torch.tensor(slot_list, dtype=torch.int32, pin_memory=True).cuda(
            non_blocking=True
        )
        bt_t = self.prepare_block_tables(seqs)

        set_context(
            False,
            slot_mapping=slot_t,
            context_lens=ctx_t,
            block_tables=bt_t,
        )
        hidden = self._run_decode_hidden(input_ids, pos_t)  # [B, hidden_size]
        reset_context()

        lm_logits = self.model.compute_logits(hidden)  # [B, vocab]
        medusa_logits = torch.stack(
            [head(hidden) for head in self.medusa_heads], dim=0
        )  # [num_heads, B, vocab]
        return lm_logits, medusa_logits

    @torch.inference_mode()
    def _run_medusa_graph(
        self, input_ids: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Run the tree-decode forward, replaying a bucketed CUDA graph when possible.

        The captured graphs are a family indexed by batch size; each graph is
        sized for bs * medusa_len input tokens.  Unused rows in the padded
        buffer are safe no-ops:
          - slot_mapping == -1 → store_kvcache_kernel skips the write
          - prefix_lens == 0   → FlashAttention reads no prefix KV
        """
        if self.enforce_eager or not hasattr(self, "medusa_graphs"):
            return self.model(input_ids, positions)

        medusa_len = self.medusa_len
        N = input_ids.size(0)
        bs = N // medusa_len
        graph_bs = next((x for x in self.medusa_graph_bs if x >= bs), None)
        if graph_bs is None:
            return self.model(input_ids, positions)
        graph_N = graph_bs * medusa_len

        context = get_context()
        gv = self.medusa_graph_vars

        # Copy live values into the pre-allocated graph-vars tensors.
        gv["input_ids"][:N] = input_ids
        gv["positions"][:N] = positions
        gv["slot_mapping"].fill_(-1)
        gv["slot_mapping"][:N] = context.slot_mapping
        gv["prefix_lens"].zero_()
        gv["prefix_lens"][:bs] = context.prefix_lens
        gv["block_tables"][:bs, : context.block_tables.size(1)] = context.block_tables

        # Re-point context to the graph-vars buffers so the replayed graph
        # reads from the same memory addresses it captured.  Each per-bucket
        # graph reads slices [:graph_N] and [:graph_bs] of these tensors.
        set_context(
            False,
            slot_mapping=gv["slot_mapping"][:graph_N],
            context_lens=gv["prefix_lens"][:graph_bs],
            block_tables=gv["block_tables"][:graph_bs],
            is_medusa_tree_decode=True,
            medusa_tree_mask=self.medusa_tree_mask,
            prefix_lens=gv["prefix_lens"][:graph_bs],
        )
        self.medusa_graphs[graph_bs].replay()
        return gv["hidden_outputs"][:N]

    @torch.inference_mode()
    def capture_medusa_cudagraph(self):
        """Capture a family of MEDUSA tree-decode CUDA graphs.

        Each graph handles a fixed batch size bs from medusa_graph_bs, sized
        for bs * medusa_len input tokens.  Smaller actual batches pad into the
        next bucket with slot_mapping=-1 / prefix_lens=0 so the padded rows are
        no-ops (verified in store_kvcache_kernel / FlashAttention semantics).

        The seed/bonus passes reuse the regular decode CUDA graph family
        (self.graphs) via run_medusa_decode_batch, so no bonus graph is
        captured here.
        """
        config = self.config
        hf_config = config.hf_config
        medusa_len = self.medusa_len
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        max_N = max_bs * medusa_len

        input_ids = torch.zeros(max_N, dtype=torch.int64)
        positions = torch.zeros(max_N, dtype=torch.int64)
        slot_mapping = torch.full((max_N,), -1, dtype=torch.int32)
        prefix_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        hidden_outputs = torch.zeros(max_N, hf_config.hidden_size)

        self.medusa_graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.medusa_graphs = {}
        pool = getattr(self, "graph_pool", None)

        for bs in reversed(self.medusa_graph_bs):
            N = bs * medusa_len
            graph = torch.cuda.CUDAGraph()
            set_context(
                False,
                slot_mapping=slot_mapping[:N],
                context_lens=prefix_lens[:bs],
                block_tables=block_tables[:bs],
                is_medusa_tree_decode=True,
                medusa_tree_mask=self.medusa_tree_mask,
                prefix_lens=prefix_lens[:bs],
            )
            # Two warmup runs to let CUDA allocate any lazy buffers before capture.
            for _ in range(2):
                hidden_outputs[:N] = self.model(input_ids[:N], positions[:N])
            torch.cuda.synchronize()

            with torch.cuda.graph(graph, pool):
                hidden_outputs[:N] = self.model(input_ids[:N], positions[:N])
            if pool is None:
                pool = graph.pool()
            self.medusa_graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_pool = pool
        self.medusa_graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            prefix_lens=prefix_lens,
            block_tables=block_tables,
            hidden_outputs=hidden_outputs,
        )
