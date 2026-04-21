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
from nanovllm.engine.medusa_utils import ResBlock
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
        self.medusa_tree_mask = medusa_buffers["medusa_attn_mask"] if medusa_buffers else None
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
        # Each head is medusa_num_layers ResBlocks followed by a linear projection
        # to vocab_size. Weights are read from medusa_lm_head.pt in the checkpoint.
        if medusa_model_path:
            hidden_size = hf_config.hidden_size
            vocab_size = hf_config.vocab_size
            self.medusa_heads = nn.ModuleList([
                nn.Sequential(
                    *[ResBlock(hidden_size) for _ in range(medusa_num_layers)],
                    nn.Linear(hidden_size, vocab_size, bias=False),
                )
                for _ in range(medusa_num_heads)
            ])
            head_ckpt = os.path.join(medusa_model_path, "medusa_lm_head.pt")
            state = torch.load(head_ckpt, map_location="cuda")
            self.medusa_heads.load_state_dict(state, strict=False)
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
                # MEDUSA mode: single tree-decode graph, no regular decode graph
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
            if hasattr(self, "medusa_graph"):
                del self.medusa_graph, self.medusa_graph_vars
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
        self, seqs: list, tree_candidates_map: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build input tensors for one MEDUSA tree-decode step (batch size = 1).

        The scheduler has pre-reserved medusa_len extra KV slots per sequence,
        so we can safely write tree-candidate K/V there.  prefix_lens is set
        to the committed length so the prefix FA call ignores those slots.
        """
        assert len(seqs) == 1, "MEDUSA tree decode requires batch size 1"
        seq = seqs[0]
        block_table = self._block_table(seq)
        committed_len = len(seq)
        medusa_len = self.medusa_len

        # generate_candidates returns [1, medusa_len]; flatten to 1-D so it
        # matches the flat input_ids layout used everywhere in ModelRunner.
        tree_tokens = tree_candidates_map[seq.seq_id].view(-1)  # [medusa_len], on CUDA

        # RoPE positions: root at committed_len, depth-d nodes at committed_len+d
        medusa_position_ids = self.medusa_buffers["medusa_position_ids"]  # [medusa_len]
        positions = committed_len + medusa_position_ids  # [medusa_len], CUDA

        # One paged-KV slot per tree node in the reserved region
        slot_list = []
        for i in range(medusa_len):
            token_idx = committed_len + i
            slot_list.append(
                block_table[token_idx // self.block_size] * self.block_size
                + token_idx % self.block_size
            )
        slot_mapping = torch.tensor(
            slot_list, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)

        prefix_lens = torch.tensor(
            [committed_len], dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)

        max_bt_len = max(len(block_table), 1)
        block_tables = torch.tensor(
            [block_table + [-1] * (max_bt_len - len(block_table))],
            dtype=torch.int32,
            pin_memory=True,
        ).cuda(non_blocking=True)

        set_context(
            False,
            slot_mapping=slot_mapping,
            context_lens=prefix_lens,   # not used in tree path but kept for consistency
            block_tables=block_tables,
            is_medusa_tree_decode=True,
            medusa_tree_mask=self.medusa_tree_mask,
            prefix_lens=prefix_lens,
        )
        return tree_tokens, positions

    @torch.inference_mode()
    def run_medusa_tree(
        self, seqs: list, tree_candidates_map: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one MEDUSA tree-decode step.

        Returns:
            lm_logits:    [medusa_len, vocab_size]
            medusa_logits:[num_medusa_heads, medusa_len, vocab_size]
        """
        input_ids, positions = self.prepare_medusa_tree(seqs, tree_candidates_map)
        hidden = self._run_medusa_graph(input_ids, positions)
        lm_logits = self.model.compute_logits(hidden)
        medusa_logits = torch.stack(
            [head(hidden) for head in self.medusa_heads], dim=0
        )
        reset_context()
        return lm_logits, medusa_logits

    @torch.inference_mode()
    def run_medusa_kv_and_seed(
        self,
        seq: "Sequence",
        new_tokens: list[int],
        start_pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write correct sequential KV for new_tokens and return seed logits.

        This serves two purposes:
          1. *Corrective pass* (after tree-decode acceptance): the tree wrote KV
             for accepted positions into non-sequential slots determined by tree
             layout.  Re-running the model on the accepted path places correct
             K/V at the sequential logical positions start_pos..start_pos+n-1,
             which prefix-attention in the next tree step will read.
          2. *Seed pass* (first decode step after prefill): writes the K/V for
             the first generated token (at start_pos = len(seq) - 1) and obtains
             LM + MEDUSA-head logits needed to build the first candidate tree.

        Args:
            seq:        The sequence (already updated with all committed tokens).
            new_tokens: Tokens to be placed at positions start_pos..start_pos+n-1.
            start_pos:  Logical position of the first new token in new_tokens.

        Returns:
            lm_logits:     [vocab_size]       — LM-head output at last new position.
            medusa_logits: [num_heads, vocab] — MEDUSA-head outputs at last position.
        """
        n = len(new_tokens)
        block_table = self._block_table(seq)

        input_ids = torch.tensor(
            new_tokens, dtype=torch.int64, pin_memory=True
        ).cuda(non_blocking=True)
        positions = torch.arange(
            start_pos, start_pos + n, dtype=torch.int64
        ).cuda(non_blocking=True)

        slot_list = [
            block_table[(start_pos + i) // self.block_size] * self.block_size
            + (start_pos + i) % self.block_size
            for i in range(n)
        ]
        slot_mapping = torch.tensor(
            slot_list, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)

        # context_lens = total KV positions seen by the LAST query token.
        # flash_attn_with_kvcache with causal=True correctly handles seqlen_q > 1:
        # query i attends to 0..context_lens-n+i (the committed prefix + prior
        # queries in this group), matching standard autoregressive semantics.
        context_lens = torch.tensor(
            [start_pos + n], dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)

        max_bt_len = max(len(block_table), 1)
        block_tables_t = torch.tensor(
            [block_table + [-1] * (max_bt_len - len(block_table))],
            dtype=torch.int32,
            pin_memory=True,
        ).cuda(non_blocking=True)

        set_context(
            False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables_t,
        )
        hidden = self.model(input_ids, positions)   # [n, hidden_size]
        reset_context()

        last_hidden = hidden[-1:]                   # [1, hidden_size]
        lm_logits = self.model.compute_logits(last_hidden).squeeze(0)   # [vocab]
        medusa_logits = torch.stack(
            [head(last_hidden).squeeze(0) for head in self.medusa_heads], dim=0
        )  # [num_heads, vocab]
        return lm_logits, medusa_logits

    @torch.inference_mode()
    def _run_medusa_graph(
        self, input_ids: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Run the model forward for tree decode, replaying the CUDA graph when possible."""
        if self.enforce_eager or not hasattr(self, "medusa_graph"):
            return self.model(input_ids, positions)

        context = get_context()
        gv = self.medusa_graph_vars
        # Copy live values into the pre-allocated graph-vars tensors
        # (these are the exact GPU buffers the captured graph reads from).
        gv["input_ids"][:] = input_ids
        gv["positions"][:] = positions
        gv["slot_mapping"][:] = context.slot_mapping
        gv["prefix_lens"][:] = context.prefix_lens
        bt = context.block_tables          # [1, actual_num_blocks]
        gv["block_tables"].zero_()
        gv["block_tables"][:1, : bt.size(1)] = bt

        # Re-point context to the graph-vars buffers so the replayed graph
        # reads from the same memory addresses it captured.
        set_context(
            False,
            slot_mapping=gv["slot_mapping"],
            context_lens=gv["prefix_lens"],
            block_tables=gv["block_tables"],
            is_medusa_tree_decode=True,
            medusa_tree_mask=self.medusa_tree_mask,
            prefix_lens=gv["prefix_lens"],
        )
        self.medusa_graph.replay()
        return gv["hidden_outputs"][:]

    @torch.inference_mode()
    def capture_medusa_cudagraph(self):
        """Capture a single CUDA graph for the MEDUSA tree-decode forward pass.

        Fixed shape: [medusa_len] input tokens, batch size 1.  The graph covers
        only the transformer forward (model.__call__); compute_logits and the
        MEDUSA heads run eagerly after replay — they're cheap linear layers.
        """
        config = self.config
        hf_config = config.hf_config
        medusa_len = self.medusa_len
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size

        input_ids = torch.zeros(medusa_len, dtype=torch.int64)
        positions = torch.zeros(medusa_len, dtype=torch.int64)
        slot_mapping = torch.full((medusa_len,), -1, dtype=torch.int32)
        prefix_lens = torch.zeros(1, dtype=torch.int32)
        block_tables = torch.zeros(1, max_num_blocks, dtype=torch.int32)
        hidden_outputs = torch.zeros(medusa_len, hf_config.hidden_size)

        set_context(
            False,
            slot_mapping=slot_mapping,
            context_lens=prefix_lens,
            block_tables=block_tables,
            is_medusa_tree_decode=True,
            medusa_tree_mask=self.medusa_tree_mask,
            prefix_lens=prefix_lens,
        )

        # Two warmup runs to let CUDA allocate any lazy buffers before capture.
        for _ in range(2):
            hidden_outputs[:] = self.model(input_ids, positions)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        pool = getattr(self, "graph_pool", None)
        with torch.cuda.graph(graph, pool):
            hidden_outputs[:] = self.model(input_ids, positions)
        if pool is None:
            self.graph_pool = graph.pool()

        torch.cuda.synchronize()
        reset_context()

        self.medusa_graph = graph
        self.medusa_graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            prefix_lens=prefix_lens,
            block_tables=block_tables,
            hidden_outputs=hidden_outputs,
        )
