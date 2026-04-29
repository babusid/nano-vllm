import atexit
import csv
from dataclasses import fields
import os
import time
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.engine import block_manager
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.speculation import SpeculationMode
from nanovllm.engine.medusa_utils import (
    generate_medusa_buffers,
    generate_candidates,
    evaluate_posterior,
    vicuna_33b_heads2_fast,
)


class LLMEngine:
    def __init__(
        self,
        model_config: Config,
        speculation_mode: SpeculationMode = SpeculationMode.NONE,
        speculator_config: list[Config] | None = None,
        speculation_length: int | None = None,
        # MEDUSA-specific params — ignored in none/naive modes
        medusa_model_path: str | None = None,
        medusa_choices: list | None = None,
        medusa_num_heads: int = 4,
        medusa_num_layers: int = 1,
        **kwargs,
    ):
        self.speculation_mode = speculation_mode
        self.model_config = model_config
        self.speculator_config = speculator_config
        self.speculation_length = speculation_length

        # tensor parallelism bookkeeping — TP not supported with any spec-dec mode
        if (
            speculation_mode is not SpeculationMode.NONE
            and model_config.tensor_parallel_size > 1
        ):
            raise NotImplementedError("Speculation not supported with TP")

        # Mode-specific param validation
        if speculation_mode is SpeculationMode.NAIVE_SPECULATION:
            if speculator_config is None:
                raise ValueError("speculator_config is required for naive speculation")
            if speculation_length is None or speculation_length < 1:
                raise ValueError(
                    "speculation_length must be a positive integer for naive speculation"
                )
        if speculation_mode is SpeculationMode.MEDUSA:
            if medusa_model_path is None:
                raise ValueError("medusa_model_path is required for MEDUSA speculation")

        # Pre-compute static MEDUSA tree buffers once at engine start.
        # These are passed to ModelRunner (for attention / head computation)
        # and Scheduler (for KV slot reservation).
        medusa_buffers: dict | None = None
        if speculation_mode is SpeculationMode.MEDUSA:
            if medusa_choices is not None:
                choices = medusa_choices
            else:
                choices = vicuna_33b_heads2_fast
            # Clip the tree to paths compatible with the available number of heads.
            # At depth d (path length d), generate_candidates maps those nodes to
            # flat-candidate indices  cur[-1] + TOPK * (d-1) + 1.  The flat vector
            # has size 1 + num_heads * TOPK, so valid depths are 1..num_heads only.
            # Paths longer than num_heads would produce out-of-range indices and
            # cause CUDA index-out-of-bounds assertions at runtime.
            choices = [c for c in choices if len(c) <= medusa_num_heads]
            if not choices:
                raise ValueError(
                    f"No valid medusa_choices for medusa_num_heads={medusa_num_heads}. "
                    "All paths in the topology exceed the number of heads."
                )
            medusa_buffers = generate_medusa_buffers(choices, device="cuda")
        self.medusa_buffers = medusa_buffers

        self.block_managers: list[BlockManager] = []
        self.model_runners: list[ModelRunner] = []
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, model_config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(
                target=ModelRunner,
                kwargs={
                    "config": model_config,
                    "rank": i,
                    "event": event,
                    "block_managers": self.block_managers,
                    "model_runner_idx": 0,
                },
            )
            process.start()
            self.ps.append(process)
            self.events.append(event)

        # setup tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config.model, use_fast=True
        )
        model_config.eos = self.tokenizer.eos_token_id

        # setup model runner(s), one for each model instance that we need to run
        # Tell the verifier how many query tokens per seq to expect during verify
        # so it can capture a matching family of CUDA graphs.
        verify_seqlen_q = (
            speculation_length + 1
            if speculation_mode is SpeculationMode.NAIVE_SPECULATION
            else None
        )
        self.model_runners.append(
            ModelRunner(
                config=model_config,
                rank=0,
                event=self.events,
                block_managers=self.block_managers,
                model_runner_idx=0,
                verify_seqlen_q=verify_seqlen_q,
                # MEDUSA params (all None/0 in non-MEDUSA modes)
                medusa_model_path=medusa_model_path,
                medusa_num_heads=medusa_num_heads,
                medusa_num_layers=medusa_num_layers,
                medusa_buffers=medusa_buffers,
            )
        )

        self.block_managers.append(
            BlockManager(
                num_blocks=model_config.num_kvcache_blocks,
                block_size=model_config.kvcache_block_size,
                block_table_idx=0,
            )
        )

        if speculation_mode is SpeculationMode.NAIVE_SPECULATION:
            if len(speculator_config) > 1:
                raise NotImplementedError(
                    "Naive Speculation with multiple models not supported"
                )
            speculator_config[0].eos = self.tokenizer.eos_token_id
            self.model_runners.append(
                ModelRunner(
                    config=speculator_config[0],
                    rank=0,
                    event=self.events,
                    block_managers=self.block_managers,
                    model_runner_idx=1,
                )
            )
            # idx should be length of current array
            block_table_idx = len(self.block_managers)
            self.block_managers.append(
                BlockManager(
                    num_blocks=speculator_config[0].num_kvcache_blocks,
                    block_size=speculator_config[0].kvcache_block_size,
                    block_table_idx=block_table_idx,
                )
            )

        # setup scheduler with access to all block managers
        medusa_reserve_tokens = None
        if medusa_buffers is not None:
            # MEDUSA can commit up to one full retrieved path plus one bonus token
            # per step; reserve enough KV slots for that worst case.
            medusa_reserve_tokens = max(
                medusa_buffers["medusa_len"],
                int(medusa_buffers["retrieve_indices"].size(1)) + 1,
            )

        self.scheduler = Scheduler(
            config=model_config,
            block_managers=self.block_managers,
            speculation_mode=self.speculation_mode,
            speculator_config=self.speculator_config,
            speculation_length=self.speculation_length,
            medusa_len=medusa_reserve_tokens,
        )

        # running acceptance stats for spec-dec modes
        self.spec_drafts_total = 0
        self.spec_accepted_total = 0

        # register cleanup hook for tp processes
        atexit.register(self.exit)

    def exit(self):
        self.model_runners[0].call("exit")
        del self.model_runners
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(
            token_ids=prompt,
            sampling_params=sampling_params,
            num_block_tables=len(self.block_managers),
        )
        self.scheduler.add(seq)

    def _naive_specdec_step(
        self, seqs: list[Sequence], is_prefill: bool
    ) -> tuple[list[list[int]], int, int]:
        # get the two model runners for regular specdec
        verifier_model_idx = 0
        drafter_model_idx = 1
        verifier = self.model_runners[verifier_model_idx]
        drafter = self.model_runners[drafter_model_idx]
        if is_prefill:
            # fill the kv of both models, but ignore the draft token
            with torch.profiler.record_function("spec.prefill.drafter_run"):
                drafter.call("run", seqs, is_prefill)
            with torch.profiler.record_function("spec.prefill.verifier_run"):
                token_ids, _ = verifier.call("run", seqs, is_prefill)
            token_ids = [[tok] for tok in token_ids]
            return token_ids, -1, -1
        else:
            # generate draft tokens
            for _ in range(self.speculation_length):
                with torch.profiler.record_function("spec.decode.drafter_run"):
                    draft_ids, draft_logits = drafter.call("run", seqs, is_prefill)
                # add draft tokens to the sequence's draft token ids

                # TODO: update this so that draft_ids is a list of lists and use extend
                # so we don't do the appending in the draft loop
                for seq, draft_id, draft_logit in zip(seqs, draft_ids, draft_logits):
                    seq.draft_token_ids[drafter_model_idx].append(draft_id)
                    seq.draft_token_logits[drafter_model_idx].append(draft_logit)

            # generate logits for the draft tokens
            # ignore the token that gets generated
            with torch.profiler.record_function("spec.decode.verifier_verify"):
                verif_token_ids, verif_logits = verifier.call(
                    "verify", seqs, drafter_model_idx
                )

            # accept/reject per sequence
            token_ids = []
            step_drafts = 0
            step_accepted = 0
            with torch.profiler.record_function("spec.decode.accept_reject"):
                for idx, seq in enumerate(seqs):
                    draft_tokens = seq.draft_token_ids[drafter_model_idx]
                    small_logits = seq.draft_token_logits[drafter_model_idx]
                    big_token_ids = verif_token_ids[idx]
                    draft_big_logits = verif_logits[idx][:-1]
                    seq_accept = []
                    step_drafts += len(draft_tokens)
                    seq_accepted_drafts = 0
                    if seq.temperature <= 0:
                        for tok, big in zip(draft_tokens, draft_big_logits):
                            verifier_argmax = int(big.argmax(dim=-1).item())
                            if tok == verifier_argmax:
                                seq_accept.append(tok)
                                seq_accepted_drafts += 1
                                continue
                            seq_accept.append(verifier_argmax)
                            break
                    else:
                        for tok, small, big in zip(
                            draft_tokens, small_logits, draft_big_logits
                        ):
                            # upcast to fp32 — fp16 logits (esp. Vicuna-33B) can
                            # overflow and poison softmax with inf/nan, which
                            # propagates into residual and trips multinomial's
                            # probability-validity assert.
                            small_prob_dist = small.float().softmax(dim=-1)
                            big_prob_dist = big.float().softmax(dim=-1)
                            p_small = small_prob_dist[tok]
                            p_big = big_prob_dist[tok]
                            accept = p_big >= p_small
                            if not accept:
                                accept = p_big.new_empty(()).uniform_() < (
                                    p_big / (p_small + 1e-12)
                                )
                            if accept:
                                seq_accept.append(tok)
                                seq_accepted_drafts += 1
                                continue
                            residual = (big_prob_dist - small_prob_dist).clamp_min(0)
                            rsum = residual.sum()
                            if rsum <= 0 or not torch.isfinite(rsum):
                                # big ≤ small everywhere (or non-finite): fall
                                # back to sampling from the target distribution
                                bonus_token = big_prob_dist.multinomial(1).item()
                            else:
                                bonus_token = (residual / rsum).multinomial(1).item()
                            seq_accept.append(bonus_token)
                            break
                    if seq_accepted_drafts == len(draft_tokens) and draft_tokens:
                        assert len(big_token_ids) == len(draft_tokens) + 1
                        seq_accept.append(big_token_ids[-1])
                    step_accepted += seq_accepted_drafts
                    if not seq_accept:
                        assert big_token_ids
                        seq_accept.append(big_token_ids[0])

                    token_ids.append(seq_accept)

            # empty draft token list
            for seq in seqs:
                seq.draft_token_ids[1] = []
                seq.draft_token_logits[1] = []

            return token_ids, step_drafts, step_accepted

    # ------------------------------------------------------------------
    # MEDUSA step logic
    # ------------------------------------------------------------------

    def _medusa_step(
        self, seqs: list[Sequence], is_prefill: bool
    ) -> tuple[list[list[int]], int, int]:
        """Execute one MEDUSA engine step for a batch of sequences.

        The entire step runs in a single batched pipeline with exactly one
        GPU→CPU synchronization point (after posterior evaluation), regardless
        of batch size:

          1. Batched seed pass (only seqs that just prefilled)
          2. Batched candidate tree construction
          3. Single batched tree-decode forward (CUDA-graphed)
          4. Batched posterior evaluation + bonus sampling + sync
          5. Batched KV scatter to commit accepted paths into sequential slots
          6. Batched bonus-token seed pass (only seqs not hitting EOS)

        Returns
        -------
        token_ids      : list[list[int]]  — new tokens per sequence (for postprocess)
        step_drafts    : int              — speculative candidates tried (-1 for prefill)
        step_accepted  : int              — speculative tokens accepted (-1 for prefill)
        """
        runner = self.model_runners[0]
        buffers = self.medusa_buffers
        retrieve_indices = buffers["retrieve_indices"]  # [num_paths, depth+1] CUDA
        tree_indices = buffers["tree_indices"]  # [medusa_len] CUDA
        medusa_len = buffers["medusa_len"]
        eos = self.model_config.eos

        # ---- Prefill ----
        if is_prefill:
            token_ids, _ = runner.call("run", seqs, True)
            # Clear MEDUSA logits so the first decode step triggers the seed pass.
            for seq in seqs:
                seq.medusa_lm_logits = None
                seq.medusa_head_logits = None
            return [[tok] for tok in token_ids], -1, -1

        # ---- Decode (any batch size) ----
        B = len(seqs)
        # Uniform-temperature assumption — keeps generate_candidates /
        # evaluate_posterior fully vectorized.  Benchmarks use a single
        # SamplingParams so this is the common case; relax to per-bucket
        # loops if mixed-temperature batches are ever needed.
        temperature = seqs[0].temperature
        if any(s.temperature != temperature for s in seqs):
            raise NotImplementedError(
                "MEDUSA batch currently requires uniform sampling temperature across seqs"
            )

        # ── [1] Batched seed pass for seqs whose logits were cleared ─────────
        seed_idx = [i for i, s in enumerate(seqs) if s.medusa_lm_logits is None]
        if seed_idx:
            seed_seqs = [seqs[i] for i in seed_idx]
            lm_s, head_s = runner.call(
                "run_medusa_decode_batch",
                seed_seqs,
                [s.last_token for s in seed_seqs],
                [len(s) - 1 for s in seed_seqs],  # seed writes KV at last committed pos
                [len(s) for s in seed_seqs],  # context_len = committed length
            )
            # lm_s: [S, V]      head_s: [H, S, V]
            for j, i in enumerate(seed_idx):
                seqs[i].medusa_lm_logits = lm_s[j : j + 1].unsqueeze(1)  # [1,1,V]
                seqs[i].medusa_head_logits = head_s[:, j : j + 1].unsqueeze(
                    2
                )  # [H,1,1,V]

        # ── [2] Assemble batched logits stacks ───────────────────────────────
        lm_stack = torch.cat([s.medusa_lm_logits for s in seqs], dim=0)  # [B,1,V]
        head_stack = torch.cat([s.medusa_head_logits for s in seqs], dim=1)  # [H,B,1,V]

        # ── [3] Batched candidate tree construction ──────────────────────────
        cart_candidates, tree_candidates = generate_candidates(
            head_stack,
            lm_stack,
            tree_indices,
            retrieve_indices,
            temperature=temperature,
        )
        # cart_candidates : [B, num_paths, depth+1]
        # tree_candidates : [B, medusa_len]

        # ── [4] Single batched tree-decode forward (CUDA-graphed) ────────────
        lm_logits, _ = runner.call("run_medusa_tree", seqs, tree_candidates)
        # lm_logits : [B*medusa_len, vocab]
        V = lm_logits.size(-1)
        lm_logits_b = lm_logits.view(B, medusa_len, V)

        # ── [5] Batched posterior evaluation ─────────────────────────────────
        path_logits = lm_logits_b[:, retrieve_indices]  # [B, num_paths, depth+1, V]
        best_candidate, accept_length = evaluate_posterior(
            path_logits,
            cart_candidates,
            temperature=temperature,
        )
        # best_candidate : [B]    accept_length : [B]

        # ── [6] Batched bonus-token sampling ─────────────────────────────────
        batch_idx = torch.arange(B, device=lm_logits_b.device)
        accept_nodes = retrieve_indices[best_candidate, accept_length]  # [B]
        bonus_logits = lm_logits_b[batch_idx, accept_nodes]  # [B, V]
        if temperature == 0:
            bonus_tokens = bonus_logits.argmax(dim=-1)  # [B]
        else:
            # fp32 upcast — same overflow concern as generate_candidates /
            # evaluate_posterior: at low temperatures, fp16 logits / temperature
            # overflow to +inf and softmax returns NaN.
            bonus_probs = torch.softmax(bonus_logits.float() / temperature, dim=-1)
            bonus_tokens = torch.multinomial(bonus_probs, 1).squeeze(-1)

        # ── [7] Gather the accepted cartesian paths (still on GPU) ──────────
        chosen_paths = cart_candidates[batch_idx, best_candidate]  # [B, depth+1]

        # ── [8] Batched KV scatter for accepted paths ────────────────────────
        # Keep accepted-path slot calculation on GPU; Python only consumes the
        # final token decisions below for scheduler/postprocess bookkeeping.
        chosen_nodes = retrieve_indices[best_candidate]  # [B, depth+1]
        runner.call("copy_accepted_kv_slots_medusa", seqs, chosen_nodes, accept_length)

        # ── [9] Batched bonus pass (all seqs, still GPU-driven) ──────────────
        # EOS sequences will be discarded during Python bookkeeping below; doing
        # the full batch here avoids a pre-bonus GPU→CPU sync for filtering.
        lm_seed, head_seed = runner.call(
            "run_medusa_bonus_batch", seqs, bonus_tokens, accept_length
        )

        # Single GPU→CPU sync for scheduler/postprocess decisions this step.
        # We pack the tensors needed by Python bookkeeping so we only
        # materialize one host transfer in the hot path.
        packed = torch.cat(
            [
                accept_length.unsqueeze(1),
                bonus_tokens.unsqueeze(1),
                chosen_paths,
            ],
            dim=1,
        )
        packed_cpu = packed.cpu().tolist()
        path_width = chosen_paths.size(1)

        # ── Build final per-seq token lists, deciding bonus eligibility ─────
        out_tokens: list[list[int]] = []
        total_accepted = 0
        for b in range(B):
            row = packed_cpu[b]
            accept_len_b = row[0]
            bonus_token_b = row[1]
            accepted = row[2 : 2 + path_width][: accept_len_b + 1]
            total_accepted += accept_len_b
            if eos in accepted:
                # Sequence will finish — skip bonus pass to avoid a post-EOS token.
                out_tokens.append(accepted)
                seqs[b].medusa_lm_logits = None
                seqs[b].medusa_head_logits = None
            else:
                out_tokens.append(accepted + [bonus_token_b])
                seqs[b].medusa_lm_logits = lm_seed[b : b + 1].unsqueeze(1)  # [1,1,V]
                seqs[b].medusa_head_logits = head_seed[:, b : b + 1].unsqueeze(
                    2
                )  # [H,1,1,V]

        step_drafts = B * (medusa_len - 1)
        step_accepted = total_accepted
        return out_tokens, step_drafts, step_accepted

    def step(self):
        with torch.profiler.record_function("llm.step.schedule"):
            seqs, is_prefill = self.scheduler.schedule()
        step_drafts = step_accepted = -1

        if self.speculation_mode is SpeculationMode.MEDUSA:
            with torch.profiler.record_function("medusa.step"):
                token_ids, step_drafts, step_accepted = self._medusa_step(
                    seqs, is_prefill
                )

        elif self.speculation_mode is SpeculationMode.NAIVE_SPECULATION:
            with torch.profiler.record_function("spec.step"):
                token_ids, step_drafts, step_accepted = self._naive_specdec_step(
                    seqs, is_prefill
                )
        else:
            with torch.profiler.record_function("base.run"):
                token_ids, _ = self.model_runners[0].call("run", seqs, is_prefill)
            token_ids = [[tok] for tok in token_ids]
            step_accepted = len(seqs)

        with torch.profiler.record_function("llm.step.postprocess"):
            self.scheduler.postprocess(seqs, token_ids)
        outputs = [
            (seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished
        ]
        step_batch_size = len(seqs)
        if is_prefill:
            step_tokens_proposed = -1
            step_tokens_accepted = -1
            num_tokens = sum(len(seq) for seq in seqs)
        elif self.speculation_mode is SpeculationMode.NONE:
            step_tokens_proposed = step_batch_size
            step_tokens_accepted = step_batch_size
            num_tokens = -step_tokens_accepted
        elif self.speculation_mode is SpeculationMode.NAIVE_SPECULATION:
            # Include verifier token in both proposed/accepted accounting.
            step_tokens_proposed = step_batch_size * (self.speculation_length + 1)
            step_tokens_accepted = step_batch_size + step_accepted
            num_tokens = -step_tokens_accepted
        else:
            # MEDUSA: proposals are all tree nodes (root + speculative nodes);
            # accepted token accounting should reflect actual emitted tokens:
            # root + accepted speculative tokens + optional bonus token
            # (bonus is skipped for EOS sequences).
            medusa_len = self.medusa_buffers["medusa_len"]
            step_tokens_proposed = step_batch_size * medusa_len
            step_tokens_accepted = sum(
                len(seq_token_ids) for seq_token_ids in token_ids
            )
            num_tokens = -step_tokens_accepted
        # step_drafts/step_accepted are -1 for prefill and non-spec steps so
        # callers can distinguish "no spec this step" from a genuine 0-draft
        # batch. Caller aggregates; see generate() / bench for reporting.
        # TODO: richer metrics — per-step distribution of accepted-run
        # lengths, time spent in drafter vs verifier, etc.
        return (
            outputs,
            num_tokens,
            step_drafts,
            step_accepted,
            step_batch_size,
            step_tokens_accepted,
            step_tokens_proposed,
        )

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        capture_throughput = os.environ.get("CAPTURE_THROUGHPUT_TRACE", "0") == "1"
        throughput_path = os.environ.get("THROUGHPUT_TRACE_PATH", "/tmp/data.csv")
        throughput_file = None
        throughput_writer = None
        cumulative_generated_tokens = 0
        if capture_throughput:
            throughput_file = open(throughput_path, "w", newline="", encoding="utf-8")
            throughput_writer = csv.writer(throughput_file)
            throughput_writer.writerow(
                [
                    "timestamp",
                    "prefill_tput_tok_s",
                    "decode_tput_tok_s",
                    "total_generated_tokens",
                    "step_batch_size",
                    "step_tokens_accepted",
                    "step_tokens_proposed",
                ]
            )
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.0
        try:
            while not self.is_finished():
                t = perf_counter()
                (
                    output,
                    num_tokens,
                    step_drafts,
                    step_accepted,
                    step_batch_size,
                    step_tokens_accepted,
                    step_tokens_proposed,
                ) = self.step()
                elapsed = perf_counter() - t
                # accumulate spec metrics regardless of use_tqdm — caller dumps
                # aggregates (see bench.py). -1 sentinel = prefill or non-spec.
                if step_drafts > 0:
                    self.spec_drafts_total += step_drafts
                    self.spec_accepted_total += step_accepted

                if num_tokens > 0:
                    prefill_throughput = num_tokens / elapsed
                    decode_throughput = 0.0
                    # Prefill emits one completion token per active sequence.
                    # Include these so CSV total_generated_tokens matches
                    # bench.py's final output-token accounting.
                    cumulative_generated_tokens += step_batch_size
                else:
                    prefill_throughput = 0.0
                    decode_throughput = -num_tokens / elapsed
                    cumulative_generated_tokens += -num_tokens

                if throughput_writer is not None:
                    throughput_writer.writerow(
                        [
                            time.time(),
                            f"{prefill_throughput:.6f}",
                            f"{decode_throughput:.6f}",
                            cumulative_generated_tokens,
                            step_batch_size,
                            step_tokens_accepted,
                            step_tokens_proposed,
                        ]
                    )
                    throughput_file.flush()

                if use_tqdm:
                    pbar.set_postfix(
                        {
                            "Prefill": f"{int(prefill_throughput)}tok/s",
                            "Decode": f"{int(decode_throughput)}tok/s",
                        }
                    )
                for seq_id, token_ids in output:
                    outputs[seq_id] = token_ids
                    if use_tqdm:
                        pbar.update(1)
        finally:
            if throughput_file is not None:
                throughput_file.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [
            {"text": self.tokenizer.decode(token_ids), "token_ids": token_ids}
            for token_ids in outputs
        ]
        if use_tqdm:
            pbar.close()
        return outputs
