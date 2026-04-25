import atexit
import os
from dataclasses import fields
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


class LLMEngine:
    def __init__(
        self,
        model_config: Config,
        speculation_mode: SpeculationMode = SpeculationMode.NONE,
        speculator_config: list[Config] | None = None,
        speculation_length: int | None = None,
        # EAGLE-3 needs a [3] list of target-layer indices to fuse for the
        # head's fc input. Not a Config, so it stays as its own kwarg.
        eagle_capture_layer_ids: list[int] | None = None,
        **kwargs,
    ):
        self.speculation_mode = speculation_mode
        self.model_config = model_config
        self.speculator_config = speculator_config
        self.speculation_length = speculation_length
        self.eagle_capture_layer_ids = eagle_capture_layer_ids
        self.eagle_debug = os.environ.get("EAGLE_DEBUG", "0") == "1"
        self._eagle_debug_printed = False
        # tensor parallelism bookkeeping
        # disable TP with specdecode for now
        if (
            speculation_mode is not SpeculationMode.NONE
            and model_config.tensor_parallel_size > 1
        ):
            raise NotImplementedError("Speculation not supported with TP")

        # Mode-specific validation. Naive needs a draft model+length; EAGLE
        # needs a head config + capture layer ids + draft length.
        if speculation_mode is SpeculationMode.NAIVE_SPECULATION:
            if speculator_config is None:
                raise ValueError("speculator_config is required for naive speculation")
            if speculation_length is None or speculation_length < 1:
                raise ValueError(
                    "speculation_length must be a positive integer "
                    "for naive speculation"
                )
        if speculation_mode is SpeculationMode.EAGLE:
            # EAGLE re-uses speculator_config[0] for the head's Config — the
            # `speculation_mode` field already disambiguates how to interpret
            # the entry (head vs separate small model).
            if speculator_config is None or len(speculator_config) != 1:
                raise ValueError(
                    "EAGLE requires speculator_config=[eagle_head_config] "
                    "(exactly one Config for the EAGLE-3 head)"
                )
            if speculation_length is None or speculation_length < 1:
                raise ValueError(
                    "speculation_length must be a positive integer for EAGLE "
                    "(this is the chain draft length K)"
                )
            if eagle_capture_layer_ids is None or len(eagle_capture_layer_ids) != 3:
                raise ValueError(
                    "eagle_capture_layer_ids must be a list of exactly 3 "
                    "target-model layer indices for EAGLE-3 hidden fusion"
                )

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

        # Verifier sees seqlen_q = K+1 query tokens during verify for both
        # naive and EAGLE; only naive can graph-capture it (EAGLE needs eager
        # for hidden-state capture, so verify_seqlen_q is None there).
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
                # When EAGLE is on, the verifier captures mid-layer hiddens
                # for fusion. This forces eager mode internally.
                eagle_capture_layer_ids=(
                    eagle_capture_layer_ids
                    if speculation_mode is SpeculationMode.EAGLE
                    else None
                ),
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

        if speculation_mode is SpeculationMode.EAGLE:
            # The EAGLE head is a dedicated runner with its own (single-layer)
            # paged KV cache. Mirrors the naive two-runner layout but with the
            # is_eagle_head flag flipping the model class to Qwen3Eagle3.
            eagle_head_config = speculator_config[0]
            eagle_head_config.eos = self.tokenizer.eos_token_id
            self.model_runners.append(
                ModelRunner(
                    config=eagle_head_config,
                    rank=0,
                    event=self.events,
                    block_managers=self.block_managers,
                    model_runner_idx=1,
                    is_eagle_head=True,
                )
            )
            block_table_idx = len(self.block_managers)
            self.block_managers.append(
                BlockManager(
                    num_blocks=eagle_head_config.num_kvcache_blocks,
                    block_size=eagle_head_config.kvcache_block_size,
                    block_table_idx=block_table_idx,
                )
            )
            # The published Eagle-3 head checkpoint appears not to ship a
            # separate token embedding table. Reuse the target model's token
            # embeddings so the head can consume target-vocab input ids.
            target_embed = self.model_runners[
                0
            ].model.model.embed_tokens.weight.detach()
            self.model_runners[1].call("load_target_embeddings", target_embed)
            if self.eagle_debug:
                vocab_stats = self.model_runners[1].call("eagle_debug_vocab_stats")
                print(
                    "EAGLE vocab stats: "
                    f"d2t_min={vocab_stats['d2t_min']}, "
                    f"d2t_max={vocab_stats['d2t_max']}, "
                    f"d2t_first8={vocab_stats['d2t_first8']}, "
                    f"t2d_true_count={vocab_stats['t2d_true_count']}"
                )

        # setup scheduler with access to all block managers
        self.scheduler = Scheduler(
            config=model_config,
            block_managers=self.block_managers,
            speculation_mode=self.speculation_mode,
            speculator_config=self.speculator_config,
            speculation_length=self.speculation_length,
        )

        # running acceptance stats — shared across naive and EAGLE
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

    def _naive_specdec_step(self):
        # todo; split the step method into dispatch pattern
        pass

    # ------------------------------------------------------------------
    # EAGLE-3 step
    # ------------------------------------------------------------------

    def _eagle_step(
        self, seqs: list[Sequence], is_prefill: bool
    ) -> tuple[list[list[int]], int, int]:
        """One engine step under EAGLE-3 chain speculation.

        Prefill: target prefill captures per-position fused hidden; eagle
        head's KV is populated over [1..L-1]; seq.eagle_target_fused holds
        the fused hidden at position (L-1) for use as the chain seed.

        Decode: K eagle chain steps produce K draft tokens, then verifier
        verifies via the standard multi-query verify path (with capture).
        Acceptance uses greedy chain matching capped at K-1 — that cap means
        we always have target_fused at the new last-committed position
        (avoids an extra target forward in the all-accept case; see the
        eagle3-open-questions note for the bonus-position-fused issue).
        """
        verifier_idx = 0
        eagle_idx = 1
        verifier = self.model_runners[verifier_idx]
        eagle = self.model_runners[eagle_idx]
        K = self.speculation_length

        if is_prefill:
            with torch.profiler.record_function("eagle.prefill.target"):
                token_ids, fused_all, cu_q = verifier.call(
                    "run_with_capture", seqs, True
                )
            # Slice fused_all per seq using prefill cu_seqlens_q.
            per_seq_fused: list[torch.Tensor] = []
            for i in range(len(seqs)):
                start, end = cu_q[i], cu_q[i + 1]
                per_seq_fused.append(fused_all[start:end])
            # Save target_fused at last prompt position; used as seed for the
            # first chain step of the next decode call.
            for seq, fused in zip(seqs, per_seq_fused):
                if fused.size(0) > 0:
                    seq.eagle_target_fused = fused[-1].clone()
            with torch.profiler.record_function("eagle.prefill.head"):
                eagle.call("eagle_prompt_prefill", seqs, per_seq_fused)
            return [[tok] for tok in token_ids], -1, -1

        # ---- Decode ----
        B = len(seqs)
        # Snapshot per-seq state needed for the chain: last_token, position,
        # and the saved target_fused (one row per seq, [3H]).
        last_tokens = [seq.last_token for seq in seqs]
        last_positions = [len(seq) - 1 for seq in seqs]
        fused_seeds = torch.stack([seq.eagle_target_fused for seq in seqs], dim=0).to(
            eagle.model_dtype
        )

        # Project once on the head runner; the resulting [B, H] is prev_hidden
        # for chain step 1.
        prev_hidden = eagle.call("project_fused_hidden", fused_seeds)

        # Stash drafts on seq.draft_token_ids[eagle_idx] so the verifier's
        # prepare_verify (uses draft_model_idx) sees them — we reuse the same
        # multi-query verify infra naive uses.
        for seq in seqs:
            seq.draft_token_ids[eagle_idx] = []

        chain_inputs_tokens = list(last_tokens)
        chain_inputs_positions = list(last_positions)
        debug_raw_drafts: list[list[int]] = []
        debug_target_drafts: list[list[int]] = []

        with torch.profiler.record_function("eagle.decode.chain_draft"):
            for step in range(K):
                hidden_out, draft_logits = eagle.call(
                    "eagle_step",
                    seqs,
                    chain_inputs_tokens,
                    chain_inputs_positions,
                    prev_hidden,
                )
                # Greedy draft sample (temperature=0 path; chain MVP doesn't
                # support typical/nucleus draft sampling). draft_logits is
                # [B, draft_vocab]; argmax gives a [B] tensor.
                draft_ids = draft_logits.argmax(dim=-1).cpu().tolist()
                # Translate draft-vocab → target-vocab via head's d2t buffer.
                # Done host-side per token; B is small (1 in the first MVP).
                target_ids = [eagle.call("eagle_translate", did) for did in draft_ids]
                if self.eagle_debug and not self._eagle_debug_printed:
                    debug_raw_drafts.append(list(draft_ids))
                    debug_target_drafts.append(list(target_ids))
                for seq, tid in zip(seqs, target_ids):
                    seq.draft_token_ids[eagle_idx].append(tid)
                # Recurrent step: next step's input is this step's drafted
                # token at position (last_position + 1), and prev_hidden is
                # the eagle's current output.
                chain_inputs_tokens = target_ids
                chain_inputs_positions = [p + 1 for p in chain_inputs_positions]
                prev_hidden = hidden_out

        with torch.profiler.record_function("eagle.decode.verify"):
            verif_token_ids, verif_logits, verif_fused = verifier.call(
                "verify_with_capture", seqs, eagle_idx
            )

        token_ids: list[list[int]] = []
        step_drafts = 0
        step_accepted = 0
        with torch.profiler.record_function("eagle.decode.accept_reject"):
            for i, seq in enumerate(seqs):
                drafts = seq.draft_token_ids[eagle_idx]
                # verif input layout: index 0 = last_committed (position L-1),
                # index 1..K = draft positions. argmax_at_index_j predicts
                # the token at position L-1+j+1 = L+j, which is where d_{j+1}
                # was placed. So d_{j+1} should match verif_token_ids[i][j].
                big_tokens = verif_token_ids[i]
                step_drafts += len(drafts)
                # Cap accept_len at K-1 so the new last-committed position is
                # always inside verify (verif covers L-1..L+K-1; capping to
                # K-1 means new committed is at most L-1+(K-1)+1 = L+K-1).
                # See eagle3-open-questions note Q3.
                cap = K - 1
                accepted: list[int] = []
                a = 0
                for j, d in enumerate(drafts):
                    if j >= cap:
                        break
                    if d == big_tokens[j]:
                        accepted.append(d)
                        a += 1
                        continue
                    # mismatch: commit verifier's token here and stop
                    accepted.append(big_tokens[j])
                    break
                else:
                    # Loop completed without break: all `cap` drafts matched
                    # (or drafts was shorter than cap). Commit the verifier's
                    # argmax at index `a` as the bonus / cap-position token.
                    if a < len(big_tokens):
                        accepted.append(big_tokens[a])
                step_accepted += a
                if not accepted:
                    # Defensive: should never happen — verify always emits at
                    # least one argmax. Take big_tokens[0].
                    assert big_tokens, "empty verifier output for seq"
                    accepted.append(big_tokens[0])
                token_ids.append(accepted)

                if self.eagle_debug and not self._eagle_debug_printed and i == 0:
                    print(
                        "EAGLE first decode debug: "
                        f"last_token={last_tokens[0]}, "
                        f"last_position={last_positions[0]}, "
                        f"raw_drafts={[step[0] for step in debug_raw_drafts if step]}, "
                        f"translated_drafts={[step[0] for step in debug_target_drafts if step]}, "
                        f"verifier_tokens={big_tokens}, "
                        f"accepted={accepted}"
                    )
                    self._eagle_debug_printed = True

                # Save target_fused at the new last-committed position. The
                # new committed token sits at position old + a (where old is
                # this seq's prev len). In the verify input, position old + a
                # is at index a + 1 (index 0 is at old - 1).
                idx_in_verify = a + 1
                fused_per_seq = verif_fused[i]
                if idx_in_verify < fused_per_seq.size(0):
                    seq.eagle_target_fused = fused_per_seq[idx_in_verify].clone()
                else:
                    # Defensive — with the K-1 cap this branch shouldn't fire.
                    seq.eagle_target_fused = fused_per_seq[-1].clone()

        # Clear drafts so the next step starts fresh.
        for seq in seqs:
            seq.draft_token_ids[eagle_idx] = []

        return token_ids, step_drafts, step_accepted

    def step(self):
        with torch.profiler.record_function("llm.step.schedule"):
            seqs, is_prefill = self.scheduler.schedule()
        # TODO: gate behavior base don speculation mode
        step_drafts = step_accepted = -1
        if self.speculation_mode is SpeculationMode.EAGLE:
            with torch.profiler.record_function("eagle.step"):
                token_ids, step_drafts, step_accepted = self._eagle_step(
                    seqs, is_prefill
                )
        elif self.speculation_mode is SpeculationMode.NAIVE_SPECULATION:
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
            else:
                # generate draft tokens
                for _ in range(self.speculation_length):
                    with torch.profiler.record_function("spec.decode.drafter_run"):
                        draft_ids, draft_logits = drafter.call("run", seqs, is_prefill)
                    # add draft tokens to the sequence's draft token ids

                    # TODO: update this so that draft_ids is a list of lists and use extend
                    # so we don't do the appending in the draft loop
                    for seq, draft_id, draft_logit in zip(
                        seqs, draft_ids, draft_logits
                    ):
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
        if self.speculation_mode is SpeculationMode.EAGLE and not is_prefill:
            # EAGLE commits up to K tokens per decode step (accept_len drafts
            # + 1 verifier token). Report the actual count so throughput
            # display reflects real token rate, not just accepted drafts.
            num_tokens = -sum(len(t) for t in token_ids)
        else:
            num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -step_accepted
        # step_drafts/step_accepted are -1 for prefill and non-spec steps so
        # callers can distinguish "no spec this step" from a genuine 0-draft
        # batch. Caller aggregates; see generate() / bench for reporting.
        # TODO: richer metrics — per-step distribution of accepted-run
        # lengths, time spent in drafter vs verifier, etc.
        return outputs, num_tokens, step_drafts, step_accepted

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.0
        idx = 0
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens, step_drafts, step_accepted = self.step()
            # accumulate spec metrics regardless of use_tqdm — caller dumps
            # aggregates (see bench.py). -1 sentinel = prefill or non-spec.
            if step_drafts > 0:
                self.spec_drafts_total += step_drafts
                self.spec_accepted_total += step_accepted
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                    decode_throughput = 0
                else:
                    prefill_throughput = 0
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix(
                    {
                        "Prefill": f"{int(prefill_throughput)}tok/s",
                        "Decode": f"{int(decode_throughput)}tok/s",
                    }
                )
                idx += 1
                print(
                    f"Step {idx}: Prefill {prefill_throughput:.2f}tok/s, Decode {decode_throughput:.2f}tok/s\n"
                )
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [
            {"text": self.tokenizer.decode(token_ids), "token_ids": token_ids}
            for token_ids in outputs
        ]
        if use_tqdm:
            pbar.close()
        return outputs
