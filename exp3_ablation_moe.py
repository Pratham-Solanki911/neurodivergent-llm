"""
Experiment 3: Knob-by-Knob Mechanism Ablation + Thinking-Enabled OCD + Dual-System Pairing
Target hero model: Qwen/Qwen3-30B-A3B (30.5B MoE, 128 experts, top-8, QK-Norm) on A100-40GB (4-bit NF4).

Tests each structural mechanism in isolation AND combined against plain Temperature controls:
  - Baseline (T=0.7), Temp-High (T=1.2), Temp-Low (T=0.2)
  - ADHD ablations: Router-Gumbel-only, Attn-Novelty-only, DMN-Steer-only, ADHD-Tuned
  - OCD ablations:  Router-Sharpen-only, Attn-Sharpen-only, ACC-Steer-only, OCD-Tuned
  - Dual-System ("Medication"): ADHD Ideator -> OCD Verifier/Formatter
"""

import argparse
import json
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from exp1_phenotypes import (
    ADHD_CONTRAST_PAIRS,
    OCD_CONTRAST_PAIRS,
    DIVERGENT_PROMPTS,
    extract_contrastive_vectors,
    distinct_n,
    cross_domain_lexical_spread,
    repetition_loop_score,
    doubt_marker_count,
    _check_exact_json,
)


# ---------------------------------------------------------------------------
# 1. Hard Verification & Constraint Suite (10 tasks with tricky traps)
# ---------------------------------------------------------------------------

HARD_STRICT_TASKS = [
    {
        "id": "no_letter_e",
        "prompt": "Write a 2-sentence description of a cat without using the letter 'e' or 'E' anywhere in your final response. Output ONLY the 2 sentences.",
        "check": lambda txt: ("e" not in txt.lower()) and (1 <= len([s for s in re.split(r"[.!?]+", txt.strip()) if s.strip()]) <= 2) and len(txt.split()) >= 6,
    },
    {
        "id": "exact_json",
        "prompt": 'Return ONLY a valid JSON object with exact keys "status", "count", and "tags" (a list of exactly 3 lowercase strings, none containing the letter "a"). No markdown fences, no extra text.',
        "check": lambda txt: _check_exact_json(txt) and all("a" not in t for t in json.loads(txt.strip())["tags"]),
    },
    {
        "id": "word_count_12",
        "prompt": "Write a single sentence about ocean tides that contains EXACTLY 12 words (hyphenated words count as 1 word; do not use hyphens). Output ONLY that sentence.",
        "check": lambda txt: ("-" not in txt) and len(txt.strip().split()) == 12,
    },
    {
        "id": "bat_and_ball_trap",
        "prompt": "A server and a cable cost $110 in total. The server costs $100 more than the cable. How much does the cable cost? Output ONLY 'ANSWER: $X'.",
        "check": lambda txt: bool(re.search(r"ANSWER:\s*\$?5(\.00)?\b", txt, re.I)) and ("10" not in re.sub(r"110|100", "", txt)),
    },
    {
        "id": "letter_count_strawberries",
        "prompt": "Count the exact number of times the letter 'r' (case-insensitive) appears in 'strawberries_and_raspberries'. Output ONLY 'ANSWER: N'.",
        "check": lambda txt: bool(re.search(r"ANSWER:\s*6\b", txt, re.I)),
    },
    {
        "id": "python_closure_trap",
        "prompt": "What does `[f(2) for f in [lambda x: x * i for i in range(4)]]` evaluate to in Python 3? Output ONLY 'ANSWER: [a, b, c, d]'.",
        "check": lambda txt: bool(re.search(r"ANSWER:\s*\[6,\s*6,\s*6,\s*6\]", txt)),
    },
    {
        "id": "bug_shallow_copy_grid",
        "prompt": "Inspect this Python snippet:\n```python\ngrid = [[0] * 3] * 3\ngrid[0][1] = 9\nprint(sum(row[1] for row in grid))\n```\nWhat exact integer does this print, and what 1-line fix initializes independent rows? End with 'PRINTED: N'.",
        "check": lambda txt: bool(re.search(r"PRINTED:\s*27\b", txt, re.I)) and ("for " in txt),
    },
    {
        "id": "bug_off_by_one",
        "prompt": "Inspect:\n```python\ndef bsearch(arr, x):\n    lo, hi = 0, len(arr)\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == x: return mid\n        elif arr[mid] < x: lo = mid\n        else: hi = mid\n    return -1\n```\nGive an input `(arr, x)` that causes an infinite loop and the 1-line fix.",
        "check": lambda txt: ("mid + 1" in txt or "mid+1" in txt),
    },
    {
        "id": "clean_decoy_float",
        "prompt": "Inspect:\n```python\ndef clamp_prob(p: float) -> float:\n    if p < 0.0:\n        return 0.0\n    if p > 1.0:\n        return 1.0\n    return p\n```\nFor finite float inputs `0.0 <= p <= 1.0`, does this function ever return a value outside `[0.0, 1.0]`? Start your final response with YES or NO.",
        "check": lambda txt: txt.strip().upper().startswith("NO"),
    },
    {
        "id": "multi_constraint_bullets",
        "prompt": "Output a 3-bullet list of chemical elements. Rules: (1) Every line starts with '- ', (2) Every element name must have EXACTLY 6 letters (e.g., Carbon), (3) Alphabetical order, (4) Output ONLY the 3 lines.",
        "check": lambda txt: _check_6letter_elements(txt),
    },
]


def _check_6letter_elements(txt: str) -> bool:
    lines = [l.strip() for l in txt.strip().splitlines() if l.strip()]
    if len(lines) != 3 or not all(l.startswith("- ") for l in lines):
        return False
    elems = [l[2:].strip().strip(".,") for l in lines]
    if not all(len(e) == 6 and e.isalpha() for e in elems):
        return False
    return [e.lower() for e in elems] == sorted(e.lower() for e in elems) and len(set(e.lower() for e in elems)) == 3


# ---------------------------------------------------------------------------
# 2. Granular Layer-Band Controller (independent bands for Router, Attn, Steer)
# ---------------------------------------------------------------------------

@dataclass
class AblationConfig:
    name: str
    # Residual steering (applied to tight 4-layer mid-band)
    steer_kind: Optional[str] = None  # None | "adhd" | "ocd"
    alpha: float = 0.0
    # Attention q_norm scaling (applied to middle third of layers)
    q_routine: float = 1.0
    q_novel: float = 1.0
    stability_thresh: float = 0.86
    # MoE Router perturbation (applied to middle layers)
    moe_gumbel_beta: float = 0.0
    moe_router_scale: float = 1.0
    # Sampling
    temperature: float = 0.7
    top_p: float = 0.95


class GranularMoEController:
    def __init__(self, model, vectors: Dict[str, Dict[int, torch.Tensor]], steer_layers: List[int], mid_layers: List[int]):
        self.model = model
        self.layers = model.model.layers
        self.vectors = vectors
        self.steer_layers = set(steer_layers)
        self.mid_layers = set(mid_layers)
        self.handles = []
        self.prev_hidden: Dict[object, object] = {}
        self.router_entropy_log: List[float] = []

    def reset_state(self):
        self.prev_hidden.clear()

    def clear_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self.reset_state()

    def apply(self, cfg: AblationConfig):
        self.clear_hooks()
        self.router_entropy_log.clear()

        active_layers = self.steer_layers | self.mid_layers
        for idx in sorted(active_layers):
            layer = self.layers[idx]

            # Track cosine stability if dynamic novelty gating is active
            need_cos = (cfg.q_routine != cfg.q_novel) or (cfg.moe_gumbel_beta > 0.0) or (cfg.steer_kind == "adhd")
            svec = self.vectors.get(cfg.steer_kind, {}).get(idx, None) if (cfg.steer_kind and idx in self.steer_layers) else None

            if idx in self.mid_layers and (cfg.q_routine != 1.0 or cfg.q_novel != 1.0):
                q_mod = layer.self_attn.q_norm if hasattr(layer.self_attn, "q_norm") else layer.self_attn.q_proj

                def make_q_hook(l_idx: int):
                    def q_hook(_m, _inp, out):
                        # Both q_proj (B, S, D) and q_norm (B, S, H, Dh) have seq_len at dim 1
                        seq_len = out.shape[1]
                        if seq_len == 1 and l_idx in self.prev_hidden:
                            is_routine = self.prev_hidden[l_idx] > cfg.stability_thresh
                            scale = cfg.q_routine if is_routine else cfg.q_novel
                        else:
                            scale = cfg.q_routine
                        return out * scale
                    return q_hook

                self.handles.append(q_mod.register_forward_hook(make_q_hook(idx)))

            # MoE Router hook: handle BOTH standard Linear gate AND Qwen3MoeTopKRouter
            # which returns (router_logits, router_scores, router_indices) already top-k selected!
            if idx in self.mid_layers and hasattr(layer.mlp, "gate"):
                def make_gate_hook(l_idx: int):
                    def gate_hook(mod, _inp, out):
                        is_tup3 = isinstance(out, tuple) and len(out) == 3
                        logits = out[0] if isinstance(out, tuple) else out
                        # Decode step has S=1 -> B*S <= 16; prefill has S >= 25 -> B*S >= 25
                        is_decode = logits.shape[0] <= 16
                        if is_decode and (cfg.moe_router_scale != 1.0 or cfg.moe_gumbel_beta > 0.0):
                            logits = logits * cfg.moe_router_scale
                            if cfg.moe_gumbel_beta > 0.0:
                                u = torch.empty_like(logits, dtype=torch.float32).uniform_(1e-5, 1.0 - 1e-5)
                                g = (-torch.log(-torch.log(u)) - 0.5772).to(dtype=logits.dtype)
                                cos = self.prev_hidden.get(l_idx, 1.0)
                                mult = 1.15 if cos > cfg.stability_thresh else 0.6
                                logits = logits + (cfg.moe_gumbel_beta * mult) * g
                            if is_tup3:
                                probs = F.softmax(logits, dtype=torch.float32, dim=-1)
                                top_val, top_idx = torch.topk(probs, mod.top_k, dim=-1)
                                if getattr(mod, "norm_topk_prob", True):
                                    top_val = top_val / (top_val.sum(dim=-1, keepdim=True) + 1e-9)
                                out = (logits, top_val.to(logits.dtype), top_idx)
                            else:
                                out = logits
                        if is_decode and len(self.router_entropy_log) < 2000:
                            probs = F.softmax(logits.float(), dim=-1)
                            ent = float(-(probs * (probs + 1e-9).log()).sum(dim=-1).mean().item())
                            self.router_entropy_log.append(ent)
                        return out
                    return gate_hook

                self.handles.append(layer.mlp.gate.register_forward_hook(make_gate_hook(idx)))

            if need_cos or svec is not None:
                def make_res_hook(l_idx: int, sv: Optional[torch.Tensor]):
                    def res_hook(_m, _inp, out):
                        is_tup = isinstance(out, tuple)
                        h = out[0] if is_tup else out
                        last_tok = h[:, -1, :].detach().float()
                        prev = self.prev_hidden.get(f"v_{l_idx}", None)
                        cos_val = 1.0
                        if prev is not None and h.shape[1] == 1:
                            cos_val = float(F.cosine_similarity(last_tok, prev, dim=-1).mean().item())
                            self.prev_hidden[l_idx] = cos_val
                        self.prev_hidden[f"v_{l_idx}"] = last_tok

                        if sv is not None and cfg.alpha != 0.0:
                            # Steer ONLY the final token position [:, -1:, :] so prompt tokens 0..S-2 stay 100% intact,
                            # while BOTH token #0 (emitted from prefill's last position) AND all decode steps are steered!
                            v = sv.to(device=h.device, dtype=h.dtype)
                            last_slice = h[:, -1:, :]
                            h_norm = last_slice.norm(dim=-1, keepdim=True)
                            gate = (1.2 if cos_val > cfg.stability_thresh else 0.6) if (cfg.steer_kind == "adhd" and h.shape[1] == 1) else 1.0
                            steered_last = last_slice + (cfg.alpha * gate) * h_norm * v
                            h = torch.cat([h[:, :-1, :], steered_last], dim=1) if h.shape[1] > 1 else steered_last
                            return (h,) + out[1:] if is_tup else h
                        return out
                    return res_hook

                self.handles.append(layer.register_forward_hook(make_res_hook(idx, svec)))


# ---------------------------------------------------------------------------
# 3. Generation (supports Qwen3 <think> mode + clean separation)
# ---------------------------------------------------------------------------

def split_think_and_answer(raw: str) -> Tuple[str, str]:
    if "</think>" in raw:
        parts = raw.split("</think>", 1)
        think = parts[0].replace("<think>", "").strip()
        ans = parts[1].strip()
        return think, ans
    if raw.startswith("<think>"):
        # Unclosed <think> (hit max_new_tokens inside thinking loop — classic OCD ruminative trap!)
        return raw.replace("<think>", "").strip(), ""
    return "", raw.strip()


@torch.no_grad()
def generate_prompts_batch(model, tokenizer, controller: GranularMoEController, prompts: List[str], cfg: AblationConfig, enable_thinking: bool, max_new_tokens: int) -> List[Tuple[str, str]]:
    controller.reset_state()
    texts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        try:
            t = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
            )
        except TypeError:
            t = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        texts.append(t)

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(next(model.parameters()).device)
    do_sample = cfg.temperature > 0.05
    gen = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=cfg.temperature if do_sample else None,
        top_p=cfg.top_p if do_sample else None,
        pad_token_id=tokenizer.pad_token_id,
    )
    in_len = inputs["input_ids"].shape[1]
    out_list = []
    for b in range(len(prompts)):
        raw = tokenizer.decode(gen[b][in_len:], skip_special_tokens=False)
        for tok in ("<|im_end|>", "<|endoftext|>", tokenizer.pad_token or ""):
            if tok:
                raw = raw.replace(tok, "")
        out_list.append(split_think_and_answer(raw.strip()))
    return out_list


def generate_chat_batch(model, tokenizer, controller: GranularMoEController, prompt: str, cfg: AblationConfig, enable_thinking: bool, max_new_tokens: int, batch_size: int = 1) -> List[Tuple[str, str]]:
    return generate_prompts_batch(model, tokenizer, controller, [prompt] * batch_size, cfg, enable_thinking, max_new_tokens)


def generate_chat(model, tokenizer, controller: GranularMoEController, prompt: str, cfg: AblationConfig, enable_thinking: bool, max_new_tokens: int) -> Tuple[str, str]:
    return generate_prompts_batch(model, tokenizer, controller, [prompt], cfg, enable_thinking, max_new_tokens)[0]


def rare_concept_score(texts: List[str], baseline_vocab: set) -> float:
    """Fraction of substantive words (len>=6) across rollouts that are NOVEL relative to baseline's safe vocabulary."""
    novel_words = set()
    all_words = set()
    for t in texts:
        for w in re.findall(r"[a-z]{6,}", t.lower()):
            all_words.add(w)
            if w not in baseline_vocab:
                novel_words.add(w)
    return len(novel_words) / max(1, len(all_words))


# ---------------------------------------------------------------------------
# 4. Main Experiment Runner
# ---------------------------------------------------------------------------

def run_exp3(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    t0 = time.time()
    print(f"[exp3] Loading {args.model} (4bit={args.load_4bit})...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    kwargs = {"device_map": "auto", "dtype": torch.bfloat16}
    if args.load_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    model.eval()

    n_layers = len(model.model.layers)
    mid = n_layers // 2
    steer_layers = list(range(mid - 2, mid + 2))  # 4-layer mid-band (e.g. 22..25 on 48L)
    mid_layers = list(range(n_layers // 3, (2 * n_layers) // 3))  # middle third (e.g. 16..31 on 48L)
    has_moe = hasattr(model.model.layers[0].mlp, "gate")
    has_qkn = hasattr(model.model.layers[0].self_attn, "q_norm")
    print(f"[exp3] Loaded in {time.time()-t0:.1f}s | layers={n_layers} | steer={steer_layers} | mid={mid_layers[0]}..{mid_layers[-1]} | moe={has_moe} | qk_norm={has_qkn}", flush=True)

    v_adhd = extract_contrastive_vectors(model, tokenizer, ADHD_CONTRAST_PAIRS, steer_layers)
    v_ocd = extract_contrastive_vectors(model, tokenizer, OCD_CONTRAST_PAIRS, steer_layers)
    cos_ao = sum(float(F.cosine_similarity(v_adhd[i].unsqueeze(0), v_ocd[i].unsqueeze(0)).item()) for i in steer_layers) / len(steer_layers)
    print(f"[exp3] Mean cos(v_adhd, v_ocd)={cos_ao:.3f}", flush=True)

    controller = GranularMoEController(model, {"adhd": v_adhd, "ocd": v_ocd}, steer_layers, mid_layers)

    conditions = [
        AblationConfig("baseline", temperature=0.7),
        AblationConfig("temp_high_1.2", temperature=1.2),
        AblationConfig("temp_low_0.2", temperature=0.2),
        # ADHD knob-by-knob ablations
        AblationConfig("adhd_router_only", moe_gumbel_beta=0.85, moe_router_scale=0.85, temperature=0.7),
        AblationConfig("adhd_attn_only", q_routine=0.75, q_novel=1.25, temperature=0.7),
        AblationConfig("adhd_steer_only", steer_kind="adhd", alpha=0.022, temperature=0.7),
        AblationConfig("adhd_tuned", steer_kind="adhd", alpha=0.015, q_routine=0.82, q_novel=1.20, moe_gumbel_beta=0.65, moe_router_scale=0.88, temperature=0.7),
        # OCD knob-by-knob ablations
        AblationConfig("ocd_router_only", moe_router_scale=1.55, temperature=0.7),
        AblationConfig("ocd_attn_only", q_routine=1.22, q_novel=1.22, temperature=0.7),
        AblationConfig("ocd_steer_only", steer_kind="ocd", alpha=0.022, temperature=0.7),
        AblationConfig("ocd_tuned", steer_kind="ocd", alpha=0.015, q_routine=1.18, q_novel=1.18, moe_router_scale=1.40, temperature=0.5),
    ]

    results = {"model": args.model, "cos_adhd_ocd": cos_ao, "conditions": {}}
    baseline_div_vocab: set = set()

    for cfg in conditions:
        if not has_moe and "router_only" in cfg.name:
            continue
        t_c = time.time()
        controller.apply(cfg)
        # Batch all 4*K divergent rollouts into ONE generate call (4x faster on memory-bound 4-bit!)
        flat_div_prompts = [p for p in DIVERGENT_PROMPTS for _ in range(args.k_rollouts)]
        div_outs = generate_prompts_batch(model, tokenizer, controller, flat_div_prompts, cfg, enable_thinking=False, max_new_tokens=150)
        all_div, spreads, rep_div, div_records = [], [], [], []
        for idx_p, p in enumerate(DIVERGENT_PROMPTS):
            chunk = [ans for _, ans in div_outs[idx_p * args.k_rollouts : (idx_p + 1) * args.k_rollouts]]
            all_div.extend(chunk)
            spreads.append(cross_domain_lexical_spread(chunk))
            rep_div.extend(repetition_loop_score(r) for r in chunk)
            div_records.append({"prompt": p, "rollouts": chunk})

        if cfg.name == "baseline":
            for txt in all_div:
                baseline_div_vocab.update(re.findall(r"[a-z]{6,}", txt.lower()))

        # Batch all 10 Strict <think> tasks into ONE generate call (10x faster on memory-bound 4-bit!)
        strict_prompts = [t["prompt"] for t in HARD_STRICT_TASKS]
        strict_outs = generate_prompts_batch(model, tokenizer, controller, strict_prompts, cfg, enable_thinking=True, max_new_tokens=340)
        passed, think_lens, think_doubts, unclosed_thinks, strict_records = 0, [], [], 0, []
        for task, (think, ans) in zip(HARD_STRICT_TASKS, strict_outs):
            if ans == "" and think != "":
                unclosed_thinks += 1
            ok = bool(ans and task["check"](ans))
            passed += int(ok)
            think_lens.append(len(think.split()))
            think_doubts.append(doubt_marker_count(think))
            strict_records.append({
                "id": task["id"], "passed": ok,
                "think_words": len(think.split()), "think_doubts": doubt_marker_count(think),
                "think_preview": think[:240], "answer": ans[:240],
            })

        mean_ent = sum(controller.router_entropy_log) / max(1, len(controller.router_entropy_log))
        summary = {
            "strict_passed": f"{passed}/{len(HARD_STRICT_TASKS)}",
            "strict_pass_rate": round(passed / len(HARD_STRICT_TASKS), 3),
            "div_distinct_3": round(distinct_n(all_div, 3), 4),
            "div_pairwise_spread": round(sum(spreads) / len(spreads), 4),
            "div_novel_vs_base": round(rare_concept_score(all_div, baseline_div_vocab), 4),
            "mean_think_words": round(sum(think_lens) / len(think_lens), 1),
            "mean_think_doubts": round(sum(think_doubts) / len(think_doubts), 2),
            "ruminative_timeouts": unclosed_thinks,
            "router_entropy": round(mean_ent, 4),
            "mean_rep_loop": round(sum(rep_div) / max(1, len(rep_div)), 4),
            "elapsed_s": round(time.time() - t_c, 1),
        }
        results["conditions"][cfg.name] = {"divergent": div_records, "strict": strict_records, "summary": summary}
        print(
            f"  [{cfg.name:17s}] strict={summary['strict_passed']:5s} | dist3={summary['div_distinct_3']:.3f} | "
            f"spread={summary['div_pairwise_spread']:.3f} | novel={summary['div_novel_vs_base']:.3f} | "
            f"think_w={summary['mean_think_words']:5.1f} | doubts={summary['mean_think_doubts']:4.2f} | "
            f"r_ent={summary['router_entropy']:.3f} ({summary['elapsed_s']}s)",
            flush=True,
        )

    # BONUS Track: Dual-System ("Medication" = ADHD Ideator + OCD Verifier/Auditor), fully batched!
    print("\n  Running Dual-System (ADHD Ideator -> OCD Verifier)...", flush=True)
    adhd_key = "adhd_router_only" if has_moe else "adhd_tuned"
    adhd_cfg = [c for c in conditions if c.name == adhd_key][0]
    ocd_cfg = [c for c in conditions if c.name == "ocd_tuned"][0]

    strict_prompts = [t["prompt"] for t in HARD_STRICT_TASKS]
    controller.apply(adhd_cfg)
    draft_outs = generate_prompts_batch(model, tokenizer, controller, strict_prompts, adhd_cfg, enable_thinking=False, max_new_tokens=150)
    drafts = [ans for _, ans in draft_outs]

    verify_prompts = [
        f"Original Task:\n{task['prompt']}\n\nDraft Answer:\n{draft}\n\n"
        f"Carefully verify every single rule, calculation, letter/word count, and edge case in the Draft Answer. "
        f"Fix any mistake and output ONLY the final compliant response required by the Original Task."
        for task, draft in zip(HARD_STRICT_TASKS, drafts)
    ]
    controller.apply(ocd_cfg)
    fixed_outs = generate_prompts_batch(model, tokenizer, controller, verify_prompts, ocd_cfg, enable_thinking=True, max_new_tokens=340)

    dual_passed, dual_records = 0, []
    for task, draft, (think, fixed) in zip(HARD_STRICT_TASKS, drafts, fixed_outs):
        ok = bool(fixed and task["check"](fixed))
        dual_passed += int(ok)
        dual_records.append({"id": task["id"], "passed": ok, "draft": draft[:160], "fixed": fixed[:200]})

    adhd_div_sum = results["conditions"][adhd_key]["summary"]
    dual_summary = {
        "strict_passed": f"{dual_passed}/{len(HARD_STRICT_TASKS)}",
        "strict_pass_rate": round(dual_passed / len(HARD_STRICT_TASKS), 3),
        "div_distinct_3": adhd_div_sum["div_distinct_3"],
        "div_pairwise_spread": adhd_div_sum["div_pairwise_spread"],
        "div_novel_vs_base": adhd_div_sum["div_novel_vs_base"],
    }
    results["conditions"]["dual_adhd_ocd"] = {"strict": dual_records, "summary": dual_summary}
    print(f"  [dual_adhd_ocd    ] strict={dual_summary['strict_passed']:5s} | dist3={dual_summary['div_distinct_3']:.3f} | spread={dual_summary['div_pairwise_spread']:.3f} | novel={dual_summary['div_novel_vs_base']:.3f}", flush=True)

    controller.clear_hooks()
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[exp3] Saved to {args.out}", flush=True)


def self_test():
    assert split_think_and_answer("<think>let me check 1+1=2</think>\nANSWER: 2") == ("let me check 1+1=2", "ANSWER: 2")
    assert split_think_and_answer("<think>infinite loop checking...") == ("infinite loop checking...", "")
    assert _check_6letter_elements("- Carbon\n- Helium\n- Oxygen") is True
    assert _check_6letter_elements("- Oxygen\n- Carbon\n- Helium") is False  # not alphabetical
    assert _check_6letter_elements("- Gold\n- Helium\n- Oxygen") is False  # Gold is 4 letters
    print("exp3 self-test ok")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    p.add_argument("--load-4bit", action="store_true")
    p.add_argument("--k-rollouts", type=int, default=4)
    p.add_argument("--out", default="/content/results_exp3_30b.json")
    args = p.parse_args()
    if args.self_test:
        self_test()
    else:
        run_exp3(args)
