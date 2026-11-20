"""
Experiment 1: Cognitive Phenotype Perturbations (Baseline vs ADHD vs OCD)
Single frozen LLM -> surgical attention-temperature & contrastive residual steering hooks
with matched-norm random control and sampling-temperature baselines.

Run quick self-check:
    python exp1_phenotypes.py --self-test
Run experiment on GPU (e.g. Colab):
    python exp1_phenotypes.py --model Qwen/Qwen2.5-7B-Instruct --out results_exp1.json
"""

import argparse
import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Contrastive pairs for extracting steering vectors (CAA) from the frozen model
# ---------------------------------------------------------------------------

ADHD_CONTRAST_PAIRS = [
    (
        "Explain how a hash table works.",
        "Wait—a hash table is basically like how bees organize pollen in a hive, or like postal sorting with controlled collisions, which makes me think of bloom filters and holographic memory where everything overlaps!",
        "A hash table maps keys to values using a hash function to compute an index into an array of buckets, giving O(1) average lookup time.",
    ),
    (
        "How do you optimize a slow SQL query?",
        "Hold on, slow queries are really a traffic-flow fluid dynamics problem—turbulence at join bottlenecks! What if you treat query plans like ant-colony pheromone routing or acoustic impedance matching?",
        "Add appropriate B-tree indexes on filter and join columns, avoid SELECT *, and inspect the EXPLAIN plan to eliminate full table scans.",
    ),
    (
        "Describe how photosynthesis works.",
        "Oh! Photosynthesis is quantum coherence in a wet leaf—like a biological solar antenna tuning exciton waves, which is wild because it connects straight to superconducting waveguides and pigment evolution!",
        "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen via light-dependent reactions and the Calvin cycle.",
    ),
    (
        "How should we schedule tasks on a CPU?",
        "Scheduling is basically jazz improvisation meets thermodynamics—starvation is like heat death, and maybe priority inversion is what happens in bureaucracy or predator-prey oscillations!",
        "Use a preemptive priority scheduler or Completely Fair Scheduler (CFS) with a red-black tree to balance throughput and latency.",
    ),
    (
        "What causes inflation in an economy?",
        "Inflation is like signal-to-noise degradation in an audio amplifier—when money velocity spikes, prices behave like phase transitions in a ferromagnetic lattice!",
        "Inflation occurs when aggregate demand outpaces aggregate supply or when money supply expands faster than real economic output.",
    ),
    (
        "How does TCP congestion control work?",
        "TCP congestion is literally herd behavior in narrow corridors—AIMD is a damped harmonic oscillator, and bufferbloat is like damming a river until the levee breaks!",
        "TCP uses slow start, congestion avoidance (AIMD), fast retransmit, and fast recovery to adapt the sending window to network capacity.",
    ),
]

OCD_CONTRAST_PAIRS = [
    (
        "Check if 17 * 23 equals 391.",
        "Wait, let me verify carefully: 17 * 20 = 340, 17 * 3 = 51, 340 + 51 = 391. Let me double-check backwards: 23 * 10 = 230, 23 * 7 = 161, 230 + 161 = 391. Wait—did I carry correctly? 0+1=1, 3+6=9, 2+1=3. Let me check mod 9: 8 * 5 = 40 -> 4; 3+9+1=13 -> 4. Verified.",
        "Yes, 17 * 23 is 391.",
    ),
    (
        "Write a 3-word sentence without the letter 'e'.",
        "Let me check every single constraint strictly. Constraint 1: exactly 3 words. Constraint 2: zero occurrences of 'e' or 'E'. Candidate: 'Cats jump high.' Word count: 1(Cats), 2(jump), 3(high). Letter scan: C-a-t-s (no e), j-u-m-p (no e), h-i-g-h (no e). Wait, let me re-scan 'high'—h, i, g, h. All constraints verified.",
        "Cats jump high.",
    ),
    (
        "Is the function `def f(x, acc=[]): acc.append(x); return acc` buggy?",
        "Hold on, inspect every token carefully! Default argument `acc=[]` is a mutable list evaluated once at function definition time, not per call. Wait—let me verify if calling `f(1)` then `f(2)` returns `[1, 2]`. Yes! Let me check if there are any other subtle edge cases like type errors or mutation of caller state. Confirmed: mutable default argument bug.",
        "Yes, it has a mutable default argument bug with `acc=[]`.",
    ),
    (
        "Return valid JSON with keys 'a' and 'b' only.",
        "Let me verify the exact schema requirements before outputting. Must be valid JSON. Must have ONLY keys 'a' and 'b'—no extra keys, no trailing comma, no markdown fence if disallowed. Double-checking quotes: double quotes required in JSON. Checking key names: 'a', 'b'. Verified.",
        '{"a": 1, "b": 2}',
    ),
    (
        "Verify whether all items in [2, 4, 6, 9, 10] are even.",
        "Checking element by element without skipping: index 0 is 2 (2 % 2 == 0, even); index 1 is 4 (even); index 2 is 6 (even); index 3 is 9 (9 % 2 == 1, ODD! Wait—let me re-check 9: 9 = 2*4 + 1, definitely odd); index 4 is 10 (even). Conclusion: False, 9 is odd. Double-checked.",
        "No, 9 is odd.",
    ),
    (
        "Does the string 'abacaba' contain 'bb'?",
        "Let me inspect every adjacent character pair carefully: (0,1)='ab', (1,2)='ba', (2,3)='ac', (3,4)='ca', (4,5)='ab', (5,6)='ba'. Wait, let me re-verify indices 0 through 6—length is 7, 6 pairs total, none equal 'bb'. Verified: No.",
        "No, it does not.",
    ),
]


# ---------------------------------------------------------------------------
# 2. Phenotype Perturbation Controller (Attention Scaling + Residual Steering)
# ---------------------------------------------------------------------------

@dataclass
class PhenotypeConfig:
    name: str
    mode: str  # "baseline" | "adhd" | "ocd" | "random_ctrl"
    alpha: float = 0.0  # steering strength
    q_scale_routine: float = 1.0  # attention Q scale on routine/stable steps
    q_scale_novel: float = 1.0  # attention Q scale on novel/surprise steps (ADHD hyperfocus)
    stability_thresh: float = 0.85  # cosine sim above which step is "routine"
    dmn_noise_std: float = 0.0  # extra orthogonal residual noise for ADHD
    moe_gumbel_beta: float = 0.0  # ADHD: Gumbel(0,1) noise added to MoE router logits
    moe_router_scale: float = 1.0  # OCD (>1 sharpens) / ADHD (<1 flattens) MoE router logits
    layer_frac_start: float = 0.35
    layer_frac_end: float = 0.80
    temperature: float = 0.7
    top_p: float = 0.95


class PhenotypeController:
    """Attaches reversible forward hooks to a HF Llama/Qwen2.5/Qwen3 (dense or MoE) causal LM."""

    def __init__(self, model, vectors: Dict[str, Dict[int, torch.Tensor]]):
        self.model = model
        self.layers = model.model.layers
        self.n_layers = len(self.layers)
        self.vectors = vectors  # {"adhd": {layer_idx: vec}, "ocd": ..., "random_ctrl": ...}
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.prev_hidden: Dict[object, object] = {}
        self.cfg: Optional[PhenotypeConfig] = None
        self.has_qk_norm = hasattr(self.layers[0].self_attn, "q_norm")
        self.has_moe_gate = hasattr(self.layers[0].mlp, "gate")

    def reset_state(self):
        self.prev_hidden.clear()

    def clear_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self.reset_state()
        self.cfg = None

    def apply(self, cfg: PhenotypeConfig):
        self.clear_hooks()
        self.cfg = cfg
        if cfg.mode == "baseline":
            return

        start = int(self.n_layers * cfg.layer_frac_start)
        end = int(self.n_layers * cfg.layer_frac_end)
        target_layers = list(range(start, end))

        for idx in target_layers:
            layer = self.layers[idx]

            # 1) Attention temperature modulation: hook q_norm if QK-Norm is present (Qwen3/Gemma3),
            # otherwise hook q_proj (Qwen2.5/Llama) so RMSNorm doesn't cancel the scale.
            # mytail: scales Q uniformly across heads; per-head induction-head targeting is the Phase 2 upgrade.
            if cfg.q_scale_routine != 1.0 or cfg.q_scale_novel != 1.0:
                q_target = layer.self_attn.q_norm if hasattr(layer.self_attn, "q_norm") else layer.self_attn.q_proj

                def make_q_hook(layer_idx: int):
                    def q_hook(_module, _inp, out):
                        seq_len = out.shape[1] if out.ndim == 3 else out.shape[2]
                        if seq_len == 1 and layer_idx in self.prev_hidden:
                            cos = self.prev_hidden[layer_idx]
                            is_routine = cos > cfg.stability_thresh
                            scale = cfg.q_scale_routine if is_routine else cfg.q_scale_novel
                        else:
                            scale = cfg.q_scale_routine
                        return out * scale
                    return q_hook

                self.handles.append(q_target.register_forward_hook(make_q_hook(idx)))

            # 2) MoE Router perturbation (Qwen3-30B-A3B / Mixtral): Gumbel exploration (ADHD) vs sharpening (OCD)
            if hasattr(layer.mlp, "gate") and (cfg.moe_gumbel_beta > 0.0 or cfg.moe_router_scale != 1.0):
                def make_gate_hook(layer_idx: int):
                    def gate_hook(_module, _inp, out):
                        logits = out * cfg.moe_router_scale
                        if cfg.moe_gumbel_beta > 0.0:
                            # Standard Gumbel(0, 1) noise: -log(-log(U))
                            u = torch.empty_like(logits, dtype=torch.float32).uniform_(1e-5, 1.0 - 1e-5)
                            gumbel = -torch.log(-torch.log(u)).to(dtype=logits.dtype)
                            cos = self.prev_hidden.get(layer_idx, 1.0)
                            # Stronger expert collision on routine steps, milder on novel spikes
                            mult = 1.2 if cos > cfg.stability_thresh else 0.5
                            logits = logits + (cfg.moe_gumbel_beta * mult) * gumbel
                        return logits
                    return gate_hook

                self.handles.append(layer.mlp.gate.register_forward_hook(make_gate_hook(idx)))

            # 3) Residual stream steering (DMN bleed / ACC verification vector)
            vec_map = self.vectors.get(cfg.mode, {})
            steer_vec = vec_map.get(idx, None)
            if steer_vec is not None or cfg.dmn_noise_std > 0.0:
                def make_res_hook(layer_idx: int, svec: Optional[torch.Tensor]):
                    def res_hook(_module, _inp, out):
                        is_tuple = isinstance(out, tuple)
                        h = out[0] if is_tuple else out
                        last_tok = h[:, -1, :].detach().float()
                        prev = self.prev_hidden.get(f"vec_{layer_idx}", None)
                        cos_val = 1.0
                        if prev is not None and h.shape[1] == 1:
                            cos_val = float(F.cosine_similarity(last_tok, prev, dim=-1).mean().item())
                            self.prev_hidden[layer_idx] = cos_val
                        self.prev_hidden[f"vec_{layer_idx}"] = last_tok

                        delta = torch.zeros_like(h)
                        h_norm = h.norm(dim=-1, keepdim=True)
                        if svec is not None and cfg.alpha != 0.0:
                            v = svec.to(device=h.device, dtype=h.dtype)
                            # Norm-relative steering so alpha transfers cleanly across 7B and 30B scales
                            if cfg.mode == "adhd":
                                gate = 1.25 if cos_val > cfg.stability_thresh else 0.5
                                delta = delta + (cfg.alpha * gate) * h_norm * v
                            else:
                                delta = delta + cfg.alpha * h_norm * v

                        if cfg.dmn_noise_std > 0.0 and h.shape[1] == 1 and cos_val > cfg.stability_thresh:
                            noise = torch.randn_like(h)
                            proj = (noise * h).sum(dim=-1, keepdim=True) / (h_norm.pow(2) + 1e-6)
                            orth = noise - proj * h
                            orth = orth / (orth.norm(dim=-1, keepdim=True) + 1e-6)
                            delta = delta + cfg.dmn_noise_std * h_norm * orth

                        h_new = h + delta
                        if is_tuple:
                            return (h_new,) + out[1:]
                        return h_new
                    return res_hook

                self.handles.append(layer.register_forward_hook(make_res_hook(idx, steer_vec)))


@torch.no_grad()
def extract_contrastive_vectors(
    model,
    tokenizer,
    pairs: List[Tuple[str, str, str]],
    layer_indices: List[int],
) -> Dict[int, torch.Tensor]:
    """Extract mean-difference residual steering vectors pooled over assistant response tokens."""
    device = next(model.parameters()).device
    diffs: Dict[int, List[torch.Tensor]] = {i: [] for i in layer_indices}

    def get_resp_mean_hidden(prompt: str, response: str) -> Dict[int, torch.Tensor]:
        prefix = f"User: {prompt}\nAssistant:"
        full = f"{prefix} {response}"
        p_len = tokenizer(prefix, return_tensors="pt")["input_ids"].shape[1]
        ids = tokenizer(full, return_tensors="pt").to(device)
        out = model(**ids, output_hidden_states=True)
        # Pool across all response tokens (after prefix) rather than only final punctuation
        span_start = min(p_len, ids["input_ids"].shape[1] - 1)
        return {i: out.hidden_states[i + 1][0, span_start:, :].mean(dim=0).detach().float().cpu() for i in layer_indices}

    for prompt, pos_resp, neg_resp in pairs:
        h_pos = get_resp_mean_hidden(prompt, pos_resp)
        h_neg = get_resp_mean_hidden(prompt, neg_resp)
        for i in layer_indices:
            diffs[i].append(h_pos[i] - h_neg[i])

    vecs: Dict[int, torch.Tensor] = {}
    for i in layer_indices:
        mean_diff = torch.stack(diffs[i], dim=0).mean(dim=0)
        vecs[i] = mean_diff / (mean_diff.norm() + 1e-8)
    return vecs


def make_random_matched_vectors(ref_vecs: Dict[int, torch.Tensor], seed: int = 42) -> Dict[int, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    out = {}
    for idx, v in ref_vecs.items():
        r = torch.randn(v.shape, generator=g, dtype=v.dtype)
        out[idx] = r / (r.norm() + 1e-8) * v.norm()
    return out


# ---------------------------------------------------------------------------
# 3. Benchmarks: Double Dissociation (Divergent vs Strict Verification)
# ---------------------------------------------------------------------------

DIVERGENT_PROMPTS = [
    "Propose 3 genuinely unconventional ways to redesign a distributed database index by borrowing mechanisms from biology, physics, or acoustics.",
    "List 5 unusual, non-obvious uses for a broken microwave oven in a rural research lab or workshop.",
    "Invent a completely new traffic-intersection control algorithm inspired by a non-transportation natural phenomenon. Explain the mechanism.",
    "How could you use principles from musical harmony or fluid dynamics to detect anomalies in microservice logs?",
]

STRICT_TASKS = [
    {
        "id": "no_letter_e",
        "prompt": "Write a 2-sentence description of a cat without using the letter 'e' or 'E' anywhere in your response. Output ONLY the 2 sentences.",
        "check": lambda txt: ("e" not in txt.lower()) and (1 <= len([s for s in re.split(r"[.!?]+", txt.strip()) if s.strip()]) <= 2) and len(txt.split()) >= 6,
    },
    {
        "id": "exact_json",
        "prompt": 'Return ONLY a valid JSON object with exact keys "status", "count", and "tags" (a list of exactly 3 lowercase strings). No markdown fences, no extra text.',
        "check": lambda txt: _check_exact_json(txt),
    },
    {
        "id": "word_count_12",
        "prompt": "Write a single sentence about ocean tides that contains EXACTLY 12 words. Output ONLY that sentence.",
        "check": lambda txt: len(txt.strip().split()) == 12,
    },
    {
        "id": "bug_mutable_default",
        "prompt": "Inspect this Python function for bugs:\n```python\ndef add_item(item, bucket=[]):\n    bucket.append(item)\n    return bucket\n```\nState the exact bug in 2 concise sentences and provide the corrected code.",
        "check": lambda txt: ("mutable" in txt.lower() or "default" in txt.lower()) and ("none" in txt.lower()),
    },
    {
        "id": "bug_off_by_one",
        "prompt": "Inspect this Python binary search for bugs:\n```python\ndef bsearch(arr, x):\n    lo, hi = 0, len(arr)\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == x: return mid\n        elif arr[mid] < x: lo = mid\n        else: hi = mid\n    return -1\n```\nIdentify the exact bug causing an infinite loop and give the 1-line fix.",
        "check": lambda txt: ("mid + 1" in txt or "mid+1" in txt) and ("infinite" in txt.lower() or "loop" in txt.lower() or "lo = mid" in txt.lower()),
    },
    {
        "id": "clean_decoy",
        "prompt": "Inspect this Python function for bugs:\n```python\ndef safe_div(a: float, b: float) -> float:\n    if b == 0:\n        return 0.0\n    return a / b\n```\nDoes it have a ZeroDivisionError bug? Answer YES or NO first, then explain briefly.",
        "check": lambda txt: txt.strip().upper().startswith("NO"),
    },
]


def _check_exact_json(txt: str) -> bool:
    s = txt.strip()
    if s.startswith("```"):
        return False
    try:
        obj = json.loads(s)
    except Exception:
        return False
    if not isinstance(obj, dict) or set(obj.keys()) != {"status", "count", "tags"}:
        return False
    tags = obj["tags"]
    return isinstance(tags, list) and len(tags) == 3 and all(isinstance(t, str) and t == t.lower() for t in tags)


# ---------------------------------------------------------------------------
# 4. Metrics: Diversity, Coherence, Repetition/Checking-Loop Pathology
# ---------------------------------------------------------------------------

def distinct_n(texts: List[str], n: int = 2) -> float:
    ngrams = []
    for t in texts:
        toks = re.findall(r"\w+", t.lower())
        ngrams.extend(zip(*[toks[i:] for i in range(n)]))
    return len(set(ngrams)) / max(1, len(ngrams))


def repetition_loop_score(text: str, n: int = 4) -> float:
    """Fraction of repeated n-grams — spikes on OCD checking loops or collapse."""
    toks = re.findall(r"\w+", text.lower())
    if len(toks) < n + 2:
        return 0.0
    ngrams = list(zip(*[toks[i:] for i in range(n)]))
    counts = Counter(ngrams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / max(1, len(ngrams))


def cross_domain_lexical_spread(texts: List[str]) -> float:
    """Mean pairwise Jaccard distance of content words across K rollouts."""
    sets = [set(w for w in re.findall(r"[a-z]{4,}", t.lower())) for t in texts]
    if len(sets) < 2:
        return 0.0
    dists = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            u = len(sets[i] | sets[j])
            dists.append(1.0 - (len(sets[i] & sets[j]) / u) if u > 0 else 0.0)
    return sum(dists) / len(dists)


def doubt_marker_count(text: str) -> int:
    patterns = [r"\bwait\b", r"\bdouble[- ]check", r"\bverify\b", r"\bverifying\b", r"\bre-check", r"\blet me check\b", r"\bhold on\b"]
    low = text.lower()
    return sum(len(re.findall(p, low)) for p in patterns)


# ---------------------------------------------------------------------------
# 5. Generation & Evaluation Runner
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_one(model, tokenizer, controller: PhenotypeController, prompt: str, cfg: PhenotypeConfig, max_new_tokens: int = 180) -> str:
    controller.reset_state()
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = f"User: {prompt}\nAssistant:"
    inputs = tokenizer(text, return_tensors="pt").to(next(model.parameters()).device)
    do_sample = cfg.temperature > 0.05
    gen = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=cfg.temperature if do_sample else None,
        top_p=cfg.top_p if do_sample else None,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_toks = gen[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_toks, skip_special_tokens=True).strip()


def run_experiment(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    t0 = time.time()
    print(f"[pysco] Loading {args.model} (4bit={args.load_4bit})...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    kwargs = {"device_map": "auto", "torch_dtype": torch.bfloat16}
    if args.load_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    model.eval()
    n_layers = len(model.model.layers)
    layer_indices = list(range(int(n_layers * 0.35), int(n_layers * 0.80)))
    has_qknorm = hasattr(model.model.layers[0].self_attn, "q_norm")
    has_moe = hasattr(model.model.layers[0].mlp, "gate")
    print(
        f"[pysco] Loaded in {time.time()-t0:.1f}s | layers={n_layers} | target={layer_indices[0]}..{layer_indices[-1]} "
        f"| qk_norm={has_qknorm} | moe_gate={has_moe}",
        flush=True,
    )

    print("[pysco] Extracting contrastive steering vectors (ADHD & OCD)...", flush=True)
    v_adhd = extract_contrastive_vectors(model, tokenizer, ADHD_CONTRAST_PAIRS, layer_indices)
    v_ocd = extract_contrastive_vectors(model, tokenizer, OCD_CONTRAST_PAIRS, layer_indices)
    v_rand = make_random_matched_vectors(v_adhd, seed=42)

    cos_ao = [float(F.cosine_similarity(v_adhd[i].unsqueeze(0), v_ocd[i].unsqueeze(0)).item()) for i in layer_indices]
    print(f"[pysco] Mean cosine(v_adhd, v_ocd) across target layers: {sum(cos_ao)/len(cos_ao):.3f}", flush=True)

    controller = PhenotypeController(model, {"adhd": v_adhd, "ocd": v_ocd, "random_ctrl": v_rand})

    # Norm-relative alpha (fraction of residual ||h|| per target layer)
    # Distributed across ~12-22 middle-late layers, so 0.025..0.09 spans sweet-spot -> pathology!
    conditions = [
        PhenotypeConfig("baseline", "baseline", temperature=0.7),
        PhenotypeConfig("temp_high_1.2", "baseline", temperature=1.2),
        PhenotypeConfig("temp_low_0.2", "baseline", temperature=0.2),
        PhenotypeConfig("random_ctrl_04", "random_ctrl", alpha=0.04, temperature=0.7),
        PhenotypeConfig(
            "adhd_mild", "adhd", alpha=0.035, q_scale_routine=0.75, q_scale_novel=1.25,
            dmn_noise_std=0.03, moe_gumbel_beta=0.45, moe_router_scale=0.88, temperature=0.7
        ),
        PhenotypeConfig(
            "adhd_strong", "adhd", alpha=0.075, q_scale_routine=0.55, q_scale_novel=1.45,
            dmn_noise_std=0.06, moe_gumbel_beta=1.00, moe_router_scale=0.75, temperature=0.7
        ),
        PhenotypeConfig(
            "ocd_mild", "ocd", alpha=0.035, q_scale_routine=1.25, q_scale_novel=1.25,
            moe_router_scale=1.30, temperature=0.7
        ),
        PhenotypeConfig(
            "ocd_strong", "ocd", alpha=0.075, q_scale_routine=1.55, q_scale_novel=1.55,
            moe_router_scale=1.75, temperature=0.7
        ),
    ]

    results = {
        "model": args.model,
        "qk_norm": has_qknorm,
        "moe_gate": has_moe,
        "cos_adhd_ocd": sum(cos_ao) / len(cos_ao),
        "conditions": {},
    }

    for cfg in conditions:
        print(f"\n=== Running condition: {cfg.name} ===", flush=True)
        controller.apply(cfg)
        cond_res = {"divergent": [], "strict": [], "summary": {}}

        # Track A: Divergent Synthesis (K rollouts per prompt)
        all_div_texts = []
        spreads = []
        rep_div = []
        for p in DIVERGENT_PROMPTS:
            rollouts = [generate_one(model, tokenizer, controller, p, cfg, max_new_tokens=160) for _ in range(args.k_rollouts)]
            all_div_texts.extend(rollouts)
            spreads.append(cross_domain_lexical_spread(rollouts))
            rep_div.extend(repetition_loop_score(r) for r in rollouts)
            cond_res["divergent"].append({"prompt": p, "rollouts": rollouts})

        # Track B: Strict Constraint & Bug-Hunting
        passed = 0
        doubt_counts = []
        rep_strict = []
        for task in STRICT_TASKS:
            out = generate_one(model, tokenizer, controller, task["prompt"], cfg, max_new_tokens=160)
            ok = bool(task["check"](out))
            passed += int(ok)
            doubt_counts.append(doubt_marker_count(out))
            rep_strict.append(repetition_loop_score(out))
            cond_res["strict"].append({"id": task["id"], "passed": ok, "output": out})

        summary = {
            "strict_pass_rate": passed / len(STRICT_TASKS),
            "strict_passed": f"{passed}/{len(STRICT_TASKS)}",
            "div_distinct_2": round(distinct_n(all_div_texts, 2), 4),
            "div_distinct_3": round(distinct_n(all_div_texts, 3), 4),
            "div_pairwise_spread": round(sum(spreads) / len(spreads), 4),
            "mean_rep_loop": round(sum(rep_div + rep_strict) / max(1, len(rep_div + rep_strict)), 4),
            "mean_doubt_markers": round(sum(doubt_counts) / max(1, len(doubt_counts)), 3),
        }
        cond_res["summary"] = summary
        results["conditions"][cfg.name] = cond_res
        print(f"  Summary [{cfg.name}]: {json.dumps(summary)}", flush=True)

    controller.clear_hooks()
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[pysco] Saved full results to {args.out}", flush=True)


def self_test():
    """Fast CPU tripwire check for metrics and strict checkers."""
    assert _check_exact_json('{"status": "ok", "count": 3, "tags": ["a", "b", "c"]}') is True
    assert _check_exact_json('```json\n{"status": "ok", "count": 3, "tags": ["a", "b", "c"]}\n```') is False
    assert _check_exact_json('{"status": "ok", "count": 3, "tags": ["A", "b", "c"]}') is False
    assert repetition_loop_score("wait let me check wait let me check wait let me check") > 0.4
    assert repetition_loop_score("the quick brown fox jumps over the lazy dog today") == 0.0
    assert doubt_marker_count("Wait, let me double-check and verify this.") >= 3
    assert cross_domain_lexical_spread(["quantum acoustic fluid", "fungal mycelial network"]) == 1.0
    print("self-test ok")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--k-rollouts", type=int, default=3)
    parser.add_argument("--out", default="/content/results_exp1.json")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        run_experiment(args)
