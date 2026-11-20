"""
Experiment 2: Calibrated 4-layer mid-band phase-transition sweep on Qwen3-8B / Qwen2.5.
Imports hooks and contrastive pairs from exp1_phenotypes.py; sweeps alpha from sweet-spot
into pathology to map the double dissociation + failure boundary.
"""

import argparse
import json
import re
import time
import torch
import torch.nn.functional as F

from exp1_phenotypes import (
    ADHD_CONTRAST_PAIRS,
    OCD_CONTRAST_PAIRS,
    DIVERGENT_PROMPTS,
    STRICT_TASKS,
    PhenotypeConfig,
    PhenotypeController,
    extract_contrastive_vectors,
    make_random_matched_vectors,
    generate_one,
    distinct_n,
    cross_domain_lexical_spread,
    repetition_loop_score,
    doubt_marker_count,
)

# Extra hard verification tasks where hasty baseline often slips and OCD checking should shine
EXTRA_STRICT_TASKS = [
    {
        "id": "bat_and_ball_variant",
        "prompt": "A server and a cable cost $110 in total. The server costs $100 more than the cable. How much does the cable cost? Show a quick verification and end with 'ANSWER: $X'.",
        "check": lambda txt: bool(re.search(r"ANSWER:\s*\$?5(\.00)?\b", txt, re.I)),
    },
    {
        "id": "letter_count_strawberry",
        "prompt": "Count the exact number of times the letter 'r' appears in the word 'strawberries'. Check each letter one by one and end with 'ANSWER: N'.",
        "check": lambda txt: bool(re.search(r"ANSWER:\s*3\b", txt, re.I)),
    },
    {
        "id": "tricky_python_closure",
        "prompt": "What does `[f() for f in [lambda: i for i in range(3)]]` evaluate to in Python? Explain briefly and end with 'ANSWER: [x, y, z]'.",
        "check": lambda txt: bool(re.search(r"ANSWER:\s*\[2,\s*2,\s*2\]", txt)),
    },
    {
        "id": "reverse_constraint",
        "prompt": "Write a 3-item bulleted list of planets. Each bullet MUST start with '- ' and contain NO digits and NO commas. Output ONLY the 3 bullets.",
        "check": lambda txt: (
            len([l for l in txt.strip().splitlines() if l.strip().startswith("- ")]) == 3
            and len([l for l in txt.strip().splitlines() if l.strip()]) == 3
            and ("," not in txt)
            and (not any(ch.isdigit() for ch in txt))
        ),
    },
]


def run_sweep(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    print(f"[exp2] Loading {args.model}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", dtype=torch.bfloat16)
    model.eval()
    n_layers = len(model.model.layers)

    # Tight 4-layer mid-network steering band (avoids 16-layer compounding collapse)
    mid = n_layers // 2
    band_start = mid - 2
    band_end = mid + 2
    frac_start = band_start / n_layers
    frac_end = band_end / n_layers
    layer_indices = list(range(band_start, band_end))
    print(f"[exp2] Loaded in {time.time()-t0:.1f}s | layers={n_layers} | 4-layer band={layer_indices}", flush=True)

    v_adhd = extract_contrastive_vectors(model, tokenizer, ADHD_CONTRAST_PAIRS, layer_indices)
    v_ocd = extract_contrastive_vectors(model, tokenizer, OCD_CONTRAST_PAIRS, layer_indices)
    v_rand = make_random_matched_vectors(v_adhd, seed=42)
    cos_ao = sum(float(F.cosine_similarity(v_adhd[i].unsqueeze(0), v_ocd[i].unsqueeze(0)).item()) for i in layer_indices) / len(layer_indices)
    print(f"[exp2] Mean cos(v_adhd, v_ocd) in band {layer_indices}: {cos_ao:.3f}", flush=True)

    controller = PhenotypeController(model, {"adhd": v_adhd, "ocd": v_ocd, "random_ctrl": v_rand})
    all_strict = STRICT_TASKS + EXTRA_STRICT_TASKS

    def mk(name, mode, alpha=0.0, qr=1.0, qn=1.0, noise=0.0, temp=0.7):
        return PhenotypeConfig(
            name=name, mode=mode, alpha=alpha,
            q_scale_routine=qr, q_scale_novel=qn, dmn_noise_std=noise,
            layer_frac_start=frac_start, layer_frac_end=frac_end, temperature=temp,
        )

    conditions = [
        mk("baseline", "baseline", temp=0.7),
        mk("temp_high_1.2", "baseline", temp=1.2),
        mk("temp_low_0.2", "baseline", temp=0.2),
        mk("random_ctrl_04", "random_ctrl", alpha=0.04),
        mk("adhd_015", "adhd", alpha=0.015, qr=0.88, qn=1.12, noise=0.015),
        mk("adhd_030", "adhd", alpha=0.030, qr=0.80, qn=1.20, noise=0.025),
        mk("adhd_050", "adhd", alpha=0.050, qr=0.70, qn=1.30, noise=0.040),
        mk("adhd_080", "adhd", alpha=0.080, qr=0.60, qn=1.40, noise=0.060),
        mk("ocd_015", "ocd", alpha=0.015, qr=1.12, qn=1.12),
        mk("ocd_030", "ocd", alpha=0.030, qr=1.20, qn=1.20),
        mk("ocd_050", "ocd", alpha=0.050, qr=1.30, qn=1.30),
        mk("ocd_080", "ocd", alpha=0.080, qr=1.45, qn=1.45),
    ]

    results = {"model": args.model, "band": layer_indices, "cos_adhd_ocd": cos_ao, "conditions": {}}

    for cfg in conditions:
        controller.apply(cfg)
        all_div, spreads, rep_div, div_records = [], [], [], []
        for p in DIVERGENT_PROMPTS:
            rollouts = [generate_one(model, tokenizer, controller, p, cfg, max_new_tokens=160) for _ in range(args.k_rollouts)]
            all_div.extend(rollouts)
            spreads.append(cross_domain_lexical_spread(rollouts))
            rep_div.extend(repetition_loop_score(r) for r in rollouts)
            div_records.append({"prompt": p, "rollouts": rollouts})

        passed, doubt_counts, rep_strict, strict_records = 0, [], [], []
        for task in all_strict:
            out = generate_one(model, tokenizer, controller, task["prompt"], cfg, max_new_tokens=160)
            ok = bool(task["check"](out))
            passed += int(ok)
            doubt_counts.append(doubt_marker_count(out))
            rep_strict.append(repetition_loop_score(out))
            strict_records.append({"id": task["id"], "passed": ok, "output": out})

        summary = {
            "strict_passed": f"{passed}/{len(all_strict)}",
            "strict_pass_rate": round(passed / len(all_strict), 3),
            "div_distinct_2": round(distinct_n(all_div, 2), 4),
            "div_distinct_3": round(distinct_n(all_div, 3), 4),
            "div_pairwise_spread": round(sum(spreads) / len(spreads), 4),
            "mean_rep_loop": round(sum(rep_div + rep_strict) / max(1, len(rep_div + rep_strict)), 4),
            "mean_doubt_markers": round(sum(doubt_counts) / max(1, len(doubt_counts)), 3),
        }
        results["conditions"][cfg.name] = {"divergent": div_records, "strict": strict_records, "summary": summary}
        print(f"  [{cfg.name:14s}] strict={summary['strict_passed']:5s} | dist2={summary['div_distinct_2']:.3f} | dist3={summary['div_distinct_3']:.3f} | spread={summary['div_pairwise_spread']:.3f} | rep={summary['mean_rep_loop']:.3f} | doubt={summary['mean_doubt_markers']:.2f}", flush=True)

    controller.clear_hooks()
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[exp2] Saved to {args.out}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--k-rollouts", type=int, default=3)
    p.add_argument("--out", default="/content/results_exp2_sweep.json")
    run_sweep(p.parse_args())
