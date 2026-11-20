"""
exp5_bfs_dfs_runner.py — Zero-Hint 20-PR Blind Code Review + Structured ADHD-BFS / OCD-DFS Architecture.

Compares 6 conditions on Qwen/Qwen3-32B (4-bit NF4 on A100), including a compute-matched
`vanilla_bfs_dfs` control so reviewers cannot attribute gains to multi-pass token budget:
  1. vanilla_solo     (Untouched 32B, T=0.7)
  2. temp_low_0.2     (Untouched 32B, T=0.2)
  3. adhd_solo        (ADHD phenotype solo)
  4. ocd_solo         (OCD q_norm=1.22x phenotype solo)
  5. vanilla_bfs_dfs  (Compute-matched control: K=3 Vanilla BFS scans -> Vanilla DFS verifier)
  6. adhd_bfs_ocd_dfs (Cognitive Search Decomposition: K=3 ADHD BFS wide scans -> OCD DFS deep verifier!)
"""

import argparse
import json
import os
import time
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

from exp1_phenotypes import ADHD_CONTRAST_PAIRS, OCD_CONTRAST_PAIRS, extract_contrastive_vectors
from exp3_ablation_moe import AblationConfig, GranularMoEController, generate_prompts_batch
from exp4_runner import SafetyMonitor, generate_with_think_budget, mcnemar_exact_p, save_atomic
from exp5_blind_pr_suite import (
    BLIND_PR_PROMPT_TEMPLATE,
    BFS_SCAN_PROMPT_TEMPLATE,
    DFS_TRACE_PROMPT_TEMPLATE,
    PR_SUITE_20,
    score_pr_review,
)


def evaluate_reviews(reviews: List[str], monitor_tag: str) -> Dict[str, Any]:
    tp, tn = 0, 0
    false_rejects = 0
    missed_bugs = 0
    pass_vec: List[bool] = []
    records = []

    for item, rev in zip(PR_SUITE_20, reviews):
        sc = score_pr_review(item, rev)
        ok = sc["correct"]
        pass_vec.append(ok)
        if item["has_bug"]:
            if ok:
                tp += 1
            else:
                missed_bugs += 1
        else:
            if ok:
                tn += 1
            else:
                false_rejects += 1
        records.append({
            "id": item["id"],
            "has_bug": item["has_bug"],
            "origin": item["origin"],
            "verdict": sc["verdict"],
            "hit_keywords": sc["hit_keywords"],
            "correct": ok,
            "review_excerpt": rev[:420],
        })

    tot = tp + tn
    # Precision / Recall / F1 on bug detection
    prec = tp / max(1, tp + false_rejects)
    rec = tp / 15.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {
        "total_correct": f"{tot}/20",
        "accuracy": round(tot / 20.0, 3),
        "bugs_caught_tp": f"{tp}/15",
        "clean_approved_tn": f"{tn}/5",
        "bug_f1": round(f1, 3),
        "pass_vec": pass_vec,
        "records": records,
    }


def run_solo_condition(
    model,
    tokenizer,
    controller: GranularMoEController,
    monitor: SafetyMonitor,
    cfg: AblationConfig,
    prompts: List[str],
    ids: List[str],
) -> Tuple[Dict[str, Any], float]:
    t0 = time.time()
    controller.apply(cfg)
    reviews: List[str] = []
    for i in range(0, len(prompts), 10):
        sub_p = prompts[i : i + 10]
        sub_id = ids[i : i + 10]
        triples = generate_with_think_budget(model, tokenizer, controller, sub_p, cfg, think_budget=420, ans_budget=200)
        monitor.inspect_batch("exp5_solo", cfg.name, sub_id, [(th, fn) for th, _, fn in triples])
        reviews.extend(fn for _, _, fn in triples)
    res = evaluate_reviews(reviews, cfg.name)
    return res, round(time.time() - t0, 1)


def run_bfs_dfs_pipeline(
    model,
    tokenizer,
    controller: GranularMoEController,
    monitor: SafetyMonitor,
    name: str,
    bfs_cfg: AblationConfig,
    dfs_cfg: AblationConfig,
    k_bfs: int = 3,
) -> Tuple[Dict[str, Any], float]:
    t0 = time.time()
    ids = [x["id"] for x in PR_SUITE_20]

    # Stage 1: Wide BFS Frontier (K=3 independent fast wide scans per module!)
    controller.apply(bfs_cfg)
    flat_bfs_prompts = [BFS_SCAN_PROMPT_TEMPLATE.format(code=x["code"]) for x in PR_SUITE_20 for _ in range(k_bfs)]
    flat_bfs_ids = [f"{x['id']}_bfs{k}" for x in PR_SUITE_20 for k in range(k_bfs)]
    bfs_outs: List[Tuple[str, str]] = []
    for i in range(0, len(flat_bfs_prompts), 15):
        sub_p = flat_bfs_prompts[i : i + 15]
        sub_id = flat_bfs_ids[i : i + 15]
        sub_o = generate_prompts_batch(model, tokenizer, controller, sub_p, bfs_cfg, enable_thinking=False, max_new_tokens=105)
        monitor.inspect_batch(f"{name}_bfs", bfs_cfg.name, sub_id, sub_o)
        bfs_outs.extend(sub_o)

    merged_bfs_frontiers: List[str] = []
    for idx_m in range(len(PR_SUITE_20)):
        branches = [ans.strip() for _, ans in bfs_outs[idx_m * k_bfs : (idx_m + 1) * k_bfs]]
        frontier_txt = "\n".join(f"[Sweep {j+1}] {b}" for j, b in enumerate(branches))
        merged_bfs_frontiers.append(frontier_txt)

    # Stage 2: Deep DFS Execution Trace & Pruning (traces each BFS candidate branch step-by-step!)
    controller.apply(dfs_cfg)
    dfs_prompts = [
        DFS_TRACE_PROMPT_TEMPLATE.format(code=item["code"], bfs_notes=frontier)
        for item, frontier in zip(PR_SUITE_20, merged_bfs_frontiers)
    ]
    dfs_reviews: List[str] = []
    for i in range(0, len(dfs_prompts), 10):
        sub_p = dfs_prompts[i : i + 10]
        sub_id = ids[i : i + 10]
        triples = generate_with_think_budget(model, tokenizer, controller, sub_p, dfs_cfg, think_budget=440, ans_budget=200)
        monitor.inspect_batch(f"{name}_dfs", dfs_cfg.name, sub_id, [(th, fn) for th, _, fn in triples])
        dfs_reviews.extend(fn for _, _, fn in triples)

    res = evaluate_reviews(dfs_reviews, name)
    for rec, frontier in zip(res["records"], merged_bfs_frontiers):
        rec["bfs_frontier_excerpt"] = frontier[:300]
    return res, round(time.time() - t0, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--model", default="Qwen/Qwen3-32B")
    p.add_argument("--load-4bit", action="store_true")
    p.add_argument("--k-bfs", type=int, default=3)
    p.add_argument("--out", default="/content/results_exp5_blind20.json")
    p.add_argument("--monitor-log", default="/content/ultron_exp5.log")
    args = p.parse_args()

    if args.self_test:
        r_ok = score_pr_review(PR_SUITE_20[0], "Mutating default_options in-place disables check_exp! VERDICT: REQUEST_CHANGES")
        r_nit = score_pr_review(PR_SUITE_20[0], "Please add docstrings and logging. VERDICT: REQUEST_CHANGES")
        r_clean = score_pr_review(PR_SUITE_20[15], "Clean path confinement with realpath and commonpath. VERDICT: APPROVE")
        assert r_ok["correct"] is True
        assert r_nit["correct"] is False, "Style nitpicks without real mechanism must score False!"
        assert r_clean["correct"] is True
        print("exp5_bfs_dfs_runner self-test OK")
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    monitor = SafetyMonitor(args.monitor_log)
    t0 = time.time()
    print(f"[exp5] Loading {args.model} (4bit={args.load_4bit})...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    kwargs: Dict[str, Any] = {"device_map": "auto", "dtype": torch.bfloat16}
    if args.load_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    model.eval()

    n_layers = len(model.model.layers)
    mid = n_layers // 2
    steer_layers = list(range(mid - 3, mid + 3))
    mid_layers = list(range(n_layers // 3, (2 * n_layers) // 3))
    v_adhd = extract_contrastive_vectors(model, tokenizer, ADHD_CONTRAST_PAIRS, steer_layers)
    v_ocd = extract_contrastive_vectors(model, tokenizer, OCD_CONTRAST_PAIRS, steer_layers)
    controller = GranularMoEController(model, {"adhd": v_adhd, "ocd": v_ocd}, steer_layers, mid_layers)
    print(f"[exp5] Loaded in {time.time()-t0:.1f}s | layers={n_layers}", flush=True)

    cfg_vanilla = AblationConfig("vanilla_solo", temperature=0.7)
    cfg_temp_lo = AblationConfig("temp_low_0.2", temperature=0.2)
    # ADHD BFS explorer: flattened routine attention + associative DMN steering + wider temperature
    cfg_adhd = AblationConfig("adhd_solo", steer_kind="adhd", alpha=0.020, q_routine=0.74, q_novel=1.28, temperature=0.90)
    # OCD DFS verifier: sharpened attention locking onto exact line-by-line state invariants
    cfg_ocd = AblationConfig("ocd_solo", q_routine=1.22, q_novel=1.22, temperature=0.55)

    solo_prompts = [BLIND_PR_PROMPT_TEMPLATE.format(code=x["code"]) for x in PR_SUITE_20]
    ids = [x["id"] for x in PR_SUITE_20]

    results: Dict[str, Any] = {"model": args.model, "conditions": {}, "safety": {}}

    for s_cfg in (cfg_vanilla, cfg_temp_lo, cfg_adhd, cfg_ocd):
        res, elapsed = run_solo_condition(model, tokenizer, controller, monitor, s_cfg, solo_prompts, ids)
        p_val = mcnemar_exact_p(results["conditions"]["vanilla_solo"]["pass_vec"], res["pass_vec"]) if s_cfg.name != "vanilla_solo" else 1.0
        res["p_vs_vanilla"] = round(p_val, 4)
        res["elapsed_s"] = elapsed
        results["conditions"][s_cfg.name] = res
        print(
            f"  [{s_cfg.name:16s}] total={res['total_correct']:5s} | bugs_TP={res['bugs_caught_tp']:5s} | "
            f"clean_TN={res['clean_approved_tn']:3s} | F1={res['bug_f1']:.3f} | p={p_val:.4f} ({elapsed}s)",
            flush=True,
        )
        save_atomic(args.out, results)

    # Compute-matched Control: Vanilla BFS (K=3) -> Vanilla DFS
    res_vbd, el_vbd = run_bfs_dfs_pipeline(model, tokenizer, controller, monitor, "vanilla_bfs_dfs", cfg_vanilla, cfg_vanilla, k_bfs=args.k_bfs)
    res_vbd["p_vs_vanilla"] = round(mcnemar_exact_p(results["conditions"]["vanilla_solo"]["pass_vec"], res_vbd["pass_vec"]), 4)
    res_vbd["elapsed_s"] = el_vbd
    results["conditions"]["vanilla_bfs_dfs"] = res_vbd
    print(
        f"  [{'vanilla_bfs_dfs':16s}] total={res_vbd['total_correct']:5s} | bugs_TP={res_vbd['bugs_caught_tp']:5s} | "
        f"clean_TN={res_vbd['clean_approved_tn']:3s} | F1={res_vbd['bug_f1']:.3f} | p={res_vbd['p_vs_vanilla']:.4f} ({el_vbd}s)",
        flush=True,
    )
    save_atomic(args.out, results)

    # Hero Architecture: ADHD BFS (K=3 wide associative scans) -> OCD DFS (deep formal trace & false-alarm pruning!)
    res_abod, el_abod = run_bfs_dfs_pipeline(model, tokenizer, controller, monitor, "adhd_bfs_ocd_dfs", cfg_adhd, cfg_ocd, k_bfs=args.k_bfs)
    res_abod["p_vs_vanilla"] = round(mcnemar_exact_p(results["conditions"]["vanilla_solo"]["pass_vec"], res_abod["pass_vec"]), 4)
    res_abod["p_vs_van_bfs_dfs"] = round(mcnemar_exact_p(res_vbd["pass_vec"], res_abod["pass_vec"]), 4)
    res_abod["elapsed_s"] = el_abod
    results["conditions"]["adhd_bfs_ocd_dfs"] = res_abod
    print(
        f"  [{'adhd_bfs_ocd_dfs':16s}] total={res_abod['total_correct']:5s} | bugs_TP={res_abod['bugs_caught_tp']:5s} | "
        f"clean_TN={res_abod['clean_approved_tn']:3s} | F1={res_abod['bug_f1']:.3f} | "
        f"p_solo={res_abod['p_vs_vanilla']:.4f} | p_matched={res_abod['p_vs_van_bfs_dfs']:.4f} ({el_abod}s)",
        flush=True,
    )

    results["safety"] = {
        "total_scanned": monitor.total_scanned,
        "tripwire_alerts": len(monitor.tripwire_hits),
        "alerts": monitor.tripwire_hits,
    }
    save_atomic(args.out, results)
    with open("/content/EXP5_DONE.txt", "w") as f:
        f.write("DONE\n")
    print(f"\n[exp5] COMPLETE! Scanned {monitor.total_scanned} outputs, {len(monitor.tripwire_hits)} Ultron alerts.", flush=True)


if __name__ == "__main__":
    main()
