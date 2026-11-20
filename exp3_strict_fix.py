"""
Quick 5-minute untruncated Strict Verification pass on Qwen/Qwen3-32B (4-bit, cached).
Fixes the 340-token <think> truncation from Stage 1 by giving <think> 950 tokens (batched B=10),
so every condition completes both its <think> deliberation AND its final answer!
"""

import json
import time
import torch

from exp1_phenotypes import ADHD_CONTRAST_PAIRS, OCD_CONTRAST_PAIRS, extract_contrastive_vectors, doubt_marker_count
from exp3_ablation_moe import (
    HARD_STRICT_TASKS,
    AblationConfig,
    GranularMoEController,
    generate_prompts_batch,
)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    t0 = time.time()
    model_id = "Qwen/Qwen3-32B"
    print(f"[strict_fix] Loading cached {model_id}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", dtype=torch.bfloat16, quantization_config=qcfg)
    model.eval()

    n_layers = len(model.model.layers)
    mid = n_layers // 2
    steer_layers = list(range(mid - 2, mid + 2))
    mid_layers = list(range(n_layers // 3, (2 * n_layers) // 3))
    print(f"[strict_fix] Loaded in {time.time()-t0:.1f}s", flush=True)

    v_adhd = extract_contrastive_vectors(model, tokenizer, ADHD_CONTRAST_PAIRS, steer_layers)
    v_ocd = extract_contrastive_vectors(model, tokenizer, OCD_CONTRAST_PAIRS, steer_layers)
    controller = GranularMoEController(model, {"adhd": v_adhd, "ocd": v_ocd}, steer_layers, mid_layers)

    conditions = [
        AblationConfig("baseline", temperature=0.7),
        AblationConfig("temp_high_1.2", temperature=1.2),
        AblationConfig("temp_low_0.2", temperature=0.2),
        AblationConfig("adhd_attn_only", q_routine=0.75, q_novel=1.25, temperature=0.7),
        AblationConfig("adhd_tuned", steer_kind="adhd", alpha=0.018, q_routine=0.80, q_novel=1.22, temperature=0.7),
        AblationConfig("ocd_attn_only", q_routine=1.22, q_novel=1.22, temperature=0.7),
        AblationConfig("ocd_steer_only", steer_kind="ocd", alpha=0.022, temperature=0.7),
        AblationConfig("ocd_tuned", steer_kind="ocd", alpha=0.016, q_routine=1.18, q_novel=1.18, temperature=0.5),
    ]

    strict_prompts = [t["prompt"] for t in HARD_STRICT_TASKS]
    out_data = {"model": model_id, "max_new_tokens": 950, "conditions": {}}

    for cfg in conditions:
        tc = time.time()
        controller.apply(cfg)
        # First test DIRECT mode (enable_thinking=False, max_new_tokens=160) -> fast impulsive/direct accuracy!
        direct_outs = generate_prompts_batch(model, tokenizer, controller, strict_prompts, cfg, enable_thinking=False, max_new_tokens=160)
        dir_pass = sum(int(bool(ans and task["check"](ans))) for task, (_, ans) in zip(HARD_STRICT_TASKS, direct_outs))

        # Second test THINKING mode (enable_thinking=True, max_new_tokens=950) -> full deliberation!
        think_outs = generate_prompts_batch(model, tokenizer, controller, strict_prompts, cfg, enable_thinking=True, max_new_tokens=950)
        thk_pass, thk_words, thk_doubts, timeouts, records = 0, [], [], 0, []
        for task, (_, d_ans), (think, t_ans) in zip(HARD_STRICT_TASKS, direct_outs, think_outs):
            if t_ans == "" and think != "":
                timeouts += 1
            ok_d = bool(d_ans and task["check"](d_ans))
            ok_t = bool(t_ans and task["check"](t_ans))
            thk_pass += int(ok_t)
            thk_words.append(len(think.split()))
            thk_doubts.append(doubt_marker_count(think))
            records.append({
                "id": task["id"],
                "direct_passed": ok_d,
                "direct_ans": d_ans[:200],
                "think_passed": ok_t,
                "think_words": len(think.split()),
                "think_doubts": doubt_marker_count(think),
                "think_ans": t_ans[:200],
            })

        summary = {
            "direct_passed": f"{dir_pass}/10",
            "think_passed": f"{thk_pass}/10",
            "mean_think_words": round(sum(thk_words) / 10.0, 1),
            "mean_think_doubts": round(sum(thk_doubts) / 10.0, 2),
            "timeouts": timeouts,
            "elapsed_s": round(time.time() - tc, 1),
        }
        out_data["conditions"][cfg.name] = {"summary": summary, "records": records}
        print(
            f"  [{cfg.name:15s}] direct={summary['direct_passed']} | think={summary['think_passed']} | "
            f"think_w={summary['mean_think_words']:5.1f} | doubts={summary['mean_think_doubts']:4.2f} | "
            f"timeouts={timeouts} ({summary['elapsed_s']}s)",
            flush=True,
        )

    # Dual-System: ADHD direct draft -> OCD <think> verifier!
    adhd_cfg = [c for c in conditions if c.name == "adhd_tuned"][0]
    ocd_cfg = [c for c in conditions if c.name == "ocd_tuned"][0]
    drafts = [r["direct_ans"] for r in out_data["conditions"]["adhd_tuned"]["records"]]
    verify_prompts = [
        f"Original Task:\n{task['prompt']}\n\nDraft Answer:\n{draft}\n\n"
        f"Carefully verify every single rule, calculation, letter/word count, and edge case in the Draft Answer. "
        f"Fix any mistake and output ONLY the final compliant response required by the Original Task."
        for task, draft in zip(HARD_STRICT_TASKS, drafts)
    ]
    controller.apply(ocd_cfg)
    dual_outs = generate_prompts_batch(model, tokenizer, controller, verify_prompts, ocd_cfg, enable_thinking=True, max_new_tokens=950)
    dual_pass = sum(int(bool(ans and task["check"](ans))) for task, (_, ans) in zip(HARD_STRICT_TASKS, dual_outs))
    out_data["conditions"]["dual_adhd_ocd"] = {
        "summary": {"think_passed": f"{dual_pass}/10"},
        "records": [{"id": t["id"], "passed": bool(a and t["check"](a)), "ans": a[:200]} for t, (_, a) in zip(HARD_STRICT_TASKS, dual_outs)],
    }
    print(f"  [dual_adhd_ocd  ] think={dual_pass}/10", flush=True)

    controller.clear_hooks()
    with open("/content/results_exp3_strict_fix_32b.json", "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    with open("/content/STRICT_FIX_DONE.txt", "w") as f:
        f.write("DONE\n")
    print("[strict_fix] Saved /content/results_exp3_strict_fix_32b.json", flush=True)


if __name__ == "__main__":
    main()
