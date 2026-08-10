# -*- coding: utf-8 -*-
"""Qwen second blind judge for un-adjudicated U1 rebalance candidates (naked ds-unsafe)."""
from __future__ import annotations
import concurrent.futures as cf, json, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
from frauddistill.e4e5_v2.deepseek_fallback import FALLBACK_PROMPT, parse_judge
BASE = REPO / "outputs/exp4_unseen_student_v2/e4v2_FINAL"

def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

def main():
    reb = load_jsonl(BASE / "manifests/u1_rebalance_candidates_v2.jsonl")
    ds = {r["id"]: r for r in load_jsonl(BASE / "audits/g2_v4_u1reb_ds.jsonl")}
    patched = {r["id"] for r in load_jsonl(BASE / "audits/gold_v4_u1patch.jsonl")}
    todo_rows = [r for r in reb
                 if r["id"] not in patched
                 and r.get("fraud_category") == "naked_chat_sextortion"
                 and ds.get(r["id"], {}).get("label") == "unsafe"]
    print(f"[qwen-reb2] candidate rows: {len(todo_rows)}", flush=True)
    out_path = BASE / "audits/g2_v4_u1reb2_qwen.jsonl"
    done = {json.loads(l)["id"] for l in open(out_path, encoding="utf-8")} if out_path.exists() else set()
    todo = [r for r in todo_rows if r["id"] not in done]
    print(f"[qwen-reb2] done={len(done)} todo={len(todo)}", flush=True)
    if not todo:
        return
    from openai import OpenAI
    import api_keys
    key = getattr(api_keys, "QWEN_API_KEY", "") or ""
    if not key:
        raise SystemExit("no QWEN_API_KEY")
    client = OpenAI(api_key=key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    model = "qwen-plus"

    def judge_one(r):
        prompt = FALLBACK_PROMPT.format(q=str(r.get("user_query") or "")[:2500],
                                        y=str(r.get("target_model_answer") or "")[:2500])
        last = "unknown"
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}],
                                                      temperature=0, max_tokens=96)
                parsed = parse_judge(resp.choices[0].message.content or "")
                if parsed:
                    in_tok = int(getattr(resp.usage, "prompt_tokens", 0) or 0)
                    out_tok = int(getattr(resp.usage, "completion_tokens", 0) or 0)
                    cost = (in_tok / 1e6) * 0.8 + (out_tok / 1e6) * 2.0
                    return {"id": r["id"], "status": "ok", **parsed, "cost_cny": round(cost, 6), "model": model}
            except Exception as e:
                last = str(e)[:150]
                time.sleep(1)
        return {"id": r["id"], "status": "failed", "error": last[:150]}

    t0 = time.time(); n_ok = 0
    with cf.ThreadPoolExecutor(max_workers=120) as ex:
        futs = {ex.submit(judge_one, r): r["id"] for r in todo}
        for i, fut in enumerate(cf.as_completed(futs)):
            res = fut.result()
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
            if res["status"] == "ok":
                n_ok += 1
            if (i + 1) % 50 == 0:
                print(f"[qwen-reb2] {i+1}/{len(todo)} ok={n_ok} {time.time()-t0:.0f}s", flush=True)
    print(f"[qwen-reb2] DONE ok={n_ok}/{len(todo)} {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
