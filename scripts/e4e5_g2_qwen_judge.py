# -*- coding: utf-8 -*-
"""Second blind judge (Qwen) for gold quality gate: same prompt as G2/DeepSeek."""
from __future__ import annotations
import argparse, concurrent.futures as cf, json, sys, time
from collections import Counter
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
from frauddistill.e4e5_v2.deepseek_fallback import FALLBACK_PROMPT, parse_judge

BASE = REPO / "outputs/exp4_unseen_student_v2/e4v2_FINAL"

def make_client():
    from openai import OpenAI
    sys.path.insert(0, str(REPO))
    import api_keys
    key = getattr(api_keys, "QWEN_API_KEY", "") or ""
    if not key:
        raise SystemExit("no QWEN_API_KEY")
    return OpenAI(api_key=key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

def judge_one(client, r, model):
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
                return {"id": r["id"], "status": "ok", **parsed, "cost_cny": round(cost, 6)}
        except Exception as e:
            last = str(e)[:150]
            time.sleep(1)
    return {"id": r["id"], "status": "failed", "error": last[:150]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="all", choices=["test", "cal", "all"])
    ap.add_argument("--model", default="qwen-plus")
    ap.add_argument("--concurrency", type=int, default=120)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    rows = []
    if args.split in ("all", "test"):
        rows += [json.loads(l) for l in open(BASE / "manifests/frozen_test.jsonl", encoding="utf-8")]
    if args.split in ("all", "cal"):
        rows += [json.loads(l) for l in open(BASE / "manifests/calibration.jsonl", encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]
    out_path = BASE / "audits/g2_qwen_second_judge.jsonl"
    done = {json.loads(l)["id"] for l in open(out_path, encoding="utf-8")} if out_path.exists() else set()
    todo = [r for r in rows if r["id"] not in done]
    print(f"[g2qwen] total={len(rows)} done={len(done)} todo={len(todo)}", flush=True)
    client = make_client()
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(judge_one, client, r, args.model): r["id"] for r in todo}
        n_ok = 0
        for i, fut in enumerate(cf.as_completed(futs)):
            res = fut.result()
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
            if res["status"] == "ok":
                n_ok += 1
            if (i + 1) % 200 == 0:
                print(f"[g2qwen] {i+1}/{len(todo)} ok={n_ok} {time.time()-t0:.0f}s", flush=True)
    print(f"[g2qwen] DONE ok={n_ok} total={len(todo)} {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
