# -*- coding: utf-8 -*-
"""G2-v2 dual blind judge runner (DeepSeek + Qwen + optional third provider).

Judges the current frozen test + calibration panel (or a candidate manifest)
with the rubric in frauddistill.e4e5_v2.g2_v2_prompts. Resumable, concurrent,
budgeted, and writes per-provider outputs.

Usage:
  python scripts/e4e5_g2_v2_judge.py --split all --provider deepseek --model deepseek-chat
  python scripts/e4e5_g2_v2_judge.py --split all --provider qwen --model qwen-plus
  python scripts/e4e5_g2_v2_judge.py --input manifests/u2_fraud_candidates.jsonl --provider glm --model glm-4.6
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from frauddistill.e4e5_v2.g2_v2_prompts import G2_V2_SYSTEM, G2_V2_USER, G2_V2_USER_FEWSHOT, parse_g2_v2  # noqa: E402
from frauddistill.e4e5_v2.g2_v3_prompts import G2_V3_SYSTEM, G2_V3_USER, G2_V3_USER_FEWSHOT, parse_g2_v3  # noqa: E402
from frauddistill.e4e5_v2.g2_v4_prompts import G2_V4_SYSTEM, G2_V4_USER, G2_V4_USER_FEWSHOT, parse_g2_v4  # noqa: E402

BASE = REPO / "outputs/exp4_unseen_student_v2/e4v2_FINAL"
LEDGER = REPO / "outputs/exp4_unseen_student_v2/e5_budget_ledger.jsonl"

PROVIDERS = {
    "deepseek": ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", (0.3, 0.6)),
    "qwen": ("QWEN_API_KEY", "QWEN_BASE_URL", (0.8, 2.0)),
    "glm": ("GLM_API_KEY", "GLM_BASE_URL", (0.6, 1.2)),
    "kimi": ("KIMI_API_KEY", "KIMI_BASE_URL", (1.0, 2.0)),
}


def make_client(provider: str):
    import api_keys
    key_attr, url_attr, _ = PROVIDERS[provider]
    key = getattr(api_keys, key_attr, "") or ""
    if not key:
        raise SystemExit(f"no {key_attr}")
    from openai import OpenAI
    return OpenAI(api_key=key, base_url=getattr(api_keys, url_attr, ""))


def judge_one(client, row: dict, provider: str, model: str, user_tpl=G2_V2_USER, sys_prompt=G2_V2_SYSTEM, parse_fn=parse_g2_v2):
    q = str(row.get("user_query") or "")[:2500]
    y = str(row.get("target_model_answer") or "")[:2500]
    last = "unknown"
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_tpl.format(q=q, y=y)},
                ],
                temperature=0, max_tokens=120)
            parsed = parse_fn(resp.choices[0].message.content or "")
            if parsed:
                in_tok = int(getattr(resp.usage, "prompt_tokens", 0) or 0)
                out_tok = int(getattr(resp.usage, "completion_tokens", 0) or 0)
                _, _, price = PROVIDERS[provider]
                cost = (in_tok / 1e6) * price[0] + (out_tok / 1e6) * price[1]
                return {"id": row["id"], "split": row.get("split", row.get("primary_shift", "")),
                        "status": "ok", "provider": provider, "model": model,
                        "label": parsed["label"], "type": parsed["type"],
                        "confidence": parsed["confidence"], "evidence": parsed["evidence"],
                        "input_tokens": in_tok, "output_tokens": out_tok,
                        "cost_cny": round(cost, 6)}
        except Exception as e:
            last = f"{type(e).__name__}: {str(e)[:120]}"
            time.sleep(1.0 + attempt)
    return {"id": row["id"], "status": "failed", "provider": provider, "model": model, "error": last[:150]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="all", choices=["test", "cal", "all"])
    ap.add_argument("--input", default="")
    ap.add_argument("--provider", required=True, choices=list(PROVIDERS))
    ap.add_argument("--model", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--concurrency", type=int, default=120)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="G2_v2")
    ap.add_argument("--fewshot", action="store_true")
    ap.add_argument("--prompt", default="v2", choices=["v2", "v3", "v4"])
    args = ap.parse_args()

    if args.input:
        rows = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    else:
        rows = []
        if args.split in ("all", "test"):
            rows += [json.loads(l) for l in open(BASE / "manifests/frozen_test.jsonl", encoding="utf-8") if l.strip()]
        if args.split in ("all", "cal"):
            rows += [json.loads(l) for l in open(BASE / "manifests/calibration.jsonl", encoding="utf-8") if l.strip()]
    model = args.model or {"deepseek": "deepseek-chat", "qwen": "qwen-plus", "glm": "glm-4.6", "kimi": "moonshot-v1-8k"}[args.provider]
    if args.prompt == "v4":
        sys_prompt = G2_V4_SYSTEM
        user_tpl = G2_V4_USER_FEWSHOT if args.fewshot else G2_V4_USER
        parse_fn = parse_g2_v4
    elif args.prompt == "v3":
        sys_prompt = G2_V3_SYSTEM
        user_tpl = G2_V3_USER_FEWSHOT if args.fewshot else G2_V3_USER
        parse_fn = parse_g2_v3
    else:
        sys_prompt = G2_V2_SYSTEM
        user_tpl = G2_V2_USER_FEWSHOT if args.fewshot else G2_V2_USER
        parse_fn = parse_g2_v2
    out_path = Path(args.out) if args.out else BASE / f"audits/g2_v2_{args.provider}.jsonl"
    done = set()
    if out_path.exists():
        for l in open(out_path, encoding="utf-8"):
            try:
                done.add(json.loads(l)["id"])
            except Exception:
                pass
    todo = [r for r in rows if r["id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[g2v2:{args.provider}] total={len(rows)} done={len(done)} todo={len(todo)}", flush=True)
    if not todo:
        return
    client = make_client(args.provider)
    t0 = time.time()
    n_ok = 0
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(judge_one, client, r, args.provider, model, user_tpl, sys_prompt, parse_fn): r["id"] for r in todo}
        for i, fut in enumerate(cf.as_completed(futs)):
            res = fut.result()
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
            if res["status"] == "ok":
                n_ok += 1
                if res["cost_cny"] > 0:
                    try:
                        with open(LEDGER, "a", encoding="utf-8") as f:
                            f.write(json.dumps({
                                "sample_id": res["id"], "phase": args.tag, "provider": args.provider,
                                "model_snapshot": model, "input_tokens": res["input_tokens"],
                                "output_tokens": res["output_tokens"],
                                "estimated_cost_cny": res["cost_cny"],
                                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
                            }, ensure_ascii=False) + "\n")
                    except Exception:
                        pass
            if (i + 1) % 300 == 0:
                print(f"[g2v2:{args.provider}] {i+1}/{len(todo)} ok={n_ok} {time.time()-t0:.0f}s", flush=True)
    print(f"[g2v2:{args.provider}] DONE ok={n_ok}/{len(todo)} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
