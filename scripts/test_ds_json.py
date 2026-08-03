import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import api_keys  # noqa: E402  (local gitignored file, see api_keys.template.py)

from openai import AsyncOpenAI

async def main():
    client = AsyncOpenAI(api_key=api_keys.DEEPSEEK_API_KEY, base_url=api_keys.DEEPSEEK_BASE_URL, timeout=60)
    try:
        resp = await client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {"role": "system", "content": "You are a specialist. Return strict JSON only with exactly these fields: {\"fraud_risk\": 0.0, \"confidence\": 0.0}"},
                {"role": "user", "content": "[USER QUERY]\nhi\n\n[MODEL ANSWER]\nhello"},
            ],
            temperature=0.0,
            max_tokens=160,
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}},
        )
        print("OK", resp.choices[0].message.content, resp.usage)
    except Exception as e:
        print("ERR", type(e).__name__, str(e)[:2000])

asyncio.run(main())
