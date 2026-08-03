import asyncio
from openai import AsyncOpenAI

async def main():
    client = AsyncOpenAI(api_key="sk-e63265dc4b06402599822fd17203256f", base_url="https://api.deepseek.com", timeout=60)
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
