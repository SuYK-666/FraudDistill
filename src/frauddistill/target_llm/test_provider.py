from __future__ import annotations

import argparse

from frauddistill.target_llm.openai_client import OpenAIJsonClient
from frauddistill.target_llm.provider_config import get_provider_config, require_api_key


def smoke_test_provider(provider: str, model: str | None = None) -> dict:
    config = get_provider_config(provider, model)
    require_api_key(config)
    client = OpenAIJsonClient(config.default_model, api_key=config.api_key, base_url=config.base_url)
    return client.complete_json(
        'Return JSON exactly like {"ok": true, "provider": "<provider>", "note": "smoke test"}',
        max_tokens=80,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["openai", "qwen", "deepseek"], required=True)
    parser.add_argument("--model")
    args = parser.parse_args()
    print(smoke_test_provider(args.provider, args.model))


if __name__ == "__main__":
    main()
