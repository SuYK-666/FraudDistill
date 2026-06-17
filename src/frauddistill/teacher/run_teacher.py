from __future__ import annotations

import argparse

from frauddistill.agents.multi_agent_teacher import MultiAgentTeacher
from frauddistill.data.schema import FraudDistillSample
from frauddistill.target_llm.openai_client import OpenAIJsonClient
from frauddistill.target_llm.provider_config import get_provider_config, require_api_key
from frauddistill.utils.io import read_jsonl, write_jsonl


def generate_teacher_signals(
    input_file: str,
    output_file: str,
    limit: int | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> int:
    client = None
    if provider:
        config = get_provider_config(provider, model)
        require_api_key(config)
        client = OpenAIJsonClient(config.default_model, api_key=config.api_key, base_url=config.base_url)
    teacher = MultiAgentTeacher(client)
    rows = []
    for idx, row in enumerate(read_jsonl(input_file)):
        if limit is not None and idx >= limit:
            break
        sample = FraudDistillSample.model_validate(row)
        rows.append(teacher.run(sample.model_dump(mode="json")))
    return write_jsonl(output_file, rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--provider", choices=["openai", "qwen", "deepseek"])
    parser.add_argument("--model")
    args = parser.parse_args()
    count = generate_teacher_signals(args.input_file, args.output_file, args.limit, args.provider, args.model)
    print(f"wrote {count} teacher signals to {args.output_file}")


if __name__ == "__main__":
    main()
