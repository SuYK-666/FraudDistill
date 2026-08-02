from __future__ import annotations

from pathlib import Path

from frauddistill.e1_final_v3.api_executor import execute_tasks, request_fingerprint
from frauddistill.e1_final_v3.io import read_jsonl, write_jsonl
from frauddistill.e1_final_v3.registry import build_v31_a_manifest


class MockClient:
    model = "mock-model"

    def __init__(self) -> None:
        self.calls = 0

    def complete_text(self, prompt: str, **kwargs):
        self.calls += 1
        return "mock answer"


def minimal_task() -> dict:
    return {
        "prompt_instance_id": "case|stage0|assistant|v",
        "canonical_case_id": "case",
        "stage_id": 0,
        "scenario": "assistant",
        "q_private": "question",
        "target_provider": "qwen",
        "requested_target_model": "mock-model",
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 8,
        "phase": "test",
    }


def test_no_run_api_means_no_provider_call(tmp_path: Path) -> None:
    client = MockClient()
    result = execute_tasks(
        [minimal_task()],
        output_path=tmp_path / "out.jsonl",
        ledger_path=tmp_path / "ledger.jsonl",
        limits={},
        run_api=False,
        confirm_budget=True,
        git_clean=True,
        provider_factory=lambda p, m: client,
    )
    assert result["created_calls"] == 0
    assert client.calls == 0


def test_no_confirm_budget_means_no_provider_call(tmp_path: Path) -> None:
    client = MockClient()
    result = execute_tasks(
        [minimal_task()],
        output_path=tmp_path / "out.jsonl",
        ledger_path=tmp_path / "ledger.jsonl",
        limits={},
        run_api=True,
        confirm_budget=False,
        git_clean=True,
        provider_factory=lambda p, m: client,
    )
    assert result["created_calls"] == 0
    assert client.calls == 0


def test_dirty_worktree_blocks_api(tmp_path: Path) -> None:
    client = MockClient()
    result = execute_tasks(
        [minimal_task()],
        output_path=tmp_path / "out.jsonl",
        ledger_path=tmp_path / "ledger.jsonl",
        limits={},
        run_api=True,
        confirm_budget=True,
        git_clean=False,
        provider_factory=lambda p, m: client,
    )
    assert result["status"] == "STOP_DIRTY_WORKTREE"
    assert client.calls == 0


def test_resume_skips_successful_fingerprint(tmp_path: Path) -> None:
    task = minimal_task()
    fp = request_fingerprint(task)
    output = tmp_path / "out.jsonl"
    write_jsonl(output, [{**task, "request_fingerprint": fp, "status": "ok", "text": "cached"}])
    client = MockClient()
    result = execute_tasks(
        [task],
        output_path=output,
        ledger_path=tmp_path / "ledger.jsonl",
        limits={},
        run_api=True,
        confirm_budget=True,
        git_clean=True,
        provider_factory=lambda p, m: client,
    )
    assert result["created_calls"] == 0
    assert result["skipped_cache"] == 1
    assert client.calls == 0


def test_corrupt_cache_is_isolated_not_fatal(tmp_path: Path) -> None:
    output = tmp_path / "out.jsonl"
    write_jsonl(output, [{"status": "ok", "text": "missing fingerprint"}])
    client = MockClient()
    result = execute_tasks(
        [minimal_task()],
        output_path=output,
        ledger_path=tmp_path / "ledger.jsonl",
        limits={},
        run_api=True,
        confirm_budget=True,
        git_clean=True,
        provider_factory=lambda p, m: client,
    )
    assert result["invalid_cache_rows"] == 1
    assert result["created_calls"] == 1
    assert client.calls == 1


def test_v31_manifest_stage_and_setting_isolation() -> None:
    import yaml

    cfg = yaml.safe_load(Path("configs/experiments/e1_final_triad_v3.yaml").read_text(encoding="utf-8"))
    prompts, tasks, audit = build_v31_a_manifest(
        raw_prompts_path=Path(cfg["data"]["fraudr1_raw_prompts"]),
        raw_base_en=Path(cfg["data"]["fraudr1_raw_base_en"]),
        raw_base_zh=Path(cfg["data"]["fraudr1_raw_base_zh"]),
        v10_registry_path=Path(cfg["data"]["v10_registry"]),
        config=cfg,
    )
    assert audit["canonical_cases"] == 2141
    assert audit["target_prompt_instances"] == 3750
    assert audit["reused_target_responses"] == 3082
    assert audit["pending_target_calls"] == 4418
    assert {p["stage_id"] for p in prompts} == {0}
    by_case = {}
    for row in prompts:
        by_case.setdefault(row["canonical_case_id"], {})[row["scenario"]] = row["exact_q_sha256"]
    paired = [v for v in by_case.values() if {"assistant", "roleplay"} <= set(v)]
    assert paired
    assert all(v["assistant"] != v["roleplay"] for v in paired[:20])
    assert all(t["stage_id"] == 0 for t in tasks)
