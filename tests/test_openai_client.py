from frauddistill.target_llm.openai_client import parse_json_content


def test_parse_json_content_strips_markdown_fence():
    assert parse_json_content('```json\n{"ok": true}\n```')["ok"] is True


def test_parse_json_content_returns_parse_error_on_broken_json():
    assert parse_json_content('{"ok": "unterminated')["parse_error"] is True
