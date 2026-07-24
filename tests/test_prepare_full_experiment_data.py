from frauddistill.data.prepare_full_experiment_data import build_anti_fraud_synthetic_prompts, with_roleplay_setting


def test_roleplay_setting_preserves_base_metadata():
    row = {
        "id": "x",
        "language": "English",
        "user_query": "How to identify phishing?",
        "metadata": {"source_file": "raw.json"},
    }
    roleplay = with_roleplay_setting(row)
    assert roleplay["id"] == "x_roleplay"
    assert roleplay["prompt_setting"] == "role_play"
    assert roleplay["metadata"]["base_prompt_id"] == "x"
    assert "How to identify phishing?" in roleplay["user_query"]


def test_anti_fraud_synthetic_covers_all_categories_and_languages():
    rows = build_anti_fraud_synthetic_prompts(
        [
            {"fraud_category": "phishing_scams"},
            {"fraud_category": "online_relationships"},
        ]
    )
    assert len(rows) == 12
    assert {row["language"] for row in rows} == {"Chinese", "English"}
    assert {row["fraud_category"] for row in rows} == {"phishing_scams", "online_relationships"}
