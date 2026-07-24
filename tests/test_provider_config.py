from frauddistill.target_llm.provider_config import get_provider_config


def test_extended_provider_defaults_are_available_without_local_keys():
    assert get_provider_config("gemini").default_model == "gemini-2.5-flash"
    assert get_provider_config("kimi").base_url == "https://api.moonshot.cn/v1"
    assert get_provider_config("glm").name == "glm"
    assert get_provider_config("openrouter").default_model == "openai/gpt-4.1-mini"
