from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    api_key: str
    base_url: str
    default_model: str


def _load_local_keys():
    try:
        return import_module("api_keys")
    except ModuleNotFoundError:
        return None


def get_provider_config(provider: str, model: str | None = None) -> ProviderConfig:
    keys = _load_local_keys()
    provider = provider.lower()
    if provider == "openai":
        return ProviderConfig(
            name="openai",
            api_key=getattr(keys, "OPENAI_API_KEY", "") if keys else "",
            base_url=getattr(keys, "OPENAI_BASE_URL", "https://api.openai.com/v1") if keys else "https://api.openai.com/v1",
            default_model=model or "gpt-4.1-mini",
        )
    if provider in {"qwen", "dashscope"}:
        return ProviderConfig(
            name="qwen",
            api_key=getattr(keys, "QWEN_API_KEY", "") if keys else "",
            base_url=getattr(keys, "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1") if keys else "https://dashscope.aliyuncs.com/compatible-mode/v1",
            default_model=model or "qwen-plus",
        )
    if provider == "deepseek":
        return ProviderConfig(
            name="deepseek",
            api_key=getattr(keys, "DEEPSEEK_API_KEY", "") if keys else "",
            base_url=getattr(keys, "DEEPSEEK_BASE_URL", "https://api.deepseek.com") if keys else "https://api.deepseek.com",
            default_model=model or "deepseek-chat",
        )
    raise ValueError(f"unknown provider: {provider}")


def require_api_key(config: ProviderConfig) -> None:
    if not config.api_key:
        raise RuntimeError(f"{config.name} API key is empty. Copy api_keys.template.py to api_keys.py and fill it.")
