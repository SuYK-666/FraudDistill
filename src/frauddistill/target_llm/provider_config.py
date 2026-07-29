from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
import os
from pathlib import Path
from typing import Any


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
        # Script entry points put their own directory first on sys.path. Resolve the
        # local, untracked key file explicitly without printing or persisting it.
        key_path = Path.cwd() / "api_keys.py"
        if not key_path.exists():
            return None
        spec = spec_from_file_location("frauddistill_local_api_keys", key_path)
        if spec is None or spec.loader is None:
            return None
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def get_provider_config(provider: str, model: str | None = None) -> ProviderConfig:
    keys = _load_local_keys()
    provider = provider.lower()
    specs: dict[str, tuple[str, str, str, str]] = {
        "openai": ("OPENAI_API_KEY", "OPENAI_BASE_URL", "https://api.openai.com/v1", "gpt-4.1-mini"),
        "qwen": ("QWEN_API_KEY", "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-plus"),
        "dashscope": ("QWEN_API_KEY", "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-plus"),
        "deepseek": ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "https://api.deepseek.com", "deepseek-chat"),
        "gemini": ("GEMINI_API_KEY", "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai", "gemini-2.5-flash"),
        "kimi": ("KIMI_API_KEY", "KIMI_BASE_URL", "https://api.moonshot.cn/v1", "moonshot-v1-8k"),
        "moonshot": ("KIMI_API_KEY", "KIMI_BASE_URL", "https://api.moonshot.cn/v1", "moonshot-v1-8k"),
        "glm": ("GLM_API_KEY", "GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4", "glm-4-flash"),
        "zhipu": ("GLM_API_KEY", "GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4", "glm-4-flash"),
        "doubao": ("DOUBAO_API_KEY", "DOUBAO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3", "doubao-seed-1-6-flash"),
        "openrouter": ("OPENROUTER_API_KEY", "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1", "openai/gpt-4.1-mini"),
    }
    if provider in specs:
        key_attr, url_attr, base_url, default_model = specs[provider]
        return ProviderConfig(
            name="qwen" if provider == "dashscope" else "kimi" if provider == "moonshot" else "glm" if provider == "zhipu" else provider,
            api_key=_get_key(keys, key_attr),
            base_url=_get_key(keys, url_attr, base_url),
            default_model=model or default_model,
        )
    raise ValueError(f"unknown provider: {provider}")


def require_api_key(config: ProviderConfig) -> None:
    if not config.api_key:
        raise RuntimeError(f"{config.name} API key is empty. Copy api_keys.template.py to api_keys.py and fill it.")


def _get_key(keys: Any, attr: str, default: str = "") -> str:
    value = os.environ.get(attr)
    if value is not None and value != "":
        return value
    return getattr(keys, attr, default) if keys else default
