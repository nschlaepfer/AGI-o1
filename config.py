"""
Configuration module for AGI-o1.

Centralizes all environment variables, model settings, and runtime configuration.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv


# Resolve repository root and load environment variables
REPO_ROOT = Path(__file__).resolve().parent
load_dotenv(REPO_ROOT / ".env")


def _bool_from_env(name: str, default: bool = False) -> bool:
    """Parse boolean environment variable with common true values."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int_from_env(name: str, default: int) -> int:
    """Parse integer environment variable with fallback."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


@dataclass(frozen=True)
class ModelConfig:
    """OpenAI model configuration settings."""
    assistant: str = "gpt-5.2"
    reasoning: str = "gpt-5.2"
    reasoning_effort: str = "high"
    code: str = "gpt-5.2-codex"
    code_max: str = "gpt-5.1-codex-max"
    realtime: str = "gpt-realtime"
    realtime_mini: str = "gpt-realtime-mini"
    embedding: str = "text-embedding-3-large"
    compaction_enabled: bool = True


@dataclass(frozen=True)
class LogConfig:
    """Logging configuration settings."""
    level: str = "INFO"
    sample_prompts: bool = True
    sample_responses: bool = False


@dataclass(frozen=True)
class ReasoningBankConfig:
    """ReasoningBank memory system configuration."""
    enabled: bool = True
    path: Path = field(default_factory=lambda: REPO_ROOT / "docs" / "reasoning_bank.json")
    max_items: int = 500


@dataclass(frozen=True)
class AppConfig:
    """Main application configuration."""
    environment: str
    api_key: str
    models: ModelConfig
    logging: LogConfig
    reasoning_bank: ReasoningBankConfig
    
    # Directory paths
    log_dir: Path = field(default_factory=lambda: REPO_ROOT / "logs")
    responses_dir: Path = field(default_factory=lambda: REPO_ROOT / "o1_responses")
    capsules_dir: Path = field(default_factory=lambda: REPO_ROOT / "insight_capsules")
    paper_path: Path = field(default_factory=lambda: REPO_ROOT / "docs" / "paper.txt")


def _resolve_api_key(env: str, key_map: Dict[str, Optional[str]]) -> Optional[str]:
    """Resolve API key for the given environment with fallback to legacy key."""
    if env in key_map and key_map[env]:
        return key_map[env]
    return os.getenv("OPENAI_API_KEY")


def load_config() -> AppConfig:
    """Load and validate application configuration from environment."""
    environment = os.getenv("OPENAI_ENVIRONMENT", "sandbox").strip().lower()
    
    api_keys: Dict[str, Optional[str]] = {
        "sandbox": os.getenv("OPENAI_API_KEY_SANDBOX"),
        "staging": os.getenv("OPENAI_API_KEY_STAGING"),
        "production": os.getenv("OPENAI_API_KEY_PRODUCTION"),
    }
    
    api_key = _resolve_api_key(environment, api_keys)
    if not api_key:
        raise ValueError(
            f"No OpenAI API key configured for environment '{environment}'. "
            "Populate `.env` or secret manager with the correct OPENAI_API_KEY_* variable."
        )
    
    models = ModelConfig(
        assistant=os.getenv("OPENAI_MODEL_ASSISTANT", os.getenv("OPENAI_MODEL_REASONING", "gpt-5.2")),
        reasoning=os.getenv("OPENAI_MODEL_REASONING", "gpt-5.2"),
        reasoning_effort=os.getenv("OPENAI_MODEL_REASONING_EFFORT", "high"),
        code=os.getenv("OPENAI_MODEL_CODE", "gpt-5.2-codex"),
        code_max=os.getenv("OPENAI_MODEL_CODE_MAX", "gpt-5.1-codex-max"),
        realtime=os.getenv("OPENAI_MODEL_REALTIME", "gpt-realtime"),
        realtime_mini=os.getenv("OPENAI_MODEL_REALTIME_MINI", "gpt-realtime-mini"),
        embedding=os.getenv("OPENAI_MODEL_EMBEDDING", "text-embedding-3-large"),
        compaction_enabled=_bool_from_env("OPENAI_MODEL_COMPACTION", True),
    )
    
    logging_config = LogConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        sample_prompts=_bool_from_env("LOG_SAMPLE_PROMPTS", True),
        sample_responses=_bool_from_env("LOG_SAMPLE_RESPONSES", False),
    )
    
    rb_path_str = os.getenv("REASONING_BANK_PATH")
    rb_path = Path(rb_path_str) if rb_path_str else REPO_ROOT / "docs" / "reasoning_bank.json"
    
    reasoning_bank = ReasoningBankConfig(
        enabled=_bool_from_env("REASONING_BANK_ENABLED", True),
        path=rb_path,
        max_items=_int_from_env("REASONING_BANK_MAX_ITEMS", 500),
    )
    
    return AppConfig(
        environment=environment,
        api_key=api_key,
        models=models,
        logging=logging_config,
        reasoning_bank=reasoning_bank,
    )


# Global config instance (loaded on import)
try:
    config = load_config()
except ValueError as e:
    # Allow import without valid config for testing
    config = None  # type: ignore
    _config_error = str(e)
