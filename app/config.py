from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from typing import Dict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    max_parallel: int = 3
    cpu_pool_size: int = 4
    whisper_model: str = "whisper-large-v3"

    vlm_provider: str = "fireworks"  # "fireworks" or "mock"
    fireworks_api_key: str = Field(default="", alias="FIREWORKS_API_KEY")
    fireworks_vlm_model: str = Field(default="accounts/fireworks/models/kimi-k2p6", alias="FIREWORKS_VLM_MODEL")
    fireworks_llm_model: str = Field(default="accounts/fireworks/models/deepseek-v4-flash", alias="FIREWORKS_LLM_MODEL")

    @classmethod
    @field_validator("fireworks_api_key", mode="before")
    def strip_quotes(cls, v: str) -> str:
        if isinstance(v, str):
            return v.strip('"' + "'")
        return v

    stage_timeouts: Dict[str, int] = {
        "download": 120,
        "extract_audio": 60,
        "transcribe": 300,
        "scene_detect": 60,
        "vlm": 120,
        "ocr": 60,
        "summarize": 60,
    }
    max_download_mb: int = 500
    temp_dir: str = "temp"

config = Settings()
