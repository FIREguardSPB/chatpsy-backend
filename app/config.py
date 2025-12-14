"""Application configuration using Pydantic Settings."""
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # OpenAI / LLM Configuration
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"
    use_llm: bool = False

    # LLM Limits
    llm_max_chars: int = 60000
    llm_max_tokens: int = 5000

    # Rate Limiting
    default_analyze_limit: int = 3
    feedback_bonus_analyses: int = 3
    recommended_bytes: int = 2548576

    # Admin
    admin_token: str = Field(default="", alias="ADMIN_TOKEN")

    # Admin-tunable defaults
    global_default_analyze_limit: int = 3

    # Payment
    payment_price_rub: int = 299  # Price for full analysis in RUB
    payment_provider: str = "yookassa"  # yookassa, stripe, etc
    payment_webhook_secret: Optional[str] = None
    payment_enabled: bool = False  # Enable/disable payment requirement
    payment_test_mode: bool = False  # Enable frontend polling/unlock behavior in test mode

    # YooKassa credentials
    yookassa_shop_id: Optional[str] = None
    yookassa_secret_key: Optional[str] = None

    # LLM behaviour
    llm_validate_json: bool = True  # Включает дополнительную валидацию JSON
    llm_retry_on_failure: int = 1   # Количество попыток при невалидном JSON

    # Temperature (чтобы менять в .env без деплоя)
    llm_temperature: float = Field(default=0.5, alias="LLM_TEMPERATURE")

    @property
    def llm_model(self) -> str:
        """Обратная совместимость: старый код может звать settings.llm_model."""
        return self.openai_model

    # Paths
    log_dir: Path = Path("logs")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.log_dir.mkdir(exist_ok=True)


# Global settings instance
settings = Settings()

