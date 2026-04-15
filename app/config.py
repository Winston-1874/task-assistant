from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "google/gemini-2.5-flash"
    llm_vision_model: str = "google/gemini-2.5-flash"

    database_url: str = "sqlite+aiosqlite:///./data/tasks.db"
    secret_key: str = "changeme"
    app_password_hash: str = ""

    daily_capacity_minutes: int = 420
    digest_enabled: bool = True
    zombie_threshold_days: int = 21
    coaching_tone: str = "concise"
    environment: str = "dev"

    @model_validator(mode="after")
    def _check_prod_secrets(self) -> "Settings":
        if self.environment == "prod":
            if self.secret_key == "changeme":
                raise ValueError("SECRET_KEY must be set in production")
            if not self.app_password_hash:
                raise ValueError("APP_PASSWORD_HASH must be set in production")
        return self


settings = Settings()
