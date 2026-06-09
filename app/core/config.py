from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App
    app_name: str = "Chasqui Core"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8090

    # Database
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "postgres"
    postgres_password: str = ""
    postgres_db: str = "chasqui"

    # Security — shared secret for gateway -> core calls (e.g. /ingest)
    internal_api_key: str | None = None

    # CORS
    cors_origins: str = "*"
    cors_allow_credentials: bool = True

    # JWT (admin authentication)
    jwt_secret_key: str = ""
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    # LLM — provider/model are swappable via LangChain's init_chat_model()
    # (Sprint 3 factory). Examples: google + gemini-2.5-flash (stable,
    # multimodal), google + gemini-3-flash-preview, anthropic + claude-*,
    # ollama + llama3.3, openai-compatible base_url for OpenRouter.
    llm_provider: str = "google"  # "google" | "anthropic" | "openai" | "ollama" | ...
    llm_model: str = "gemini-2.5-flash"
    google_api_key: str | None = None
    anthropic_api_key: str | None = None

    # Modality overrides for models the capability registry doesn't know
    # (see app/core/llm_capabilities.py). None = auto-detect by model name.
    llm_supports_vision: bool | None = None
    llm_supports_audio: bool | None = None

    # Embeddings (RAG over pgvector)
    embedding_model: str = "text-embedding-004"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
