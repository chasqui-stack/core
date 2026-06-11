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
    # (see app/core/llm.py). Examples: google + gemini-2.5-flash (stable,
    # multimodal), google + gemini-3-flash-preview, anthropic + claude-*,
    # ollama + llama3.3, openrouter + <vendor/model>.
    llm_provider: str = "google"  # "google" | "anthropic" | "openai" | "openrouter" | "ollama"
    llm_model: str = "gemini-2.5-flash"
    llm_temperature: float = 0.7
    google_api_key: str | None = None
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    openai_base_url: str | None = None  # any OpenAI-compatible server
    openrouter_api_key: str | None = None
    ollama_base_url: str | None = None  # default http://localhost:11434

    # Conversation history window fed to the agent each turn
    history_limit: int = 20

    # Sent verbatim to the end user when a turn fails — the ONE user-facing
    # literal in the backend, so it's operator-configurable (set it in your
    # users' language). Everything LLM-facing is English; the system prompt
    # rule "reply in the user's language" localizes the agent itself.
    fallback_reply: str = (
        "Sorry, I had trouble processing your message. "
        "Could you try again in a moment?"
    )

    # Modality overrides for models the capability registry doesn't know
    # (see app/core/llm_capabilities.py). None = auto-detect by model name.
    llm_supports_vision: bool | None = None
    llm_supports_audio: bool | None = None

    # Embeddings (RAG over pgvector) — provider-swappable via
    # init_embeddings(), like the LLM (see app/core/embeddings.py).
    embedding_provider: str = "google"  # "google" | "openai" | "ollama" | ...
    embedding_model: str = "gemini-embedding-001"

    # PROVISION-TIME setting: the vector column width is created from this on
    # the first `alembic upgrade`. Changing it afterwards requires a column
    # migration + re-embedding every row. Index strategy (Sprint 4) adapts:
    # <=2000 HNSW on vector · 2001-4000 HNSW on halfvec · >4000 exact scan.
    embedding_dim: int = 768

    # Media storage (ADR-003) — S3-compatible API via boto3. One client
    # covers AWS S3, Cloudflare R2, Spaces, B2 and MinIO (docker-compose).
    # OPTIONAL: leave unset and media is processed in-turn but not persisted
    # (degraded, not broken). endpoint_url unset = AWS S3 default.
    storage_endpoint_url: str | None = None
    storage_bucket: str | None = None
    storage_access_key: str | None = None
    storage_secret_key: str | None = None
    storage_region: str | None = None
    # Split-horizon deployments only (e.g. docker-compose): presigned URLs
    # embed the signing endpoint, and the browser may not resolve the
    # internal one (http://minio:9000). Set this to the browser-reachable
    # endpoint; unset = same as STORAGE_ENDPOINT_URL.
    storage_public_endpoint_url: str | None = None

    # Canonical outbound seam (ADR-004) — one var per channel, the mirror of
    # /ingest. The core POSTs operator messages to the channel gateway's
    # /send with the same INTERNAL_API_KEY. Unset = sends from the admin
    # panel fail with a clear error for that channel.
    channel_whatsapp_send_url: str | None = None

    # Handoff notifications (ADR-004) — both OPTIONAL, both best-effort
    # (failures log, never break the turn). Webhook covers Slack/Zapier/n8n;
    # SMTP covers email via any relay (Brevo, Mailgun, SES, Gmail app
    # password) — "where is your relay", never "which provider".
    notify_webhook_url: str | None = None
    smtp_host: str | None = None
    smtp_port: int = 587  # 587 = STARTTLS, 465 = implicit SSL
    smtp_user: str | None = None
    smtp_password: str | None = None
    smtp_from: str | None = None
    notify_email_to: str | None = None  # comma-separated recipients

    @property
    def smtp_configured(self) -> bool:
        return bool(self.smtp_host and self.smtp_from and self.notify_email_to)

    @property
    def storage_configured(self) -> bool:
        return bool(
            self.storage_bucket and self.storage_access_key and self.storage_secret_key
        )

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
