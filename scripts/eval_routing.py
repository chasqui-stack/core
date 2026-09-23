#!/usr/bin/env python
"""Routing eval: does the agent pick `faq_search` vs `search_documents` right?

Runs the REAL agent (your LLM_MODEL + embeddings from .env) over 11 questions:
5 answerable only from the FAQ, 5 only from the uploaded documents, and 1 that
neither store answers. Each question is a fresh conversation. Per answerable
question it scores three things:

    F  the FIRST tool call was the expected one
    U  the expected tool was USED at some point (the sibling handover worked)
    G  the reply is GROUNDED: it contains the seeded fact (regex)

Re-run it whenever a tool docstring or return string of `faq`/`knowledge`
changes, or when you switch models (AGENTS.md, ADR-013).

It never touches your dev data: it (re)creates a `<POSTGRES_DB>_eval` database,
seeds it from `scripts/eval_data/` (embedding calls are real, and cheap) and
runs there. LangSmith tracing is forced off unless you pass --trace.

Usage:
    make eval-routing              # 1 run
    make eval-routing runs=5       # 5 runs, totals out of 25 per side
    uv run python scripts/eval_routing.py --runs 5 --model gemini-3.8-flash

G is a regex, so read the replies when it fails: a correct "1:00 p.m." once
failed a check that only looked for "13".
"""

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_DIR = Path(__file__).resolve().parent / "eval_data"

# Seed. The FAQ and the documents are disjoint on purpose, except the price
# list's returns/warranty sections, which are near-misses of the FAQ's returns
# entry and of the manual's warranty steps — that overlap is what makes the
# routing hard.
FAQS = [
    ("¿Cuál es el horario de atención?", "Atendemos de lunes a viernes de 9:00 a 18:00, y sábados de 9:00 a 13:00 (hora de Lima)."),
    ("¿Cuál es la política de devoluciones?", "Aceptamos devoluciones dentro de los 30 días con comprobante de compra. El reembolso se procesa en 5-7 días hábiles."),
    ("¿Hacen envíos a provincia?", "Sí, enviamos a todo el Perú vía Olva Courier. Lima: 24-48h. Provincias: 3-5 días hábiles. Envío gratis desde S/150."),
    ("¿Qué medios de pago aceptan?", "Aceptamos tarjetas Visa y Mastercard, Yape, Plin y transferencia bancaria (BCP e Interbank). No aceptamos pago contra entrega."),
    ("¿Tienen tienda física?", "Sí, estamos en Av. Larco 1234, Miraflores, Lima. También puedes escribirnos a hola@tiendaandina.example o llamar al 01 555 0199."),
]
DOCUMENTS = ["manual-mochila-chasqui-30l.md", "lista-de-precios-2026.md"]

# (expected side, question, grounding regex). side "none" = neither store
# answers it: read the reply, it must not invent a product.
QUESTIONS = [
    ("faq", "¿A qué hora abren los sábados?", r"13|1(:00)?\s*(p\.?\s*m|de la tarde)"),
    ("faq", "Hola, ¿hacen envíos a Arequipa? ¿cuánto demora?", r"3\s*(-|a|y)\s*5"),
    ("faq", "¿Puedo pagar con Yape?", r"yape"),
    ("faq", "¿Dónde queda su tienda? quiero ir en persona", r"larco"),
    ("faq", "Compré algo hace dos semanas y no me gustó, ¿puedo devolverlo?", r"30"),
    ("docs", "¿Cuánto cuesta la mochila Chasqui de 30 litros?", r"189"),
    ("docs", "¿Puedo meter la mochila Chasqui a la lavadora?", r"mano"),
    ("docs", "¿Cuántos kilos aguanta la mochila de 30L?", r"12"),
    ("docs", "Se me rompió el cierre de la mochila, ¿cómo hago válida la garantía?", r"garantias@|foto"),
    ("docs", "¿Qué garantía tiene la botella térmica de 1 litro?", r"24"),
    ("none", "¿Venden carpas para 4 personas?", None),
]
EXPECT = {"faq": "faq_search", "docs": "search_documents"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FAQ vs documents routing eval")
    parser.add_argument("--runs", type=int, default=1, help="passes over the questions (default: 1)")
    parser.add_argument("--model", help="override LLM_MODEL for this run")
    parser.add_argument("--trace", action="store_true", help="keep LangSmith tracing as configured")
    parser.add_argument("--no-seed", action="store_true", help="reuse the existing eval DB as is")
    return parser.parse_args()


async def prepare_database(admin_url: str, eval_db: str) -> None:
    """Create `eval_db` if missing, build the schema and empty it."""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import NullPool
    from sqlmodel import SQLModel

    from app.core.config import settings

    admin_engine = create_async_engine(admin_url, isolation_level="AUTOCOMMIT")
    async with admin_engine.connect() as conn:
        exists = await conn.scalar(
            text("SELECT 1 FROM pg_database WHERE datname = :db"), {"db": eval_db}
        )
        if not exists:
            await conn.execute(text(f'CREATE DATABASE "{eval_db}"'))
    await admin_engine.dispose()

    engine = create_async_engine(settings.database_url, poolclass=NullPool)
    async with engine.begin() as conn:
        await conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"'))
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(SQLModel.metadata.create_all)
        tables = ", ".join(t.name for t in SQLModel.metadata.sorted_tables)
        await conn.execute(text(f"TRUNCATE {tables} CASCADE"))
    await engine.dispose()


async def seed() -> None:
    from app.db import session as db_session
    from app.modules.faq import service as faq_service
    from app.modules.knowledge import service as kb
    from app.modules.knowledge.models import Document

    async with db_session.async_session_factory() as session:
        for question, answer in FAQS:
            await faq_service.create_entry(session, question=question, answer=answer)
        docs = []
        for filename in DOCUMENTS:
            data = (DATA_DIR / filename).read_bytes()
            doc = await kb.create_document(session, filename=filename, mime_type="text/markdown", data=data)
            docs.append((doc.id, data))
        await session.commit()

    for doc_id, data in docs:
        await kb.process_document(doc_id, data)
    async with db_session.async_session_factory() as session:
        for doc_id, _ in docs:
            doc = await session.get(Document, doc_id)
            if doc.status != kb.STATUS_READY:
                sys.exit(f"seed failed: {doc.filename} is {doc.status} ({doc.error_detail})")


async def ask(question: str) -> tuple[list[str], str]:
    """One fresh conversation through the real agent; rolled back afterwards."""
    from langchain_core.messages import AIMessage, HumanMessage

    from app.db import session as db_session
    from app.models import Contact, Conversation
    from app.modules import registry
    from app.services import agent_config_service, orchestrator
    from app.services.agent_context import TurnContext

    async with db_session.async_session_factory() as session:
        try:
            config = await agent_config_service.get_config(session)
            contact = Contact(channel="eval", external_id=f"eval-{abs(hash(question))}")
            session.add(contact)
            await session.flush()
            conversation = Conversation(contact_id=contact.id)
            session.add(conversation)
            await session.flush()
            ctx = TurnContext(
                session=session, contact_id=contact.id, conversation_id=conversation.id, config=config
            )
            fragments = await registry.get_prompt_fragments(ctx, question)
            messages = [
                orchestrator._system_message(config, fragments, orchestrator._capabilities()),
                HumanMessage(content=question),
            ]
            result = await orchestrator._get_agent().ainvoke({"messages": messages}, context=ctx)
            calls = [
                call["name"]
                for message in result["messages"]
                if isinstance(message, AIMessage)
                for call in (message.tool_calls or [])
            ]
            return calls, orchestrator._extract_text(result["messages"][-1]).strip()
        finally:
            await session.rollback()


async def main() -> None:
    args = parse_args()
    if not args.trace:
        os.environ["LANGSMITH_TRACING"] = "false"

    from app.core.config import settings

    if args.model:
        settings.llm_model = args.model
    admin_url = settings.database_url
    eval_db = f"{settings.postgres_db}_eval"
    # Before ANY app.db import: the app's engine is built from settings at
    # import time, so from here on everything talks to the eval DB.
    settings.postgres_db = eval_db

    import app.models  # noqa: F401 — every core table for create_all
    from app.modules import registry

    registry.discover()
    if not args.no_seed:
        await prepare_database(admin_url, eval_db)
        await seed()

    print(f"MODEL {settings.llm_provider}:{settings.llm_model} · DB {eval_db} · runs {args.runs}")
    totals = {side: {"first": 0, "used": 0, "grounded": 0} for side in EXPECT}
    for run in range(1, args.runs + 1):
        for side, question, fact in QUESTIONS:
            calls, reply = await ask(question)
            flags = "   "
            if side in EXPECT:
                first = bool(calls) and calls[0] == EXPECT[side]
                used = EXPECT[side] in calls
                grounded = bool(re.search(fact, reply, re.I))
                t = totals[side]
                t["first"] += first
                t["used"] += used
                t["grounded"] += grounded
                flags = f"{'F' if first else '-'}{'U' if used else '-'}{'G' if grounded else '-'}"
            print(
                f"run{run} [{side:4}] {flags} {question}\n"
                f"        tools={calls}\n        reply={reply[:220]!r}",
                flush=True,
            )
    per_side = args.runs * 5
    print(f"TOTALS (of {per_side} per side)", json.dumps(totals))


if __name__ == "__main__":
    asyncio.run(main())
