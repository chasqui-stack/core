"""Agent orchestrator — STUB.

Sprint 1 ships a canned echo so the ingest pipeline is testable end-to-end.
Sprint 3 replaces this with the real LangGraph state machine
(router → tools → respond) fed by the Tool Registry (§8) and memory (§6).
The signature is the seam: ingest_service only knows this function.
"""

from app.models import Conversation
from app.schemas.ingest import InboundMessage, OutboundMessage


async def run_turn(
    conversation: Conversation,
    inbound: InboundMessage,
) -> list[OutboundMessage]:
    """Produce the agent's reply (1..N messages) for one inbound message.

    Stub behavior: echo text back (or acknowledge non-text types).
    """
    if inbound.type == "text" and inbound.text:
        reply = f"Echo: {inbound.text}"
    else:
        reply = f"Recibí tu mensaje ({inbound.type}). Pronto podré procesarlo."

    return [OutboundMessage(type="text", text=reply)]
