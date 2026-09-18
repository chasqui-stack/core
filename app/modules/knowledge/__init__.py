"""Knowledge module — Document-RAG over uploaded files.

Operators upload documents (pdf, docx, txt, md, html); the module extracts
their text, splits it into overlapping chunks and embeds each chunk into
pgvector, tracking per-document state (`pending → processing → ready | error`).

Coexists with the FAQ module: FAQ is curated, atomic Q&A; this is bulk
reference material (manuals, catalogs, policies). The `search_documents`
agent tool lands separately — for now the module contributes tables and
admin routes only.
"""

# Imported at module level so the tables reach SQLModel.metadata on discover()
from app.modules.knowledge.models import Document, DocumentChunk


class KnowledgeModule:
    """Document knowledge base: upload, chunk, embed, search."""

    name = "knowledge"

    def register_tools(self):
        return []

    def register_models(self):
        return [Document, DocumentChunk]

    def register_admin_routes(self, router):
        from app.modules.knowledge.admin import register

        register(router)


module = KnowledgeModule()
