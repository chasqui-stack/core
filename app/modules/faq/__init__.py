"""FAQ module — search the company knowledge base.

STUB in Sprint 3: the real RAG (pgvector over the admin-managed FAQ
collection) lands in Sprint 4. The tool exists now so the agent's behavior
and the module contract are exercised end-to-end.
"""

from langchain.tools import tool


@tool
def faq_search(query: str) -> str:
    """Busca en la base de conocimiento (preguntas frecuentes) de la empresa.

    Usa esta herramienta cuando el usuario pregunte por información propia
    del negocio: productos, servicios, precios, horarios, políticas, etc.

    Args:
        query: Conceptos clave de lo que el usuario necesita saber
            (ej: "horario de atención", "política de devoluciones").
    """
    # Sprint 4: embed `query` + pgvector similarity over the FAQ collection.
    return (
        "La base de conocimiento aún no tiene contenido. "
        "Indica al usuario que por ahora no tienes ese dato."
    )


class FaqModule:
    """FAQ knowledge-base search (RAG in Sprint 4)."""

    name = "faq"

    def register_tools(self):
        return [faq_search]


module = FaqModule()
