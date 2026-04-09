"""Esquemas Pydantic para respuestas del modelo."""

from pydantic import BaseModel, ConfigDict, Field


class OCRPageOutput(BaseModel):
    """Salida estricta para extracción de una página escaneada."""

    model_config = ConfigDict(extra="forbid")

    markdown_text: str = Field(
        description=(
            "Transcripción completa del texto visible en la imagen, en Markdown. "
            "Orden de lectura natural (columnas y bloques como en la página). "
            "Usa #/##/### si hay títulos claros, listas con - o 1., tablas en Markdown si se distinguen. "
            "Conserva fórmulas y símbolos lo más fielmente posible (LaTeX entre $ si aplica). "
            "Mantén el idioma original. Si algo es ilegible, marca [ilegible]. "
            "Sin introducción, sin conclusiones, sin 'aquí está el texto': solo el contenido de la página."
        )
    )


class CornellTopicBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(
        description=(
            "Nombre del tema alineado con la estructura del documento: capítulo, sección, subtítulo o temática "
            "cuando el texto las muestre (#/##/###, numeración, títulos destacados). Sin prefijos meta tipo 'Tema:'. "
            "No uses números de página, rangos ('páginas X–Y'), ni 'fragmento' en el título."
        )
    )
    cues: list[str] = Field(
        description=(
            "Lista de pistas tipo Cornell: palabras clave o preguntas cortas de repaso (una idea por elemento). "
            "Evita frases largas; 3 a 10 ítems según densidad del tema."
        )
    )
    notes: str = Field(
        description=(
            "Síntesis académica densa: definiciones, hipótesis, procedimientos, fórmulas (en texto o LaTeX ligero), "
            "relaciones causa-efecto y condiciones límite del modelo o experimento cuando el original las mencione. "
            "No pegues párrafos literales extensos. "
            "Solo si el fragmento termina antes de cerrar el tema y el origen no lo ciere aquí, indica al final: "
            "(continúa en el siguiente fragmento)."
        )
    )
    topic_summary: str = Field(
        description=(
            "Cierre del tema: 2 a 5 frases que integren la idea central, supuestos clave y utilidad en el contexto del texto."
        )
    )


class CornellSummaryStructured(BaseModel):
    """Resumen por temas estilo Cornell; coincide con el esquema JSON enviado al modelo."""

    model_config = ConfigDict(extra="forbid")

    topics: list[CornellTopicBlock] = Field(
        description=(
            "Lista ordenada según la secuencia del documento en este fragmento: cada elemento corresponde a una "
            "temática, sección o subtítulo explícito o implícito del texto (no al orden artificial de ## Página N). "
            "Unifica en un solo topic lo que el autor trata como la misma sección repartida en varias páginas. "
            "Si el texto es muy breve, un solo tema puede bastar. "
            "Todo en español claro salvo términos técnicos habituales en el original."
        )
    )


class WindowSummaryCheckpoint(BaseModel):
    """Una ventana ya resumida (archivo en summary_partials)."""

    model_config = ConfigDict(extra="forbid")

    version: int = 1
    start_p: int
    end_p: int
    body_sha256: str
    structured: CornellSummaryStructured
