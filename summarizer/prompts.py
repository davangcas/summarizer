"""Plantillas de prompt para OCR y resúmenes."""

OCR_PROMPT = """Rol: transcriptor OCR de documentos académicos.
Tarea: rellena únicamente el campo del esquema con el texto de la imagen.

Reglas:
- Transcribe todo lo legible; conserva jerarquía (títulos, listas, enumeraciones) en Markdown.
- No añadas explicaciones, saludos, comentarios sobre la imagen ni resúmenes.
- No inventes texto donde no haya; usa [ilegible] en huecos.
- Idioma: el mismo que aparece en la página."""

SUMMARY_CORNELL_USER_PREFIX = """Rol: tutor de estudio y síntesis para textos académicos en español.
Salida: cumple EXACTAMENTE el esquema JSON indicado (solo claves permitidas; lista `topics` con objetos title, cues, notes, topic_summary).

Instrucciones:
1. Tras el separador --- está el documento fuente. Particiona y resume por la estructura discursiva del autor: capítulos, secciones, subtítulos, apartados numerados o temáticas claras (incluidas en encabezados Markdown del propio texto o en negritas/títulos implícitos). Los marcadores `## Página N` solo delimitan el contenido disponible, no son títulos de salida.
2. En `topics`, ordena los elementos en el mismo orden en que aparecen esas secciones/temáticas en el fragmento. Si la misma sección continúa en varias páginas del bloque, unifica en un único topic.
3. Por tema: `title` debe reflejar esa sección o temática (adaptado o acortado si hace falta); nunca pongas en `title` números de página, rangos tipo "páginas X–Y", "Pág.", ni metadatos de fragmento. cues como repaso; notes densa (definiciones, pasos, supuestos, fórmulas cuando existan); topic_summary cierra la idea y utilidad.
4. Prioriza rigor: hechos, definiciones, datos y razonamiento del texto; no inventes citas, referencias ni detalles inexistentes en el material.
5. Si el documento mezcla idiomas, sintetiza en español salvo nombres propios o términos técnicos estándar.
6. No incluyas texto fuera del JSON (sin markdown envolvente, sin comentarios).
7. No crees temas "meta" que solo repitan el título del libro o del documento sin contenido sustantivo (definiciones, hechos o argumentos del texto). Si un bloque no aporta más que una frase genérica, no lo incluyas como topic separado.
8. Evita duplicar el mismo capítulo o temática con títulos casi idénticos (p. ej. título del libro y "título del libro: origen"); unifica en un solo topic cuando sea el mismo asunto."""

SUMMARY_CHUNK_WRAPPER = """Contexto: este bloque es el fragmento {part} de {total} de un documento largo (no tienes el resto).

Qué hacer:
- Extrae solo los temas que se apoyen en el contenido de ESTE fragmento.
- Si un tema empieza aquí y seguramente sigue después, en `notes` indica al final: (continúa en el siguiente fragmento).
- No inventes contenido de otras partes del documento.

---
{body}"""

SUMMARY_WINDOW_WRAPPER = """Contexto: el bloque siguiente contiene el texto de las páginas {start}–{end} del PDF original, separadas por marcadores `## Página N`. Eso es solo delimitación de contexto (no repitas ni uses esos rangos o números de página en los campos `title` del JSON).

Qué hacer:
- Extrae `topics` siguiendo títulos, subtítulos, secciones y temáticas del propio contenido (no una lista por página).
- Si la misma sección continúa en varias páginas dentro de este bloque, unifica en un solo `topic` con un único `title`.
- Resalta definiciones, fórmulas, procedimientos, hipótesis y límites que aparezcan en el texto (sin inventar).
- Si el texto está en otro idioma, sintetiza en español salvo nombres propios y términos técnicos habituales.
- Evita duplicar el mismo tema cerrado solo porque cambia el marcador `## Página`; en solape con otra ventana, prioriza información nueva sin repetir el mismo `title` si el contenido es redundante.

---
{body}"""

BOOK_CHAPTER_OUTLINE_PREFIX = """

Lista de referencia de capítulos o partes del libro (orden del original). Cuando el fragmento cubra alguno, alinea el `title` del topic con el nombre más cercano o una variante clara; no inventes capítulos que no aparezcan en el texto del fragmento.
Capítulos:
{outline_lines}
"""

UNIFY_ASSEMBLED_CORNELL_PROMPT = """Rol: editor académico de apuntes estilo Cornell.

Recibes el Markdown completo de un resumen ya generado: incluye `## Índice` con enlaces y bloques `###` con Pistas, Notas y Resumen del tema. Puede haber temas duplicados o solapados, títulos redundantes, o secciones triviales/meta sin sustancia.

Objetivo: devuelve UN ÚNICO JSON con el esquema habitual: lista `topics` donde cada elemento tiene title, cues, notes, topic_summary.

Reglas:
1. Fusiona temas que traten el mismo asunto (títulos similares o contenido redundante); integra cues y notes sin repetir ideas.
2. Elimina temas triviales: solo dicen que el autor es conocido, repiten el título del libro sin contenido nuevo, o son frases genéricas sin datos del texto.
3. Ordena los temas en secuencia lógica coherente con el documento (orden de exposición del autor, no orden aleatorio).
4. Mantén rigor: no inventes hechos; síntesis densa en notes; topic_summary cierra la utilidad del tema.
5. Salida: solo JSON, en español, sin markdown fuera del JSON.

---
{combined}
"""

UNIFY_SUMMARIES_PROMPT = """Rol: editor de apuntes. Recibes varios resúmenes parciales del MISMO documento (Markdown) tras ---.

Objetivo: producir un único JSON del esquema con una lista `topics` coherente para todo el documento.

Reglas:
1. Fusiona temas duplicados o muy similares; unifica cues y notes sin repetir ideas.
2. Ordena los temas en secuencia lógica (orden del libro o del razonamiento, no orden de fragmentos).
3. Mantén el estilo Cornell (title, cues, notes, topic_summary) en cada tema.
4. Elimina contradicciones; prioriza consistencia.
5. Salida: solo el JSON del esquema, en español.

---
{combined}"""
