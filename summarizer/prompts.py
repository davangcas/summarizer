"""Plantillas de prompt para OCR y resúmenes."""

OCR_PROMPT = """Rol: transcriptor OCR de documentos académicos.
Tarea: rellena únicamente el campo del esquema con el texto de la imagen.

Reglas:
- Transcribe todo lo legible; conserva jerarquía (títulos, listas, enumeraciones) en Markdown.
- No añadas explicaciones, saludos, comentarios sobre la imagen ni resúmenes.
- No inventes texto donde no haya; usa [ilegible] en huecos.
- Idioma: el mismo que aparece en la página."""

SUMMARY_CORNELL_USER_PREFIX = """Rol: tutor de estudio y redactor de apuntes académicos en español (nivel universitario).
Salida: cumple EXACTAMENTE el esquema JSON indicado (solo claves permitidas; lista `topics` con objetos title, cues, notes, topic_summary).

Instrucciones:
1. Tras el separador --- está el documento fuente. Particiona y resume por la estructura discursiva del autor: capítulos, secciones, subtítulos, apartados numerados o temáticas claras (incluidas en encabezados Markdown del propio texto o en negritas/títulos implícitos). Los marcadores `## Página N` solo delimitan el contenido disponible, no son títulos de salida.
2. En `topics`, ordena los elementos en el mismo orden en que aparecen esas secciones/temáticas en el fragmento. Si la misma sección continúa en varias páginas del bloque, unifica en un único topic.
3. Por tema, respeta este reparto (no acortes todo en `topic_summary`):
   - `title`: refleja la sección o temática; nunca números de página, rangos "páginas X–Y", "Pág." ni metadatos de fragmento.
   - `cues`: solo pistas breves de repaso (palabras clave o preguntas cortas); 3 a 10 ítems según densidad.
   - `notes`: AQUÍ va el contenido de estudio principal. Debe ser sustancial y utilizable como material de repaso serio: varios párrafos cortos (puedes separarlos con líneas en blanco dentro del string JSON) y/o líneas que empiecen con "- " para listar puntos. Incluye, cuando el original las tenga: definiciones formales o informales, hipótesis, notación, fórmulas (LaTeX ligero con $...$ si aplica), pasos de procedimientos o algoritmos, condiciones de aplicación, casos particulares, relaciones entre conceptos, advertencias o límites del modelo. Parafrasea y condensa sin vaciar el contenido: evita una sola frase telegráfica si el texto ofrece más matices.
   - `topic_summary`: cierre breve (2 a 5 frases) que integre idea central y utilidad; no dupliques todo lo ya dicho en `notes`, pero puede remarcar el hilo conductor.
4. Prioriza rigor: hechos, definiciones, datos y razonamiento del texto; no inventes citas, referencias ni detalles inexistentes en el material.
5. Si el documento mezcla idiomas, sintetiza en español salvo nombres propios o términos técnicos estándar.
6. No incluyas texto fuera del JSON (sin markdown envolvente, sin comentarios).
7. No crees temas "meta" que solo repitan el título del libro o del documento sin contenido sustantivo (definiciones, hechos o argumentos del texto). Si un bloque no aporta más que una frase genérica, no lo incluyas como topic separado.
8. Evita duplicar el mismo capítulo o temática con títulos casi idénticos (p. ej. título del libro y "título del libro: origen"); unifica en un solo topic cuando sea el mismo asunto."""

SUMMARY_CHUNK_WRAPPER = """Contexto: este bloque es el fragmento {part} de {total} de un documento largo (no tienes el resto).

Qué hacer:
- Extrae solo los temas que se apoyen en el contenido de ESTE fragmento.
- En cada tema, `notes` debe ser el campo más extenso y detallado (material de estudio), no un resumen de una línea.
- Si un tema empieza aquí y seguramente sigue después, en `notes` indica al final: (continúa en el siguiente fragmento).
- No inventes contenido de otras partes del documento.

---
{body}"""

SUMMARY_WINDOW_WRAPPER = """Contexto: el bloque siguiente contiene el texto de las páginas {start}–{end} del PDF original, separadas por marcadores `## Página N`. Eso es solo delimitación de contexto (no repitas ni uses esos rangos o números de página en los campos `title` del JSON).

Qué hacer:
- Extrae `topics` siguiendo títulos, subtítulos, secciones y temáticas del propio contenido (no una lista por página).
- Si la misma sección continúa en varias páginas dentro de este bloque, unifica en un solo `topic` con un único `title` y acumula el detalle en `notes` (un solo bloque de notas rico, no varias versiones sucesivas telegráficas).
- En cada tema, desarrolla `notes` con el mismo nivel de detalle que exigirías en apuntes para un examen: definiciones, fórmulas, pasos y matices presentes en estas páginas (sin inventar).
- Si el texto está en otro idioma, sintetiza en español salvo nombres propios y términos técnicos habituales.
- Evita duplicar el mismo tema cerrado solo porque cambia el marcador `## Página`; en solape con otra ventana, prioriza información nueva sin repetir el mismo `title` si el contenido es redundante.

---
{body}"""

BOOK_CHAPTER_OUTLINE_PREFIX = """

Lista de referencia de capítulos o partes del libro (orden del original). Cuando el fragmento cubra alguno, alinea el `title` del topic con el nombre más cercano o una variante clara; no inventes capítulos que no aparezcan en el texto del fragmento.
Capítulos:
{outline_lines}
"""

UNIFY_ASSEMBLED_CORNELL_PROMPT = """Rol: editor académico de apuntes estilo Cornell (documento para estudio serio).

Recibes el Markdown completo de un resumen ya generado: incluye `## Índice` con enlaces y bloques `###` con Pistas, Notas y Resumen del tema. Puede haber temas duplicados o solapados, títulos redundantes, o secciones triviales/meta sin sustancia.

Objetivo: devuelve UN ÚNICO JSON con el esquema habitual: lista `topics` donde cada elemento tiene title, cues, notes, topic_summary.

Reglas:
1. Fusiona temas que traten el mismo asunto (títulos similares o contenido redundante). Al fusionar, une el contenido de las notas en un solo texto `notes` más completo: conserva definiciones, fórmulas, pasos y matices de todas las entradas fusionadas; elimina solo repeticiones literales obvias. No sustituyas párrafos de notas por una frase más corta si eso pierde información útil.
2. Si algún bloque tiene `notes` demasiado escueto (una o dos frases genéricas) pero las `Pistas` o el `Resumen del tema` sugieren más sustancia, reescribe `notes` para que sea el cuerpo de estudio principal (sin inventar datos que no estén en el material recibido).
3. Elimina temas triviales: solo dicen que el autor es conocido, repiten el título del libro sin contenido nuevo, o son frases genéricas sin datos del texto.
4. Ordena los temas en secuencia lógica coherente con el documento (orden de exposición del autor, no orden aleatorio).
5. Mantén rigor: no inventes hechos. `notes` debe seguir siendo el campo más detallado de cada tema; `topic_summary` solo cierra en pocas frases.
6. Salida: solo JSON, en español, sin markdown fuera del JSON.

---
{combined}
"""

UNIFY_SUMMARIES_PROMPT = """Rol: editor de apuntes académicos. Recibes varios resúmenes parciales del MISMO documento (Markdown) tras ---.

Objetivo: producir un único JSON del esquema con una lista `topics` coherente para todo el documento.

Reglas:
1. Fusiona temas duplicados o muy similares. Integra `notes` en un solo texto enriquecido (definiciones, fórmulas, pasos): quita redundancia pero no comprimas hasta dejar notas telegráficas.
2. Ordena los temas en secuencia lógica (orden del libro o del razonamiento, no orden de fragmentos).
3. Mantén el estilo Cornell (title, cues, notes, topic_summary): `notes` sigue siendo el bloque más extenso y detallado por tema.
4. Elimina contradicciones; prioriza consistencia.
5. Salida: solo el JSON del esquema, en español.

---
{combined}"""
