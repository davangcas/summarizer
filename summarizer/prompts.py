"""Plantillas de prompt para OCR y resúmenes Cornell.

Separamos el contenido en (a) un prompt de **sistema** estable que se envía
una vez por llamada (y que LM Studio/llama.cpp puede cachear como prefijo
entre slots), y (b) prompts de **usuario** específicos por tarea (ventana,
fragmento, unificación). Esto reduce significativamente los tokens
repetidos cuando se resumen libros largos por ventanas.
"""

OCR_PROMPT = """Rol: transcriptor OCR de documentos académicos.
Tarea: rellena únicamente el campo del esquema con el texto de la imagen.

Reglas:
- Transcribe todo lo legible; conserva jerarquía (títulos, listas, enumeraciones) en Markdown.
- No añadas explicaciones, saludos, comentarios sobre la imagen ni resúmenes.
- No inventes texto donde no haya; usa [ilegible] en huecos.
- Idioma: el mismo que aparece en la página."""


SUMMARY_CORNELL_SYSTEM_PROMPT = """Rol: tutor de estudio y redactor de apuntes académicos en español (nivel universitario) estilo Cornell.

Esquema y campos por tema:
- `title`: refleja la sección o temática del autor. NO uses números de página, rangos ("páginas X–Y"), "Pág.", "Fragmento", ni metadatos.
- `cues`: 3 a 10 palabras clave o preguntas cortas de repaso (una idea por elemento).
- `notes`: cuerpo de estudio principal y el campo MÁS EXTENSO del tema. Varios párrafos cortos (separados por líneas en blanco dentro del string JSON) y/o viñetas con "- ". Incluye, cuando el original lo tenga: definiciones, hipótesis, notación, fórmulas, pasos de procedimientos/algoritmos, condiciones de aplicación, casos particulares, relaciones entre conceptos y advertencias. Parafrasea y condensa sin vaciar el contenido.
- `topic_summary`: cierre de 2 a 5 frases con la idea central y utilidad. No dupliques todo lo de `notes`; remarca el hilo conductor.

Estructura del resumen:
- Particiona por la estructura discursiva del autor (capítulos, secciones, subtítulos, apartados numerados o temáticas claras). Los marcadores `## Página N` son sólo delimitadores de contexto: NO son títulos de salida.
- Si la misma sección se reparte en varias páginas o bloques, unifícala en un único `topic`.
- Orden de `topics`: el mismo orden de aparición en el fragmento.
- No crees temas "meta" (que solo repitan el título del libro sin contenido sustantivo).
- Evita duplicar la misma temática con títulos casi idénticos: unifica.

Idioma y rigor:
- Sintetiza en español. Mantén nombres propios y términos técnicos estándar.
- No inventes citas, referencias ni datos ausentes en el material.

Fórmulas y matemáticas:
- Usa LaTeX entre `$...$` (inline) o `$$...$$` (bloque): `$\\\\frac{a}{b}$`, `$\\\\sqrt{x^2}$`, `$10\\\\,\\\\text{m/s}^2$`.
- CRÍTICO: en JSON los backslashes de LaTeX deben ir DOBLES (`\\\\text`, `\\\\frac`). Si no puedes garantizar el doble escape, prefiere Unicode (`m/s²`, `·`, `π`, `≈`, `≤`, `≥`, super/subíndices Unicode).
- Mantén cada fórmula en una sola línea dentro de su span `$...$` (sin saltos de línea).

Salida: SOLO el JSON del esquema, en español, sin Markdown envolvente ni comentarios."""

CORNELL_DEPTH_HIGH_SUFFIX = """Perfil de profundidad alta:
- Prefiere varios `topics` (uno por subapartado o argumento) sobre un tema monolítico.
- Granularidad fina en `notes`: cada definición, ejemplo o caso en su propia viñeta/párrafo; refleja la granularidad del original.
- No reduzcas varias definiciones o ejemplos distintos a una sola frase si cada uno aporta matices."""

SUMMARY_CHUNK_WRAPPER = """Contexto: fragmento {part} de {total} de un documento largo (no tienes el resto).

Qué hacer:
- Extrae sólo los temas apoyados en ESTE fragmento.
- En cada tema, `notes` es el campo más extenso; no una frase telegráfica.
- Si un tema empieza aquí y seguramente sigue después, indica al final de `notes`: (continúa en el siguiente fragmento).
- No inventes contenido de otras partes del documento.

---
{body}"""

SUMMARY_WINDOW_WRAPPER = """Contexto: secciones {start}–{end} del documento, delimitadas por marcadores `## Página N`. Esos marcadores son sólo contexto (NO los uses como `title` ni cites rangos de página).

Qué hacer:
- Extrae `topics` siguiendo títulos, subtítulos, secciones y temáticas del contenido.
- Si la misma sección continúa en varios bloques de este fragmento, unifica en un solo `topic` con `title` único y `notes` consolidado.
- En solape con otra ventana, prioriza información nueva; no repitas el mismo `title` si el contenido es redundante.

---
{body}"""

BOOK_CHAPTER_OUTLINE_PREFIX = """

Lista de referencia de capítulos o partes del libro (orden del original). Cuando el fragmento cubra alguno, alinea el `title` del topic con el nombre más cercano o una variante clara; no inventes capítulos que no aparezcan en el texto del fragmento.
Capítulos:
{outline_lines}
"""

UNIFY_ASSEMBLED_CORNELL_PROMPT = """Tarea: fusión final de un resumen Cornell ya generado en Markdown.

Recibes tras --- el Markdown completo (con `## Índice` y bloques `### tema` que contienen Pistas, Notas y Resumen del tema). Puede haber temas duplicados, solapados o triviales.

Devuelve UN JSON del esquema con la lista `topics` consolidada.

Reglas:
1. Fusiona temas con el mismo asunto. Al fusionar, ENRIQUECE `notes` uniendo definiciones, fórmulas, pasos y matices; elimina sólo repeticiones literales obvias. NO sustituyas párrafos de notas por una frase corta si eso pierde información.
2. Si un `notes` es escueto pero las `Pistas` o `Resumen` sugieren más sustancia, reescribe `notes` con ese contenido (sin inventar).
3. Elimina temas triviales: frases genéricas, "el autor es conocido", repeticiones del título del libro sin contenido.
4. Orden lógico coherente con la exposición del autor.
5. Mantén rigor: sin invenciones. `notes` sigue siendo el campo más detallado.

---
{combined}
"""

UNIFY_ASSEMBLED_CORNELL_BATCH_PROMPT = """Tarea: lote parcial de fusión Cornell.

Este es el LOTE {part} de {total} del resumen (bloques `### tema`). Otros lotes existen antes/después; NO inventes contenido externo.

Devuelve UN JSON del esquema cubriendo SÓLO este lote.

Reglas:
1. Fusiona dentro de este lote temas con el mismo asunto. Concatena y enriquece `notes` (definiciones, fórmulas, pasos); elimina sólo repeticiones literales obvias.
2. No descartes temas distintos por ahorrar espacio.
3. Orden de aparición en este lote.

---
{combined}
"""

UNIFY_SUMMARIES_PROMPT = """Tarea: unifica varios resúmenes parciales del MISMO documento (Markdown, tras ---).

Devuelve UN JSON del esquema con la lista `topics` coherente para todo el documento.

Reglas:
1. Fusiona temas duplicados o similares. Integra `notes` enriqueciéndolas; no comprimas a líneas telegráficas.
2. Orden lógico (orden del libro o del razonamiento, no orden de fragmentos).
3. Estilo Cornell: `notes` es el bloque más extenso y detallado.
4. Elimina contradicciones; prioriza consistencia.

---
{combined}"""
