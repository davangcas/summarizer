# Summarizer

Pipeline para extraer texto y generar resúmenes Cornell desde archivos fuente.

## Formatos soportados

- `.pdf` (extracción directa o por visión para escaneados).
- `.docx` (extracción directa con librería Python, incluyendo tablas en formato simple).
- `.doc` (no soportado en modo solo Python; convertir antes a `.docx`).

## Extracción de PDFs y OCR

Hay dos rutas para sacar texto de un PDF:

1. **Extracción directa**: el PDF lleva el texto embebido (cualquier
   PDF generado por Word, LaTeX, navegadores, etc.). Se copia tal
   cual, es perfecta y casi instantánea. La hace `pymupdf` /
   `pymupdf4llm`.
2. **OCR (visión)**: la página es una **imagen** (escaneo, foto,
   captura) sin texto seleccionable. Se renderiza la página a PNG y
   se manda al modelo de visión (vía MarkItDown) para que la
   transcriba. Es lenta y puede tener errores; sólo tiene sentido
   cuando la primera ruta no aporta nada.

El pipeline aplica una estrategia **híbrida por página**: para cada
página se prueba primero la extracción directa, y sólo se manda a OCR
**si esa página específica** no tiene texto seleccionable. Una sola
página escaneada no obliga a OCR-ear las demás 174 páginas digitales
de un libro.

Para detectar si una página tiene texto se combinan dos APIs (la
básica `page.get_text()` que es robusta, y `pymupdf4llm` que produce
mejor Markdown estructurado): se conserva la salida más rica entre las
dos y `has_text=False` queda únicamente cuando ninguna recupera nada.
Esto evita falsos positivos donde `pymupdf4llm` falla en libros con
fuentes especiales y dispara OCR sobre páginas perfectamente
extraíbles.

Variables relacionadas:

- **`SUMMARIZER_HYBRID_OCR`** (`true` / `false`, default `true`): si se
  pone en `false`, se vuelve al comportamiento histórico "todo o nada"
  (OCR del documento completo si falta alguna página).
- Si la visión está desactivada y hay páginas sin texto, se conserva el
  texto de las demás y se inserta el marcador
  `[contenido no extraíble en esta página]` en las páginas afectadas, en
  lugar de descartar el archivo entero.

## Requisito para Word

Para procesar `.docx`, se usa `python-docx` (definido en `requirements.txt`).

## Selección de origen

Al iniciar, puede:

- seleccionar una carpeta (se procesan archivos soportados recursivamente), o
- seleccionar archivos específicos.

Variables de entorno disponibles:

- `SUMMARIZER_FILES_DIRECTORY`: carpeta base a procesar.
- `SUMMARIZER_SOURCE_FILES`: lista de archivos específicos (`;` o `os.pathsep`).
- `SUMMARIZER_PDF_FILES`: alias legacy equivalente a `SUMMARIZER_SOURCE_FILES`.

## Resumen Cornell: detalle, unificación y rendimiento

Libros largos se resumen **por ventanas** (páginas del Markdown completado) y, por defecto, se **unifica** el resultado. Para evitar un resumen demasiado compacto:

- **`SUMMARIZER_SUMMARY_UNIFY_WINDOWS`** (`true` / `false`, default `true`): si `false`, no se llama al modelo para unificar todo el ensamblaje; el PDF/Markdown final conserva todos los temas de cada ventana (más largo, con posible solape entre ventanas).
- **`SUMMARIZER_SUMMARY_UNIFY_MODE`** (`none` / `lmless` / `hierarchical` / `aggressive`, default `hierarchical`): selecciona el estilo de la unificación post-ensamblado. Los 4 modos resuelven el mismo problema (consolidar duplicados entre ventanas) con distinto balance entre detalle y síntesis:

  - `none`: devuelve el ensamblaje (con el dedup Jaccard ya aplicado durante el ensamblado) sin tocar. **Cero llamadas LLM extra**. Es lo más fiel al material original; puede quedar algún duplicado blando que el Jaccard no detectó.
  - `lmless`: aplica una **segunda pasada Jaccard** con umbral relajado (`SEMANTIC_DEDUP_THRESHOLD - 0.1`) sobre el ensamblaje, fusionando títulos, cues y notas. **Cero llamadas LLM extra**. Recomendado para libros largos donde se quiere preservar al máximo el detalle (uso como texto académico).
  - `hierarchical` (default): unificación por lotes con LLM, una llamada por lote (cubre libros que no caben en un solo prompt). Es el comportamiento heredado tras los ajustes de no-compresión.
  - `aggressive`: `hierarchical` + una pasada FINAL single-pass del LLM sobre todo el resultado. Tiende a comprimir mucho; úsalo solo si prefieres una versión más sintética a costa de detalle.

- **`SUMMARIZER_SUMMARY_UNIFY_HIERARCHICAL`** (`true` / `false`, default `true`): solo afecta a los modos `hierarchical` y `aggressive`. Si `false`, en esos modos no se trocea por lotes (se intenta un único prompt y se omite si no cabe).
- **`SUMMARIZER_SUMMARY_FINAL_UNIFY_PASS`** (`true` / `false`, default `false`): **flag legacy** equivalente a forzar `SUMMARY_UNIFY_MODE=aggressive` cuando este último no se pasa explícito. Si `SUMMARY_UNIFY_MODE` se setea, esta variable se ignora.
- **`SUMMARIZER_SUMMARY_DUAL_OUTPUT`** (`true` / `false`, default `false`): además del resumen principal, escribe `nombre_full.md` y `nombre_full.pdf` en la misma carpeta de resúmenes con el ensamblaje **antes** de la unificación (solo si difiere del final).
- **`SUMMARIZER_SUMMARY_KEEP_PARTIALS`** (`true` / `false`, default `false`): si `true`, tras éxito **no** se borra `summary_partials/` (checkpoints por ventana y `_combined_windows.md`).
- **`SUMMARIZER_CORNELL_DEPTH`**: `normal` (default) o `high` / `deep` / `alto` — instrucciones extra para más temas y notas más granularizadas.
- **`SUMMARIZER_SUMMARY_MAX_PAGES_PER_WINDOW`**: páginas fuente por ventana (default `3`). Valores menores suelen dar más llamadas al modelo y más temas sueltos.
- **`SUMMARIZER_SUMMARY_PAGE_OVERLAP`**: solape entre ventanas (default `1`).
- **`SUMMARIZER_MAX_PARALLEL_WINDOW_SUMMARIES`**: workers para resumir ventanas en paralelo (default `2`). Con `1` se fuerza orden secuencial.
- **`SUMMARIZER_PROMPT_CONTEXT_RATIO`**, **`SUMMARIZER_PROMPT_CONTEXT_RATIO_START`**, etc.: fracción del contexto del modelo usada como techo del prompt (ver `summarizer/config.py`).

En LM Studio conviene aumentar el **límite de tokens de salida** del modelo para respuestas JSON largas.

## Fórmulas y matemáticas

Las fórmulas se manejan en dos etapas y se pre-renderizan a imagen antes
de generar el PDF final (el motor por defecto, `markdown_pdf`, no
interpreta LaTeX por sí mismo):

1. **Saneador** (`summarizer/math_sanitize.py`): repara los errores típicos
   de los LLM locales al rellenar campos JSON con LaTeX (p. ej. `\text`
   sin doblar barras genera `\t` como TAB y aparece `<TAB>ext{m/s}^2` en
   la salida parseada). El saneador restaura los caracteres de control a
   sus secuencias de escape originales y reinserta el backslash de
   comandos comunes (`\frac`, `\text`, `\cdot`, símbolos griegos, etc.)
   dentro de spans `$...$` y `$$...$$`.
2. **Pre-render** (`summarizer/math_render.py`): usa `matplotlib.mathtext`
   para convertir cada span matemático a un PNG cacheado por sha256 y
   sustituye el span por `![](ruta.png)`. Pipeline puro Python sin
   dependencias del sistema.

Subset de LaTeX soportado por `matplotlib.mathtext`: comandos básicos
(`\frac`, `\sqrt`, super/subíndices `^` y `_`, símbolos griegos minúsculas
y mayúsculas, `\text`, integrales/sumas/productos, símbolos de
desigualdad, operadores). **No soporta** entornos como `align`,
`gather`, `pmatrix`/`bmatrix`, ni paquetes externos como `\physics` o
`\siunitx`. Para notación que cae fuera del subset, en el prompt se
sugiere al modelo caer a Unicode (`m/s²`, `·`, `π`, `≈`, `≤`, `≥`).

Cuando una expresión queda fuera del subset (típicamente matrices,
`\begin{...}`, etc.) y el render falla, el LaTeX original se preserva en
el PDF como **bloque de código** (fenced code block) en vez de descartarse,
para que al menos quede recuperable y legible en monospace.

Las imágenes generadas se guardan en `{stem}_math/` junto al PDF de
salida y se referencian mediante **rutas relativas URL-encoded** (p. ej.
`![](stem_math/math_abc123.png)`). El `Section` de `markdown_pdf` recibe
ese mismo directorio como `archive root`, de modo que `fitz.Story` puede
embeber el PNG aunque la carpeta destino tenga espacios o caracteres
especiales en el nombre. Versiones anteriores generaban paths absolutos
sin encodear que el parser CommonMark dejaba como texto literal en el
PDF (`![](D:/...png)` visible como referencia rota).

Variables:

- **`SUMMARIZER_MATH_RENDER`** (`true` / `false`, default `true`):
  activa/desactiva el pre-render. Con `false`, los spans `$...$` se
  pasan tal cual al motor de PDF (se ven como texto literal).
- **`SUMMARIZER_MATH_DPI`** (entero, default `200`): resolución usada al
  rasterizar fórmulas a PNG.

## Robustez del resumen

- **Checkpoint parcial**: si el resumen falla a la mitad (timeouts,
  parada manual, etc.) y ya hay ventanas completadas, se escribe
  `summarized_texts/{nombre}.partial.md` con el ensamblaje hasta ese
  punto. Los checkpoints por ventana en `summary_partials/{nombre}/`
  permiten reanudar desde la primera ventana sin completar en el
  siguiente run. Al terminar con éxito, ambos artefactos se limpian.
- **Reintentos LLM**: además del retry sobre timeouts/conexión, se
  reintenta una vez sobre `BadRequestError` no-overflow y se reintenta
  cuando la respuesta no cumple el schema (`ValidationError`). Si el
  servidor no soporta `response_format` estructurado, se cae a un modo
  free-form inyectando el JSON schema en el prompt.
- **Idempotencia en sync de PDFs OCR**: se omite la copia cuando el
  destino existe con el mismo `size` y `mtime`.
- **Dedup semántico de temas** entre ventanas: en vez de comparar
  títulos literales, se usa similitud Jaccard combinada (tokens del
  título normalizado y bigramas del inicio de notas). Cuando se detecta
  un duplicado, se **fusiona** el contenido en lugar de descartarlo.

Variables:

- **`SUMMARIZER_SEMANTIC_DEDUP_THRESHOLD`** (decimal `0..1`, default
  `0.5`): umbral del Jaccard para considerar dos temas como duplicados.
  Subirlo lo hace más estricto (menos fusiones); bajarlo, más agresivo.
- **`SUMMARIZER_ASSEMBLE_DEDUP_GLOBAL`** y
  **`SUMMARIZER_ASSEMBLE_DEDUP_BORDER`**: activan dedup global y de
  bordes inmediatos (defaults conservadores, ver `summarizer/config.py`).

## Heurística de outline

- **`SUMMARIZER_BOOK_OUTLINE_HEURISTIC`** (`true` / `false`, default
  `true`): activa la detección de índice del libro al inicio del texto
  completado para alinear los títulos de ventana.
- **`SUMMARIZER_BOOK_OUTLINE_SCAN_CHARS`** (entero, default `64000`):
  ventana inicial escaneada. Si no se halla índice, también se escanea
  una ventana secundaria entre las páginas 5 y 40 (útil cuando el
  prólogo es muy extenso).

## Caches de arranque

- `.cache/tokenizer_id.txt`: id del primer tokenizer que arrancó con
  éxito, para no repetir la cadena de fallbacks de Hugging Face en cada
  ejecución. Se regenera automáticamente si se borra.

## Pruebas

```text
python -m unittest discover tests -v
```
