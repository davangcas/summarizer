# Summarizer

Pipeline para extraer texto y generar resúmenes Cornell desde archivos fuente.

## Formatos soportados

- `.pdf` (extracción directa o por visión para escaneados).
- `.docx` (extracción directa con librería Python, incluyendo tablas en formato simple).
- `.doc` (no soportado en modo solo Python; convertir antes a `.docx`).

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
- **`SUMMARIZER_SUMMARY_UNIFY_HIERARCHICAL`** (`true` / `false`, default `true`): si `true` y la unificación está activa, cuando el ensamblaje no cabe en un solo prompt se unifica **por lotes** y en varios niveles en lugar de omitirse o forzar un único JSON enorme.
- **`SUMMARIZER_SUMMARY_DUAL_OUTPUT`** (`true` / `false`, default `false`): además del resumen principal, escribe `nombre_full.md` y `nombre_full.pdf` en la misma carpeta de resúmenes con el ensamblaje **antes** de la unificación (solo si difiere del final).
- **`SUMMARIZER_SUMMARY_KEEP_PARTIALS`** (`true` / `false`, default `false`): si `true`, tras éxito **no** se borra `summary_partials/` (checkpoints por ventana y `_combined_windows.md`).
- **`SUMMARIZER_CORNELL_DEPTH`**: `normal` (default) o `high` / `deep` / `alto` — instrucciones extra para más temas y notas más granularizadas.
- **`SUMMARIZER_SUMMARY_MAX_PAGES_PER_WINDOW`**: páginas fuente por ventana (default `3`). Valores menores suelen dar más llamadas al modelo y más temas sueltos.
- **`SUMMARIZER_SUMMARY_PAGE_OVERLAP`**: solape entre ventanas (default `1`).
- **`SUMMARIZER_MAX_PARALLEL_WINDOW_SUMMARIES`**: workers para resumir ventanas en paralelo (default `2`). Con `1` se fuerza orden secuencial.
- **`SUMMARIZER_PROMPT_CONTEXT_RATIO`**, **`SUMMARIZER_PROMPT_CONTEXT_RATIO_START`**, etc.: fracción del contexto del modelo usada como techo del prompt (ver `summarizer/config.py`).

En LM Studio conviene aumentar el **límite de tokens de salida** del modelo para respuestas JSON largas.

## Pruebas

```text
python -m unittest tests.test_cornell_helpers -v
```
