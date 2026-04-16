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
