# Papers — Agent Guide

## Directory Structure

Each paper lives under `papers/<short-name>/`:

```text
<short-name>/
├── paper.pdf       # source PDF, preferably from arXiv
├── paper.md        # MinerU-extracted searchable text
├── images/         # figures extracted alongside paper.md
└── analysis.md     # optional, project-focused reading notes
```

Do not keep parser-specific intermediate artifacts (`*_model.json`,
`*_middle.json`, `*_content_list_v2.json`, layout PDFs) in the library.

## Adding a New Paper

1. **Find arXiv ID** — search arxiv.org for the exact title.
2. **Create directory** — `mkdir -p papers/<short-name>/images`
3. **Download PDF**
   ```bash
   curl -sL -o papers/<short-name>/paper.pdf "https://arxiv.org/pdf/<id>"
   ```
4. **Extract with MinerU** (remote API on server_4090)
   ```bash
   cd ~/apps/mineru
   uv run mineru -p <absolute-path-to-pdf> -o ./output --api-url http://10.42.1.254:8000
   ```
5. **Move results**
   ```bash
   cp ~/apps/mineru/output/paper/hybrid_auto/paper.md  papers/<short-name>/paper.md
   cp ~/apps/mineru/output/paper/hybrid_auto/images/*  papers/<short-name>/images/
   ```
6. **Clean up** — remove the MinerU output for this run:
   ```bash
   rm -rf ~/apps/mineru/output/paper
   ```
7. **Update index** — add a row to the table in `papers/README.md`.

## Naming Convention

Use a short, lowercase, hyphen-separated name that captures the paper's key
idea (e.g. `reguide`, `physics-grounded-diffusion`, `lpb`).
