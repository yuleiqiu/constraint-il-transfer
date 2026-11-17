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
4. **Extract with MinerU on server_4090** (verified 2026-11-17). MinerU 4.1.7
   runs only on the server: `mineru-kit api-server` (tier `advanced`, tmux
   session `mineru-api`) listens on `127.0.0.1:8001` and is not reachable from
   this machine, so copy the PDF over and run the client there. Use `-f zip`;
   the default markdown format inlines figures as base64.
   ```bash
   N=<short-name>; R=/tmp/mineru_$N
   ssh server_4090 "mkdir -p $R/in" && scp papers/$N/paper.pdf server_4090:$R/in/
   ssh server_4090 "cd ~/apps/mineru && source env.sh && \
     ~/.local/bin/uv run --locked mineru-kit parse $R/in/paper.pdf -o $R/out \
       -f zip --remote-url http://127.0.0.1:8001 --tier advanced"
   scp server_4090:$R/out/paper.zip /tmp/mineru_$N.zip
   unzip -q /tmp/mineru_$N.zip -d /tmp/mineru_$N
   ```
   If parsing fails, check `tmux ls` and `~/apps/mineru/state/api.log` on the
   server; restart with `~/apps/mineru/start-mineru-api.sh`.
5. **Keep only text and figures** — the zip holds `markdown.md`, `images/`,
   `middle_json.json` and `structured_content.json`. Keep `markdown.md` as
   `paper.md` plus only the images it references (display equations are
   already LaTeX, so the `*_equation_*.jpg` crops are unused):
   ```bash
   cp /tmp/mineru_$N/markdown.md papers/$N/paper.md
   grep -o '](images/[^)]*)' papers/$N/paper.md | sed 's/](//;s/)//' \
     | xargs -I{} cp /tmp/mineru_$N/{} papers/$N/{}
   ```
6. **Clean up** — `ssh server_4090 "rm -rf $R"; rm -rf /tmp/mineru_$N /tmp/mineru_$N.zip`
7. **Update index** — add a row to the table in `papers/README.md`.

## Naming Convention

Use a short, lowercase, hyphen-separated name that captures the paper's key
idea (e.g. `reguide`, `physics-grounded-diffusion`, `lpb`).
