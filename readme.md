# Model Runner Hub

A small toolkit for **running local LLM benchmarks** (Python + Tk) and **exploring model scores** in a **static web dashboard**. Benchmark numbers in `web/data/benchmarks.json` are **data-driven**—swap the JSON for your own exports or measured runs.

## Repository layout

| Path | Purpose |
|------|---------|
| `ModelRunner.py` | Tkinter UI: pick a model, run prompts, append results to `llmSummary.csv` (requires GPU/CPU setup per your env). |
| `Visualizer.py` | Reads `llmSummary.csv` and writes histogram PNGs under `histograms/` (created at runtime). |
| `web/` | Benchmark explorer: `index.html`, `styles.css`, `app.js`, and `data/benchmarks.json`. |
| `serve_web.py` | Serves `web/` over HTTP (needed for fetch; avoids file-URL quirks). If the default port is busy, the next free port is used automatically. |

## Web dashboard (recommended)

From the repo root:

```bash
python serve_web.py
```

Open the URL printed in the terminal (default **8765**, or the next free port). Use **Reload** after editing `web/data/benchmarks.json`, or pass a hosted JSON with:

`http://127.0.0.1:<port>/?data=https://example.com/benchmarks.json`

Features include weighted presets (reasoning, coding, SWE, math, balanced), filters, sortable matrix, cohort percentile bars, CSV export, and multi-model radar compare.

## Local inference GUI (optional)

Requires dependencies appropriate for your hardware (CUDA, `llama-cpp-python`, etc.):

```bash
pip install codecarbon huggingface_hub llama_cpp transformers memory_profiler pandas matplotlib numpy memory_profiler
```

Then:

```bash
python ModelRunner.py
```

Results append to `llmSummary.csv`. Run `Visualizer.py` to regenerate histograms from that CSV.

**Note:** `ModelRunner.py` references paths and model IDs you may need to align with your machine; treat it as a starting point.

## Git

- **`.gitignore`** excludes Python caches, virtualenvs, editor junk, and generated artifacts such as `llmSummary.csv` and `histograms/`.
- Initialize or clone as usual, then:

```bash
git add .
git status
git commit -m "Describe your change."
git push origin <branch>
```

## License

Add a `LICENSE` file if you plan to open-source the project.
