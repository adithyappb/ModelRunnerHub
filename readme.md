# Model Runner Hub

Local **LLM benchmarking** (Python + Tk) and a **static web dashboard** for exploring benchmark-style metrics. CSV results use a **v2 schema**; the web UI reads **`web/data/benchmarks.json`** (swap or host your own JSON).

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS / Linux

pip install -r requirements.txt
python ModelRunner.py
```

**Recommended default model:** `Qwen/Qwen2.5-0.5B-Instruct` (Hugging Face / `transformers` path). It is preselected in the UI and works **without** `llama-cpp-python`. GGUF models are optional and need a separate install (see below).

For higher Hugging Face download limits, set `HF_TOKEN` in the environment before running.

## Repository layout

| Path | Purpose |
|------|---------|
| `benchmark_utils.py` | CSV v2 schema, stats helpers, migration from legacy `llmSummary.csv` headers. |
| `ModelRunner.py` | GUI: model + prompts, background run, append row to `llmSummary.csv`. |
| `Visualizer.py` | Plots from `llmSummary.csv` → `histograms/` (and versioned snapshots when used). |
| `web/` | Dashboard: `index.html`, `styles.css`, `app.js`, `data/benchmarks.json`. |
| `serve_web.py` | HTTP server for `web/` (avoids `file://` fetch limits). Default port **8765**, or next free in a short range. |
| `requirements.txt` | Core stack; no GGUF C++ build. |
| `requirements-gguf.txt` | Notes + optional `llama-cpp-python` wheels. |
| `install_llama_windows.ps1` | Helper to try Windows-friendly `llama-cpp-python` installs. |

## Web dashboard

From the repo root:

```bash
python serve_web.py
```

Open the printed URL (default **8765**, or override with `PORT` / `python serve_web.py <port>`). Optional query: `?data=https://…/benchmarks.json` to load remote JSON.

## Local inference GUI

1. **Hugging Face models** — install `requirements.txt`. First run downloads weights into the Hugging Face cache (not in this repo).

2. **GGUF models** — `llama-cpp-python` is **not** in `requirements.txt` (Windows often fails to compile from source). Try wheels in order:

   ```bash
   pip install llama-cpp-python --prefer-binary
   pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
   ```

   Or run `install_llama_windows.ps1`. If install still fails, stay on HF-only models.

Each finished run **appends one row** to `llmSummary.csv` (v2 columns):

| Column | Meaning |
|--------|---------|
| `run_id`, `timestamp` | UUID and UTC time |
| `model_name`, `backend` | HF id; `transformers` or `gguf_llama_cpp` |
| `num_parameters` | Parameter count when known (`-1` if unknown) |
| `n_queries` | Number of prompts |
| `latency_*`, `memory_*` | Wall-clock latency and RSS samples (`psutil` if installed) |
| `tokens_per_sec_mean` | Estimated output throughput |

Legacy v1 CSV rows are migrated in place on first append (`backend=legacy_import` where inferred).

**Plots**

```bash
python Visualizer.py
```

`ModelRunner.py` also refreshes plots after a successful run. Set `MPL_SHOW=1` to open figures interactively; default is save-only under `histograms/`.

Edit model lists at the top of `ModelRunner.py` if you change repos or GGUF filenames.

## Environment variables

| Variable | Role |
|----------|------|
| `HF_TOKEN` | Authenticated Hugging Face Hub access (higher rate limits, fewer download issues). |
| `LLAMA_N_GPU_LAYERS` | GGUF: `0` (CPU, default), `-1` or `N` for GPU offload when your build supports it. |
| `MPL_SHOW` | Set to `1` / `true` to show Matplotlib windows from `Visualizer.py`. |
| `PORT` | Used by `serve_web.py` if no CLI port is given. |
| `CODECARBON_LOG_LEVEL` | App defaults this to reduce console noise if CodeCarbon is installed. |

The GUI sets `HF_HUB_DISABLE_SYMLINKS_WARNING` on Windows to quiet symlink cache notices (cache still works).

## Git and ignored files

See **`.gitignore`**: Python caches, virtualenvs, editor junk, and **regenerated / local run outputs** such as `llmSummary.csv`, `histograms/`, and **`emissions.csv`** (CodeCarbon log if present from older runs or external tools).

Track `web/data/benchmarks.json` if it is your checked-in sample; use a branch or fork-specific file for private numbers if needed.

## License

Add a `LICENSE` file if you open-source the project.
