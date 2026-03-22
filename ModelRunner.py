"""
Tkinter UI for local LLM benchmark runs. Appends structured rows to llmSummary.csv
and optionally opens Visualizer plots.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
import traceback
import warnings
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk

import pandas as pd

# Quieter console: CodeCarbon is optional for UX; HF symlink warning off on Windows.
os.environ.setdefault("CODECARBON_LOG_LEVEL", "error")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
logging.getLogger("codecarbon").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

from benchmark_utils import (
    CSV_COLUMNS_V2,
    append_result_row_v2,
    latency_stats,
    memory_stats,
    new_run_id,
    utc_now_iso,
)

try:
    import psutil
except ImportError:
    psutil = None


def _norm_key(s: str) -> str:
    return s.strip().lower()


# --- GGUF models: display label -> (hf_repo_id, filename) ---
GGUF_MODEL_FILES: dict[str, tuple[str, str]] = {
    "thebloke/tinyllama-1.1b-chat-gguf": (
        "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    ),
    "thebloke/llama-2-13b-chat-ggml": (
        "TheBloke/Llama-2-13B-chat-GGML",
        "llama-2-13b-chat.ggmlv3.q5_1.bin",
    ),
    "thebloke/open-llama-7b-open-instruct-ggml": (
        "TheBloke/open-llama-7b-open-instruct-GGML",
        "open-llama-7B-open-instruct.ggmlv3.q4_0.bin",
    ),
    "thebloke/koala-13b-ggml": ("TheBloke/koala-13B-GGML", "koala-13B.ggmlv3.q4_0.bin"),
}

# --- Hugging Face causal LM models (transformers generate path; no llama-cpp-python) ---
HF_CAUSAL_MODELS: frozenset[str] = frozenset(
    {
        # Small / fast — good for llmSummary.csv without heavy VRAM
        "Qwen/Qwen2.5-0.5B-Instruct",
        "HuggingFaceTB/SmolLM2-360M-Instruct",
        "microsoft/Phi-3-mini-4k-instruct",
        "tiiuae/falcon-7b-instruct",
        # Larger
        "codellama/CodeLlama-13b-hf",
        "thebloke/stable-vicuna-13b-hf",
        "tiiuae/falcon-40b-instruct",
    }
)

HF_BY_NORM: dict[str, str] = {_norm_key(m): m for m in HF_CAUSAL_MODELS}

# Default selection (works on Windows without llama-cpp-python; small download).
RECOMMENDED_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# "light" = small / 7B or less; "heavy" = 13B+ / 40B. Skip-big-models filters to "light" only.
MODEL_TIER: dict[str, str] = {
    "thebloke/tinyllama-1.1b-chat-gguf": "light",
    "thebloke/open-llama-7b-open-instruct-ggml": "light",
    "qwen/qwen2.5-0.5b-instruct": "light",
    "huggingfacetb/smolm2-360m-instruct": "light",
    "microsoft/phi-3-mini-4k-instruct": "light",
    "tiiuae/falcon-7b-instruct": "light",
    "thebloke/llama-2-13b-chat-ggml": "heavy",
    "thebloke/koala-13b-ggml": "heavy",
    "codellama/CodeLlama-13b-hf": "heavy",
    "thebloke/stable-vicuna-13b-hf": "heavy",
    "tiiuae/falcon-40b-instruct": "heavy",
}

# Shown if llama_cpp import fails (Windows wheels first).
LLAMA_CPP_INSTALL_HINT = (
    "GGUF models need the llama-cpp-python package.\n\n"
    "On Windows, install a prebuilt wheel (try in order):\n"
    "  1) pip install llama-cpp-python --prefer-binary\n"
    "  2) pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu\n\n"
    "Or skip GGUF entirely: uncheck nothing needed — pick a Hugging Face model above "
    "(e.g. Qwen2.5-0.5B, SmolLM2-360M, Phi-3-mini, Falcon-7B) and run without llama-cpp-python.\n"
    "See requirements-gguf.txt for details."
)

# Short, high-signal prompts: reasoning, code (HumanEval-style), math, instruction-following, knowledge, structured output.
DEFAULT_QUESTIONS = "\n".join(
    [
        "Math (show steps): A shirt is on sale for $24 after a 25% discount. What was the original price? Give the equation and answer.",
        "Code (Python only): def has_duplicate(nums: list[int]) -> bool: return True if any value appears more than once, else False. Use O(n) time with a set. No imports.",
        "Reasoning: All cats are mammals. Fluffy is not a mammal. Can Fluffy be a cat? Answer Yes or No, then one sentence.",
        "Instruction: Reply with exactly 3 lines: line1='TASK:OK', line2='MODE:lite', line3='END'. No extra text.",
        "Knowledge: In at most 3 sentences, explain the difference between TCP and UDP for real-time voice chat.",
        "JSON: Output ONLY a single JSON object with keys 'even' and 'odd' whose values are the counts of even and odd digits in 202514. No markdown code fences.",
    ]
)


def _llama_n_params(llm) -> int:
    try:
        n = getattr(llm, "n_params", None)
        if callable(n):
            n = n()
        if n is not None:
            return int(n)
    except Exception:
        pass
    return -1


def _rss_mb() -> float | None:
    if psutil is None:
        return None
    try:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def _put_recommended_first(ids: list[str]) -> list[str]:
    if RECOMMENDED_MODEL_ID in ids:
        rest = [x for x in ids if x != RECOMMENDED_MODEL_ID]
        return [RECOMMENDED_MODEL_ID] + rest
    return ids


def format_run_error(exc: BaseException) -> str:
    """Human-readable message; avoids useless bare 'type' style dialogs."""
    if isinstance(exc, ImportError):
        msg = str(exc).strip()
        low = msg.lower()
        if "llama" in low or "llama_cpp" in low:
            return LLAMA_CPP_INSTALL_HINT
        return f"Missing package (ImportError):\n{msg}"
    if isinstance(exc, (OSError, RuntimeError)):
        return f"{type(exc).__name__}\n{exc}"
    name = type(exc).__name__
    msg = str(exc).strip()
    if not msg or msg == repr(exc):
        msg = traceback.format_exception_only(type(exc), exc)[-1].strip()
    if len(msg) < 3 or msg.lower() == "type":
        alt = traceback.format_exception_only(type(exc), exc)[-1].strip()
        if len(alt) >= len(msg):
            msg = alt
    low = msg.lower()
    # Hugging Face / HTTP
    if "huggingface" in name.lower() or "http" in name.lower() or "connection" in low:
        return f"{name}\n{msg}\n\nCheck your network, VPN, and Hugging Face status. Set HF_TOKEN for higher rate limits."
    if "flash_attn" in low or "flash attention" in low:
        return f"{name}\n{msg}\n\nThis app uses eager attention; try updating transformers or restart the run."
    return f"{name}\n{msg}"


class ModelRunner:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Model Runner — Local benchmarks")
        self.root.minsize(520, 420)

        self._busy = False
        self.results_df = pd.DataFrame(columns=CSV_COLUMNS_V2)
        self._skip_big_var = tk.BooleanVar(value=True)

        self._build_ui()

    @staticmethod
    def _all_model_ids() -> list[str]:
        ids = sorted(set(GGUF_MODEL_FILES.keys()) | HF_CAUSAL_MODELS, key=str.lower)
        return _put_recommended_first(ids)

    def _light_model_ids(self) -> list[str]:
        ids = sorted(
            [m for m in self._all_model_ids() if MODEL_TIER.get(_norm_key(m), "heavy") == "light"],
            key=str.lower,
        )
        return _put_recommended_first(ids)

    def _default_model_choice(self, vals: list[str]) -> str:
        if RECOMMENDED_MODEL_ID in vals:
            return RECOMMENDED_MODEL_ID
        return vals[0] if vals else ""

    def _refresh_model_combo(self) -> None:
        if self._skip_big_var.get():
            vals = self._light_model_ids()
            hint = "light models only"
        else:
            vals = self._all_model_ids()
            hint = "all models"
        self.model_combo["values"] = vals
        self.status_var.set(f"Model list: {hint} ({len(vals)} choices).")
        default = self._default_model_choice(vals)
        cur = (self.model_var.get() or "").strip()
        if cur and cur not in vals:
            self.model_var.set(default)
        elif not cur:
            self.model_var.set(default)

    def _build_ui(self) -> None:
        pad = {"padx": 10, "pady": 6}
        style = ttk.Style()
        if sys.platform == "win32":
            try:
                style.theme_use("vista")
            except tk.TclError:
                pass
        style.configure("MR.Title.TLabel", font=("Segoe UI", 14, "bold"))
        style.configure("MR.Sub.TLabel", font=("Segoe UI", 9), foreground="#5c5c5c")

        ttk.Label(self.root, text="Model Runner Hub", style="MR.Title.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w", padx=12, pady=(14, 2)
        )
        ttk.Label(
            self.root,
            text=f"Recommended (reliable on Windows): {RECOMMENDED_MODEL_ID}",
            style="MR.Sub.TLabel",
        ).grid(row=1, column=0, columnspan=3, sticky="w", padx=12, pady=(0, 6))

        ttk.Label(self.root, text="Model").grid(row=2, column=0, sticky="w", **pad)
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(
            self.root,
            textvariable=self.model_var,
            width=52,
            values=[],
            state="readonly",
        )
        self.model_combo.grid(row=2, column=1, sticky="ew", **pad)

        skip_row = ttk.Frame(self.root)
        skip_row.grid(row=2, column=2, sticky="nw", padx=(0, 10))
        ttk.Checkbutton(
            skip_row,
            text="Skip big models\n(7B-class / lighter)",
            variable=self._skip_big_var,
            command=self._refresh_model_combo,
        ).pack(anchor="w")

        self.status_var = tk.StringVar(value="Idle.")
        ttk.Label(self.root, textvariable=self.status_var, foreground="#666666").grid(
            row=4, column=0, columnspan=3, sticky="w", padx=12
        )

        self._refresh_model_combo()

        ttk.Label(self.root, text="Prompts (one per line)").grid(row=3, column=0, sticky="nw", **pad)
        self.questions_text = tk.Text(self.root, height=12, width=54, wrap="word", font=("Consolas", 10))
        self.questions_text.grid(row=3, column=1, columnspan=2, sticky="nsew", **pad)
        self.questions_text.insert("1.0", DEFAULT_QUESTIONS)

        self.progress = ttk.Progressbar(self.root, mode="determinate", length=420)
        self.progress.grid(row=5, column=0, columnspan=3, sticky="ew", padx=12, pady=6)

        btn_row = ttk.Frame(self.root)
        btn_row.grid(row=6, column=0, columnspan=3, pady=(4, 14))
        self.run_btn = ttk.Button(btn_row, text="Run benchmark", command=self._on_run_clicked)
        self.run_btn.pack(side="left", padx=4)
        ttk.Button(btn_row, text="Quit", command=self.root.quit).pack(side="left", padx=4)

        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(3, weight=1)

    def _on_run_clicked(self) -> None:
        if self._busy:
            return
        model = (self.model_var.get() or "").strip()
        if not model:
            messagebox.showerror("Model", "Select a model.")
            return

        raw = self.questions_text.get("1.0", tk.END)
        questions = [q.strip() for q in raw.splitlines() if q.strip()]
        if len(questions) < 1:
            messagebox.showerror("Prompts", "Enter at least one non-empty line.")
            return
        if len(questions) > 64:
            messagebox.showerror("Prompts", "Too many prompts (max 64).")
            return

        allowed = self._light_model_ids() if self._skip_big_var.get() else self._all_model_ids()
        if model not in allowed:
            messagebox.showerror(
                "Model",
                "Selected model is not in the current list. Uncheck 'Skip big models' or pick a light model.",
            )
            return

        self._busy = True
        self.run_btn.state(["disabled"])
        self.progress["value"] = 0
        self.status_var.set("Running…")

        key = _norm_key(model)
        t = threading.Thread(target=self._worker, args=(key, questions), daemon=True)
        t.start()

    def _worker(self, model_key: str, questions: list[str]) -> None:
        try:
            if model_key in HF_BY_NORM:
                row = self._run_transformers(HF_BY_NORM[model_key], questions)
            elif model_key in GGUF_MODEL_FILES:
                repo, filename = GGUF_MODEL_FILES[model_key]
                row = self._run_gguf_llama(repo, filename, questions)
            else:
                raise ValueError(
                    f"Unknown model key '{model_key}'. Pick from the list (GGUF or HF entries)."
                )
            # Default-arg captures values for deferred Tk callbacks (exception `e` is cleared after `except`)
            self.root.after(0, lambda r=row: self._finish_ok(r))
        except Exception as e:  # noqa: BLE001 — surface to UI
            self.root.after(0, lambda err=e: self._finish_err(err))

    def _finish_ok(self, row: dict) -> None:
        self._busy = False
        self.run_btn.state(["!disabled"])
        self.progress["value"] = 100
        self.status_var.set("Done. Appended row to llmSummary.csv")

        new_df = pd.DataFrame([row], columns=CSV_COLUMNS_V2)
        if self.results_df.empty:
            self.results_df = new_df
        else:
            self.results_df = pd.concat([self.results_df, new_df], ignore_index=True)
        try:
            append_result_row_v2("llmSummary.csv", row)
        except Exception as e:
            messagebox.showerror("CSV", f"Could not write llmSummary.csv: {e}")
            return

        messagebox.showinfo(
            "Results",
            f"Latency mean: {row['latency_mean_s']:.4f} s\n"
            f"P95: {row['latency_p95_s']:.4f} s\n"
            f"Memory mean: {row['memory_mean_mb']:.1f} MB\n"
            f"Throughput: {row['tokens_per_sec_mean']:.2f} tok/s (est.)",
        )

        try:
            from Visualizer import visualize_data

            visualize_data("llmSummary.csv")
        except Exception as e:
            messagebox.showwarning("Plots", f"Benchmark saved, but plots failed: {e}")

    def _show_run_error(self, title: str, message: str) -> None:
        if len(message) < 380:
            messagebox.showerror(title, message)
            return
        win = tk.Toplevel(self.root)
        win.title(title)
        win.transient(self.root)
        win.minsize(480, 280)
        ttk.Label(win, text=title, font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=12, pady=(12, 6))
        st = scrolledtext.ScrolledText(win, width=78, height=16, wrap="word", font=("Consolas", 10))
        st.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        st.insert("1.0", message)
        st.configure(state="disabled")
        ttk.Button(win, text="OK", command=win.destroy).pack(pady=(0, 12))

    def _finish_err(self, err: BaseException) -> None:
        self._busy = False
        self.run_btn.state(["!disabled"])
        self.progress["value"] = 0
        self.status_var.set("Error — see dialog.")
        self._show_run_error("Run failed", format_run_error(err))

    def _progress(self, i: int, n: int) -> None:
        def _ui() -> None:
            self.progress["value"] = 100.0 * (i + 1) / max(n, 1)
            self.status_var.set(f"Query {i + 1} / {n}…")

        self.root.after(0, _ui)

    def _run_gguf_llama(self, repo_id: str, filename: str, questions: list[str]) -> dict:
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise ImportError(LLAMA_CPP_INSTALL_HINT) from e
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=repo_id, filename=filename)
        # 0 = CPU-only; set LLAMA_N_GPU_LAYERS=-1 to offload all layers to GPU when supported.
        n_gpu_layers = int(os.environ.get("LLAMA_N_GPU_LAYERS", "0"))
        llm = Llama(
            model_path=path,
            n_threads=max(1, (os.cpu_count() or 4) - 1),
            n_batch=512,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

        n_params = _llama_n_params(llm)
        times: list[float] = []
        mem_samples: list[float] = []
        total_out_tokens = 0

        for i, question in enumerate(questions):
            self._progress(i, len(questions))
            m0 = _rss_mb()
            t0 = time.perf_counter()
            try:
                out = llm(
                    prompt=question,
                    max_tokens=256,
                    temperature=0.5,
                    top_p=0.95,
                    repeat_penalty=1.15,
                    top_k=64,
                )
            except TypeError:
                out = llm(
                    question,
                    max_tokens=256,
                    temperature=0.5,
                    top_p=0.95,
                    repeat_penalty=1.15,
                    top_k=64,
                )
            dt = time.perf_counter() - t0
            times.append(dt)
            m1 = _rss_mb()
            if m0 is not None and m1 is not None:
                mem_samples.append(max(m0, m1))
            elif m1 is not None:
                mem_samples.append(m1)

            text = ""
            try:
                text = out["choices"][0]["text"]
            except Exception:
                pass
            try:
                toks = len(llm.tokenize(text.encode("utf-8")))
            except Exception:
                toks = max(1, len(text) // 4)
            total_out_tokens += toks

        ls = latency_stats(times)
        ms = memory_stats(mem_samples) if mem_samples else {"mean": float("nan"), "std": float("nan")}
        total_time = sum(times)
        tps = (total_out_tokens / total_time) if total_time > 0 else float("nan")

        return {
            "run_id": new_run_id(),
            "timestamp": utc_now_iso(),
            "model_name": repo_id,
            "backend": "gguf_llama_cpp",
            "num_parameters": int(n_params),
            "n_queries": int(len(questions)),
            "latency_mean_s": float(ls["mean"]),
            "latency_std_s": float(ls["std"]),
            "latency_p95_s": float(ls["p95"]),
            "memory_mean_mb": float(ms["mean"]),
            "memory_std_mb": float(ms["std"]),
            "tokens_per_sec_mean": float(tps),
        }

    def _run_transformers(self, model_id: str, questions: list[str]) -> dict:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        dm = {"device_map": "auto"} if device == "cuda" else {}
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=dtype,
                attn_implementation="eager",
                trust_remote_code=True,
                **dm,
            )
        except TypeError:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    dtype=dtype,
                    trust_remote_code=True,
                    **dm,
                )
            except TypeError:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                    **dm,
                )
        if device == "cpu":
            model = model.to(device)

        n_params = int(sum(p.numel() for p in model.parameters()))
        model.eval()

        times: list[float] = []
        mem_samples: list[float] = []
        total_out_tokens = 0

        for i, question in enumerate(questions):
            self._progress(i, len(questions))
            inputs = tok(question, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            m0 = _rss_mb()
            t0 = time.perf_counter()
            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.5,
                    top_p=0.95,
                    pad_token_id=tok.pad_token_id,
                )
            dt = time.perf_counter() - t0
            times.append(dt)
            m1 = _rss_mb()
            if m0 is not None and m1 is not None:
                mem_samples.append(max(m0, m1))
            elif m1 is not None:
                mem_samples.append(m1)

            in_len = int(inputs["input_ids"].shape[1])
            out_len = int(out.shape[1])
            total_out_tokens += max(0, out_len - in_len)

        ls = latency_stats(times)
        ms = memory_stats(mem_samples) if mem_samples else {"mean": float("nan"), "std": float("nan")}
        total_time = sum(times)
        tps = (total_out_tokens / total_time) if total_time > 0 else float("nan")

        return {
            "run_id": new_run_id(),
            "timestamp": utc_now_iso(),
            "model_name": model_id,
            "backend": "transformers",
            "num_parameters": int(n_params),
            "n_queries": int(len(questions)),
            "latency_mean_s": float(ls["mean"]),
            "latency_std_s": float(ls["std"]),
            "latency_p95_s": float(ls["p95"]),
            "memory_mean_mb": float(ms["mean"]),
            "memory_std_mb": float(ms["std"]),
            "tokens_per_sec_mean": float(tps),
        }


def main() -> int:
    """
    Launch the Tk GUI. Prints one line to stdout so the terminal is not blank;
    errors go to stderr with a non-zero exit code.
    """
    warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")
    warnings.filterwarnings("ignore", message=".*flash.attention.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*flash_attn.*", category=UserWarning)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers.generation").setLevel(logging.ERROR)

    try:
        root = tk.Tk()
    except tk.TclError as e:
        print(
            "Tkinter could not open a display (headless session, SSH without X11, or broken Tcl/Tk).",
            file=sys.stderr,
        )
        print(f"Detail: {e}", file=sys.stderr)
        return 1

    try:
        ModelRunner(root)
    except Exception as e:
        print(f"Model Runner Hub: failed to build UI — {e}", file=sys.stderr)
        try:
            root.destroy()
        except Exception:
            pass
        return 1

    # Bring window forward once (common on Windows when it opens behind other apps).
    try:
        root.lift()
        root.attributes("-topmost", True)
        root.after(250, lambda: root.attributes("-topmost", False))
    except tk.TclError:
        pass

    print("Model Runner Hub — window open. Close it or press Quit to exit.", flush=True)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
