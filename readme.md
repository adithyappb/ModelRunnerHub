# ModelRunnerHub

## ModelRunner.py

- This script provides a GUI for selecting a model, entering questions, and running model inference.
- Results are saved to a CSV file (`llmSummary.csv`).
- Visualizes the performance of different models.

## HistogramGenerator.py

- Reads `llmSummary.csv` and generates histograms for various metrics.
- Histograms are saved in the `histograms` directory.

## Usage

1. Run `ModelRunner.py` to execute model inference and save results.
2. Run `HistogramGenerator.py` to generate histograms based on the saved results.

Ensure dependencies are installed using:
#bash
     pip install codecarbon huggingface_hub llama_cpp transformers memory_profiler
Feel free to customize the model options, questions, and metrics.

##Requirements
Python 3.x
Libraries: tkinter, ttk, codecarbon, huggingface_hub, llama_cpp, transformers, memory_profiler, time, numpy, pandas, matplotlib
