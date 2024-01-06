import tkinter as tk
from tkinter import ttk, messagebox
from codecarbon import track_emissions
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from transformers import AutoTokenizer
import transformers
from memory_profiler import memory_usage
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

class ModelRunner:
    def __init__(self, root):
        self.root = root
        self.root.title("Model Execution and Visualization")

        self.create_widgets()
        self.initialize_data_storage()

    def create_widgets(self):
        # Model selection
        self.model_label = ttk.Label(self.root, text="Select Model:")
        self.model_label.grid(row=0, column=0, pady=10)

        self.model_var = tk.StringVar()
        self.model_combobox = ttk.Combobox(self.root, textvariable=self.model_var)
        self.model_combobox["values"] = [
            "TheBloke/Llama-2-13B-chat-GGML",
            "TheBloke/open-llama-7b-open-instruct-GGML",
            "codellama/CodeLlama-13b-hf",
            "tiiuae/falcon-7b-instruct",
            "TheBloke/stable-vicuna-13B-HF",
            "tiiuae/falcon-40b-instruct",
            "TheBloke/open-llama-7b-open-instruct-GGML"
        ]
        self.model_combobox.grid(row=0, column=1, pady=10)

        # Questions entry
        self.questions_label = ttk.Label(self.root, text="Enter 10 Questions (one per line):")
        self.questions_label.grid(row=1, column=0, pady=10)

        self.questions_text = tk.Text(self.root, height=10, width=40)
        self.questions_text.grid(row=1, column=1, pady=10)

        # Run button
        self.run_button = ttk.Button(self.root, text="Run", command=self.run_models)
        self.run_button.grid(row=2, column=0, columnspan=2, pady=20)

    def initialize_data_storage(self):
        self.data_columns = ["Model Name", "Number of Model Parameters", "Average Query Execution Time",
                             "Standard Deviation of Query Execution Time", "Average Memory Usage",
                             "Standard Deviation of Memory Usage"]

        self.results_data = pd.DataFrame(columns=self.data_columns)

    @track_emissions()
    def run_llama_inference(self, model_name_or_path, model_basename, questions):
        lcpp_llm = Llama(
            model_path=hf_hub_download(repo_id=model_name_or_path, filename=model_basename),
            n_threads=1,  # CPU cores
            n_batch=32,  # Adjust as needed
            n_gpu_layers=4  # Adjust based on your model and GPU VRAM
        )

        # Results per model
        model_results = {"Model Name": model_name_or_path, "Number of Model Parameters": lcpp_llm.params.n_gpu_layers}

        # Iterate over questions
        query_times = []
        memory_usage_values = []

        for question in questions:
            start_time = time.time()

            # Perform inference
            response = lcpp_llm(
                prompt=question, max_tokens=256, temperature=0.5, top_p=0.95, repeat_penalty=1.2, top_k=150
            )
            print(question, response["choices"])

            # Measure time taken for the query
            query_time = time.time() - start_time
            query_times.append(query_time)

            # Memory profiling for each question
            mem_usage = memory_usage(
                (lcpp_llm, (question,),
                 {"max_tokens": 256, "temperature": 0.5, "top_p": 0.95, "repeat_penalty": 1.2, "top_k": 150})
            )
            memory_usage_values.append(max(mem_usage))

        # Calculate averages and standard deviations
        avg_query_time = sum(query_times) / len(query_times)
        std_dev_query_time = (sum((t - avg_query_time) ** 2 for t in query_times) / len(query_times)) ** 0.5

        avg_memory_usage = sum(memory_usage_values) / len(memory_usage_values)
        std_dev_memory_usage = (
                sum((m - avg_memory_usage) ** 2 for m in memory_usage_values) / len(memory_usage_values)
        ) ** 0.5

        # Update model results
        model_results["Average Query Execution Time"] = avg_query_time
        model_results["Standard Deviation of Query Execution Time"] = std_dev_query_time
        model_results["Average Memory Usage"] = avg_memory_usage
        model_results["Standard Deviation of Memory Usage"] = std_dev_memory_usage

        return model_results

    def run_models(self):
        model_name = self.model_var.get()

        if not model_name:
            messagebox.showerror("Error", "Please select a model.")
            return

        questions_text = self.questions_text.get("1.0", tk.END).strip()
        questions = questions_text.split("\n")

        if len(questions) != 5:
            messagebox.showerror("Error", "Please enter exactly 5 questions.")
            return

        if model_name.lower() == "codellama/codellama-13b-hf" or model_name.lower() == "tiiuae/falcon-7b-instruct" or model_name.lower() == "thebloke/stable-vicuna-13b-hf" or model_name.lower() == "tiiuae/falcon-40b-instruct":
            results = self.run_text_generation(model_name, questions)
        else:
            model_info = None
            if model_name.lower() == "thebloke/llama-2-13b-chat-ggml":
                model_info = ("TheBloke/Llama-2-13B-chat-GGML", "llama-2-13b-chat.ggmlv3.q5_1.bin")
            elif model_name.lower() == "thebloke/open-llama-7b-open-instruct-ggml":
                model_info = ("TheBloke/open-llama-7b-open-instruct-GGML", "open-llama-7B-open-instruct.ggmlv3.q4_0.bin")
            elif model_name.lower() == "thebloke/koala-13b-ggml":
                model_info = ("TheBloke/koala-13B-GGML", "koala-13B.ggmlv3.q4_0.bin")
            results = self.run_llama_inference(model_info[0], model_info[1], questions)

        # Save results to CSV
        csv_filename = "llmSummary.csv"
        self.save_results_to_csv(csv_filename, results)

        # Visualize data for the specific model
        visualize_data(csv_filename)

    def show_results(self, results):
        messagebox.showinfo("Results", f"Execution Time: {results['Average Query Execution Time']:.4f} s\n"
                                       f"Memory Usage: {results['Average Memory Usage']:.2f} MB\n")

        # Save results to DataFrame
        self.results_data = pd.concat([self.results_data, pd.DataFrame([results], columns=self.data_columns)], ignore_index=True)

    def save_results_to_csv(self, filename, results):
        self.show_results(results)
        self.results_data.to_csv(filename, index=False)

if __name__ == "__main__":
    root = tk.Tk()
    app = ModelRunner(root)
    root.mainloop()



