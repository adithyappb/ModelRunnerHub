import pandas as pd
import matplotlib.pyplot as plt
import os

def visualize_data(filename):
    df = pd.read_csv(filename)

    # Create a directory to save the histograms
    output_directory = 'histograms'
    os.makedirs(output_directory, exist_ok=True)

    # Unique model names
    unique_models = df['Model Name'].unique()

    # Metrics to visualize
    metrics = ['Number of Model Parameters', 'Average Query Execution Time', 'Standard Deviation of Query Execution Time',
               'Average Memory Usage', 'Standard Deviation of Memory Usage']

    # Set a color palette for better differentiation
    colors = plt.cm.get_cmap('tab10', len(unique_models))

    # Plot histograms for each metric and save them
    for metric in metrics:
        plt.figure(figsize=(12, 8))
        plt.title(f'Distribution of {metric} for Different Models', fontsize=16)

        # Determine the range of values across all models
        data_range = df[metric].min(), df[metric].max()

        for model, color in zip(unique_models, colors.colors):
            model_data = df[df['Model Name'] == model]
            plt.hist(model_data[metric], bins=20, alpha=0.7, label=model, color=color, log=True, range=data_range)  # Log scaling and fixed range

        plt.xlabel(metric, fontsize=14)
        plt.ylabel('Frequency (log scale)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.tick_params(axis='both', which='major', labelsize=12)

        # Save the histogram
        histogram_filename = os.path.join(output_directory, f'{metric}_histogram.png')
        plt.savefig(histogram_filename, bbox_inches='tight')
        plt.show()

