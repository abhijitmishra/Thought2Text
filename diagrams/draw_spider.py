import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import pi

# Load the input ODS file
file_path = './SubjectWise.ods'
df = pd.read_excel(file_path, engine='odf')

# Separate the data into ALL and NO_STAGE2 variants
df_no_stage2 = df[df['Model'].str.contains('NO_STAGE2')]
df_all = df[~df['Model'].str.contains('NO_STAGE2')]

# List of models and subjects to process
models = ['Llama-3-8B', 'Mistral-7B-v0.3', 'Qwen2.5-7B']
metrics_selected = ['GPT4 Adequacy', 'Mean BLEU Unigram Score']
subjects = ['subject-1', 'subject-2', 'subject-3', 'subject-4', 'subject-5', 'subject-6']

# Function to plot spider chart with smaller legend markers and adjusted placement
def plot_spider_with_very_small_legend(metric_name, model_name, data_all, data_no_stage2, subjects, file_name):
    # Number of variables (subjects)
    categories = subjects
    N = len(categories)
    

    # What will be the angle of each axis in the plot? (we divide the plot / number of variables)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # Draw one axe per variable and add labels
    plt.xticks(angles[:-1], categories)

    # ALL variant (in blue)
    values = data_all.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="ALL", color='blue')
    ax.fill(angles, values, 'b', alpha=0.1)

    # NO_STAGE2 variant (in red)
    values = data_no_stage2.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="NO_STAGE2", color='red')
    ax.fill(angles, values, 'r', alpha=0.1)

    # Adjust the legend to have smaller markers and place it farther away
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 0.85), frameon=False, handlelength=0.5, handletextpad=0.2, fontsize='small')

    # Save the plot to a file
    plt.tight_layout()
    plt.yticks(fontsize=9)
    plt.savefig(file_name, format='png')  # Save as PNG, change 'png' to 'pdf' or others as needed
    plt.close()  # Close the figure to avoid displaying in interactive environments

# Loop through the two selected metrics and three models, plotting and saving the spider charts
for model in models:
    for metric in metrics_selected:
        all_values_metric = df_all[df_all['Model'].str.contains(f'{model}')][metric].values[:6]  # First 6 subjects for "ALL"
        no_stage2_values_metric = df_no_stage2[df_no_stage2['Model'].str.contains(f'{model}')][metric].values[:6]  # First 6 subjects for "NO_STAGE2"
        if all_values_metric.size > 0 and no_stage2_values_metric.size > 0:
            # Define the file name for each plot
            file_name = f"{model}_{metric.replace(' ', '_')}.png"
            plot_spider_with_very_small_legend(metric, model, all_values_metric, no_stage2_values_metric, subjects, file_name)

