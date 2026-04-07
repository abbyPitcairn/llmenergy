import pandas as pd
import matplotlib.pyplot as plt
"""
Create a histogram showing the distribution of prompt lengths in the dataset. 
"""
def prompt_length_histogram(graph_dir: str, dataset_path: str):
    df = pd.read_csv(dataset_path)
    plt.figure(figsize=(8,5))

    plt.hist(df["prompt_length"], bins=20)

    plt.xlabel("Prompt Length")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(graph_dir + "/histogram.pdf")