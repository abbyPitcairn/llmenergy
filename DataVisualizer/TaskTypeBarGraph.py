import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
"""
Create a Bar Graph showing the distribution of the six task types in the dataset.
"""
def task_type_bar_graph(graph_dir: str, dataset_path: str, task_type_legend_elements: list):
    df = pd.read_csv(dataset_path)

    task_counts = df["task_type"].value_counts().sort_index()

    plt.figure()
    plt.bar(task_counts.index, task_counts.values)
    plt.xlabel("Task Type")
    plt.ylabel("Number of Prompts")
    plt.xticks([1,2,3,4,5,6])

    plt.legend(handles=task_type_legend_elements, title="Task Type")
    plt.savefig(graph_dir + "/bargraph.pdf")