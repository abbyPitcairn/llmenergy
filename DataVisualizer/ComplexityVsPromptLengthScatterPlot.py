import matplotlib.pyplot as plt
import pandas as pd
"""
Create a scatter plot showing complexity on the x axis and prompt length on the y axis.
"""
def complexity_vs_prompt_length_scatter_plot(graph_dir: str, dataset_path: str, task_type_legend_elements: list):
    df = pd.read_csv(dataset_path)
    plt.scatter(df["task_type"], df["prompt_length"])
    plt.xlabel("Task Type")
    plt.ylabel("Prompt Length (words)")

    plt.legend(handles=task_type_legend_elements, title="Task Types")
    plt.savefig(graph_dir + "/scatterplot.pdf")