import pandas as pd
import matplotlib.pyplot as plt
"""
Create a box plot showing complexity on the x axis and prompt length on the y axis.
"""
def prompt_length_vs_complexity_box_plot(graph_dir: str, dataset_path: str):
    df = pd.read_csv(dataset_path)

    # Group prompt lengths by complexity
    data = [
        df[df["complexity"] == 0]["prompt_length"],
        df[df["complexity"] == 1]["prompt_length"],
        df[df["complexity"] == 2]["prompt_length"]
    ]

    plt.figure(figsize=(8, 5))
    plt.boxplot(
        data,
        widths=0.6,
        patch_artist=True,
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        medianprops=dict(linewidth=2)
    )

    plt.xlabel("Task Complexity", fontsize=12)
    plt.ylabel("Prompt Length", fontsize=12)
    plt.xticks([1, 2, 3], [0, 1, 2], fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(graph_dir + "/boxplot.pdf")