import pandas as pd
import matplotlib.pyplot as plt
"""
Create a stacked bar graph showing the distribution of complexity levels for each task type.
"""
def task_type_vs_complexity_stacked_bar(graph_dir: str, dataset_path: str):
    df = pd.read_csv(dataset_path)

    counts = pd.crosstab(df["task_type"], df["complexity"])
    counts.plot(
        kind="bar",
        stacked=True,
        figsize=(8,5)
    )

    plt.xlabel("Task Type")
    plt.ylabel("Number of Prompts")

    task_names = [
        "Casual", "Explanation", "Writing",
        "Coding", "Productivity", "Creativity"
    ]

    plt.xticks(range(len(task_names)), task_names, rotation=30)
    plt.legend(title="Complexity")

    plt.tight_layout()
    plt.savefig(graph_dir + "/stackedbar.pdf")
