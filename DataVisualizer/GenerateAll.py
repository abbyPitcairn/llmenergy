import subprocess
from matplotlib.lines import Line2D
import ComplexityVsPromptLengthScatterPlot
import PromptLengthHistogram
import PromptLengthVsComplexityBoxPlot
import PromptWordCloud
import TaskTypeBarGraph
import TaskTypeVsComplexityStackedBar

# Path to plotting scripts folder, output graph folder and dataset file path
SCRIPTS_DIR = "DataVisualizer/PlottingScripts"
GRAPHS_DIR = "DataVisualizer/Graphs"
DATASET_PATH = "Data/dataset.csv"

# Custom legend elements for task type legend
TASK_TYPE_LEGEND = [
    Line2D([0], [0], marker='o', linestyle='None', label='1 - Casual'),
    Line2D([0], [0], marker='o', linestyle='None', label='2 - Explanation'),
    Line2D([0], [0], marker='o', linestyle='None', label='3 - Writing'),
    Line2D([0], [0], marker='o', linestyle='None', label='4 - Coding'),
    Line2D([0], [0], marker='o', linestyle='None', label='5 - Productivity'),
    Line2D([0], [0], marker='o', linestyle='None', label='6 - Creative')
]

# Generate the dataset
dataset_generated = subprocess.run(["python", "DatasetGenerator.py"], capture_output=True, text=True)
if dataset_generated.returncode != 0:
    raise Exception("Failed to generate dataset")

# Run each plotting script
ComplexityVsPromptLengthScatterPlot.complexity_vs_prompt_length_scatter_plot(GRAPHS_DIR, DATASET_PATH, TASK_TYPE_LEGEND)
PromptLengthHistogram.prompt_length_histogram(GRAPHS_DIR, DATASET_PATH)
PromptLengthVsComplexityBoxPlot.prompt_length_vs_complexity_box_plot(GRAPHS_DIR, DATASET_PATH)
PromptWordCloud.prompt_word_cloud(GRAPHS_DIR, DATASET_PATH)
TaskTypeBarGraph.task_type_bar_graph(GRAPHS_DIR, DATASET_PATH, TASK_TYPE_LEGEND)
TaskTypeVsComplexityStackedBar.task_type_vs_complexity_stacked_bar(GRAPHS_DIR, DATASET_PATH)