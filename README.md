# LLM Energy Consumption - Experimental Research

The aim of this research is to discover the average energy cost of a single LLM text-to-text-generation prompt. 

The projected final release date for this project is May 5, 2026.

## Description

First, we develop a dataset representative of the average LLM prompt by combining popular HuggingFace datasets with LLM-generated prompts. Then we attempt to measure the average energy cost of a single prompt to an LLM by running this set of 1,000 total prompts through three different popular models. We query the hardware for CPU/GPU usage every 0.5 seconds and average over the full model response time. Finally, the results are analyzed and a final estimate of energy cost is given in Watts per hour. 

The dataset is made of 500 AI-generated prompts and 500 prompts pulled from verified online datasets. For online datasets, five were selected from the most downloaded page on `HuggingFace.com` with datasets filtered for text-to-text generation and question-answering tasks; then, 100 rows were taken from each dataset using a random seed. The AI-generated prompts come from ChatGPT's online API and must be generated, saved and uploaded separately as a .csv file at `Data/AI_generated_prompts.csv`. See the dataset and experiment sections below for more details about the process. 

Please note that the full experiment pipeline takes approximately four hours to execute. See the experiment section below for hardware details. 

## Execution

### Main Experiment

To run the program:

* Download project files.
* Generate and save a HuggingFace.com access token.
* Copy/paste your HF token into `bin/config`.
* Navigate to the project directory.
* Run the commands to install and run:
  
```
./bin/install
./bin/run
```

### Results Analysis

To run the analysis script on one of the output files, navigate to the `Results` directory and use the following command. 

Parameters:
- `--input` specifies the results file you'd like to analyze.
- `--k` specifies how many results you would like to see for the top-k highest and lowest values from each output column.
- `--output` specifies where to save the script output to; format is a csv file. 

```
python analyze_results.py --input results.csv --k 5
```

### Data Visualization

To generate the six dataset visualization plots, run the command:

```
python DataVisualization/GenerateAll.py
```

Graphs will save to `DataVisualization/Graphs` as .pdf. 

**Note:** data visualization can be done separately from the main experiment and does not require the dataset to be explicitly generated, because running the `GenerateAll.py` script will re-generate the dataset. 


## Configuration

### config file

In the `config` file you will need to set your HuggingFace access token, otherwise gated models will not be downloaded in the experiment.

Other settings in `config`:

* MODEL_NAMES: space-separated string of model ID's from HuggingFace. See models section below.
* MAX_TOKENS: maximum number of tokens for a prompt and response. This helps keep computation cost reasonable. Default = 256.
* SAMPLES_PER_DATASET: number of prompts per HuggingFace dataset that get added to our final dataset. Default = 100.
* LLM_PROMPT_TARGET: number of prompts in the LLM-generated prompt file. This just checks that we have the correct number of prompts. Default = 500.
* LLM_PROMPTS_CSV: path to the LLM-generated prompt file. Default = "Data/AI_generated_prompts.csv".
* DATASET_PATH: path to the final combined dataset. This is output for `DatasetGenerator` and input for `Experiment`. Default ="Data/dataset.csv".
* OUTPUT_DIR: directory to store energy measurement result files in. Default ="Results".
* CPU_TDP_WATTS: fallback estimate for CPU power if `pyRAPL` is not available. Default = 150.
* POWER_INTERVAL_MS: how often, in milliseconds, `PowerMonitor` queries for CPU/GPU wattage. Default = 500.
* NUM_RUNS: number of times that each model goes through the entire prompt set. This allows us to average each prompt over ten runs for more accurate results. Default = 5.
* RANDOM_SEED: for reproducibility. Default = 42.

### Models

* **LLaMA:** meta-llama/Llama-3.1-8B 
* **GPT2:** openai-community/gpt2 
* **Qwen:** Qwen/Qwen2.5-7B

## Dataset

In the dataset file, there are seven columns: 
* **ID**: identifying number for the prompt, assigned based on row number. 
* **Prompt**: the prompt text that will be given to the LLM while we measure energy usage.
* **Prompt Length**: the number of words/tokens in the prompt.
* **Length Bucket**: the prompt is sorted into one of four buckets based on its length relative to the rest of the dataset. 
* **Task Type**: based on a key with 6 different task types labeled 1-6: Casual, Explanation, Writing, Coding, Productivity, Creative.
* **Complexity**: an annotation by the author to sort prompts by complexity of the task: 0 - low complexity, 1 - medium, 2 - high.
* **Origin**: the origin of the prompt; either ChatGPT or the name of the HuggingFace dataset.

### HuggingFace Datasets

* `HuggingFaceH4/MATH-500`
* `cais/mmlu`
* `openai/openai_humaneval`
* `crownelius/Opus-4.6-Reasoning-3300x`
* `google/simpleqa-verified`

### Example Data:

| Prompt | Prompt Length | Task Type | Complexity | Output Length | Origin |
|------|------|------|------|------|------|
| "At 50 miles per hour, how far would a car travel in $2\frac{3}{4}$ hours? Express your answer as a mixed number." | 21 | 2 | 4 | 2 | MATH-500 |
| Who was known for playing the trombone for The Jazzmen at the time of Kenny Ball's death? | 17 | 1 | 1 |0 | SimpleQA |
| Where should I go on vacation? | 6 | 0 | 1 |0 | ChatGPT |

## Experiment

In initial experiments, models tried to always generate the maximum number of tokens, even if the final output ends up being the same phrase repeated 10-20 times. 
To mitigate this, in our model outputs we set `no_repeat_ngram_size=3`. This way, if our model starts generating very repetitive output, 
it will get cut off. This helps give a more accurate final response, since most conversational LLMs don't repeat themselves as much as the "instruct" versions of models. 

### Experiment Outline

* For each of three models:
  * Repeat the experiment 5 times to normalize:
    * Run 1,000 prompts through the model:
      * For each prompt, we record:
        * Power consumption in CPU and GPU
        * Estimated total FLOPs (floating-point operations)
        * Memory usage in CPU and GPU

### Hardware Specifications

Using a linux server equipped with an Intel Xeon Gold 5215 (10-core, 2.50 GHz), 64 GB RAM, and dual NVIDIA GeForce RTX 2080 Ti GPUs, running Ubuntu 22.04.5 LTS, the full experiment with five runs for each of three models takes approximately **four hours** to execute. 

## Authors

* **Lead Author:** Abigail Pitcairn [abigail.pitcairn@maine.edu]

## Release History

* **March 4, 2026:** Initial Release of Dataset Generator
* **March 10, 2026:** Added Data Visualization Scripts for Graph Generation
* **March 23, 2026:** Data Visualization Update, Initial Experiment Script Upload
* **April 7, 2026:** Moved all files to a new repository due to a broken remote connection, implemented a single prompt experiment. 
* **April 8, 2026:** First implementation of a full working experiment, runnable from ./bin/run.
* **April 13, 2026:** Updated Experiment.py to run on only one GPU, significantly reducing overall experiment runtime. 
* **May 5, 2026:** Projected Final Release Date
