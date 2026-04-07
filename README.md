# LLM Energy Consumption - Experimental Research

The aim of this research is to discover the average energy cost of prompting an LLM for text-to-text generation. Currently, this work only generates the dataset to be used. The projected final release date for this project is May 5, 2026.


### Description

First, we develop a dataset representative of the average LLM prompt and then test the energy cost by running this set of prompts through a handful of different LLMs. Finally, these experimental results will be analyzed and a final estimate of energy cost will be given in Watts per hour. 

The dataset is comprised of 500 AI-generated prompts and 500 prompts pulled from verified online datasets. For online datasets, five were selected from the most downloaded page on `HuggingFace.com` with datasets filtered for text-to-text generation and question-answering tasks; then, 100 rows were taken from each dataset using a random seed. These datasets are:

* `HuggingFaceH4/MATH-50`
* `cais/mmlu`
* `openai/openai_humaneval`
* `crownelius/Opus-4.6-Reasoning-3300x`
* `google/simpleqa-verified`

The AI-generated prompts come from ChatGPT's online API and must be generated, saved and uploaded separately as a .csv file at `Data/AI_generated_prompts.csv`. 

`DatasetGenerator.py` will create the complete dataset from this saved .csv and using the HuggingFace API. 

### Execution

To run the program:

* Download project files.
* Generate and save a HuggingFace.com access token.
* Run the Python commands below, and enter HF token when prompted.
* Now your dataset is generated at the output csv file.
  
```
pip install -r requirements.txt
huggingface-cli login
python DatasetGenerator.py
```

To generate the six data visualization plots, run the command:

```
python DataVisualization/GenerateAll.py
```

Graphs will save to `DataVisualization/Graphs` as .pdf. 

**Note:** it is not necessary to generate the dataset prior to generating the plots, because running the `GenerateAll.py` script will re-generate the dataset. 

### Dataset

In the dataset file, there are seven columns: 
* **ID**: identifying number for the prompt, assigned based on row number. 
* **Prompt**: the prompt text that will be given to the LLM while we measure energy usage.
* **Prompt Length**: the number of words/tokens in the prompt.
* **Length Bucket**: the prompt is sorted into one of four buckets based on its length relative to the rest of the dataset. 
* **Task Type**: based on a key with 6 different task types labeled 1-6: Casual, Explanation, Writing, Coding, Productivity, Creative.
* **Complexity**: an annotation by the author to sort prompts by complexity of the task: 0 - low complexity, 1 - medium, 2 - high.
* **Origin**: the origin of the prompt; either ChatGPT or the name of the HuggingFace dataset.

#### Example Data:

| Prompt | Prompt Length | Task Type | Complexity | Output Length | Origin |
|------|------|------|------|------|------|
| "At 50 miles per hour, how far would a car travel in $2\frac{3}{4}$ hours? Express your answer as a mixed number." | 21 | 2 | 4 | 2 | MATH-500 |
| Who was known for playing the trombone for The Jazzmen at the time of Kenny Ball's death? | 17 | 1 | 1 |0 | SimpleQA |
| Where should I go on vacation? | 6 | 0 | 1 |0 | ChatGPT |

### Experiment

Not Fully Implemented Yet

### Authors

* **Lead Author:** Abigail Pitcairn [abigail.pitcairn@maine.edu]

### Release History

* **March 4, 2026:** Initial Release of Dataset Generator
* **March 10, 2026:** Added Data Visualization Scripts for Graph Generation
* **March 23, 2025:** Data Visualization Update, Initial Experiment Script Upload
* **May 5, 2026:** Projected Final Release Date
