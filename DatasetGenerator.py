"""
Build unified prompt dataset for energy experiments.

Sources:
- LLM-generated prompts from a local CSV
- HuggingFace datasets:
    - HuggingFaceH4/MATH-500
    - cais/mmlu
    - openai/openai_humaneval
    - crownelius/Opus-4.6-Reasoning-3300x
    - google/simpleqa-verified

Output columns:
- prompt              : str
- prompt_length       : int  (word count)
- length_bucket       : int  in {0,1,2,3} (quartiles by prompt_length)
- task_type           : int  in {1..6}
- complexity          : int  in {0,1,2}  (0=low, 1=medium, 2=high)
- origin              : str  (e.g., "ChatGPT", "code_contests", ...)
"""

import os
import json
import pandas as pd
from typing import Optional, List, Dict
import re
import datasets

# ── Load config from environment ────────────────────────────
LLM_PROMPTS_CSV:     Optional[str] = os.getenv("LLM_PROMPTS_CSV") or None
OUTPUT_CSV:          str           = os.getenv("OUTPUT_CSV", "Data/dataset.csv")
RANDOM_SEED:         int           = int(os.getenv("RANDOM_SEED") or "42")
SAMPLES_PER_DATASET: int           = int(os.getenv("SAMPLES_PER_DATASET") or "100")
LLM_PROMPT_TARGET:   int           = int(os.getenv("LLM_PROMPT_TARGET") or "500")
MAX_PROMPT_WORDS:    int           = int(os.getenv("MAX_PROMPT_WORDS") or "250")

# ── Load HF dataset configs from bin/datasets.json ───────────────────────────
_REPO_ROOT   = os.path.dirname(os.path.abspath(__file__))
_DATASETS_PATH = os.path.join(_REPO_ROOT, "bin", "datasets.json")

with open(_DATASETS_PATH) as f:
    HF_DATASETS: List[Dict] = json.load(f)

# Helper Methods

def clean_prompt_text(text: str) -> str:
    """
    Normalize prompt text by:
    - Replacing all non-space whitespace with a space
    - Removing special characters beyond standard punctuation
    - Collapsing multiple spaces into one
    - Stripping leading/trailing spaces
    """

    if not isinstance(text, str):
        return ""

    # Replace all whitespace except space
    text = re.sub(r"[^\S ]+", " ", text)

    # Remove characters that are NOT:
    # letters, numbers, space, or standard punctuation
    text = re.sub(r"[^a-zA-Z0-9 .,!?;:'\"()\-\[\]{}<>/\\@#$%^&*_+=|`~]", "", text)

    # Collapse multiple spaces to one
    text = re.sub(r" +", " ", text)

    return text.strip()


def compute_word_count(text: str) -> int:
    """Token/word count based on whitespace splitting."""
    if not isinstance(text, str):
        return 0
    return len(text.strip().split())


def assign_length_buckets(lengths: pd.Series) -> pd.Series:
    """
    Assign each length to a bucket 0-3 so that approximately 25%
    of the dataset falls into each bucket.
    """
    q1, q2, q3 = lengths.quantile([0.25, 0.5, 0.75])

    def bucket(l: float) -> int:
        if l <= q1:
            return 0
        elif l <= q2:
            return 1
        elif l <= q3:
            return 2
        else:
            return 3

    return lengths.apply(bucket)


def extract_prompt_from_example(example: dict, candidates: List[str]) -> Optional[str]:
    """
    Try a list of candidate field names to get a prompt string from a HuggingFace example.
    Returns None if nothing works.
    """
    for field in candidates:
        if field in example and example[field] is not None:
            value = example[field]
            if isinstance(value, str):
                return value
    print(f"No prompt found for example {format(example)}")
    return None


def is_valid_prompt(text: Optional[str]) -> Tuple[bool, int]:
    """
    Validate a prompt and return (is_valid, word_count).
    A prompt is valid if it is a non-empty string within MAX_PROMPT_WORDS.
    Word count is computed immediately on the raw text before any cleaning,
    so that length filtering happens at download time.
    """
    if not text or not isinstance(text, str):
        return False, 0
    word_count = compute_word_count(text)
    if word_count == 0 or word_count > MAX_PROMPT_WORDS:
        return False, word_count
    return True, word_count


def load_hf_prompts() -> pd.DataFrame:
    """
    Load exactly SAMPLES_PER_DATASET valid prompts from each configured
    HuggingFace dataset. A prompt is valid if it is non-empty and does not
    exceed MAX_PROMPT_WORDS (checked immediately on the raw downloaded text).

    The dataset is shuffled and iterated in order; if a prompt is too long
    or missing, the next candidate in the shuffled dataset is used as a
    replacement until the exact target count is reached.

    Warns if the dataset is too small to supply enough valid prompts.
    """
    records = []

    for cfg in HF_DATASETS:
        print(f"\nLoading HF dataset: {cfg['name']}")

        try:
            if cfg["subset"]:
                ds = datasets.load_dataset(
                    cfg["name"],
                    cfg["subset"],
                    split=cfg["split"],
                )
            else:
                ds = datasets.load_dataset(
                    cfg["name"],
                    split=cfg["split"],
                )
        except Exception as e:
            print(f"  Failed to load {cfg['name']}: {e}")
            continue

        ds = ds.shuffle(seed=RANDOM_SEED)
        total_available = len(ds)

        if total_available < SAMPLES_PER_DATASET:
            print(
                f"  Warning: {cfg['name']} only has {total_available} rows total, "
                f"which is fewer than the target of {SAMPLES_PER_DATASET}. "
                f"Will collect as many valid prompts as possible."
            )

        collected = 0
        skipped_too_long = 0
        skipped_no_text = 0

        # Iterate through the entire shuffled dataset until we hit the target
        # or exhaust all available rows. This ensures long prompts are replaced
        # by the next candidate rather than simply reducing the final count.
        for ex in ds:
            if collected >= SAMPLES_PER_DATASET:
                break

            prompt_text = extract_prompt_from_example(ex, cfg["prompt_fields"])

            # Compute word count immediately on raw downloaded text
            valid, word_count = is_valid_prompt(prompt_text)

            if not valid:
                if word_count > MAX_PROMPT_WORDS:
                    skipped_too_long += 1
                else:
                    skipped_no_text += 1
                # Continue to next candidate as a replacement
                continue

            records.append(
                {
                    "prompt": prompt_text.strip(),
                    "prompt_length": word_count,
                    "task_type": cfg["task_type"],
                    "complexity": cfg["complexity"],
                    "origin": cfg["origin"],
                }
            )
            collected += 1

        print(
            f"  -> Collected {collected}/{SAMPLES_PER_DATASET} prompts. "
            f"Skipped {skipped_too_long} too long, {skipped_no_text} empty/missing."
        )
        if collected < SAMPLES_PER_DATASET:
            print(
                f"  Warning: Could only collect {collected} valid prompts from "
                f"{cfg['name']} after exhausting all {total_available} rows."
            )

    return pd.DataFrame.from_records(records)


def load_llm_prompts(csv_path: str) -> pd.DataFrame:
    """
    Load exactly LLM_PROMPT_TARGET valid prompts from the AI-generated CSV.

    - Word count is computed immediately on the raw prompt text.
    - Prompts that are empty or exceed MAX_PROMPT_WORDS are skipped (replaced
      by the next row in the file).
    - If the CSV contains more than LLM_PROMPT_TARGET valid prompts, only the
      first LLM_PROMPT_TARGET are kept.
    - Warns if the CSV cannot supply enough valid prompts.

    Expected columns:
      - prompt (str)
      - task_type (int in {1..6})
      - complexity (int in {0,1,2})
    """
    print(f"\nLoading AI-generated prompts from: {csv_path}")
    df = pd.read_csv(csv_path)

    if "prompt" not in df.columns:
        raise ValueError("LLM prompts CSV must contain a 'prompt' column.")

    # Standardize optional columns
    if "task_type" not in df.columns:
        df["task_type"] = pd.NA
    if "complexity" not in df.columns:
        df["complexity"] = pd.NA

    df["origin"] = "ChatGPT"

    # Compute word count immediately on raw downloaded text
    df["prompt_length"] = df["prompt"].apply(compute_word_count)

    # Filter: remove empty or over-length prompts (replacement by next valid row)
    total_raw = len(df)
    valid_mask = (df["prompt_length"] > 0) & (df["prompt_length"] <= MAX_PROMPT_WORDS)
    skipped = (~valid_mask).sum()
    df = df[valid_mask].reset_index(drop=True)

    print(
        f"  -> {total_raw} rows in CSV. "
        f"Skipped {skipped} invalid/too-long prompts. "
        f"{len(df)} valid candidates available."
    )

    # Enforce exact target count
    if len(df) < LLM_PROMPT_TARGET:
        print(
            f"  Warning: Only {len(df)} valid AI-generated prompts available; "
            f"target is {LLM_PROMPT_TARGET}. Final dataset will be smaller than intended."
        )
    else:
        df = df.iloc[:LLM_PROMPT_TARGET]
        print(f"  -> Using exactly {LLM_PROMPT_TARGET} AI-generated prompts.")

    return df[["prompt", "prompt_length", "task_type", "complexity", "origin"]]


# Main Pipeline

def main():
    # Load HF dataset prompts (5 datasets × 100 = 500 prompts)
    hf_df = load_hf_prompts()

    # Load AI-generated prompts (target: 500 prompts)
    if LLM_PROMPTS_CSV is not None:
        llm_df = load_llm_prompts(LLM_PROMPTS_CSV)
        combined = pd.concat([hf_df, llm_df], ignore_index=True)
    else:
        combined = hf_df.copy()
        print("No AI-generated prompts CSV configured; using HF prompts only.")

    # Report total before cleaning
    print(f"\nTotal prompts before cleaning: {len(combined)}")

    # Add unique ID to each prompt
    combined["id"] = combined.index

    # Clean all prompts
    combined["prompt"] = combined["prompt"].apply(clean_prompt_text)

    # Recompute word count after cleaning in case cleaning changed lengths
    combined["prompt_length"] = combined["prompt"].apply(compute_word_count)

    # Compute length_bucket so ~25% of samples fall into each bucket
    combined["length_bucket"] = assign_length_buckets(combined["prompt_length"])

    # Confirm correct ordering of columns for consistency
    combined = combined[
        [
            "id",
            "prompt",
            "prompt_length",
            "length_bucket",
            "task_type",
            "complexity",
            "origin",
        ]
    ]

    # Save dataset to CSV
    combined.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved combined dataset with {len(combined)} rows to: {OUTPUT_CSV}")

    # Breakdown by origin for a quick sanity check
    print("\nPrompt counts by origin:")
    print(combined["origin"].value_counts().to_string())


if __name__ == "__main__":
    main()
