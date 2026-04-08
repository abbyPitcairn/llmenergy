"""
Prompt Power Measurement
======================================
Runs a set of prompts through a model and record all power / energy / FLOPs / memory metrics to a .csv.
"""

import os
import re
import csv
import time
import threading
import subprocess
import platform
from typing import Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables, type cast if necessary
MODEL_NAMES = [m.strip() for m in (os.getenv("MODEL_NAMES") or "").split(",") if m.strip()]
HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_PATH = os.getenv("DATASET_PATH")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "../Results")
CPU_TDP_WATTS = float(os.getenv("CPU_TDP_WATTS", "150"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))

# ── Optional dependencies ─────────────────────────────────────────────────────
try:
    from codecarbon import EmissionsTracker
    HAS_CODECARBON = True
except ImportError:
    HAS_CODECARBON = False
    print("[WARN] codecarbon not installed — carbon tracking disabled. pip install codecarbon")

try:
    import pyRAPL
    pyRAPL.setup()
    HAS_PYRAPL = True
except ImportError:
    HAS_PYRAPL = False
    print("[WARN] pyRAPL not installed — pip install pyRAPL")
except (PermissionError, Exception) as e:
    HAS_PYRAPL = False
    print(f"[WARN] pyRAPL unavailable ({e}) — falling back to /proc/stat × TDP estimate")

# ── /proc/stat CPU utilisation helper (Linux RAPL fallback) ──────────────────
def _read_proc_stat() -> Optional[Tuple[int, int]]:
    try:
        with open("/proc/stat") as f:
            line = f.readline()
        fields = line.split()
        if fields[0] != "cpu":
            return None
        values = list(map(int, fields[1:]))
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        return idle, sum(values)
    except OSError:
        return None

def _read_cpu_utilization(interval_s: float) -> Optional[float]:
    s1 = _read_proc_stat()
    if s1 is None:
        return None
    time.sleep(interval_s)
    s2 = _read_proc_stat()
    if s2 is None:
        return None
    d_total = s2[1] - s1[1]
    d_idle = s2[0] - s1[0]
    return 0.0 if d_total == 0 else 1.0 - (d_idle / d_total)

# ── Power Monitor ─────────────────────────────────────────────────────────────
class PowerMonitor:
    """
    Cross-platform CPU/GPU power sampler (500 ms polling).

    macOS  → powermetrics (requires sudo)
    Linux  → pyRAPL (Intel RAPL) or /proc/stat × TDP fallback
    GPU    → nvidia-smi (any OS with NVIDIA driver)
    """

    def __init__(self, interval_ms: int = 500):
        self.interval_ms = interval_ms
        self.running = False
        self.thread = None
        self.cpu_samples: list[float] = []
        self.gpu_samples: list[float] = []
        self._system = platform.system()
        self._has_nvidia = self._check_nvidia()

    def _check_nvidia(self) -> bool:
        try:
            subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, check=True, timeout=5)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False

    def _sample_gpu_watts(self) -> Optional[float]:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                text=True,
                timeout=5,  # FIX: prevent nvidia-smi from hanging indefinitely
            )
            return sum(float(v.strip()) for v in out.strip().splitlines() if v.strip())
        except Exception:
            return None

    def _monitor_macos(self):
        proc = subprocess.Popen(
            ["sudo", "powermetrics", "--samplers", "smc", "-i", str(self.interval_ms)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        while self.running:
            line = proc.stdout.readline()
            m = re.search(r"CPU Power:\s*(\d+\.?\d*)\s*W", line)
            if m:
                self.cpu_samples.append(float(m.group(1)))
            if self._has_nvidia:
                w = self._sample_gpu_watts()
                if w is not None:
                    self.gpu_samples.append(w)
        proc.terminate()

    def _monitor_linux(self):
        meter = pyRAPL.Measurement("cpu") if HAS_PYRAPL else None
        while self.running:
            if meter:
                meter.begin()
                time.sleep(self.interval_ms / 1000)
                meter.end()
                duration_s = meter.result.duration / 1e6
                energy_uj = sum(meter.result.pkg or [0])
                if duration_s > 0:
                    self.cpu_samples.append((energy_uj / 1e6) / duration_s)
            else:
                util = _read_cpu_utilization(self.interval_ms / 1000)
                if util is not None:
                    self.cpu_samples.append(util * CPU_TDP_WATTS)
                else:
                    time.sleep(self.interval_ms / 1000)
            if self._has_nvidia:
                w = self._sample_gpu_watts()
                if w is not None:
                    self.gpu_samples.append(w)

    def _monitor_fallback(self):
        while self.running:
            time.sleep(self.interval_ms / 1000)
            if self._has_nvidia:
                w = self._sample_gpu_watts()
                if w is not None:
                    self.gpu_samples.append(w)

    def start(self):
        self.running = True
        target = (self._monitor_macos if self._system == "Darwin" else
                  self._monitor_linux if self._system == "Linux" else
                  self._monitor_fallback)
        self.thread = threading.Thread(target=target, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=10)  # FIX: increased from 5s to allow clean shutdown

    def avg_cpu_watts(self) -> float:
        return sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0.0

    def avg_gpu_watts(self) -> float:
        return sum(self.gpu_samples) / len(self.gpu_samples) if self.gpu_samples else 0.0

# ── FLOPs Estimator ───────────────────────────────────────────────────────────
def estimate_flops(model, input_tokens: int, output_tokens: int) -> float:
    """
    FLOPs ≈ 2 × N_params × T_tokens  (Kaplan et al. 2020)
           + attention FLOPs: 4 × L × H × d_head × T²
    """
    total_params = sum(p.numel() for p in model.parameters())
    T = input_tokens + output_tokens
    base_flops = 2 * total_params * T
    attn_flops = 0.0
    cfg = getattr(model, "config", None)
    if cfg:
        n_layers = getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", 0))
        n_heads  = getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", 0))
        hidden   = getattr(cfg, "hidden_size", getattr(cfg, "n_embd", 0))
        d_head   = hidden // n_heads if n_heads else 0
        attn_flops = 4 * n_layers * n_heads * d_head * (T ** 2)
    return base_flops + attn_flops

# ── Peak Memory ───────────────────────────────────────────────────────────────
def get_peak_memory_mb() -> dict:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return {"peak_gpu_mem_mb": torch.cuda.max_memory_allocated() / 1024**2,
                "peak_cpu_mem_mb": 0.0}
    try:
        import psutil, os
        rss = psutil.Process(os.getpid()).memory_info().rss / 1024**2
        return {"peak_gpu_mem_mb": 0.0, "peak_cpu_mem_mb": rss}
    except ImportError:
        return {"peak_gpu_mem_mb": 0.0, "peak_cpu_mem_mb": 0.0}

# ── Run a single prompt and return a result dict ──────────────────────────────
def run_prompt(prompt_id: str, prompt: str, model, tokenizer) -> dict:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_token_count = inputs["input_ids"].shape[1]

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    carbon_tracker = None
    if HAS_CODECARBON:
        carbon_tracker = EmissionsTracker(log_level="error", save_to_file=False,
                                          tracking_mode="machine")
        carbon_tracker.start()

    monitor = PowerMonitor(interval_ms=500)
    monitor.start()
    t0 = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,  # FIX: allow early stopping at EOS token
            no_repeat_ngram_size=3,
        )

    t1 = time.perf_counter()
    monitor.stop()
    carbon_kg = carbon_tracker.stop() if carbon_tracker else None

    response_sec       = t1 - t0
    output_token_count = outputs.shape[1] - input_token_count
    tokens_per_sec     = output_token_count / response_sec if response_sec > 0 else 0.0

    avg_cpu_w       = monitor.avg_cpu_watts()
    avg_gpu_w       = monitor.avg_gpu_watts()
    total_avg_w     = avg_cpu_w + avg_gpu_w
    hours           = response_sec / 3600
    cpu_energy_wh   = avg_cpu_w   * hours
    gpu_energy_wh   = avg_gpu_w   * hours
    total_energy_wh = total_avg_w * hours
    total_energy_j  = total_energy_wh * 3600
    joules_per_token = total_energy_j / output_token_count if output_token_count > 0 else 0.0

    total_flops     = estimate_flops(model, input_token_count, output_token_count)
    flops_per_token = total_flops / output_token_count if output_token_count > 0 else 0.0

    mem = get_peak_memory_mb()

    return {
        "prompt_id":        prompt_id,
        "input_tokens":     input_token_count,
        "output_tokens":    output_token_count,
        "response_sec":     response_sec,
        "tokens_per_sec":   tokens_per_sec,
        "avg_cpu_watts":    avg_cpu_w,
        "avg_gpu_watts":    avg_gpu_w,
        "avg_total_watts":  total_avg_w,
        "cpu_energy_wh":    cpu_energy_wh,
        "gpu_energy_wh":    gpu_energy_wh,
        "total_energy_wh":  total_energy_wh,
        "total_energy_j":   total_energy_j,
        "joules_per_token": joules_per_token,
        "total_flops":      total_flops,
        "flops_per_token":  flops_per_token,
        "peak_gpu_mem_mb":  mem["peak_gpu_mem_mb"],
        "peak_cpu_mem_mb":  mem["peak_cpu_mem_mb"],
        "carbon_kg":        carbon_kg if carbon_kg is not None else "",
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def run_one(model_name: str, run_number: int, model, tokenizer, rows: list):
    """Run one pass of all prompts through an already-loaded model."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    safe_model_name = model_name.replace("/", "_")
    output_path = os.path.join(OUTPUT_DIR, f"{safe_model_name}_run_{run_number:02d}.csv")

    total = len(rows)
    output_fields = [
        "prompt_id", "input_tokens", "output_tokens",
        "response_sec", "tokens_per_sec",
        "avg_cpu_watts", "avg_gpu_watts", "avg_total_watts",
        "cpu_energy_wh", "gpu_energy_wh", "total_energy_wh", "total_energy_j",
        "joules_per_token", "total_flops", "flops_per_token",
        "peak_gpu_mem_mb", "peak_cpu_mem_mb", "carbon_kg",
    ]

    print(f"    Output: {output_path}")
    with open(output_path, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=output_fields)
        writer.writeheader()

        for i, (prompt_id, prompt) in enumerate(rows, start=1):
            if i % 100 == 0:
                print(f"  [{i}/{total}]")
            try:
                result = run_prompt(prompt_id, prompt, model, tokenizer)
                writer.writerow(result)
                out_f.flush()
            except Exception as e:
                print(f"[ERROR] prompt_id={prompt_id}: {e}")
                writer.writerow({"prompt_id": prompt_id, **{k: "" for k in output_fields if k != "prompt_id"}})
                out_f.flush()

    print(f"==> Done. Results saved to: {output_path}")


def main():
    if not DATASET_PATH or not os.path.isfile(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at: {DATASET_PATH}")

    with open(DATASET_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [(row["id"], row["prompt"]) for row in reader if row.get("prompt", "").strip()]

    print(f"==> Loaded {len(rows)} prompts from {DATASET_PATH}")

    NUM_RUNS = int(os.getenv("NUM_RUNS", "10"))
    for model_name in MODEL_NAMES:

        # Load model once for this model
        print(f"\n{'='*60}")
        print(f"Loading model: {model_name}")
        print(f"{'='*60}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            token=HF_TOKEN,
        )
        model.eval()
        print("==> Model loaded\n")

        # Do all runs with this model
        for run_number in range(1, NUM_RUNS + 1):
            print(f"\n--- {model_name}  |  Run {run_number}/{NUM_RUNS} ---")
            run_one(model_name, run_number, model, tokenizer, rows)

        # Free memory before loading next model
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"\n==> Model {model_name} unloaded.")


if __name__ == "__main__":
    if not MODEL_NAMES:
        raise ValueError(
            "MODEL_NAMES environment variable is not set or empty. "
            "Set it as a comma-separated list, e.g.: "
            "'meta-llama/Llama-3.1-8B-Instruct,mistralai/Mistral-7B-v0.1'"
        )
    main()
