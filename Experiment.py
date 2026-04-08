"""
Prompt Power Measurement
======================================
Runs a set of prompts through a model and records all power / energy / FLOPs / memory metrics to a .csv.
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
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── Environment variables ─────────────────────────────────────────────────────
MODEL_NAMES    = [m.strip() for m in (os.getenv("MODEL_NAMES") or "").split(",") if m.strip()]
HF_TOKEN       = os.getenv("HF_TOKEN")
DATASET_PATH   = os.getenv("DATASET_PATH")
OUTPUT_DIR     = os.getenv("OUTPUT_DIR", "../Results")
CPU_TDP_WATTS  = float(os.getenv("CPU_TDP_WATTS", "150"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))
NUM_RUNS       = int(os.getenv("NUM_RUNS", "10"))

# ── Optional: pyRAPL ─────────────────────────────────────────────────────────
try:
    import pyRAPL
    pyRAPL.setup()
    HAS_PYRAPL = True
except ImportError:
    HAS_PYRAPL = False
    print("[WARN] pyRAPL not installed — pip install pyRAPL")
except Exception as e:
    HAS_PYRAPL = False
    print(f"[WARN] pyRAPL unavailable ({e}) — falling back to /proc/stat x TDP estimate")

# ── Optional: pynvml for fast GPU power (no subprocess overhead) ─────────────
try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_HANDLES = [
        pynvml.nvmlDeviceGetHandleByIndex(i)
        for i in range(pynvml.nvmlDeviceGetCount())
    ]
    HAS_PYNVML = True
except Exception:
    HAS_PYNVML = False
    _NVML_HANDLES = []
    print("[WARN] pynvml not available — GPU power via nvidia-smi subprocess (slower)")

# ── /proc/stat CPU utilisation helper ────────────────────────────────────────
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

def _cpu_util_between(s1: Tuple[int, int], s2: Tuple[int, int]) -> float:
    d_total = s2[1] - s1[1]
    d_idle  = s2[0] - s1[0]
    return 0.0 if d_total == 0 else 1.0 - (d_idle / d_total)

# ── Power Monitor ─────────────────────────────────────────────────────────────
class PowerMonitor:
    """
    Cross-platform CPU/GPU power sampler.

    macOS  -> powermetrics (requires sudo)
    Linux  -> pyRAPL (Intel RAPL) or /proc/stat x TDP fallback
    GPU    -> pynvml (preferred, zero-subprocess) or nvidia-smi fallback

    The background thread is daemon=True so it never blocks process exit.
    stop() waits at most 2 x INTERVAL_S + 1 s, not indefinitely.
    """

    INTERVAL_S = 0.5  # 500 ms polling

    def __init__(self):
        self.running         = False
        self._thread         = None
        self.cpu_samples: list[float] = []
        self.gpu_samples: list[float] = []
        self._system         = platform.system()

    # ── GPU sampling ─────────────────────────────────────────────────────────
    def _sample_gpu_watts(self) -> Optional[float]:
        if HAS_PYNVML:
            try:
                return sum(
                    pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0  # mW -> W
                    for h in _NVML_HANDLES
                )
            except Exception:
                return None
        # Fallback: nvidia-smi subprocess
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=power.draw",
                 "--format=csv,noheader,nounits"],
                text=True, timeout=3,
            )
            return sum(float(v.strip()) for v in out.strip().splitlines() if v.strip())
        except Exception:
            return None

    # ── macOS monitor ─────────────────────────────────────────────────────────
    def _monitor_macos(self):
        proc = subprocess.Popen(
            ["sudo", "powermetrics", "--samplers", "smc",
             "-i", str(int(self.INTERVAL_S * 1000))],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        try:
            while self.running:
                line = proc.stdout.readline()
                if not line:
                    break
                m = re.search(r"CPU Power:\s*(\d+\.?\d*)\s*W", line)
                if m:
                    self.cpu_samples.append(float(m.group(1)))
                w = self._sample_gpu_watts()
                if w is not None:
                    self.gpu_samples.append(w)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    # ── Linux monitor ─────────────────────────────────────────────────────────
    def _monitor_linux(self):
        meter = pyRAPL.Measurement("cpu") if HAS_PYRAPL else None

        while self.running:
            # ── CPU power ────────────────────────────────────────────────────
            if meter:
                try:
                    meter.begin()
                    time.sleep(self.INTERVAL_S)
                    meter.end()
                    duration_s = meter.result.duration / 1e6
                    energy_uj  = sum(meter.result.pkg or [0])
                    if duration_s > 0:
                        self.cpu_samples.append((energy_uj / 1e6) / duration_s)
                except Exception:
                    time.sleep(self.INTERVAL_S)
            else:
                # /proc/stat: two reads bracketing a sleep
                s1 = _read_proc_stat()
                time.sleep(self.INTERVAL_S)
                s2 = _read_proc_stat()
                if s1 and s2:
                    self.cpu_samples.append(
                        _cpu_util_between(s1, s2) * CPU_TDP_WATTS
                    )

            # ── GPU power ────────────────────────────────────────────────────
            w = self._sample_gpu_watts()
            if w is not None:
                self.gpu_samples.append(w)

    # ── Fallback (GPU-only, no CPU power path) ────────────────────────────────
    def _monitor_fallback(self):
        while self.running:
            time.sleep(self.INTERVAL_S)
            w = self._sample_gpu_watts()
            if w is not None:
                self.gpu_samples.append(w)

    # ── Public API ────────────────────────────────────────────────────────────
    def start(self):
        self.running = True
        target = (self._monitor_macos   if self._system == "Darwin" else
                  self._monitor_linux   if self._system == "Linux"  else
                  self._monitor_fallback)
        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=self.INTERVAL_S * 2 + 1)
            self._thread = None

    def avg_cpu_watts(self) -> float:
        return sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0.0

    def avg_gpu_watts(self) -> float:
        return sum(self.gpu_samples) / len(self.gpu_samples) if self.gpu_samples else 0.0


# ── FLOPs Estimator ───────────────────────────────────────────────────────────
def estimate_flops(model, input_tokens: int, output_tokens: int) -> float:
    """
    FLOPs ~= 2 x N_params x T_tokens  (Kaplan et al. 2020)
           + attention FLOPs: 4 x L x H x d_head x T^2
    """
    total_params = sum(p.numel() for p in model.parameters())
    T            = input_tokens + output_tokens
    base_flops   = 2 * total_params * T
    attn_flops   = 0.0
    cfg          = getattr(model, "config", None)
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
        return {
            "peak_gpu_mem_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
            "peak_cpu_mem_mb": 0.0,
        }
    try:
        import psutil
        rss = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
        return {"peak_gpu_mem_mb": 0.0, "peak_cpu_mem_mb": rss}
    except ImportError:
        return {"peak_gpu_mem_mb": 0.0, "peak_cpu_mem_mb": 0.0}


# ── Run a single prompt ───────────────────────────────────────────────────────
def run_prompt(prompt_id: str, prompt: str, model, tokenizer) -> dict:
    inputs            = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_token_count = inputs["input_ids"].shape[1]

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    monitor = PowerMonitor()
    monitor.start()
    t0 = time.perf_counter()

    gen_config = GenerationConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
    )
    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=gen_config)

    t1 = time.perf_counter()
    monitor.stop()

    response_sec       = t1 - t0
    output_token_count = outputs.shape[1] - input_token_count
    tokens_per_sec     = output_token_count / response_sec if response_sec > 0 else 0.0

    avg_cpu_w        = monitor.avg_cpu_watts()
    avg_gpu_w        = monitor.avg_gpu_watts()
    total_avg_w      = avg_cpu_w + avg_gpu_w
    hours            = response_sec / 3600
    cpu_energy_wh    = avg_cpu_w   * hours
    gpu_energy_wh    = avg_gpu_w   * hours
    total_energy_wh  = total_avg_w * hours
    total_energy_j   = total_energy_wh * 3600
    joules_per_token = total_energy_j / output_token_count if output_token_count > 0 else 0.0

    total_flops      = estimate_flops(model, input_token_count, output_token_count)
    flops_per_token  = total_flops / output_token_count if output_token_count > 0 else 0.0

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
    }


# ── Output schema ─────────────────────────────────────────────────────────────
OUTPUT_FIELDS = [
    "prompt_id", "input_tokens", "output_tokens",
    "response_sec", "tokens_per_sec",
    "avg_cpu_watts", "avg_gpu_watts", "avg_total_watts",
    "cpu_energy_wh", "gpu_energy_wh", "total_energy_wh", "total_energy_j",
    "joules_per_token", "total_flops", "flops_per_token",
    "peak_gpu_mem_mb", "peak_cpu_mem_mb",
]


# ── Run all prompts for one model / one run ───────────────────────────────────
def run_one(model_name: str, run_number: int, model, tokenizer, rows: list):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    safe_name   = model_name.replace("/", "_")
    output_path = os.path.join(OUTPUT_DIR, f"{safe_name}_run_{run_number:02d}.csv")
    total       = len(rows)

    print(f"    Output -> {output_path}")
    with open(output_path, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()

        for i, (prompt_id, prompt) in enumerate(rows, start=1):
            if i % 50 == 0 or i == 1:
                print(f"  [{i:>4}/{total}] prompt_id={prompt_id}")
            try:
                result = run_prompt(prompt_id, prompt, model, tokenizer)
                writer.writerow(result)
            except Exception as e:
                print(f"[ERROR] prompt_id={prompt_id}: {e}")
                writer.writerow({
                    "prompt_id": prompt_id,
                    **{k: "" for k in OUTPUT_FIELDS if k != "prompt_id"},
                })
            out_f.flush()  # never lose partial results

    print(f"  ==> Run {run_number} complete ({total} prompts) -> {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not DATASET_PATH or not os.path.isfile(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at: {DATASET_PATH!r}")

    with open(DATASET_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows   = [(row["id"], row["prompt"]) for row in reader
                  if row.get("prompt", "").strip()]

    print(f"==> Loaded {len(rows)} prompts from {DATASET_PATH}")

    for model_name in MODEL_NAMES:
        print(f"\n{'='*60}")
        print(f"  Loading model: {model_name}")
        print(f"{'='*60}")

        tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            offload_buffers=True, # for large models
            token=HF_TOKEN,
            no_repeat_ngram_size=3
        )
        model.eval()
        print("==> Model loaded\n")

        for run_number in range(1, NUM_RUNS + 1):
            print(f"\n--- {model_name}  |  Run {run_number}/{NUM_RUNS} ---")
            run_one(model_name, run_number, model, tokenizer, rows)

        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"\n==> Model {model_name} unloaded.")


if __name__ == "__main__":
    if not MODEL_NAMES:
        raise ValueError(
            "MODEL_NAMES env var is not set. "
            "Example: export MODEL_NAMES=meta-llama/Llama-3.1-8B-Instruct"
        )
    main()