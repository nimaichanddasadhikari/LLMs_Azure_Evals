# Ollama LLM Benchmarking — Learnings & Recommendations

**Project:** Benchmarking Mistral & DeepSeek (and other Ollama-served LLMs) on open-source datasets
**Author:** Nimai Chand Das Adhikari
**Team:** Microsoft / Azure-Migrate
**Companion notebooks:**
- [phi2_qlora_finetune.ipynb](phi2_qlora_finetune.ipynb) — Phi-2 QLoRA fine-tuning on synthetic data
- [ollama_benchmark.ipynb](ollama_benchmark.ipynb) — v1 benchmark (initial)
- [ollama_benchmark_v2.ipynb](ollama_benchmark_v2.ipynb) — v2 benchmark (current)

---

## 1. Version history

### v1 — `ollama_benchmark.ipynb`
First end-to-end benchmark harness.

**What it does**
- Verifies the local Ollama daemon at `http://localhost:11434`
- Captures device specs: OS, CPU cores, RAM, disk, GPU (`pynvml` → `GPUtil` fallback)
- Pulls a configurable list of Mistral & DeepSeek variants
- Loads small slices (`N_SAMPLES=20`) of GSM8K, HellaSwag, ARC-Easy, TruthfulQA, MMLU
- For every (model, item): records wall-clock latency, Ollama-reported `total/load/prompt_eval/eval` durations, **tokens/sec**, prompt + completion tokens, RAM delta, GPU memory delta
- Auto-scores MCQ + GSM8K via regex; TruthfulQA stays open-ended
- Outputs per-model and per-(model, dataset) summaries, latency/throughput boxplots, accuracy heatmap
- Persists CSV + JSON in a timestamped folder

**What was missing / problems found**
- No insight into **how big a context the model can take** — important for prompt-heavy workloads.
- Sample sizes were arbitrary (`[:20]`) and not stratified by class.
- No way to **resume** a partial run after a crash or interrupt.
- No early-exit if a model was hopelessly slow.
- File became corrupted by overlapping edits during in-place patching → led to v2 being a clean rewrite.

---
Dataset	Full        size	    Why it's slow
GSM8K (test)	    1,319	    Each item generates up to 384 tokens of chain-of-thought
HellaSwag (val)	    10,042	    Big — even at 8 tokens/call, 10k calls hurts
ARC-Easy (test)	    2,376	    MCQ but still thousands of calls
TruthfulQA (val)	817	        Open-ended, up to 256 tokens each
MMLU subject (test)	~237	    Manageable
---

### v2 — `ollama_benchmark_v2.ipynb`
Clean rewrite with everything v1 was missing, plus CPU optimizations.

**Additions over v1**
1. **Section 4b — model capability probe** via `/api/show`:
   - `context_length` (max total tokens)
   - parameter count, quantization, family, on-disk size
   - exposed as `MODEL_INFO` dict and `model_capabilities.csv`
2. **Per-call `total_tokens`** (`prompt_tokens + completion_tokens`) and **`context_utilization`** (= `total_tokens / context_length`) on every result row.
3. **Stratified sampling** via `_take()` and `DATASET_CAPS` (random shuffle + per-label balancing using `SAMPLE_SEED=42`).
4. **Resumable runs**: results are appended to `RUN_DIR/results.jsonl`. Re-running with `BENCH_RUN_DIR=...` skips already-completed `(model, id)` pairs.
5. **Early-stop guard** `EARLY_STOP_SECONDS` abandons a model whose moving-average wall latency exceeds the threshold (after ≥10 calls).
6. **Per-task generation budgets** (`GEN_CONFIG`) with **stop sequences** to prevent rambling outputs.
7. **DeepSeek-R1 thinking suppression** — `extra_stops_for(model)` adds `<think>`/`</think>` stops automatically.
8. **`keep_alive`** sent on every request so the model isn't unloaded between calls.
9. Plots expanded to 2×2: latency, throughput, **total tokens per call**, accuracy heatmap.
10. `report.json` records `dataset_caps` and `gen_config` for reproducibility.

**v2 patch — CPU tuning**
- New **section 3b** with PowerShell instructions for `OLLAMA_NUM_PARALLEL`, `OLLAMA_KEEP_ALIVE`, `OLLAMA_NUM_THREAD`.
- `MODELS` switched to CPU-friendly: `mistral:7b-instruct-q4_K_M`, `deepseek-r1:1.5b`, `llama3.2:1b-instruct-q4_K_M`, `phi3:mini-4k-instruct-q4_K_M`. Removed `deepseek-r1:7b`.
- `ollama_generate` now passes `num_thread = physical cores` and `num_ctx = 2048`; `keep_alive='60m'`; `request_timeout=1800`.
- `DATASET_CAPS` halved: `gsm8k=50, hellaswag=150, arc_easy=150, truthfulqa=50, mmlu=100` (~500 items total).
- `GEN_CONFIG` tightened: MCQ `num_predict=2`, numeric `128`, open `32`.
- `EARLY_STOP_SECONDS` raised to `600` so slow CPU runs don't bail.

---

## 2. Key lessons learned

### 2.1 The cost of a benchmark is `n_calls × tokens_per_call × seconds_per_token`
On CPU, `seconds_per_token` is large (50 ms–1 s). The biggest wins always come from:
1. Reducing `n_calls` — **smart, stratified sub-sampling** beats `[:20]`.
2. Reducing `tokens_per_call` — **`num_predict` + `stop` sequences** are the biggest single optimisation. MCQ tasks should produce **2 tokens**, not 256.
3. Reducing reload overhead — **`keep_alive`** keeps the model warm between calls.

### 2.2 Stop sequences > generous `num_predict`
Models will fill `num_predict` if you let them. Adding `stop=['\n', '.', 'Question:', 'Context:']` for MCQ trims wasted tokens by 90%+.

### 2.3 Reasoning models (DeepSeek-R1) — *do not just stop on `<think>`*
`deepseek-r1:*` emits `<think>...</think>` traces. **Initial mistake:** we added `<think>` and `</think>` to the stop list to "save tokens". Result on the first benchmark run was effectively **0% accuracy across all datasets** for `deepseek-r1:1.5b` — generation was halted at token 1 (the opening `<think>`), so no answer was ever produced. The recorded `prediction` was empty / `<think>` only and the regex parsed nothing.

**Correct handling:**
1. **Do NOT add `<think>` to stops.** Let the think block complete.
2. Increase `num_predict` substantially for R1: at least **512** for MCQ, **1024+** for GSM8K. Reasoning traces are long.
3. Post-process the response before scoring:
   ```python
   import re
   answer = re.sub(r"<think>.*?</think>", "", response, flags=re.S).strip()
   ```
4. Expect 5–10× the latency of a comparable non-reasoning model. Budget accordingly.
5. For pure throughput / MCQ benchmarks, prefer a non-reasoning model (`qwen2.5:*-instruct`, `llama3.2:1b-instruct`, `gemma3:*`) and benchmark R1 separately on reasoning-heavy datasets only (GSM8K, MATH, BBH).

### 2.3a Model families now in the v2 model list
Updated `MODELS` covers four families so we can compare apples-to-apples:

| Family | CPU-friendly tags | Notes |
|---|---|---|
| **Mistral** | `mistral:7b-instruct-q4_K_M` | 32k ctx, slow on CPU but strong baseline |
| **Llama 3.2** | `llama3.2:1b-instruct-q4_K_M` | fastest on CPU |
| **Phi-3** | `phi3:mini` | 3.8B; *use plain tag* — `phi3:mini-4k-instruct-q4_K_M` returned 404 in our pull |
| **Qwen 2.5** | `qwen2.5:1.5b-instruct`, `qwen2.5:3b-instruct`, `qwen2.5:7b-instruct-q4_K_M` | added in this revision; very strong instruction following at small sizes |
| **Gemma 3** | `gemma3:1b`, `gemma3:4b` (CPU); `gemma3:12b`, `gemma3:27b` (GPU) | **user requested "Gemma 4 26B"** — no `gemma4` tag exists on Ollama yet; closest is `gemma3:27b` (~17 GB q4, GPU only). Swap to `gemma4:*` if/when published |
| **DeepSeek-R1** | (disabled by default) | reasoning model; needs the §2.3 handling — do not benchmark with `<think>` in stops |

### 2.4 Always probe the model's context window
`/api/show` returns `model_info` with arch-prefixed keys like `llama.context_length`. Don't hard-code 4096 — Mistral instruct is 32k, Phi-3-mini-4k is 4k, etc.

### 2.5 Resumability is non-negotiable for long runs
Anything that runs for hours **will** be interrupted (laptop sleep, OS update, kernel restart). Append-only JSONL with stable IDs (`f"{dataset}::{i}"`) is the simplest reliable mechanism.

### 2.6 Per-call instrumentation must be cheap
Sampling `psutil.virtual_memory()` and `pynvml` once before/after each call is essentially free. Calling `nvidia-smi` via subprocess is not — it adds 100+ ms per call.

### 2.7 Bitsandbytes is GPU-only and platform-fragile
QLoRA training in [phi2_qlora_finetune.ipynb](phi2_qlora_finetune.ipynb) needs CUDA. On Windows native, install often fails — prefer WSL2 + Linux wheels, Colab, or a Linux VM.

### 2.8 Notebook editing is risky for big files
While iteratively patching v1, multiple `replace_string_in_file` ops on a JSON `.ipynb` produced a corrupted file (cell sources mashed together). Lessons:
- For non-trivial structural changes to `.ipynb`, prefer creating a v2 file fresh.
- Use `ConvertFrom-Json` / `ConvertTo-Json` (PowerShell) for structural inserts.
- Keep the notebook open in VS Code while editing — autosave + diff lets you catch breakage fast.

---

## 3. Detailed recommendations

### 3.1 Recommended workflow per model
1. **Pull** with `ollama pull <name>` (notebook does this for you).
2. **Probe** `/api/show` to record `context_length`, params, quantization, file size.
3. **Warm up** with a 4-token call so the first measured call doesn't include load time.
4. **Loop** datasets, appending to JSONL. Tag each row with `model`, `id`, `dataset`, `task_type`, `gold`, `prediction`, all timing fields, and tokens.
5. **Aggregate** with pandas: per-model summary + per-(model, dataset) accuracy & latency.
6. **Persist** raw + summaries + `device_info.json` + `report.json` in a timestamped run dir.

### 3.2 Hardware-aware configuration

| Resource | `MODELS` | `DATASET_CAPS` total | `GEN_CONFIG` MCQ `num_predict` | `num_ctx` | `EARLY_STOP_SECONDS` |
|---|---|---|---|---|---|
| **CPU only (8-16 GB RAM)** | 1B–3B q4 only | ~500 | 2 | 2048 | 600 |
| **CPU + 32 GB RAM** | up to 7B q4 | ~1,000 | 2–4 | 2048–4096 | 600 |
| **GPU 8–12 GB VRAM** | 7B q4 / 8B q4 | ~2,000 | 4 | 4096 | 120 |
| **GPU 24+ GB VRAM** | 7B fp16 / 13B / Mixtral q4 | full splits | 4 | 8192+ | 60 |

### 3.3 Prompt patterns by task type
- **MCQ**: instruct *"Respond with only the letter, no reasoning."* Set `num_predict=2`, stop on `\n`/`.`/`Question:`/`Context:`.
- **Numeric (GSM8K)**: ask the model to end with `'#### <number>'`. Stop on `Question:` and `\n\n`. Cap at 128 tokens on CPU, 256 on GPU.
- **Open-ended (TruthfulQA)**: ask for *one sentence*. Stop on `\n`. Cap at 32–64 tokens.

### 3.4 Sampling strategy
- Use **stratified random sampling by `gold` label** for MCQ to avoid imbalanced subsets giving misleading accuracy.
- Pin `SAMPLE_SEED` for reproducibility.
- For confidence-interval-grade results you need **≥300 items per dataset**. Smaller is fine for **relative** model comparisons on the same subset.

### 3.5 Metrics to record per call
| Field | Why |
|---|---|
| `wall_seconds` | True user-observed latency |
| `load_duration_s` | Cold-start cost (should be ~0 after warm-up) |
| `prompt_eval_s`, `prompt_tokens` | Prefill cost, scales with prompt length |
| `completion_eval_s`, `completion_tokens` | Decoding cost, scales with output |
| `tokens_per_sec` | `eval_count / eval_duration` — generation throughput |
| `total_tokens` | Prompt + completion |
| `context_utilization` | How close to the model's max context |
| `ram_before/after/delta_gb` | System RAM impact |
| `gpu_before/after/delta_gb` | VRAM impact (None on CPU) |
| `correct`, `parsed_guess` | Quality |

### 3.6 Aggregations worth reporting
- **Per model**: `n`, mean accuracy, mean / median / p95 latency, mean tokens/sec, mean & max `total_tokens`, mean RAM/GPU delta, `context_length`.
- **Per (model, dataset)**: same metrics, plus `avg_ctx_utilization` to spot prompts approaching the model's limit.

### 3.7 Plots that have proven useful
1. Latency boxplot per model (skew shows outliers / cold starts).
2. Tokens/sec boxplot per model (shows raw decode speed).
3. Total tokens per call boxplot (catches runaway thinking traces).
4. Accuracy heatmap (model × dataset).

### 3.8 Reproducibility checklist
- [ ] `SAMPLE_SEED` set
- [ ] `temperature=0.0`
- [ ] `gen_config` and `dataset_caps` saved in `report.json`
- [ ] `device_info.json` saved
- [ ] `model_capabilities.csv` saved
- [ ] All raw rows saved as JSONL (one row per call, append-only)

### 3.9 Operational tips
- On Windows, set `OLLAMA_*` env vars **before** launching the Ollama desktop app (it spawns the daemon).
- `OLLAMA_NUM_PARALLEL=1` for CPU; otherwise threads compete.
- `OLLAMA_NUM_THREAD = physical cores` (not logical) — hyperthreading often hurts.
- Use `keep_alive='60m'` on long runs; default is 5 min.
- Force-evict between models with `keep_alive: 0` to free RAM/VRAM.

### 3.10 Pitfalls to avoid
- ❌ Don't `[:N]` — that's not a sample, it's the first N. Shuffle.
- ❌ Don't measure first-call latency without a warm-up — `load_duration` will dominate.
- ❌ Don't enable two models concurrently on a single GPU unless you've verified they fit.
- ❌ Don't run reasoning models on MCQ without `<think>` stops — accuracy unchanged, latency 5–10×.
- ❌ Don't keep results only in memory — persist incrementally.
- ❌ Don't trust nominal "context length" without testing — sometimes Ollama caps `num_ctx` to a smaller default.

---

## 4. Future work

- **Concurrency across models** when total VRAM permits (separate Ollama clients).
- **Token-cost dashboards** (matplotlib → Plotly) for interactive exploration.
- **Add more datasets**: BBH, IFEval, HumanEval (code), MT-Bench (judge-graded).
- **Judge-based scoring** for `truthfulqa` and other open-ended tasks (use a stronger judge model via the same Ollama pipeline, or an API model).
- **Cost models**: convert `tokens_per_sec` + `wall_seconds` into $-per-1k-tokens equivalents for hosted alternatives.
- **CI run**: a tiny `DATASET_CAPS` (10 each) smoke-test that runs on every push.
- **Phi-2 QLoRA → full evaluation loop**: feed the fine-tuned adapter through the benchmark harness via a custom Ollama Modelfile for end-to-end "tune → bench" cycles.

---

## 5. File index
- [phi2_qlora_finetune.ipynb](phi2_qlora_finetune.ipynb) — fine-tune Phi-2 with QLoRA on synthetic data
- [ollama_benchmark.ipynb](ollama_benchmark.ipynb) — v1 benchmark
- [ollama_benchmark_v2.ipynb](ollama_benchmark_v2.ipynb) — v2 benchmark with CPU tuning
- This document — `LEARNINGS_AND_RECOMMENDATIONS.md`
