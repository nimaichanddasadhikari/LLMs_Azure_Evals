# Ollama LLM Benchmarking — README

**Project:** Benchmarking Mistral, DeepSeek, Qwen, Gemma, Phi-3, Llama (via Ollama) on open-source datasets
**Author:** Nimai Chand Das Adhikari
**Team:** Microsoft / Azure-Migrate
**Companion notebooks:**
- [phi2_qlora_finetune.ipynb](phi2_qlora_finetune.ipynb) — Phi-2 QLoRA fine-tuning on synthetic data
- [ollama_benchmark.ipynb](ollama_benchmark.ipynb) — v1 benchmark (initial)
- [ollama_benchmark_v2.ipynb](ollama_benchmark_v2.ipynb) — v2 benchmark (CPU-tuned, some minor errors in Gemma4:26B)
- [ollama_benchmark_v3.ipynb](ollama_benchmark_v3.ipynb) — **v3 benchmark (current)** — adds reasoning-model support (Gemma 4, DeepSeek-R1)

## Quick start
1. Install [Ollama](https://ollama.com) and start the daemon (`ollama serve` or launch the desktop app).
2. Open [ollama_benchmark_v3.ipynb](ollama_benchmark_v3.ipynb).
3. Edit the `MODELS` list in the "Configure & pull models" cell.
4. Run all cells. Results land in `benchmark_run_<timestamp>/` (JSONL + CSVs + `report.json`).

See §1.3 below for what's new in v3, and §2.3 / §2.9 for the reasoning-model handling that fixes the `gemma4:26b` 0% accuracy bug from v2.

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

### v3 — `ollama_benchmark_v3.ipynb`
Fixes the `gemma4:26b` 0% accuracy bug discovered in v2's `benchmark_run_20260512_164937/` output, and generalises the fix to all reasoning models.

**Root cause of the v2 bug**
`gemma4:26b` *is* a real Ollama tag — Mixture-of-Experts, 25.8B total / 3.8B active params, Q4_K_M (~18 GB), 256K context (see https://ollama.com/library/gemma4:26b). It is **not** a phantom tag. v2 scored it at 0% because **Gemma 4 is a reasoning model with thinking mode**: its raw output begins with control tokens like `<|think|>` and a `<|channel>thought\n ... <channel|>` block before the user-visible answer. v2's MCQ config (`num_predict=2`, `stop=['\n', '.', 'Question:', 'Context:']`) terminated *inside* that thought-control preamble, so the parsed prediction was always empty and the regex scorer recorded `correct=0` for all 477 items.

**Additions over v2**
1. **Reasoning-model classifier** `is_reasoning_model(model)` covers `gemma4*` and `deepseek-r1*`.
2. **Separate `REASONING_GEN_CONFIG`** with much larger `num_predict` (MCQ=512, numeric=1024, open=512) and **no `\n` stop**, so the model can complete its hidden thought block and then emit the real answer.
3. **`clean_reasoning_response()`** strips the thought block before scoring:
   - Gemma 4: `<|channel>thought\n ... <channel|>` (paired) and truncated openers.
   - DeepSeek-R1: `<think> ... </think>` (paired) and truncated openers.
   - Lone stray control tokens (e.g. a leading `<|think|>`).
   - **Order matters:** strip *paired* blocks first, only then handle truncated openers — otherwise a greedy `.*\Z` regex eats the real answer that follows a properly closed block. (This bug was caught and fixed during v3 unit testing.)
4. **`extra_stops_for(model)` returns `[]` for reasoning models** — the v2 mistake of stopping on `<think>` is what gave DeepSeek-R1 ~0% accuracy. Stripping post-hoc is the correct approach.
5. **Per-row `raw_prediction`** captured for reasoning models (first 600 chars), so the thought-stripping is auditable.
6. **Reasoning-aware warm-up** (256 tokens instead of 4) and the empty-streak guard runs on *post-clean* text.
7. **Post-pull validation guard**: any model that didn't actually install (per `/api/tags`) is dropped from `MODELS` before the benchmark loop, so a failed pull can't silently produce 0% rows again.
8. **`empty_response_rate` and `suspect_misconfigured`** columns added to `model_summary` to make this class of bug visually obvious in future runs.

**Verified with unit tests** against realistic Gemma 4 / DeepSeek-R1 / standard outputs (closed thought blocks, truncated thought blocks, prose-with-letter answers, GSM8K-style numeric answers, and standard non-reasoning passthrough) — all pass.

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

### 2.3a Model families now in the v3 model list

| Family | CPU-friendly tags | Notes |
|---|---|---|
| **Mistral** | `mistral:7b-instruct-q4_K_M` | 32k ctx, slow on CPU but strong baseline |
| **Llama 3.2** | `llama3.2:1b-instruct-q4_K_M` | fastest on CPU |
| **Phi-3** | `phi3:mini` | 3.8B; *use plain tag* — `phi3:mini-4k-instruct-q4_K_M` returned 404 in our pull |
| **Qwen 2.5** | `qwen2.5:1.5b-instruct`, `qwen2.5:3b-instruct`, `qwen2.5:7b-instruct-q4_K_M` | very strong instruction following at small sizes |
| **Gemma 3** | `gemma3:1b`, `gemma3:4b` (CPU); `gemma3:12b`, `gemma3:27b` (GPU) | non-reasoning; standard `GEN_CONFIG` |
| **Gemma 4** | `gemma4:26b` (GPU); `gemma4:e2b`, `gemma4:e4b` (edge); `gemma4:31b` (workstation) | **reasoning model** (auto-routed to `REASONING_GEN_CONFIG`); MoE 25.8B/3.8B active for `26b`; 256K ctx |
| **DeepSeek-R1** | (disabled by default) | reasoning model; needs the §2.3 / §2.9 handling — *never* add `<think>` to stops |

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

### 2.9 An apparent 0% accuracy is almost always a *configuration* bug, not a model bug
v2's `benchmark_run_20260512_164937/` showed `gemma4:26b` at exactly 0.000 accuracy across all 477 scoreable items. Three diagnostic signals pointed at configuration, not model quality:
1. The recorded `prediction` field was the **same value (empty string) on every row** — a real model would produce *some* variety even at chance accuracy.
2. `avg completion_tokens ≈ 2` matched `num_predict` exactly — the model was being cut off at the budget.
3. The reported throughput (~30 tok/s for a supposed 26B model on CPU) was implausibly high — another sign generation never really started, only the prefill ran.

**Generalised lesson:** before re-pulling a model or assuming it's broken, always inspect a handful of raw `prediction` strings. If they are uniformly empty or contain only control tokens, the bug is in your generation config / post-processing — not the model. v3 makes this easier to spot via `empty_response_rate` and `suspect_misconfigured` columns in `model_summary`.

### 2.10 Post-process reasoning-model output with two passes (paired, then truncated)
A naive truncated-fallback regex like `r"<\|?think\|?>.*\Z"` will greedily eat the visible answer that comes *after* a properly closed thought block. The v3 cleaner does paired-block stripping **first**, and only applies a truncated-opener strip when an unmatched opener still remains. Without this ordering, 3 of 4 realistic Gemma 4 outputs cleaned to empty strings — turning the v2 bug into a *different* 0% bug.

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

## 4. Results & analysis

All runs were executed on the same Windows 11 development workstation:
- Intel Xeon-class CPU, **8 physical / 16 logical cores**, **~68 GB RAM**, no GPU available to Ollama at the time of the runs.
- Ollama daemon configured per §3.9 (`OLLAMA_NUM_PARALLEL=1`, `OLLAMA_KEEP_ALIVE=60m`, `OLLAMA_NUM_THREAD=8`).
- Generation: `temperature=0.0`, `num_ctx=2048`, `keep_alive=60m`, `request_timeout=1800 s`.
- Sampling: `SAMPLE_SEED=42`, stratified by gold label per dataset.

Numbers below are taken directly from each run's `per_model_summary.csv` and `per_model_dataset.csv`. TruthfulQA accuracy is intentionally blank — it is open-ended and not auto-scored.

### 4.1 v1 run — `benchmark_results_20260511_203433/`
- **Notebook:** [ollama_benchmark.ipynb](ollama_benchmark.ipynb)
- **Sample size:** 20 items per dataset (`N_SAMPLES=20`, not stratified, first-N slice)
- **Models:** `mistral:7b-instruct-q4_K_M`, `deepseek-r1:1.5b`, `deepseek-r1:7b`

#### Per-model summary
| Model | n | avg accuracy | avg latency (s) | median (s) | p95 (s) | tok/s |
|---|---:|---:|---:|---:|---:|---:|
| `mistral:7b-instruct-q4_K_M` | 100 | **0.475** | 11.60 | 7.54 | 24.36 | 12.6 |
| `deepseek-r1:1.5b` | 100 | 0.075 | 6.05 | 4.32 | 12.11 | 47.5 |
| `deepseek-r1:7b` | 100 | 0.025 | 16.88 | 11.47 | 38.06 | 13.0 |

#### Per (model, dataset)
| Model | GSM8K | HellaSwag | ARC-Easy | MMLU world history |
|---|---:|---:|---:|---:|
| `mistral:7b` | 0.25 | 0.35 | 0.60 | 0.70 |
| `deepseek-r1:1.5b` | 0.30 | 0.00 | 0.00 | 0.00 |
| `deepseek-r1:7b` | 0.10 | 0.00 | 0.00 | 0.00 |

#### Findings
1. **Mistral-7B is a competent CPU baseline** (47.5 % avg, strong on MMLU/ARC).
2. **DeepSeek-R1's catastrophic 0 % on the three MCQ tasks** is the v1-era manifestation of the reasoning-model bug (see §2.3). R1 was given the same 8-token MCQ budget as Mistral; it produced only the opening of its thought trace and never reached the answer letter. R1 still managed 30 % / 10 % on GSM8K because numeric tasks had `num_predict≈384`, enough to occasionally complete the trace.
3. **R1's apparent 47 tok/s** is a misleading artefact: the model is decoding `<think>...</think>` tokens fast, but those tokens contain no scoreable answer.

### 4.2 v2 run #1 — `benchmark_run_20260512_012649/` (DeepSeek-R1 stop-token regression)
- **Notebook:** [ollama_benchmark_v2.ipynb](ollama_benchmark_v2.ipynb), early revision
- **Sample size:** 477 stratified items (gsm8k 50, hellaswag 148, arc_easy 129, mmlu 100, truthfulqa 50)
- **Models:** `mistral:7b-instruct-q4_K_M`, `llama3.2:1b-instruct-q4_K_M`, `deepseek-r1:1.5b`

#### Per-model summary
| Model | n | avg accuracy | avg latency (s) | tok/s | avg total tokens |
|---|---:|---:|---:|---:|---:|
| `llama3.2:1b-instruct-q4_K_M` | 477 | 0.248 | 3.12 | 103.1 | 198.6 |
| `mistral:7b-instruct-q4_K_M` | 477 | 0.042 | 9.77 | 12.6 | 207.0 |
| `deepseek-r1:1.5b` | 477 | **0.000** | 3.42 | n/a | 171.3 |

#### Findings
1. **DeepSeek-R1 collapsed to 0.0 % across the board.** This run added `<think>` and `</think>` to the stop list to "save tokens"; R1 was halted at token 1 (the opening `<think>`) so no answer was ever produced. `avg_completion_tokens=1` on every dataset is the smoking gun.
2. **Mistral collapsed too**, but for a different reason: the MCQ stop list `['\n', '.', 'Question:', 'Context:']` plus `num_predict=2` left Mistral emitting only a punctuation token on `arc_easy` (acc 0.0) and `hellaswag` (acc 0.0). The instruction template Mistral uses tends to start its reply with newlines.
3. **Llama-3.2-1B held up** because its quantised template emits the answer letter immediately.
4. **Lesson recorded in §2.3 / §2.10:** never gate a reasoning model on its own thought-control tokens; always inspect a sample of `prediction` strings before trusting an aggregate accuracy of 0.

### 4.3 v2 run #2 — `benchmark_run_20260512_164937/` (the canonical v2 run; broad model sweep)
- **Notebook:** [ollama_benchmark_v2.ipynb](ollama_benchmark_v2.ipynb), final revision
- **Sample size:** 477 stratified items (same caps as §4.2)
- **Models:** 9 — Mistral, Llama-3.2, Phi-3, three Qwen-2.5 sizes, two Gemma-3 sizes, plus the **bogus `gemma4:26b` config**

#### Per-model summary
| Model | n | avg accuracy | avg latency (s) | tok/s | avg total tokens | ctx len |
|---|---:|---:|---:|---:|---:|---:|
| `qwen2.5:7b-instruct-q4_K_M` | 477 | **0.607** | 8.11 | 21.2 | 203.2 | 32 768 |
| `qwen2.5:3b-instruct` | 477 | 0.560 | 4.77 | 45.5 | 203.5 | 32 768 |
| `phi3:mini` | 477 | 0.548 | 6.11 | 32.3 | 214.7 | 131 072 |
| `gemma3:4b` | 477 | 0.501 | 5.85 | 31.6 | 192.8 | 131 072 |
| `qwen2.5:1.5b-instruct` | 477 | 0.494 | 3.46 | 81.0 | 202.9 | 32 768 |
| `gemma3:1b` | 477 | 0.354 | 3.25 | 89.7 | 190.5 | 32 768 |
| `llama3.2:1b-instruct-q4_K_M` | 477 | 0.248 | 3.30 | 94.8 | 198.6 | 131 072 |
| `mistral:7b-instruct-q4_K_M` | 477 | 0.042 | 9.81 | 12.6 | 207.0 |  32 768 |
| `gemma4:26b` | 477 | **0.000** ⚠ | 6.05 | 28.2 | 195.4 | 262 144 |

#### Per (model, dataset) accuracy heatmap
| Model | GSM8K | HellaSwag | ARC-Easy | MMLU world history |
|---|---:|---:|---:|---:|
| `qwen2.5:7b-instruct-q4_K_M` | 0.02 | 0.757 | 0.488 | **0.83** |
| `qwen2.5:3b-instruct` | 0.02 | 0.682 | 0.442 | 0.80 |
| `phi3:mini` | 0.12 | 0.649 | 0.481 | 0.70 |
| `gemma3:4b` | 0.14 | 0.554 | 0.419 | 0.71 |
| `qwen2.5:1.5b-instruct` | 0.02 | 0.554 | 0.457 | 0.69 |
| `gemma3:1b` | 0.12 | 0.392 | 0.341 | 0.43 |
| `llama3.2:1b-instruct-q4_K_M` | 0.00 | 0.270 | 0.264 | 0.32 |
| `mistral:7b-instruct-q4_K_M` | 0.16 | 0.000 | 0.000 | 0.10 |
| `gemma4:26b` | **0.00** | **0.00** | **0.00** | **0.00** |

#### Findings
1. **Qwen-2.5 dominates the size-for-size comparison.** Even the 1.5B Qwen beats Llama-3.2-1B by ~25 absolute points; Qwen-7B at 60.7 % avg is the strongest single model and the only one above 80 % on MMLU.
2. **Phi-3-mini (3.8B) punches above its weight** at 54.8 %, second only to Qwen-3B in the sub-7B tier.
3. **GSM8K is universally hard for non-reasoning models on this CPU budget** (best: 0.16 for Mistral). `num_predict=128` plus `temperature=0` is enough to *try* chain-of-thought but rarely enough to land the right number.
4. **Mistral-7B's MCQ collapse** repeats the §4.2 issue (template emits a leading newline that the `'\n'` stop fires on). Mistral's MMLU score (0.10) is much worse than its v1 score (0.70 on the same task) precisely because v1 used `num_predict=8` for MCQ — see §2.2 for the trade-off.
5. **`gemma4:26b` 0.000 across all 4 scoreable datasets.** Investigation of `results.jsonl` showed the `prediction` field was an empty string on every single one of the 477 rows — the diagnostic pattern flagged in §2.9. Root cause: Gemma 4 is a reasoning model; v2's `num_predict=2` and `stop=['\n', ...]` truncated inside the `<|think|>...<|channel>thought\n ... <channel|>` preamble. This is exactly the bug v3 fixes.

### 4.4 v3 partial validation run — `benchmark_run_20260513_130928/` (gemma4:26b)
- **Notebook:** [ollama_benchmark_v3.ipynb](ollama_benchmark_v3.ipynb)
- **Sample size:** the run was an early-validation slice — only the 50 GSM8K items were attempted; 38 completed before a manual stop.
- **Model:** `gemma4:26b` only, routed through `REASONING_GEN_CONFIG` (num_predict 1024 for numeric, no `\n` stop).

#### Per-model summary (raw output of v3)
| Model | n | avg accuracy | avg latency (s) | median | p95 | tok/s | avg total tokens | empty rate | flagged? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `gemma4:26b` | 38 | **0.421** | 66.85 | 77.75 | 80.65 | 14.06 | 964.3 | 0.579 | `suspect_misconfigured=True` |

#### Per (model, dataset)
| Model | dataset | n | accuracy | avg latency | tok/s | avg prompt | avg completion | avg total | ctx util |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `gemma4:26b` | gsm8k | 38 | 0.421 | 66.85 | 14.1 | 95.4 | 868.9 | 964.3 | 0.4 % |

#### Findings
1. **The v3 fix works.** v2 reported `gemma4:26b = 0/477` (0.0 %) on the same `num_ctx=2048` CPU setup. v3 reports **42.1 % on GSM8K** with non-empty cleaned predictions — the model is genuinely solving problems, not silently producing empty strings.
2. **Reasoning trace cost is real.** Average completion length jumped from 73.9 tokens (v2 truncated) to **868.9 tokens**, almost a 12× increase. At 14 tok/s on CPU that translates into ~67 s per item — exactly what §3.2 warned about.
3. **The 57.9 % `empty_response_rate` and `suspect_misconfigured=True` flag are *expected*** for this run, not a regression: those rows are GSM8K items where the thought block didn't finish within `num_predict=1024`, leaving a truncated opener that `clean_reasoning_response()` correctly drops. Those items score 0 and inflate the empty-rate. The 42.1 % accuracy figure is computed only over the 38 completed rows, so the metric itself is honest; the flag is a *workload* signal, not a *configuration* signal. Increasing `num_predict` to 2048 (or `num_ctx` to 4096 to match) should recover most of those rows on a GPU.
4. **`avg_ctx_utilization=0.004`** confirms the 256K context window is wildly over-provisioned for our workload — Gemma 4's strength is long context, which our short-prompt benchmarks don't exercise. Worth adding a long-context dataset (LongBench, RULER) to a future v4.
5. **Throughput parity with `gemma3:27b` would have been ~0.3–0.8 tok/s** per published reports for dense 27B on CPU. Gemma 4's MoE (3.8B active) sustains ~14 tok/s — roughly 20–40× faster. This is a strong argument for prioritising MoE architectures on memory-rich, compute-poor hosts.

### 4.5 Cross-version comparison: same model, different configs

| Model | v1 (203433) | v2-#1 (012649) | v2-#2 (164937) | v3 (130928) |
|---|---:|---:|---:|---:|
| `mistral:7b-instruct-q4_K_M` | 0.475 | 0.042 | 0.042 | (not run) |
| `llama3.2:1b-instruct-q4_K_M` | — | 0.248 | 0.248 | (not run) |
| `deepseek-r1:1.5b` | 0.075 | 0.000 | (disabled) | (disabled) |
| `qwen2.5:7b-instruct-q4_K_M` | — | — | **0.607** | (not run) |
| `gemma4:26b` | — | — | **0.000** ⚠ | **0.421** ✓ |

**Headline:** the same physical model can score 0 % or 42 % on the same prompts depending entirely on the generation config and post-processing. v3 closes the reasoning-model gap; the next step is to re-run the full v2-#2 model sweep on v3 so that `mistral`, `llama3.2`, `qwen2.5*`, `phi3`, `gemma3*` and `gemma4:26b` are all measured under the new harness on identical inputs.

### 4.6 Performance envelope on this CPU host
From the v2-#2 numbers (the most complete sweep):

| Tier | Models | Throughput band | Suitable for |
|---|---|---|---|
| **Fast (≥ 80 tok/s)** | `llama3.2:1b`, `qwen2.5:1.5b`, `gemma3:1b` | 90–105 tok/s | Real-time MCQ, latency-sensitive UX |
| **Mid (20–50 tok/s)** | `qwen2.5:3b`, `phi3:mini`, `gemma3:4b`, `qwen2.5:7b`, `gemma4:26b` MoE | 21–45 tok/s | Batched eval, agentic tasks |
| **Slow (< 15 tok/s)** | `mistral:7b-instruct-q4_K_M` | 12–13 tok/s | Avoid on CPU unless quality-critical |

**Memory:** none of the models in the v2-#2 sweep moved system RAM by more than ~13 MB per call (`avg_ram_delta_gb ≈ 0`), confirming that quantised weights stay paged-in once warm. Gemma 4's 18 GB resident footprint did not crash this 68 GB host.

---

## 5. Future work

- **Concurrency across models** when total VRAM permits (separate Ollama clients).
- **Token-cost dashboards** (matplotlib → Plotly) for interactive exploration.
- **Add more datasets**: BBH, IFEval, HumanEval (code), MT-Bench (judge-graded).
- **Judge-based scoring** for `truthfulqa` and other open-ended tasks (use a stronger judge model via the same Ollama pipeline, or an API model).
- **Cost models**: convert `tokens_per_sec` + `wall_seconds` into $-per-1k-tokens equivalents for hosted alternatives.
- **CI run**: a tiny `DATASET_CAPS` (10 each) smoke-test that runs on every push.
- **Phi-2 QLoRA → full evaluation loop**: feed the fine-tuned adapter through the benchmark harness via a custom Ollama Modelfile for end-to-end "tune → bench" cycles.

---

## 6. File index
- [phi2_qlora_finetune.ipynb](phi2_qlora_finetune.ipynb) — fine-tune Phi-2 with QLoRA on synthetic data
- [ollama_benchmark.ipynb](ollama_benchmark.ipynb) — v1 benchmark
- [ollama_benchmark_v2.ipynb](ollama_benchmark_v2.ipynb) — v2 benchmark with CPU tuning
- [ollama_benchmark_v3.ipynb](ollama_benchmark_v3.ipynb) — **v3 benchmark (current)** — adds reasoning-model support
- `benchmark_results_20260511_203433/` — v1 run (Mistral + DeepSeek-R1, 100 items each)
- `benchmark_run_20260512_012649/` — v2 run #1 (DeepSeek-R1 stop-token regression captured here)
- `benchmark_run_20260512_164937/` — v2 run #2 (canonical 9-model sweep; surfaced the `gemma4:26b` 0 % bug)
- `benchmark_run_20260513_130928/` — v3 validation slice (`gemma4:26b` GSM8K @ 42.1 %)
- `benchmark_run_<timestamp>/` — generic per-run output (raw JSONL, CSV summaries, `report.json`, `device_info.json`, `model_capabilities.csv`)
- This document — `README.md`
