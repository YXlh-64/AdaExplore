## How to collect the skill memory

This repo uses two complementary skill memories, both injected into the same prompts:

- **Negative memory (experience guidance)**: *log-grounded constraints* distilled from **failed** kernels (compile/runtime errors).
  Format is a line-based memory file: `You cannot ...||<score (frequency)>`.
- **Positive memory (best-practice guidance)**: *log-grounded techniques* distilled from **successful** kernels (correct, and at least as fast as the PyTorch baseline by default).
  Format is the same line-based file, with a different rule shape: `You should ...||<score (frequency)>`.

Below is the end-to-end guidance to collect and use both.

---

## Prerequisites: what logs look like

Both collectors assume you already have an agent run under `outputs/...`, with per-problem folders containing step artifacts like:

- `step_<k>.py`: generated kernel code at step \(k\)
- `step_<k>_metrics.(txt|json)`: evaluation metrics (compiled/correctness/runtime/speedup, etc.)
- `step_<k>_prompt.txt`: the prompt used at step \(k\) (optional but recommended)

Example:

- `outputs/example_run/2_1/step_7.py`
- `outputs/example_run/2_1/step_7_metrics.json`
- `outputs/example_run/2_1/step_7_prompt.txt`

---

## Negative memory collection ("you cannot...")

### What it is

Negative memory is extracted from **error logs only** (CE/RE-type failures where `correctness=False`).  
The extractor asks an LLM to produce **exactly one** minimal rule:

- Must start with: `You cannot ...`
- Must be strictly grounded in the error message
- If unclear, it outputs `no guidance` and will be skipped

The collected memory is stored in a line-based memory file (for example under `results/memory/`), where each line is:

```
<knowledge_item>||<score>
```

The `<score>` is a simple frequency-like counter; it increases when the new item is judged as a duplicate of an existing one (LLM-based duplicate judge in `skill_memory/deduplicate_knowledge.py`).

### How it is used by the agent

At runtime, the proposer prompt optionally injects items whose score is above a threshold:

- Memory file path: `--general_memory_path`
- Enable updating (write-back): `--memory_update`
- Injection threshold (in proposer prompt): `--knowledge_1_threshold` (default: `3`)

So, **collect many**, then **control how much gets injected** with the threshold.

```bash
  --filter-max-difference
```

### Option A: update skill memory during the agent run (online update, recommended)

When running the agent via `agent/agent_entry.py`, you can enable auto-update:

```bash
python agent/agent_entry.py \
  --config config/<your_config>.yaml \
  --general_memory_path results/memory/general_memory_v1_200.txt \
  --memory_update \
  --knowledge_1_threshold 3
```

Notes:

- This path uses a file lock (`<memory>.lock`) to avoid concurrent write corruption.
- The update happens after a problem finishes, by calling `skill_memory/skill_memory.update_memory(...)` on that problem’s log folder.
- As of this version, the **same** `--memory_update` flag also builds the positive
  ("you should...") memory described below, against a sibling file
  (`results/memory/general_memory_v1_200_positive.txt` in this example) unless you
  override it with `--general_memory_positive_path`. See "Positive memory collection" below.

### Option B: collect skill memory from an existing `outputs/<run>/<problem>/...` folder

If you have a specific run folder (e.g. `outputs/example_run`) and want to update a memory file from the errors inside it, you can run:

```bash
python skill_memory/skill_memory.py \
  --log-dir outputs/example_run \
  --knowledge-store-path results/memory/general_memory_v1_200.txt \
  --server azure \
  --model-name gpt-5-mini \
  --max-logs 3000 \
  --seed 42
```

Notes:

- It scans `outputs/<run>/*/*_metrics.(txt|json)` and only keeps logs where **`correctness=False`**.
  This is the default `--mode errors`; see "Positive memory collection" below for `--mode successes`.
- To drop “max_difference” related cases (i.e., correctness issue cases, often noisy), add:

```bash
  --filter-max-difference
```

---

## Positive memory collection ("you should...")

### What it is

Positive memory is the mirror image of negative memory: it's extracted from **successful
logs only** — `correctness=True` **and** `fast_p` at or above a minimum speedup
(`--min-speedup` / `--min_speedup_for_best_practice`, default `1.0`, i.e. at least as fast
as the PyTorch baseline). The extractor asks the LLM for **exactly one** minimal,
code-grounded rule:

- Must start with: `You should ...`
- Must describe a concrete technique actually present in the successful kernel (a fusion,
  a tiling/block-size choice, a masking pattern, a numerically-stable reformulation, ...) —
  not vague praise
- If the code doesn't clearly demonstrate one well-defined technique, it outputs
  `no guidance` and is skipped

It's stored in the same `<knowledge_item>||<score>` format, deduplicated against existing
entries the same way (LLM-based duplicate judge), just in its own file so "must not do X"
and "should do Y" rules don't get mixed together in one undifferentiated list.

### How it is used by the agent

Same injection mechanism as negative memory, via its own path/threshold:

- Memory file path: `--general_memory_positive_path` (auto-derived from
  `--general_memory_path` if `--memory_update` is set and this isn't given explicitly)
- Enable updating (write-back): same `--memory_update` flag as negative memory
- Minimum speedup to qualify as a positive example: `--min_speedup_for_best_practice` (default `1.0`)
- Injection threshold (in proposer/reviser/tuner prompts): same `--knowledge_1_threshold` as negative memory

### Option A: online update during the agent run (recommended, automatic)

No extra flags needed beyond what you already pass for negative memory — `--memory_update`
builds both files:

```bash
python agent/agent_entry.py \
  --config config/<your_config>.yaml \
  --general_memory_path results/memory/general_memory_v1_200.txt \
  --memory_update \
  --knowledge_1_threshold 3
```

This also writes `results/memory/general_memory_v1_200_positive.txt`. Pass
`--general_memory_positive_path` explicitly if you want a different location.

### Option B: collect positive memory from an existing run folder

Same script as negative memory, with `--mode successes` and a **different**
`--knowledge-store-path` (don't point it at the negative file — the two memories must stay
in separate files):

```bash
python skill_memory/skill_memory.py \
  --log-dir outputs/example_run \
  --knowledge-store-path results/memory/general_memory_v1_200_positive.txt \
  --server azure \
  --model-name gpt-5-mini \
  --mode successes \
  --min-speedup 1.0 \
  --max-logs 3000 \
  --seed 42
```

---

## Refreshing the PyTorch layer/activation list

`synthesis/generate_data.py` (composite prompt style) reads `skill_memory/pytorch_layers_activations.json`, a snapshot of all `torch.nn` layers and activations scraped from the official PyTorch docs. The snapshot is committed in this repo, so you do **not** need to run the scraper to reproduce the paper results.

Re-run it only if you want to refresh against a newer PyTorch version:

```bash
python skill_memory/scrape_pytorch_layers.py
```

This fetches `https://docs.pytorch.org/docs/stable/nn.html` and overwrites both `pytorch_layers_activations.json` (machine-readable) and `pytorch_layers_activations.txt` (human-readable) in this directory.

---

## Quick sanity checks

- **Negative memory**: open your memory file (for example `results/memory/general_memory_v1_200.txt`) and confirm each line looks like:
  - starts with `You cannot ...`
  - ends with `||<number>`
- **Positive memory**: open its sibling file (for example `results/memory/general_memory_v1_200_positive.txt`) and confirm each line looks like:
  - starts with `You should ...`
  - ends with `||<number>`