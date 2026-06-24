# Reproducing the OpenRouter Sample-Benchmark Workflow

This is the runbook for the local experiment setup we built on top of upstream
AdaExplore: running the **Explore** stage through a free OpenRouter model,
against a small, fixed, reproducible sample drawn from the *unmodified*
KernelBench level-2 benchmark, then evaluating and comparing results across
settings/checkpoints as you iterate.

It assumes a single consumer GPU (this machine: RTX 2050, 4GB VRAM, Ampere /
compute capability 8.6) rather than the paper's A6000 (48GB), so everything
here is scoped down in *quantity* (fewer problems, fewer steps) without
changing the benchmark's task definitions, shapes, or distribution.

---

## 0. One-time code change already applied

`agent/inference_server.py` did not support OpenRouter out of the box (only
Azure/OpenAI/Anthropic). We added a branch to `create_inference_server()`:

```python
elif server_type == "openrouter":
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY must be set when server_type='openrouter'.")
    return openai.OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
```

and added `"openrouter"` to the `--server_type` argparse `choices` in:
`agent/agent_entry.py`, `agent/large_loop.py`, `agent/mcts.py`,
`agent/small_loop.py`, `synthesis/generate_data.py`, `agent/inference_server.py`.

This is already committed to your working tree — nothing to redo. If you ever
diff against upstream AdaExplore, this is the delta that makes
`--server_type openrouter` valid.

---

## 1. One-time environment setup

```bash
cd "AdaExplore"
conda env create -f environment.yml      # creates the `adaexplore` env (Python 3.12, torch 2.5.0+cu124)
conda activate adaexplore
```

Verify CUDA works:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Get a key from [openrouter.ai/keys](https://openrouter.ai/keys), then in every
shell you use to run the agent:
```bash
export OPENROUTER_API_KEY=sk-or-...
```
(Never commit this key or put it in a config file — keep it as an env var only.)

---

## 2. Start the local judge (foreground, its own terminal)

The judge (`online_judge/app_with_queue.py`) is what actually compiles and
times each generated kernel on your GPU. Run it in its own terminal/pane and
leave it running for the whole session:

```bash
conda activate adaexplore
HOST=127.0.0.1 AVAILABLE_GPUS=0 bash online_judge/start_server.sh
```

We bind to `127.0.0.1` (not the script's default `0.0.0.0`) because
`/evaluate` executes arbitrary submitted code with no auth — no reason to
expose that beyond this machine.

Sanity check from a second terminal:
```bash
curl http://127.0.0.1:12017/health
```
Should report `"status":"healthy"` and your GPU under `devices`.

---

## 3. The sample benchmark: which problems, and why

File: `config/smoke/test_list_kbl2_sample5.txt`
```
2 4,15,36,82,95
```
This is the standard `test_list` syntax (`"<level> <id1>,<id2>,..."`) the
paper's own `config/test_list/test_list_2.txt` uses — just naming 5 IDs
instead of the full 1–100 range. **No task file was modified** — these are
the original KernelBench level-2 problems, referenced as-is.

The 5 IDs are an unbiased, reproducible sample:
```bash
python3 -c "
import random
random.seed(42)
print(sorted(random.sample(range(1, 101), 5)))
"
# -> [4, 15, 36, 82, 95]
```

### Problem documentation

| ID | File | Operations | Input shape | VRAM risk on 4GB |
|----|------|------------|-------------|-------------------|
| 4  | `4_Conv2d_Mish_Mish.py` | Conv2d → Mish → Mish | `(64, 64, 256, 256)` | **High** — see note below |
| 15 | `15_ConvTranspose3d_BatchNorm_Subtract.py` | ConvTranspose3d → BatchNorm3d → subtract spatial mean | `(16, 16, 16, 32, 32)` | Low |
| 36 | `36_ConvTranspose2d_Min_Sum_GELU_Add.py` | ConvTranspose2d → min(dim=1) → sum(dim=2) → GELU → +bias | `(16, 64, 128, 128)` | Moderate |
| 82 | `82_Conv2d_Tanh_Scaling_BiasAdd_Max.py` | Conv2d → tanh → ×scale → +bias → MaxPool2d | `(128, 8, 256, 256)` | Moderate–High |
| 95 | `95_Matmul_Add_Swish_Tanh_GELU_Hardtanh.py` | Linear(8192→8192) → +bias → Swish → Tanh → GELU → Hardtanh | `(1024, 8192)` | Low (one big matmul, no spatial blow-up) |

Full reference definitions (unmodified, copied here for convenience —
canonical copies live in `datasets/KernelBench/level2/`):

<details>
<summary>4_Conv2d_Mish_Mish.py</summary>

```python
class Model(nn.Module):
    """Simple model that performs a convolution, applies Mish, and another Mish."""
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = torch.nn.functional.mish(x)
        x = torch.nn.functional.mish(x)
        return x

batch_size, in_channels, out_channels, height, width, kernel_size = 64, 64, 128, 256, 256, 3
```
</details>

<details>
<summary>15_ConvTranspose3d_BatchNorm_Subtract.py</summary>

```python
class Model(nn.Module):
    """A 3D convolutional transpose layer followed by Batch Normalization and subtraction."""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        x = x - torch.mean(x, dim=(2, 3, 4), keepdim=True)
        return x

batch_size, in_channels, out_channels = 16, 16, 32
depth, height, width = 16, 32, 32
kernel_size, stride, padding = 3, 2, 1
```
</details>

<details>
<summary>36_ConvTranspose2d_Min_Sum_GELU_Add.py</summary>

```python
class Model(nn.Module):
    """Convolution transpose, minimum, sum, GELU, addition."""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv_transpose(x)
        x = torch.min(x, dim=1, keepdim=True)[0]
        x = torch.sum(x, dim=2, keepdim=True)
        x = torch.nn.functional.gelu(x)
        x = x + self.bias
        return x

batch_size, in_channels, out_channels = 16, 64, 128
height, width, kernel_size, stride, padding, output_padding = 128, 128, 3, 2, 1, 1
```
</details>

<details>
<summary>82_Conv2d_Tanh_Scaling_BiasAdd_Max.py</summary>

```python
class Model(nn.Module):
    """Convolution, tanh, scaling, bias add, max-pool."""
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = scaling_factor
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.max_pool = nn.MaxPool2d(pool_kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = torch.tanh(x)
        x = x * self.scaling_factor
        x = x + self.bias
        x = self.max_pool(x)
        return x

batch_size, in_channels, out_channels = 128, 8, 64
height, width, kernel_size = 256, 256, 3
scaling_factor, pool_kernel_size = 2.0, 4
```
</details>

<details>
<summary>95_Matmul_Add_Swish_Tanh_GELU_Hardtanh.py</summary>

```python
class Model(nn.Module):
    """Matmul, add, Swish, Tanh, GELU, Hardtanh."""
    def __init__(self, in_features, out_features, add_value_shape):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.add_value = nn.Parameter(torch.randn(add_value_shape))

    def forward(self, x):
        x = self.matmul(x)
        x = x + self.add_value
        x = torch.sigmoid(x) * x   # Swish
        x = torch.tanh(x)
        x = torch.nn.functional.gelu(x)
        x = torch.nn.functional.hardtanh(x, min_val=-1, max_val=1)
        return x

batch_size, in_features, out_features = 1024, 8192, 8192
```
</details>

**To draw a different sample size or level**, change `random.sample(range(1, 101), 5)`
(level 2 range is 1–100; level 3 is 1–50, full ranges in `DATASET_RANGES`,
`agent/utils.py`), rewrite the test list, and **reuse that exact same list
across every setting you compare** — changing the sample between runs makes
comparisons meaningless.

---

## 4. The experiment config

File: `config/smoke/config_kbl2_sample5_openrouter.yaml`

```yaml
test_source: KB
agent_type: MCTS
test_list_path: config/smoke/test_list_kbl2_sample5.txt
dtype_str: fp32
gpu_name: NVIDIA GeForce RTX 2050
gpu_architecture: Ampere
gpu_ids: [0]
num_processes: 1                  # keep at 1: free-tier rate limits + single GPU + MAX_CONCURRENT_EVALS=1
use_remote_eval: true
remote_eval_url: http://127.0.0.1:12017
server_type: openrouter
model_name: "nvidia/nemotron-3-ultra-550b-a55b:free"
max_completion_tokens: 16384

total_steps: 5                    # MCTS steps per problem
max_memory_round: 5
pool_size: 5
disable_reviewer: false

save_path: outputs/MCTS_kbl2_sample5_openrouter   # MUST contain "MCTS" — see Gotcha #3
general_memory_path: results/memory/general_memory_v1_200.txt
memory_update: false              # don't mutate the shared skill memory during eval runs

debug: false
exploration_weight: 0.3
expand_exploration_ratio: 1.0
reward_alpha: 0.0
small_step_limit: 2
p_large: 0.2
pool_size_extra_max: 0
softmax_temperature: 1.0
dummy: false
geometric_p: 0.6
```

### Fields you'll actually want to change between experiments/checkpoints

| Field | What it controls |
|---|---|
| `model_name` | Which OpenRouter model proposes/reviews kernels. Check current free-tier slugs at [openrouter.ai/collections/free-models](https://openrouter.ai/collections/free-models) before reusing an old one — the lineup changes often. |
| `total_steps` | Search budget per problem. More steps = more chances to find a correct/fast kernel, more API calls. |
| `agent_type` | `MCTS` (tree search, what we're using), or `IRS`/`IRL`/`IRB`/`IRLE`/`PS` for the simpler baselines the paper compares against. |
| `disable_reviewer` | Turn off the reviewer pass (halves LLM calls/step, less self-correction). |
| `exploration_weight`, `p_large`, `reward_alpha`, `small_step_limit`, `geometric_p` | MCTS search-shape knobs (see paper §Explore). |
| `general_memory_path` | Which Adapt-stage skill memory to inject into prompts (see §8). |
| `save_path` | **Always give each experiment/checkpoint a distinct path, and always include the `agent_type` substring** (e.g. `outputs/MCTS_kbl2_sample5_<experiment-name>`) — required by `stats.py`, see Gotcha #3. |

**Do not change** `test_list_path` between settings you intend to compare —
that's the fixed sample from §3.

---

## 5. Running an experiment (foreground — watch it live)

```bash
conda activate adaexplore
export OPENROUTER_API_KEY=sk-or-...
python agent/agent_entry.py --config config/smoke/config_kbl2_sample5_openrouter.yaml
```

Run this in the foreground in its own terminal so you see the `tqdm` progress
bar and step-by-step logging directly. Expect roughly **3–17+ minutes per
MCTS step** on the free Nemotron tier — it's a large reasoning model
(550B/55B-active MoE) that spends real completion-token budget on internal
chain-of-thought before emitting kernel code, and free-tier requests queue
behind paid traffic. This is normal, not a hang (see Gotcha #2).

To try a different setting, **copy the YAML**, change only the field(s) under
test plus `save_path`, and run again:
```bash
cp config/smoke/config_kbl2_sample5_openrouter.yaml config/smoke/config_kbl2_sample5_<experiment>.yaml
# edit save_path (keep "MCTS" in it) + the one field you're testing
python agent/agent_entry.py --config config/smoke/config_kbl2_sample5_<experiment>.yaml
```

If a run is interrupted, `worker_process` in `agent_entry.py` skips any
`{level}_{problem_id}` folder under `save_path` that already has a
`global_best_kernel_{total_steps}.py` + metrics file, and deletes/redoes any
folder that doesn't (i.e. partially-failed problems are retried from
scratch, not resumed mid-search) — so re-running the exact same command is
the correct way to retry only the problems that didn't finish. There's also
`--resume_from <prior_save_path>` for resuming a specific completed run under
a new `save_path`.

---

## 6. How the MCTS search and evaluation actually work

### 6.1 The tree: large steps vs. small steps

Each MCTS **node** = one candidate `ModelNew` kernel + its eval result. The
tree starts at a dummy root (no kernel yet) and grows one node per step, two
ways (`agent/mcts.py::expand_large` / `expand_small`):

- **Large step** — "propose something new." The proposer LLM is shown a
  *pool* of the best kernels found anywhere in the tree so far (size
  `pool_size`, selected by the diversity logic in
  `_get_diverse_pool_for_large_step`) and writes a brand-new `ModelNew` from
  scratch. This is what creates a new branch (sibling).
- **Small step** — "refine what we have." A reviewer agent (skippable via
  `disable_reviewer`) critiques the most recent kernel on the path back to
  the last large-step ancestor (up to `max_memory_round` kernels of
  context), then a tuner agent emits `<old_str>/<new_str>` edits applied to
  that kernel. This extends a branch one level deeper (parent → child).

### 6.2 One MCTS step, end to end

Every iteration `step_idx` from 1 to `total_steps` does, in order
(`MCTSKernelOptimizer.step`):

1. **Select** — walk down from the root, picking the child with the best
   UCB1 score at each level (`exploration_weight` controls how much it
   favors under-explored branches over the highest-scoring one), until it
   decides to expand the current node instead of descending further.
2. **Expand** — decide large vs. small step: always large on the dummy
   root; otherwise small step, unless the node already has
   `small_step_limit` small-step children (forces a large step) or a
   `p_large` coin flip says large.
3. **Evaluate** — actually compile/run the new kernel against the
   reference model (see §6.4) — the only part of a step that costs an LLM
   call **and** a GPU evaluation.
4. **Backpropagate** — convert the eval result into a reward and add it to
   `visits`/`total_reward`/`max_reward` on every ancestor up to the root, so
   the next step's UCB1 calculation reflects it.

### 6.3 The reward, and what this experiment's knobs do to it

`MCTSNode.reward` (`agent/mcts.py` lines 33-92) turns the eval result into a
single number:

| Outcome | Reward |
|---|---|
| Didn't compile | `0.0` |
| Compiled, wrong output | `0.05` |
| Correct, `fast_p` clipped to `[0.1, 10.0]` | linearly mapped to `[0.4, 1.6]` |

UCB1's exploitation term blends a node's `max_reward` and `avg_reward`:
`reward_alpha * max + (1 - reward_alpha) * avg`. Our config sets
`reward_alpha: 0.0` — pure average, so a branch with one lucky great
attempt and several bad ones is *not* favored over a branch that's
consistently mediocre; this trades "chase the single best find" for "chase
the most reliable branch."

Two more of our config's values are worth calling out concretely:
- `p_large: 0.2` / `small_step_limit: 2` — only 1-in-5 expansions is a fresh
  kernel design; the rest are refinements, capped at 2 refinements per
  branch before it's forced to branch out again.
- `pool_size_extra_max: 0` — the "sample extra kernels from unrelated
  branches" step in `_get_diverse_pool_for_large_step` is fully disabled
  (`max_extra_nodes` computes to `0`), so `geometric_p: 0.6` in our config is
  set but currently inert — the large-step pool is built purely from branch
  points on the path back to root.

**`total_steps: 5` is a tiny budget.** Since the very first step out of the
dummy root is forced to be a large step, a 5-step run only ever adds 5 nodes
total to the tree per problem — there's barely room for the small-step
refinement loop to do anything. At this scale you're closer to
"smoke-testing the pipeline end to end" than running a search with room to
converge; the `fast_p ≈ 1.0` result for `2_15` in §8 is a plausible
consequence of this, not necessarily a sign the approach can't find a real
speedup — raise `total_steps` before drawing conclusions about kernel
quality.

### 6.4 Correctness and speedup measurement (`src/eval.py`)

This is the part of `src/eval.py` that runs inside the judge for every
candidate kernel (`eval_kernel_against_ref` / `run_and_check_correctness` /
`time_execution_with_cuda_event`):

**Correctness** (`run_and_check_correctness`):
- Runs `num_correct_trials` trials (search-time default: **5**, see
  `agent/actions.py`), each with a different deterministic random seed
  derived from `seed_num=42`.
- Each trial feeds the same random inputs to the original PyTorch `Model`
  and the candidate `ModelNew`, and compares outputs with
  `torch.allclose(output, output_new, atol=5e-2, rtol=5e-2)`.
- **All trials must pass** for `correctness: true`. The pass count is
  recorded as `metadata["correctness_trials"] = "(pass_count / num_correct_trials)"`.
- Any exception during the candidate's forward pass (illegal memory access,
  OOM, shape mismatch, etc.) is caught and recorded as `correctness: false`,
  not a hard crash of the whole run.

**Performance** (`time_execution_with_cuda_event`, only runs if correctness passed):
- 10 warmup iterations (untimed), then `num_perf_trials` timed iterations
  (search-time default: **100**) measured with `torch.cuda.Event`.
- The top and bottom 5% of timings are trimmed as outliers
  (`outlier_remove_ratio=0.1`, removed from both ends) — that's why our
  results show `"num_trials": 90` for 100 requested trials.
- `mean`/`std`/`min`/`max` are computed on the trimmed set (`get_timing_stats`).
- The **baseline** (original PyTorch model's time) is either loaded from a
  recorded file at `results/timing/<gpu_name>/baseline_time_torch.json`
  (currently only exists for `A100-SXM4-40GB`, `A6000`, `L40S`, `B200` — **not**
  for an RTX 2050) or, if absent, measured fresh on your own GPU the same way.
  Ours always falls back to `"baseline_type": "measured"` — meaning our
  speedup numbers are self-consistent (same GPU, same measurement method) but
  **not directly comparable in absolute terms** to the paper's numbers on
  A6000/A100/etc.
- `fast_p = baseline_mean / candidate_mean`. **`fast_p > 1` = faster than
  PyTorch eager, `fast_p < 1` = slower, `fast_p ≈ 1` = no real speedup.**

`tool_scripts/re_evaluate.py` re-runs this with higher trial counts
(`--num_correct_trials 5 --num_perf_trials 100`, configurable) against the
already-recorded best kernel, for a more rigorous final number before you
report/compare it.

---

## 7. Reading results: real walkthrough from this run

### 7.1 What's saved per problem (the full artifact list)

Inside `outputs/<save_path>/<level>_<problem_id>/`, per MCTS run:

| File | Contents |
|---|---|
| `reference_src.py` | The unmodified `Model` class for this problem (copy of the input) |
| `step_N.py` | The candidate kernel generated at MCTS step N |
| `step_N_metrics.json` | That kernel's eval result (compiled/correctness/runtime/`fast_p`) |
| `step_N_prompt.txt` | The exact prompt sent to the LLM for step N (proposer, or reviser+tuner concatenated) |
| `step_N_log.json` | Tree bookkeeping at step N: node id, parent id, `visits`/`total_reward`/`max_reward`/`avg_reward`, and the global-best score *so far* |
| `tree_stats.json` | Written once at the end of the run: node counts, depth, large-vs-small-step split |
| `tree_structure.txt` | Written once at the end: a full text dump of the tree (every node's metrics/reward/UCB) |
| `global_best_kernel_<total_steps>.py` | The single best-scoring kernel across the *whole* tree, not just the last step |
| `global_best_metrics_<total_steps>.json` | That kernel's eval result |

`step_0` is always the dummy root (no kernel, all-zero metrics) — that's why
problems that died before any LLM call (our `2_36`/`2_82`/`2_95` DNS failure
in §8) still have a `step_0` folder with nothing useful in it.

### 7.2 One kernel's metrics directly
```bash
python3 -c "
import json
d = json.load(open('outputs/MCTS_kbl2_sample5_openrouter/2_15/global_best_metrics_5.json'))
print(json.dumps(d, indent=2))
"
```
Our actual output for problem `2_15`:
```json
{
  "compiled": true,
  "correctness": true,
  "metadata": {
    "hardware": "NVIDIA GeForce RTX 2050",
    "correctness_trials": "(5 / 5)"
  },
  "runtime": 56.7,
  "runtime_stats": {
    "mean": 56.7, "std": 0.0852, "min": 56.6, "max": 56.9,
    "num_trials": 90,
    "fast_p": 1.0018,
    "baseline_type": "measured",
    "baseline_time": 56.8
  }
}
```
Read this as: the generated Triton kernel passed all 5 correctness trials,
and ran in 56.7ms (mean of 90 trimmed trials) vs. a freshly-measured PyTorch
baseline of 56.8ms — i.e. **essentially tied with PyTorch eager, not a real
optimization** (`fast_p` barely above 1.0).

### 7.3 Aggregate stats across the sample
```bash
python tool_scripts/stats.py --log_folder outputs/MCTS_kbl2_sample5_openrouter --step 5 --verbose \
  --load_eval_task_ids_path config/smoke/eval_ids_completed.txt
```
Our actual output (restricted to the 2 problems that finished — see Gotcha #4
for why the filter file is needed):
```
Task ID: 2 15, Fast P: 1.0018
WARNING: Bug folder does not have a corresponding normal folder, use the bug folder as the normal folder
Task ID: 2 4, Fast P: 0

Statistics for outputs/MCTS_kbl2_sample5_openrouter:
  Total: 2, Correct: 1, Accuracy: 0.5000
  Avg speedup: 0.5009, GM(10): 0.3165
  Avg speedup (correct only): 1.0018, GM(10) (correct only): 1.0018
  Bug eval count: 1
  Count 1.2: 0
  Count 2: 0
```
- `Accuracy` = fraction of problems with a correct kernel.
- `Avg speedup` divides by *all* problems (incorrect ones count as 0);
  `Avg speedup (correct only)` divides only by problems that passed.
- `Bug eval count` = problems where `stats.py` detected an infra-level error
  (OOM, "evaluation failed after 3 attempts", etc.) somewhere in that
  problem's step history — `stats.py` renames that problem's folder to
  `<level>_<id>_BUG` so it's visible and excluded from being silently counted
  as a clean wrong-answer. (Ours: problem `2_4` OOM'd on one step — see §8.)

### 7.4 Re-measure more rigorously before reporting a final number
```bash
python tool_scripts/re_evaluate.py \
  --log_folder outputs/MCTS_kbl2_sample5_openrouter \
  --num_correct_trials 5 --num_perf_trials 100 \
  --use_remote_eval --remote_eval_url http://127.0.0.1:12017
```

### 7.5 Evaluate one specific kernel file by hand
```bash
python tool_scripts/eval_one_kernel.py outputs/MCTS_kbl2_sample5_openrouter/2_15/global_best_kernel_5.py \
  --level 2 --problem_id 15 \
  --use_remote_eval --remote_eval_url http://127.0.0.1:12017
```

---

## 8. Actual result of the first full run (2026-06-23)

| Problem | Result |
|---|---|
| `2_4` (Conv2d_Mish_Mish) | Compiled, **not correct** — also hit a real `torch.OutOfMemoryError` on one step (`Tried to allocate 1.97 GiB`, only 623MiB free) while checking correctness. Confirms the VRAM risk flagged in §3: at `(64, 64, 256, 256)` input, this problem is genuinely tight on a 4GB card. Folder renamed `2_4_BUG` by `stats.py`. |
| `2_15` (ConvTranspose3d_BatchNorm_Subtract) | **Correct** (5/5 trials), `fast_p ≈ 1.002` — ties PyTorch eager, no real speedup. |
| `2_36`, `2_82`, `2_95` | **Failed before any kernel was generated** — `httpx.ConnectError: Temporary failure in name resolution` on the very first OpenRouter call. This was a transient local DNS/network blip on this machine, confirmed unrelated to OpenRouter rate-limiting (a direct probe immediately after returned `HTTP 200` in ~5s). Each of these folders only contains `step_0` (the no-LLM-needed baseline eval) — nothing was lost, just retry. |

**To retry the 3 failed problems** (network is healthy now), just re-run the
same command from §5 — it will skip `2_15` and `2_4_BUG` automatically and
redo only `2_36`/`2_82`/`2_95` from scratch.

---

## 9. Adapt-stage skill memory (what's feeding the prompts)

Explore-stage runs (including ours) read a skill-memory text file via
`general_memory_path` and inject it into every proposer/reviewer prompt. We
use the bundled, already-built Adapt-stage output —
`results/memory/general_memory_v1_200.txt` — with `memory_update: false`, so
**no Adapt re-run is needed** to reproduce these results.

If you want to test whether memory *built by the free OpenRouter model
itself* (instead of the paper's gpt-5-mini-derived memory) changes outcomes,
that's a full Adapt-stage run:
```bash
python agent/agent_entry.py --config config/SYN-v1/config_SYN-v1_none_MCTS.yaml
```
(edit that YAML's `server_type`/`model_name` to your OpenRouter setup first —
it currently points at `openai`/`gpt-5-mini`), which incrementally builds
`outputs/SYN-v1_example_run/general_memory.txt` as it runs. Point a new
Explore config's `general_memory_path` at that file to compare.

---

## 10. Known gotchas

1. **YAML string vs int**: if you ever drive a *single* problem without a
   `test_list_path` (top-level `level:`/`problem_id:` keys instead), they
   **must** be quoted YAML strings (`level: "2"`), or the judge rejects the
   request with `400 Bad Request: Input should be a valid string`. The
   `test_list_path` mechanism (what this guide uses throughout) doesn't have
   this bug because `load_tasks_from_test_list` (`agent/utils.py`) always
   casts both fields to `str`.

2. **Free-tier latency**: a step that takes ~3 min early in a run can take
   15+ min later — the model itself is large and slow under free-tier
   queueing, separate from any hard rate-limit block. The client already
   retries with backoff (`agent/inference_server.py::query_inference_server`,
   plus the `openai` SDK's own internal retries). If you want to check
   whether you're actually being throttled right now (vs. just slow), probe
   directly: `curl -H "Authorization: Bearer $OPENROUTER_API_KEY" https://openrouter.ai/api/v1/chat/completions -d '{"model":"...","messages":[...],"max_tokens":5}'` and look at the HTTP status/latency.

3. **`save_path` must contain the agent-type substring** (e.g. `"MCTS"`).
   `tool_scripts/stats.py::extract_step()` only recognizes `step_N.py`/
   `step_N_metrics.json` filenames when `"MCTS"`/`"MH"`/`"CHAIN"` appears
   *in the log folder path string itself* — otherwise it raises
   `ValueError: log folder does not match the file name pattern`. Always name
   experiment `save_path`s like `outputs/MCTS_<experiment-name>`.

4. **`stats.py` crashes on problems with zero completed steps.** If a problem
   failed before producing even one candidate kernel (e.g. our DNS failure on
   `2_36`/`2_82`/`2_95`, which only have `step_0`), `stats.py` throws
   `FileNotFoundError: .../<level>_<id>/.py` trying to find a "best step" that
   doesn't exist. Work around it with `--load_eval_task_ids_path <file>`
   (format: one `"<level> <problem_id>"` per line) listing only the problems
   that actually finished, until you've retried the rest.

5. **Judge exposure**: always start the judge with `HOST=127.0.0.1`, not the
   script's `0.0.0.0` default, on a personal machine — `/evaluate` runs
   arbitrary code with no authentication.

6. **VRAM**: 4GB is much smaller than the paper's A6000. Confirmed in
   practice on problem `2_4` (`torch.OutOfMemoryError`). Watch for this in
   step logs, especially on level-3 problems (full architectures) or
   high-resolution level-2 problems like `2_4`/`2_82` (256×256 inputs).
