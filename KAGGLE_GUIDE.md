# Running AdaExplore on Kaggle (No Conda)

Kaggle Notebooks run one fixed Python kernel inside an ephemeral container —
there's no `conda env create`, no separate terminal pane to leave a server
running in by default, and the filesystem mostly resets between sessions.
This guide adapts the OpenRouter workflow from
[`OPENROUTER_SAMPLE_BENCHMARK.md`](OPENROUTER_SAMPLE_BENCHMARK.md) and
[`OPENROUTER_PIPELINE_GUIDE.md`](OPENROUTER_PIPELINE_GUIDE.md) to that
environment. Everything below is meant to be pasted into Kaggle notebook
cells in order — each fenced block is one cell unless noted otherwise.

---

## 0. What's actually different on Kaggle

- **No conda.** `environment.yml` is just a thin wrapper around
  `pip install -r requirements.txt` anyway (see the comment at the top of
  that file) — on Kaggle you skip the wrapper and pip-install straight into
  the notebook's existing Python.
- **GPU is bigger.** Kaggle's free accelerators are **T4 ×2** (16GB each) or
  **P100** (16GB) — both far above the 4GB RTX 2050 used in the other two
  guides. The VRAM/OOM caution called out there (KernelBench problem `2_4`)
  is unlikely to reproduce here.
- **`/kaggle/working` is the only directory you should write to.** It's
  writable and is what gets preserved if you commit the notebook
  ("Save Version"); everything else in the container can be read-only or
  reset.
- **Sessions are time-boxed.** Interactive GPU sessions disconnect after
  ~20 minutes of idling and are capped around 9 hours of continuous compute;
  "Save & Run All" (commit) jobs get the same ~9h ceiling but don't need the
  browser open. Both matter here because free-tier OpenRouter latency is
  slow (3–17+ min/step, per the other guides) — plan accordingly (§10).
- **No background terminal by default.** The local judge
  (`online_judge/app_with_queue.py`) needs to keep running *while* the agent
  script also runs in the same container — on Kaggle that means starting it
  as a detached subprocess from a cell, not "open a second terminal."
- **Secrets, not hardcoded keys.** Notebooks can end up public; the
  OpenRouter key should go through Kaggle's Secrets add-on, never typed
  into a cell that gets saved.

---

## 1. Notebook settings (before writing any cell)

In the notebook editor's right-hand panel:
- **Accelerator** → `GPU T4 x2` (or `GPU P100` if that's what's offered) —
  not "None".
- **Internet** → **On**. Required for both `pip install` and every
  OpenRouter call; if it's off, installation fails immediately, so check
  this first if anything below fails at the first network step.
- For a run you expect to take more than an hour or so unattended, use
  **Save & Run All (Commit)** rather than staying purely interactive — it
  uses the same compute quota but isn't tied to your browser staying open.

---

## 2. Get the code onto Kaggle

Your local working tree currently has uncommitted changes (the OpenRouter
branch in `agent/inference_server.py` etc., plus the new
`OPENROUTER_*.md` guides and `config/smoke/*` files) — a plain `git clone`
of the public upstream repo would **not** include any of that. Pick one:

**Option 1 — Kaggle Dataset (recommended, no GitHub account needed)**

From your local machine: zip the repo as-is (including your uncommitted
changes) and upload it at kaggle.com → *Datasets* → *New Dataset* (or via
the `kaggle` CLI: `kaggle datasets create -p <zipped-folder>`). Then in the
notebook, attach that dataset (*Add Input*) and copy it into the writable
working directory:

```python
import shutil
shutil.copytree("/kaggle/input/<your-dataset-slug>", "/kaggle/working/AdaExplore")
%cd /kaggle/working/AdaExplore
```

**Option 2 — git clone (only if you push your changes to your own remote
first)**

This guide doesn't push anything on your behalf — if you want this path,
push your current branch to a fork/remote yourself, then:

```python
%cd /kaggle/working
!git clone https://github.com/<your-user>/<your-fork>.git AdaExplore
%cd AdaExplore
```

---

## 3. Install dependencies — pip only

Two plans, try A first:

**Plan A — full pinned stack** (matches `requirements.txt` exactly):
```python
!pip install -q --no-cache-dir -r requirements.txt
```

Then verify the GPU stack actually came up correctly:
```python
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
```
If `torch.cuda.is_available()` is `True` here, you're done — skip Plan B.

**Plan B — keep Kaggle's preinstalled torch/triton** (fallback, only if
Plan A's check above prints `False`, which would mean the pinned
`torch==2.5.0` cu124 wheel didn't bind to Kaggle's driver):
```python
!grep -vE '^(torch|triton)==' requirements.txt > /tmp/requirements_kaggle.txt
!pip install -q --no-cache-dir -r /tmp/requirements_kaggle.txt
```
This leaves Kaggle's own (already CUDA-matched) torch/triton pair in place
and only installs everything else AdaExplore needs on top of it. Re-run the
verification cell above to confirm.

**Optional — trim install time.** `sentence-transformers` in
`requirements.txt` is only used by an *alternate*, embedding-based
duplicate-detection path in `skill_memory/deduplicate_knowledge.py`
(`get_embeddings_sentence_transformers`) — the path actually used by
`update_memory()` (the one the OpenRouter pipeline guide's Adapt-stage
workflow exercises) is the LLM-judge path, `llm_judge_duplicate_batch`,
which doesn't need it. If install time matters more than completeness,
drop it from the requirements file before installing.

---

## 4. Store the OpenRouter key as a Kaggle Secret

Notebook editor → *Add-ons* → *Secrets* → add a secret named
`OPENROUTER_API_KEY` with your key from
[openrouter.ai/keys](https://openrouter.ai/keys). Then, in a cell:

```python
from kaggle_secrets import UserSecretsClient
import os
os.environ["OPENROUTER_API_KEY"] = UserSecretsClient().get_secret("OPENROUTER_API_KEY")
```

Never `print()` this value or echo it in a shell command — anything a cell
outputs can end up saved in the notebook if you commit it, and Kaggle
notebooks are frequently shared/forked publicly.

---

## 5. Start the local judge as a background process

Kaggle has no second terminal pane open by default, so launch the judge as
a detached subprocess from a cell (`start_new_session=True` keeps it alive
once the cell returns) instead of the `bash online_judge/start_server.sh`
foreground pattern used on a normal machine:

```python
import subprocess, os, time

judge_log = open("/kaggle/working/judge.log", "w")
judge_env = {**os.environ, "AVAILABLE_GPUS": "0,1"}  # "0" only if you got a single-GPU P100, not T4x2
judge_proc = subprocess.Popen(
    ["python3", "-m", "uvicorn", "online_judge.app_with_queue:app",
     "--host", "127.0.0.1", "--port", "12017"],
    cwd="/kaggle/working/AdaExplore",
    env=judge_env,
    stdout=judge_log, stderr=subprocess.STDOUT,
    start_new_session=True,
)
print("judge pid:", judge_proc.pid)
time.sleep(5)
```

We bind `127.0.0.1`, not `0.0.0.0` — same reasoning as the other guides:
`/evaluate` runs arbitrary submitted code with no auth, no reason to widen
that beyond loopback even inside an already-isolated container.

Health check, in a separate cell:
```python
import requests
print(requests.get("http://127.0.0.1:12017/health").json())
```
Should report `"status":"healthy"` and your GPU(s) under `devices`. If this
fails, check `/kaggle/working/judge.log` before anything else.

---

## 6. Tell the configs what GPU you actually got

`gpu_name`/`gpu_architecture` are purely descriptive — they only get
interpolated into the LLM prompt text (`agentprompt/prompt_modules.py::generate_hardware_information_prompt`),
nothing in the eval/compile path branches on them — so just match them to
whatever `torch.cuda.get_device_name(0)` printed in §3:

| Kaggle accelerator | `--gpu_name` | `--gpu_architecture` |
|---|---|---|
| GPU T4 x2 | `Tesla T4` | `Turing` |
| GPU P100 | `Tesla P100` | `Pascal` |

You don't need to edit the YAML files for this — CLI flags override config
values (`agent/utils.py::load_config_from_yaml`, "Command line arguments
take precedence over YAML values"), so just append them to the run command
in §7/§8.

---

## 7. Run Explore-only on the 5 sampled kernels

Same config and test list as `OPENROUTER_SAMPLE_BENCHMARK.md` §3-§5 — no
edits needed beyond the GPU override:

```python
!python -u agent/agent_entry.py \
  --config config/smoke/config_kbl2_sample5_openrouter.yaml \
  --gpu_name "Tesla T4" --gpu_architecture "Turing"
```

(`-u` keeps stdout unbuffered so `tqdm`/log lines actually stream into the
cell output instead of arriving in one burst at the end.) On T4 x2, you
could also raise `num_processes`/`gpu_ids` in a copy of the config to
parallelize across both GPUs — not necessary for a 5-problem smoke run,
worth it for a larger sample.

## 8. (Optional) Run the Adapt-stage smoke config the same way

From `OPENROUTER_PIPELINE_GUIDE.md` §3.1:
```python
!python -u agent/agent_entry.py \
  --config config/smoke/config_adapt_syn5_openrouter.yaml \
  --gpu_name "Tesla T4" --gpu_architecture "Turing"
```

---

## 9. Evaluation, inspection, and getting your results off Kaggle

The tools are identical to the other two guides — `tool_scripts/stats.py`,
`tool_scripts/re_evaluate.py`, `tool_scripts/eval_one_kernel.py`, reading
`step_N_metrics.json` / `global_best_metrics_*.json` directly — run them as
notebook cells exactly as documented in
`OPENROUTER_SAMPLE_BENCHMARK.md` §7 and `OPENROUTER_PIPELINE_GUIDE.md` §5.
The one Kaggle-specific step is **getting `outputs/` out before the
session ends**, since `/kaggle/working` is only preserved if you commit:

```python
!cd /kaggle/working/AdaExplore && zip -r /kaggle/working/outputs.zip outputs/
```
Then either click **Save Version** so `/kaggle/working/outputs.zip`
appears under that version's *Output* tab for download, or, if you want it
to survive into a *different* notebook/session, push it as a new Kaggle
Dataset via the API (`kaggle datasets version -p /kaggle/working ...`).
Don't rely on it just sitting in `/kaggle/working` across a fresh,
uncommitted session — that directory resets.

---

## 10. Kaggle-specific gotchas

1. **Session limits vs. free-tier OpenRouter latency.** A single MCTS step
   can take 3–17+ minutes on a free model under load. `total_steps: 5`
   across 5 problems (§7) is comfortably inside one session; if you scale
   up `total_steps` or the problem count, you may hit the ~9h ceiling
   mid-run. `agent_entry.py`'s `worker_process` already skips any
   `{level}_{problem_id}` folder that has a finished
   `global_best_kernel_{total_steps}.py`, and `--resume_from` exists for
   continuing a specific run — so a session cutoff loses at most the one
   problem that was in flight, not the whole run. Re-launch with the same
   `--config` (same `save_path`) in a fresh session to pick up where you
   left off.
2. **Idle disconnects in interactive mode.** If you're babysitting the
   notebook live rather than using "Save & Run All," Kaggle will disconnect
   an idle kernel — keep the tab active, or switch to a commit run for
   anything long.
3. **Disk quota (~20GB shared across input+working+output).** `pip`
   wheel caches and accumulating `outputs/` from multiple runs can eat into
   this; `--no-cache-dir` on pip installs (already used in §3) helps, and
   periodically zipping-and-clearing old `outputs/<run>` folders helps too.
4. **Never let the API key hit a saved cell output.** Use Secrets (§4) and
   don't `print(os.environ["OPENROUTER_API_KEY"])` — a committed,
   forked, or shared notebook would leak it.
5. **`AVAILABLE_GPUS` must match what you actually got.** `"0,1"` for T4x2,
   `"0"` for a single P100 — passed as an env var to the judge process in
   §5, separate from the agent's own `--gpu_name`/`--gpu_architecture`
   (§6) and `gpu_ids`/`num_processes` config fields.
6. **Internet toggle is checked once, fails loud.** If it's off, `pip
   install` and the first OpenRouter call both fail immediately at startup
   — that's the first thing to check if §3 or §7 errors out right away
   rather than partway through.
