# Running Adapt + Explore (or Explore Alone) via OpenRouter

This is the decision-and-command guide for which stage(s) of AdaExplore to
run through a free OpenRouter model, and how to inspect what came out the
other end. It assumes you've already done the one-time setup in
[`OPENROUTER_SAMPLE_BENCHMARK.md`](OPENROUTER_SAMPLE_BENCHMARK.md) §0-§2
(conda env, `OPENROUTER_API_KEY`, the local judge running on
`127.0.0.1:12017`) — this guide doesn't repeat that, it builds on it.

For the deep-dive on MCTS internals (large/small steps, UCB1, the reward
formula) and on exactly how correctness/speedup are measured, see that same
doc's §6. This guide stays one level up: which stage to run, in what order,
and how to read the results.

---

## 1. The two stages, and the one decision that matters

From [`readme.md`](readme.md):

- **Adapt** — run the agent on a pile of *synthetic* throwaway tasks,
  collect its compile/runtime failures, and distill the recurring ones into
  a reusable text file of rules (`results/memory/general_memory_v1_200.txt`
  is the bundled, pre-built example). This file gets injected into every
  proposer/reviser/tuner prompt as "things not to do."
- **Explore** — the MCTS tree search (`agent/mcts.py`) that actually tries
  to speed up *your* target kernel(s). This always runs — it's the thing
  that produces a `global_best_kernel.py`.

**The decision**: Adapt only ever *feeds* Explore a memory file. You do not
need to run Adapt to run Explore — a pre-built memory file already ships in
this repo. You only need to run Adapt yourself if you want to:
- see how the failure-distillation process works end to end, or
- build memory from *your* model (the free Nemotron model you're using)
  instead of the paper's `gpt-5-mini`-derived memory, to test whether that
  changes outcomes.

| You want to... | Run |
|---|---|
| Just optimize the 5 sample kernels, fastest path | §2 (Explore-only, bundled memory) |
| See the failure→memory distillation loop, cheaply, reusing OpenRouter | §3.1 (Adapt, online update on 5 tiny synthetic tasks) |
| Build memory for free from a run you already did | §3.2 (Adapt, offline build from existing logs) |
| Generate brand-new synthetic tasks (not reuse the bundled 500) | §3.3 (optional, expensive) |
| Compare bundled-memory vs. your-own-memory Explore results | §4 |

---

## 2. Path A — Explore only, no Adapt (the 5 sampled kernels)

This is the workflow already fully documented in
`OPENROUTER_SAMPLE_BENCHMARK.md`. One-line recap:

```bash
conda activate adaexplore
export OPENROUTER_API_KEY=sk-or-...
python agent/agent_entry.py --config config/smoke/config_kbl2_sample5_openrouter.yaml
```

The config's `general_memory_path: results/memory/general_memory_v1_200.txt`
with `memory_update: false` is what makes this Explore-only: it *reads*
the bundled memory on every prompt but never writes to it. Nothing here
touches Adapt. For the full problem list, config field reference, expected
timing, and troubleshooting gotchas, see that doc's §3-§5 and §10.

---

## 3. Path B — Build your own memory via OpenRouter (Adapt), then Explore with it

### 3.1 Option B1 (recommended): online memory update on a handful of synthetic tasks

This is the cheapest way to see real Adapt-stage behavior. It runs MCTS
(same machinery as Explore) on a few *synthetic* tasks instead of real
benchmark problems, and after each problem finishes, automatically mines
that problem's failures into a memory file.

**Dataset gotcha first**: the paper's full synthetic test list
(`config/test_list/test_list_syn.txt`) requests problems `1-2007`, but the
synthetic dataset actually bundled in this repo
(`datasets/KernelBench_syn/syn_v1/`) only has **500** files (`1.py`...
`500.py`). Don't point the full list at the bundled dataset — it'll index
out of range past problem 500. That's why a scoped-down list is needed for
a smoke run.

Two new files are already created for you:

`config/smoke/test_list_syn_sample5.txt`:
```
1 1-5
```
(`"1"` here is the synthetic dataset *version*, i.e. `syn_v1` —
`load_test_source`/`construct_synthesized_data_dataset` use `level` as the
version number for `test_source: SYN`, not a difficulty level like KB. IDs
`1-5` are well inside the bundled 500.)

`config/smoke/config_adapt_syn5_openrouter.yaml` (key fields):
```yaml
test_source: SYN
test_list_path: config/smoke/test_list_syn_sample5.txt
server_type: openrouter
model_name: "nvidia/nemotron-3-ultra-550b-a55b:free"
total_steps: 3                    # small: this run's job is to generate failures, not to optimize
disable_reviewer: true            # matches the paper's own Adapt-stage config (config/SYN-v1/...)
save_path: outputs/MCTS_adapt_syn5_openrouter
general_memory_path: outputs/MCTS_adapt_syn5_openrouter/general_memory.txt
memory_update: true                # <-- this is the one flag that makes it an Adapt run
```

Run it:
```bash
conda activate adaexplore
export OPENROUTER_API_KEY=sk-or-...
python agent/agent_entry.py --config config/smoke/config_adapt_syn5_openrouter.yaml
```

**What happens, mechanically** (`agent_entry.py::agent_entry`, lines
105-161): for each of the 5 synthetic problems, MCTS runs for 3 steps same
as any Explore run. Once that problem's search finishes, since
`memory_update: true`, it acquires a file lock and calls
`skill_memory.update_memory(memory_path=..., log_path=<that problem's
folder>, server=<your openrouter client>, model_name=..., filter_max_difference=True)`.
That function scans every `step_N_metrics.(json|txt)` in the folder
*except* `step_0` (the dummy root) for `correctness=False`, and for each
one asks the LLM for exactly one grounded `"You cannot ..."` rule (or
`"no guidance"` if the error doesn't support a clean rule). New rules are
checked against existing ones with an LLM-based duplicate judge
(`skill_memory/deduplicate_knowledge.py`) — duplicates bump an existing
rule's score instead of adding a new line.

The same `--memory_update` flag also mines the **positive** side: any step
that's `correctness=True` *and* at least as fast as the PyTorch baseline
(`fast_p >= --min_speedup_for_best_practice`, default `1.0`) gets asked for
one grounded `"You should ..."` rule instead — a concrete technique
actually present in that kernel (a fusion, a tiling choice, a masking
pattern, ...), not vague praise. This writes to a *second*, separate file
(`skill_memory/skill_memory.py::update_positive_memory`) so "must not do X"
and "should do Y" rules never get mixed into one undifferentiated list.
`general_memory_positive_path` defaults to a sibling of
`general_memory_path` (`general_memory.txt` → `general_memory_positive.txt`)
if you don't set it explicitly — nothing extra to add to the config above
to get both.

**Cost/time expectations**: 5 problems × 3 steps = 15 LLM-driven kernel
attempts, *plus* one extra LLM call per failed attempt (memory
distillation) and one more per dedup comparison once the file has entries.
At the same free-tier Nemotron latency documented in
`OPENROUTER_SAMPLE_BENCHMARK.md` §5 (3-17+ min per step), budget **at least
1-2 hours**, possibly more. Run it in the background or its own terminal;
if you just want a fast sanity check that the plumbing works, edit
`test_list_syn_sample5.txt` down to `1 1-1` and `total_steps: 2` first.

### 3.2 Option B2 (free — reuses work you already did)

You already have `outputs/MCTS_kbl2_sample5_openrouter/` on disk from the
Explore-only run in §2, and it already contains real failures (confirmed:
`2_15` has several `correctness: false` steps, `2_4_BUG` has an OOM
failure). You can mine memory from that existing run, at zero additional
GPU cost beyond what you've already spent, no new MCTS run needed. This
builds the *negative* file (failures → `"you cannot..."`):

```bash
python skill_memory/skill_memory.py \
  --log-dir outputs/MCTS_kbl2_sample5_openrouter \
  --knowledge-store-path outputs/skill_memory_from_kbl2_sample5.txt \
  --server openrouter \
  --model-name "nvidia/nemotron-3-ultra-550b-a55b:free" \
  --filter-max-difference \
  --seed 42
```

Run it a second time with `--mode successes` and a **different**
`--knowledge-store-path` to build the *positive* file (successes →
`"you should..."`) from the same run — the two memories must stay in
separate files, never mixed:

```bash
python skill_memory/skill_memory.py \
  --log-dir outputs/MCTS_kbl2_sample5_openrouter \
  --knowledge-store-path outputs/skill_memory_from_kbl2_sample5_positive.txt \
  --server openrouter \
  --model-name "nvidia/nemotron-3-ultra-550b-a55b:free" \
  --mode successes \
  --min-speedup 1.0 \
  --seed 42
```

Notes:
- `--server` in `skill_memory.py` isn't restricted by an argparse
  `choices` list, and `create_inference_server()` already has the
  `openrouter` branch (`agent/inference_server.py` lines 139-148) — so this
  works today, no code changes needed.
- `step_0` files are excluded automatically by both modes' filename
  filter, so the dummy-root entries can't pollute either file.
- Don't expect much output from only 2 finished problems — this is meant
  as a "see the mechanism work," not "build a real memory base" exercise.
  For a real memory base you'd point `--log-dir` at a much larger
  Explore/Adapt run. The positive run in particular only has `2_15`'s
  steps 1-2 to draw from (verified: `correctness=True`, `fast_p` ≈ 1.0,
  ≈ 1.002 — see §8) — `2_4_BUG`'s failure doesn't qualify here.

### 3.3 Option C (optional, expensive): regenerate the synthetic dataset itself

Only do this if you want *new* synthetic tasks beyond the bundled 500 —
options B1/B2 above already exercise the full Adapt loop using the
existing dataset. `synthesis/generate_data.py` already supports
`--server_type openrouter` (confirmed in this repo's working tree diff).
Scoped down for free-tier:

```bash
python synthesis/generate_data.py \
  --server_type openrouter \
  --model_name "nvidia/nemotron-3-ultra-550b-a55b:free" \
  --prompt_style composite \
  --input_levels 1 \
  --num_generations 5 \
  --num_examples_per_request 3 \
  --temperature 1.0

python synthesis/rename.py \
  --source_path outputs/data_generation/<generated_data_dir_from_previous_step> \
  --data_path datasets/KernelBench_syn/syn_v2 \
  --force
```

The paper's own default is `--num_generations 200`; at free-tier latency
that's hours of additional LLM calls just to produce *tasks*, before any
Adapt/Explore search even starts on them. Keep `--num_generations` small
unless you specifically need a bigger or different synthetic pool.

---

## 4. Closing the loop: re-run Explore with the memory you built

To see whether your own memory changes Explore outcomes, copy the §2
config and swap only `general_memory_path` and `save_path`:

```bash
cp config/smoke/config_kbl2_sample5_openrouter.yaml \
   config/smoke/config_kbl2_sample5_ownmemory.yaml
```

Edit `config_kbl2_sample5_ownmemory.yaml`:
```yaml
general_memory_path: outputs/MCTS_adapt_syn5_openrouter/general_memory.txt  # or outputs/skill_memory_from_kbl2_sample5.txt
general_memory_positive_path: outputs/MCTS_adapt_syn5_openrouter/general_memory_positive.txt  # set explicitly -- see note below
memory_update: false        # Explore-time should not also be rewriting memory mid-run
save_path: outputs/MCTS_kbl2_sample5_ownmemory   # must contain "MCTS"; must differ from the original run
```

`general_memory_positive_path` only gets **auto-derived** when
`--memory_update`/`memory_update: true` is set (it's a convenience for the
run that's *building* memory). Here `memory_update: false` — you're only
*reading* memory this time — so if you built a positive file in §3.1/§3.2
and want this Explore run to actually use it, you must set
`general_memory_positive_path` yourself, same as `general_memory_path`.

Then run it the same way as §2, and compare against the original bundled-
memory run with `stats.py` (§5.4) — same fixed 5-problem sample both times,
so the comparison is meaningful (don't change `test_list_path` between the
two).

---

## 5. Evaluation & inspection, end to end

### 5.1 Is the judge alive?
```bash
curl http://127.0.0.1:12017/health
```
Should report `"status":"healthy"` and your GPU under `devices`. Do this
before any run if something looks stuck — a dead judge is a much more
common cause of a hang than free-tier latency.

### 5.2 What's on disk after a run finishes

Every run (Adapt or Explore — both use the `MCTS` agent type and the same
logging code path in `agent/mcts.py`) writes the same artifact set per
problem, under `<save_path>/<level>_<problem_id>/`:

| File | Contents |
|---|---|
| `reference_src.py` | The unmodified reference `Model` class for this problem |
| `step_N.py` | Candidate kernel generated at step N |
| `step_N_metrics.json` | That kernel's eval result (compiled/correctness/runtime/`fast_p`) |
| `step_N_prompt.txt` | Exact prompt sent to the LLM at step N |
| `step_N_log.json` | Tree bookkeeping: visits, rewards, global-best-so-far |
| `tree_stats.json` / `tree_structure.txt` | Written once, at the end of that problem's search |
| `global_best_kernel_<total_steps>.py` / `global_best_metrics_<total_steps>.json` | The winning kernel for that problem |

(Full description: `OPENROUTER_SAMPLE_BENCHMARK.md` §7.1.) For an Adapt
run, there's one more side effect outside any problem folder: the
`general_memory_path` file itself gets updated after each problem
finishes (see §5.7).

### 5.3 One problem's result, directly
```bash
python3 -c "
import json
d = json.load(open('outputs/<save_path>/<level>_<id>/global_best_metrics_<total_steps>.json'))
print(json.dumps(d, indent=2))
"
```

### 5.4 Aggregate stats across a sample
```bash
python tool_scripts/stats.py --log_folder outputs/<save_path> --step <total_steps> --verbose
```
Reports accuracy (fraction with a correct kernel) and average speedup
(`fast_p`), both over all problems and over correct-only problems. Works
identically for an Explore-only run or an Adapt run, as long as
`save_path` contains `"MCTS"` (`tool_scripts/stats.py::extract_step()`
requires that substring in the path — both `config_kbl2_sample5_openrouter`
and the new `config_adapt_syn5_openrouter` already satisfy this). If any
problem has zero completed steps (e.g. it died before the first LLM call),
pass `--load_eval_task_ids_path <file>` listing only the finished problems,
or `stats.py` will throw a `FileNotFoundError` looking for a step that
doesn't exist.

### 5.5 Re-measuring more rigorously before reporting a final number
```bash
python tool_scripts/re_evaluate.py \
  --log_folder outputs/<save_path> \
  --num_correct_trials 5 --num_perf_trials 100 \
  --use_remote_eval --remote_eval_url http://127.0.0.1:12017
```
Re-runs the already-recorded best kernels with higher trial counts —
useful since search-time defaults already use 5 correctness trials / 100
perf trials, but you may want to confirm stability before citing a number.

### 5.6 Spot-checking one kernel file by hand
```bash
python tool_scripts/eval_one_kernel.py outputs/<save_path>/<level>_<id>/global_best_kernel_<total_steps>.py \
  --level <level> --problem_id <id> \
  --use_remote_eval --remote_eval_url http://127.0.0.1:12017
```

### 5.7 Inspecting the skill-memory files themselves (Adapt-specific)

An Adapt run with `memory_update: true` writes **two** files, both plain
text, one rule per line, same `<rule>||<score>` shape, deliberately kept
separate so constraint-rules and style-suggestion-rules never get mixed in
one list the model has to disambiguate on its own:
```
general_memory.txt:           You cannot ...||<score>     # from failed steps
general_memory_positive.txt:  You should ...||<score>     # from successful, >= baseline-speed steps
```
`<score>` is a frequency counter in both — it increments each time the
LLM-based duplicate judge decides a newly-mined rule restates an existing
one, instead of appending a new line.

To check whether an Adapt run actually changed anything:
```bash
wc -l outputs/MCTS_adapt_syn5_openrouter/general_memory.txt            # negative: line count before vs. after
wc -l outputs/MCTS_adapt_syn5_openrouter/general_memory_positive.txt   # positive: same check
diff <(sort old_memory.txt) <(sort outputs/MCTS_adapt_syn5_openrouter/general_memory.txt)
```
Or just watch the run's stdout live — `update_memory()`/`update_positive_memory()`
each print their own file's first entry/score before processing (labeled
`"skill memory (negative: ...)"` / `"skill memory (positive: ...)"`), and
the dedup judge logs each decision as it runs.

Expect the positive file to fill slower than the negative one in a small
run like our `total_steps: 3` / 5-problem smoke config: it only fires on
steps that are *both* correct *and* at least `min_speedup_for_best_practice`
(default `1.0`) — a higher bar than "any failure," which is most of what a
weak free-tier model produces early on. That's expected, not a bug.

Two things that look like bugs but aren't:
- **An empty or unchanged memory file after a run.** `update_memory` only
  fires on steps where `correctness=False` (excluding `step_0`), and even
  then the LLM is allowed to answer `"no guidance"` if the failure doesn't
  cleanly support a one-sentence rule — that attempt is silently skipped.
  Few problems × few steps (our `total_steps: 3` smoke config) means few
  chances for this to trigger at all.
- **A `general_memory_path + ".lock"` (and `general_memory_positive_path + ".lock"`)
  file appearing during the run.** Those are the two file locks
  `agent_entry.py` takes — one per memory file — before calling
  `update_memory()`/`update_positive_memory()`, to stop two parallel workers
  corrupting either file (`num_processes` > 1 case). They should be
  released and effectively inert once the run exits cleanly — if a `.lock`
  file lingers *and* a new run hangs acquiring it, that indicates the
  previous run was killed
  uncleanly while holding it, not a problem with your config.

---

## 6. All commands, in one place

```bash
# one-time
conda activate adaexplore
export OPENROUTER_API_KEY=sk-or-...
HOST=127.0.0.1 AVAILABLE_GPUS=0 bash online_judge/start_server.sh   # separate terminal, leave running
curl http://127.0.0.1:12017/health                                  # sanity check

# Path A: Explore only (5 sample kernels, bundled memory)
python agent/agent_entry.py --config config/smoke/config_kbl2_sample5_openrouter.yaml

# Path B1: Adapt, online update on 5 tiny synthetic tasks
python agent/agent_entry.py --config config/smoke/config_adapt_syn5_openrouter.yaml

# Path B2: Adapt, offline build from a run you already have
python skill_memory/skill_memory.py \
  --log-dir outputs/MCTS_kbl2_sample5_openrouter \
  --knowledge-store-path outputs/skill_memory_from_kbl2_sample5.txt \
  --server openrouter --model-name "nvidia/nemotron-3-ultra-550b-a55b:free" \
  --filter-max-difference --seed 42

# Close the loop: re-run Explore with your own memory (§4), then compare
python tool_scripts/stats.py --log_folder outputs/MCTS_kbl2_sample5_openrouter --step 5 --verbose
python tool_scripts/stats.py --log_folder outputs/MCTS_kbl2_sample5_ownmemory --step 5 --verbose
```

Before reusing `nvidia/nemotron-3-ultra-550b-a55b:free` (or picking a
different one), check current free-tier slugs at
[openrouter.ai/collections/free-models](https://openrouter.ai/collections/free-models)
— the free lineup changes often.
