# Collect error messages from logs and summarize them into a skill memory file.

import os
import argparse
from tqdm import tqdm
import sys
import random
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.inference_server import create_inference_server, query_inference_server
from agent.utils import read_metrics
from skill_memory.deduplicate_knowledge import llm_judge_duplicate_batch

PROMPT_TEMPLATE = """You are an assistant that extracts *minimal, log-grounded constraints* from Triton kernel error messages
and records them as one-line "You cannot ..." rules in a skill memory file.

Given the code block and the error message:

- Summarize the error into **exactly one sentence**
- The sentence MUST begin with "You cannot ..."
- The sentence MUST describe the **minimal prohibited action** implied by the error
- The sentence MUST NOT include:
  - suggestions, fixes, or alternatives
  - reasons, explanations, or consequences
  - assumptions beyond what is directly implied by the error message
- Do NOT generalize beyond the specific scope indicated by the error
  (e.g., inside a Triton kernel, at compile time, for constexpr parameters, etc.)
- **Do NOT guess, infer, or hallucinate a rule**
- **If the error message does not clearly imply a single, well-defined prohibited action,
  output exactly: `no guidance`**

Only use information that is directly supported by the error message.

<Code Block>
{code_block}
</Code Block>

<Error Message>
{error_message}
</Error Message>

Now write the one-sentence constraint.
"""

POSITIVE_PROMPT_TEMPLATE = """You are an assistant that extracts *minimal, log-grounded best practices* from successful Triton kernel implementations
and records them as one-line "You should ..." rules in a skill memory file.

Given the code block and its measured results:

- Summarize the technique behind this kernel's correctness/performance into **exactly one sentence**
- The sentence MUST begin with "You should ..."
- The sentence MUST describe a **concrete, transferable technique** actually present in the code
  (e.g. a fusion of specific operators, a memory-access or tiling pattern, a block/grid size choice,
  a numerically-stable reformulation, a boundary/masking pattern, an autotuning configuration)
- The sentence MUST NOT include:
  - vague praise ("write efficient code", "optimize the kernel")
  - reasons, explanations, or consequences
  - assumptions beyond what is directly visible in the code
- Do NOT generalize beyond the specific scope indicated by the code
  (e.g., inside a Triton kernel, for this operator pattern, etc.)
- **Do NOT guess, infer, or hallucinate a rule**
- **If the code does not clearly demonstrate a single, well-defined transferable technique,
  output exactly: `no guidance`**

Only use information that is directly supported by the code and metrics below.

<Code Block>
{code_block}
</Code Block>

<Metrics>
{metrics}
</Metrics>

Now write the one-sentence best practice.
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Collect error messages from logs and summarize them into a skill memory file")
    parser.add_argument("file", type=str, nargs="?", default=None,
                        help="Optional: Process a single metrics file instead of batch processing")
    parser.add_argument("--log-dir", type=str, default="outputs", 
                        help="Directory containing log files (default: outputs)")
    parser.add_argument("--server", type=str, default="azure", 
                        help="Inference server type (default: azure)")
    parser.add_argument("--model-name", type=str, default="gpt-5-mini", 
                        help="Model name to use for inference (default: gpt-5-mini)")
    parser.add_argument("--knowledge-store-path", type=str, default="knowledge_store.txt", 
                        help="Path to store the skill memory file (default: knowledge_store.txt)")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for shuffling logs (default: 42)")
    parser.add_argument("--max-logs", type=int, default=3000, 
                        help="Maximum number of logs to process (default: 3000)")
    parser.add_argument("--filter-max-difference", action="store_true",
                        help="Filter out logs containing 'max_difference' (default: False)")
    parser.add_argument("--progress_file", type=str, default=None,
                        help="Path to store the progress (default: None)")
    parser.add_argument("--mode", type=str, default="errors", choices=["errors", "successes"],
                        help="'errors' collects negative \"you cannot...\" memory from failed logs (default); "
                             "'successes' collects positive \"you should...\" memory from correct, "
                             "at-least-as-fast-as-baseline logs. Run twice (once per mode, with two "
                             "different --knowledge-store-path values) to build both.")
    parser.add_argument("--min-speedup", type=float, default=1.0,
                        help="Only used with --mode successes: minimum fast_p (speedup over PyTorch "
                             "baseline) a correct kernel must reach to count as a best-practice example (default: 1.0)")
    return parser.parse_args()

def check_error_exists(file: str, filter_max_difference: bool = True):
    with open(file, "r") as f:
        content = f.read()
        if "correctness=False" not in content or (filter_max_difference and "max_difference" in content):
            return False, "Not a CE or RE error" # only collect experience from CE and RE
    return True, "Error exists"

def check_success_exists(file: str, min_speedup: float = 1.0):
    """A log counts as a positive (best-practice) example if it's correct and at least
    min_speedup faster than the PyTorch baseline (fast_p semantics: >1 means faster)."""
    try:
        correctness, fast_p = read_metrics(file)
    except Exception:
        return False, "Could not parse metrics"
    if not correctness:
        return False, "Not a successful kernel"
    if fast_p < min_speedup:
        return False, "Below speedup threshold"
    return True, "Successful kernel"

def collect_experience_from_single_log(file: str, server: callable, model_name: str, filter_max_difference: bool = False):
    if file.endswith("_metrics.txt"):
        with open(file, "r") as f:
            content = f.read()
            if "correctness=False" not in content or (filter_max_difference and "max_difference" in content):
                return False, "Not a CE or RE error" # only collect experience from CE and RE
    elif file.endswith("_metrics.json"):
        metrics = json.load(open(file, "r"))
        if metrics is None:
            return False, "Metrics file not found"
        if metrics["correctness"] == True or (filter_max_difference and "max_difference" in str(metrics)):
            return False, "Not a CE or RE error" # only collect experience from CE and RE
        content = str(metrics)
    else:
        return False, "Unknown metrics file format"
    code_file = file.replace("_metrics.txt", ".py").replace("_metrics.json", ".py")
    if not os.path.exists(code_file):
        print(f"Warning: Code file not found: {code_file}, skipping...")
        return False, "Code file not found"
    with open(code_file, "r") as f:
        code_block = f.read()
    error_message = content
    prompt = PROMPT_TEMPLATE.format(code_block=code_block, error_message=error_message)
    response = query_inference_server(server, model_name, prompt)
    if "no guidance" in response.lower():
        return False, "No guidance"
    return True, response.strip().split("\n")[-1]

def collect_best_practice_from_single_log(file: str, server: callable, model_name: str, min_speedup: float = 1.0):
    success, reason = check_success_exists(file, min_speedup=min_speedup)
    if not success:
        return False, reason
    code_file = file.replace("_metrics.txt", ".py").replace("_metrics.json", ".py")
    if not os.path.exists(code_file):
        print(f"Warning: Code file not found: {code_file}, skipping...")
        return False, "Code file not found"
    with open(code_file, "r") as f:
        code_block = f.read()
    with open(file, "r") as f:
        metrics_content = f.read()
    prompt = POSITIVE_PROMPT_TEMPLATE.format(code_block=code_block, metrics=metrics_content)
    response = query_inference_server(server, model_name, prompt)
    if "no guidance" in response.lower():
        return False, "No guidance"
    return True, response.strip().split("\n")[-1]

def _scan_log_files(log_path: str) -> list:
    """List every step's metrics file in a problem's log folder, excluding the dummy
    root (step_0) and the already-aggregated best_metrics summary."""
    files = os.listdir(log_path)
    return [
        os.path.join(log_path, file) for file in files
        if (file.endswith("_metrics.txt") or file.endswith("_metrics.json"))
        and "best_metrics" not in file and "step_0" not in file
    ]

def _load_knowledge_base(memory_path: str):
    knowledge_base, score_base = [], []
    if os.path.exists(memory_path):
        with open(memory_path, "r") as f:
            for line in f.readlines():
                assert "||" in line, "Knowledge store file is not formatted correctly"
                item, score = line.strip().split("||")
                knowledge_base.append(item.strip())
                score_base.append(float(score.strip()))
    return knowledge_base, score_base

def _write_knowledge_base(memory_path: str, knowledge_base: list, score_base: list):
    with open(memory_path, "w") as f:
        for item, score in zip(knowledge_base, score_base):
            f.write(f"{item}||{score}\n")

def _absorb_into_knowledge_base(response: str, knowledge_base: list, score_base: list, server: callable, model_name: str) -> bool:
    """Append response to the knowledge base, or bump an existing duplicate's score instead. Returns dup_flag."""
    dup_flag = False
    if len(knowledge_base) > 0:
        result = llm_judge_duplicate_batch(
            candidate_text=response,
            existing_texts=knowledge_base,
            existing_unique_indices=list(range(len(knowledge_base))),
            model_name=model_name,
            existing_server=server
        )
        if result is not None:
            matched_unique_idx, confidence = result
            if confidence > 0.5:
                score_base[matched_unique_idx] += 1
                dup_flag = True
    if not dup_flag:
        knowledge_base.append(response)
        score_base.append(1.0)
    return dup_flag

def update_memory(memory_path: str, log_path: str, server: callable, model_name: str, filter_max_difference: bool = False):
    log_path_list = _scan_log_files(log_path)
    knowledge_base, score_base = _load_knowledge_base(memory_path)

    print("-" * 100
          + "\nskill memory (negative: \"you cannot...\")"
          + "\n"
          + "-" * 100
          + "\n")
    print("example skill memory entry:")
    print(knowledge_base[0] if len(knowledge_base) > 0 else "No skill memory")
    print("example score base:")
    print(score_base[0] if len(score_base) > 0 else "No score base")

    for file in log_path_list:
        try:
            success, response = collect_experience_from_single_log(
                file=file, server=server, model_name=model_name, filter_max_difference=filter_max_difference)
            if success:
                dup_flag = _absorb_into_knowledge_base(response, knowledge_base, score_base, server, model_name)
                print("response:")
                print(response)
                print("dup_flag:", dup_flag)
                print("-" * 100)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise e

    _write_knowledge_base(memory_path, knowledge_base, score_base)

def update_positive_memory(memory_path: str, log_path: str, server: callable, model_name: str, min_speedup: float = 1.0):
    log_path_list = _scan_log_files(log_path)
    knowledge_base, score_base = _load_knowledge_base(memory_path)

    print("-" * 100
          + "\nskill memory (positive: \"you should...\")"
          + "\n"
          + "-" * 100
          + "\n")
    print("example best-practice entry:")
    print(knowledge_base[0] if len(knowledge_base) > 0 else "No best-practice memory")

    for file in log_path_list:
        try:
            success, response = collect_best_practice_from_single_log(
                file=file, server=server, model_name=model_name, min_speedup=min_speedup)
            if success:
                dup_flag = _absorb_into_knowledge_base(response, knowledge_base, score_base, server, model_name)
                print("response:")
                print(response)
                print("dup_flag:", dup_flag)
                print("-" * 100)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise e

    _write_knowledge_base(memory_path, knowledge_base, score_base)

if __name__ == "__main__":
    args = parse_args()
    log_dir = args.log_dir
    logs = sorted([log for log in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, log))])
    server = create_inference_server(args.server)
    model_name = args.model_name

    # --mode selects which kind of memory this invocation builds: negative ("you cannot...",
    # the default, unchanged from before) from failed logs, or positive ("you should...")
    # from correct, at-least-min-speedup logs. Run the script twice with two different
    # --knowledge-store-path values to build both files.
    if args.mode == "successes":
        collector_fn = lambda file: collect_best_practice_from_single_log(
            file=file, server=server, model_name=model_name, min_speedup=args.min_speedup)
        precount_fn = lambda file: check_success_exists(file, args.min_speedup)
        precount_label = "successful"
    else:
        collector_fn = lambda file: collect_experience_from_single_log(
            file=file, server=server, model_name=model_name, filter_max_difference=args.filter_max_difference)
        precount_fn = lambda file: check_error_exists(file, args.filter_max_difference)
        precount_label = "error"

    if args.file:
        # single log
        success, response = collector_fn(args.file)
        if success:
            print(response)
        else:
            print(f"Error: {response}")
        exit()

    # load all log paths
    log_path_list = []
    for log in logs:
        files = os.listdir(os.path.join(log_dir, log))
#        if not log.split("_")[-1].isdigit() or int(log.split("_")[-1]) > 20: continue # skip logs after 20 steps
        for file in files:
            if (file.endswith("_metrics.txt") or file.endswith("_metrics.json")) and not file.endswith("_best_metrics.txt") and not "step_0" in file:
                log_path_list.append(os.path.join(log_dir, log, file))

    if args.progress_file is not None:
        with open(args.progress_file, "r") as f:
            processed_log_path_list = f.readlines()
            processed_log_path_list = [line.strip() for line in processed_log_path_list]
    else:
        processed_log_path_list = []

    random.seed(args.seed)
    random.shuffle(log_path_list)

    print(f"Total {len(log_path_list)} logs to process")
    total_precount = 0
    for log_path in log_path_list:
        success, response = precount_fn(log_path)
        if success:
            total_precount += 1
    print(f"Total {total_precount} {precount_label} logs to process")

    # Load the existing skill memory file.
    knowledge_store_path = args.knowledge_store_path
    knowledge_base, score_base = _load_knowledge_base(knowledge_store_path)

    log_path_list = log_path_list[:args.max_logs]
    new_log_path_list = []
    processed_count = 0
    pbar = tqdm(enumerate(log_path_list), total=len(log_path_list), desc="Processing logs")
    for idx, log_path in pbar:
        if log_path in processed_log_path_list:
            continue
        new_log_path_list.append(log_path)
        try:
            success, response = collector_fn(log_path)
            if success:
                _absorb_into_knowledge_base(response, knowledge_base, score_base, server, model_name)
            else:
                tqdm.write(f"Warning: {response}")
        except Exception as e:
            tqdm.write(f"Warning: Error processing {log_path}: {e}")

        processed_count += 1
        # Update progress bar with skill memory size.
        pbar.set_postfix({"KB size": len(knowledge_base)})
        # Persist the skill memory every 100 logs.
        if processed_count % 100 == 0:
            tqdm.write(f"Processed {processed_count} logs, {len(knowledge_base)} knowledge items collected")
            # Update the skill memory file.
            _write_knowledge_base(knowledge_store_path, knowledge_base, score_base)
            # add the new log path list to the progress file
            if args.progress_file is not None:
                with open(args.progress_file, "a") as f:
                    f.write("\n".join(new_log_path_list) + "\n")
                new_log_path_list = []

    # Save the final skill memory file.
    _write_knowledge_base(knowledge_store_path, knowledge_base, score_base)

    if args.progress_file is not None:
        with open(args.progress_file, "a") as f:
            f.write("\n".join(new_log_path_list) + "\n")

    print(f"Processed {processed_count} logs, {len(knowledge_base)} knowledge items collected")