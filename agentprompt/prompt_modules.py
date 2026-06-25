import os

EXPERIENCE_GUIDANCE = """## Common Pitfalls (from previous optimization attempts)

The following are frequently observed failure patterns from prior kernel optimization runs. **Avoid** these patterns in your implementation:

{experience_guidance}

"""

BEST_PRACTICE_GUIDANCE = """## Best Practices (from previous successful kernels)

The following are recurring techniques behind correct, at-least-as-fast-as-baseline kernels from prior optimization runs. **Apply** these practices where relevant in your implementation:

{best_practice_guidance}

"""

HARDWARE_INFORMATION = """## Hardware Information

Here is some information about the underlying hardware that you should keep in mind:

- The GPU that will run the kernel is NVIDIA {gpu_name}, {gpu_architecture} architecture.

"""

def generate_experience_guidance_prompt(experience_guidance_path: str, threshold: int=2) -> str:
    """
    threshold: the threshold of the experience guidance, if the experience guidance is less than the threshold, it will not be included in the experience guidance
    """
    if experience_guidance_path is None or not os.path.exists(experience_guidance_path):
        return ""
    with open(experience_guidance_path, "r") as f:
        lines = f.readlines()
        experience_guidance_content = []
        for line in lines:
            if float(line.strip().split("||")[1]) < threshold:
                continue
            experience_guidance_content.append(line.strip().split("||")[0])
        experience_guidance_content = "\n".join(experience_guidance_content)
    return EXPERIENCE_GUIDANCE.format(experience_guidance=experience_guidance_content)

def generate_best_practice_prompt(best_practice_path: str, threshold: int=2) -> str:
    """
    threshold: same semantics as generate_experience_guidance_prompt, but applied to the
    positive ("you should...") memory store instead of the negative ("you cannot...") one.
    """
    if best_practice_path is None or not os.path.exists(best_practice_path):
        return ""
    with open(best_practice_path, "r") as f:
        lines = f.readlines()
        best_practice_content = []
        for line in lines:
            if float(line.strip().split("||")[1]) < threshold:
                continue
            best_practice_content.append(line.strip().split("||")[0])
        best_practice_content = "\n".join(best_practice_content)
    return BEST_PRACTICE_GUIDANCE.format(best_practice_guidance=best_practice_content)

def generate_hardware_information_prompt(gpu_name: str, gpu_architecture: str) -> str:
    if gpu_name is None or gpu_architecture is None:
        return ""
    return HARDWARE_INFORMATION.format(gpu_name=gpu_name, gpu_architecture=gpu_architecture)