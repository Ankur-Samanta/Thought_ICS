"""Utility functions from StepCo (verbatim from wzy6642/StepCo/StepCo/utils.py).

Source: https://github.com/wzy6642/StepCo/blob/main/StepCo/utils.py

Note: We omit `post_process_value` because StepCo's version uses eval() on
numeric strings, which doesn't work for our non-math benchmarks (CSQA, GPQA).
We use the harness-wide `normalize_answer` from dataset_loaders.py instead.
"""

import re


def get_reasoning_steps(solution: str) -> list:
    """Parse <Step N>...</Step N>-tagged reasoning into a list of step strings."""
    steps = re.split(r'<Step(?: \d+)?>', solution)
    steps = [step.strip() for step in steps]
    pattern = r'<Step(?: \d+)?>'
    steps = [re.sub(pattern, '', step) for step in steps]
    pattern = r'</Step(?: \d+)?>'
    steps = [re.sub(pattern, '', step) for step in steps]
    for idx in range(len(steps)):
        if len(steps[0]) < 5:
            dec = 1
        else:
            dec = 0
        if idx != len(steps) - 1 and len(steps[idx]) >= 5:
            steps[idx] = f'<Step {idx + 1 - dec}> ' + steps[idx] + f' </Step {idx + 1 - dec}>'
    return steps


def find_first_smaller_index(arr, threshold):
    """Return 1-indexed position of the first array element below `threshold`,
    or 0 if no element is below threshold."""
    for i in range(len(arr)):
        if arr[i] < threshold:
            return i + 1
    return 0
