#!/usr/bin/env python3
"""
Test batch scheduler with dummy API calls to verify logic before notebook integration.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from collections import deque
import random

# =============================================================================
# Mock API for Testing
# =============================================================================

def mock_batch_inference(prompts: List[str]) -> List[str]:
    """Mock inference that returns dummy responses."""
    responses = []
    for prompt in prompts:
        if "Initial solution" in prompt:
            # 70% chance of wrong answer
            if random.random() < 0.7:
                responses.append("Solution: The answer is 42. \\boxed{42}")
            else:
                responses.append("Solution: The answer is 7. \\boxed{7}")

        elif "check for errors" in prompt or "verify" in prompt:
            # 80% chance of detecting error if wrong
            if random.random() < 0.8:
                responses.append("I found an error. \\boxed{ERROR}")
            else:
                responses.append("Solution is correct. \\boxed{CORRECT}")

        elif "regenerate" in prompt or "Error analysis" in prompt:
            # 30% chance of fixing error
            if random.random() < 0.3:
                responses.append("Corrected solution. \\boxed{7}")
            else:
                responses.append("Still wrong. \\boxed{99}")

        else:
            responses.append("\\boxed{42}")

    return responses


def extract_boxed_answer(text: str) -> str:
    """Simple extraction for testing."""
    import re
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    return match.group(1) if match else "NO_ANSWER"


# =============================================================================
# Problem State Management
# =============================================================================

@dataclass
class ProblemState:
    """State for a single problem being evaluated."""
    problem_id: str
    problem: str
    ground_truth: str

    # Current state
    current_solution: str = ""
    current_answer: str = ""
    iteration: int = 0

    # History
    iterations_data: List[Dict] = field(default_factory=list)
    error_reasoning: str = ""

    # Control flow
    next_action: str = "initial_solution"  # "initial_solution", "error_detection", "regeneration"
    is_done: bool = False
    success: bool = False

    # Config
    max_iterations: int = 10
    autonomy_level: int = 3


@dataclass
class BatchSlot:
    """A slot in the batch that can hold a problem."""
    state: Optional[ProblemState] = None

    def is_empty(self) -> bool:
        return self.state is None

    def clear(self):
        self.state = None


# =============================================================================
# Prompt Building
# =============================================================================

def build_prompt(state: ProblemState) -> str:
    """Build the next prompt for this problem based on its state."""

    if state.next_action == "initial_solution":
        return f"""Initial solution for problem {state.problem_id}:
Problem: {state.problem}
Provide solution with answer in \\boxed{{}}."""

    elif state.next_action == "error_detection":
        if state.autonomy_level == 1:
            # Oracle
            prompt = f"""Problem: {state.problem}

Current solution (WRONG):
{state.current_solution}

Correct answer: {state.ground_truth}

Find the error and respond with \\boxed{{ERROR}} or \\boxed{{CORRECT}}"""
        elif state.autonomy_level == 2:
            # Binary feedback
            prompt = f"""Problem: {state.problem}

Current solution (WRONG):
{state.current_solution}

Your answer is incorrect. Check for errors.
Respond with \\boxed{{ERROR}} or \\boxed{{CORRECT}}"""
        else:
            # Autonomous
            prompt = f"""Problem: {state.problem}

Current solution:
{state.current_solution}

Carefully verify your solution and check for errors.
Respond with \\boxed{{ERROR}} or \\boxed{{CORRECT}}"""

        return prompt

    elif state.next_action == "regeneration":
        return f"""Problem: {state.problem}

Previous attempt:
{state.current_solution}

Error analysis:
{state.error_reasoning}

Based on the error, regenerate the solution with answer in \\boxed{{}}."""

    else:
        raise ValueError(f"Unknown action: {state.next_action}")


# =============================================================================
# Response Processing
# =============================================================================

def process_response(state: ProblemState, response: str) -> None:
    """Process response and update problem state."""

    if state.next_action == "initial_solution":
        # Got initial solution
        state.current_solution = response
        state.current_answer = extract_boxed_answer(response)

        # Record iteration
        state.iterations_data.append({
            'iteration': state.iteration,
            'solution': state.current_solution,
            'answer': state.current_answer,
            'correct': state.current_answer == state.ground_truth
        })

        # Check if correct
        if state.current_answer == state.ground_truth:
            state.is_done = True
            state.success = True
            print(f"  [{state.problem_id}] ✓ CORRECT on iter {state.iteration}")
        else:
            # Move to error detection
            state.next_action = "error_detection"
            print(f"  [{state.problem_id}] Wrong answer: {state.current_answer} (expected {state.ground_truth})")

    elif state.next_action == "error_detection":
        # Got error detection result
        state.error_reasoning = response
        boxed = extract_boxed_answer(response)
        has_error = (boxed == "ERROR")

        if has_error:
            # Model found error, will regenerate
            state.next_action = "regeneration"
            print(f"  [{state.problem_id}] Error detected, will regenerate")
        else:
            # Model thinks it's correct but it's not - stop
            state.is_done = True
            state.success = False
            print(f"  [{state.problem_id}] ✗ No error found, stopping")

    elif state.next_action == "regeneration":
        # Got regenerated solution
        state.current_solution = response
        state.current_answer = extract_boxed_answer(response)
        state.iteration += 1

        # Record iteration
        state.iterations_data.append({
            'iteration': state.iteration,
            'solution': state.current_solution,
            'answer': state.current_answer,
            'correct': state.current_answer == state.ground_truth,
            'error_reasoning': state.error_reasoning
        })

        # Check if correct now
        if state.current_answer == state.ground_truth:
            state.is_done = True
            state.success = True
            print(f"  [{state.problem_id}] ✓ CORRECTED on iter {state.iteration}")
        elif state.iteration >= state.max_iterations:
            state.is_done = True
            state.success = False
            print(f"  [{state.problem_id}] ✗ Max iterations reached")
        else:
            # Try again
            state.next_action = "error_detection"
            print(f"  [{state.problem_id}] Still wrong: {state.current_answer}, checking again")


# =============================================================================
# Batch Scheduler
# =============================================================================

class BatchScheduler:
    """Manages batch evaluation with dynamic slot filling."""

    def __init__(self, batch_size: int, max_iterations: int = 10, autonomy_level: int = 3):
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.autonomy_level = autonomy_level

        # Initialize empty batch slots
        self.slots: List[BatchSlot] = [BatchSlot() for _ in range(batch_size)]

        # Problem queue
        self.problem_queue: deque = deque()

        # Completed results
        self.completed_results: List[Dict] = []

        # Stats
        self.total_batch_calls = 0

    def add_problems(self, problems: List[Dict]):
        """Add problems to the queue."""
        self.problem_queue.extend(problems)

    def fill_empty_slots(self):
        """Fill any empty slots with problems from queue."""
        for slot in self.slots:
            if slot.is_empty() and self.problem_queue:
                problem_data = self.problem_queue.popleft()
                slot.state = ProblemState(
                    problem_id=problem_data['id'],
                    problem=problem_data['problem'],
                    ground_truth=problem_data['answer'],
                    max_iterations=self.max_iterations,
                    autonomy_level=self.autonomy_level
                )
                print(f"[FILL] Loaded {slot.state.problem_id} into slot")

    def has_active_work(self) -> bool:
        """Check if there's any work left to do."""
        return any(not slot.is_empty() for slot in self.slots) or len(self.problem_queue) > 0

    def get_active_states(self) -> List[ProblemState]:
        """Get all active problem states (non-empty slots)."""
        return [slot.state for slot in self.slots if not slot.is_empty()]

    def run(self):
        """Main evaluation loop."""
        print(f"\n{'='*80}")
        print(f"Starting Batch Scheduler (batch_size={self.batch_size})")
        print(f"Total problems: {len(self.problem_queue)}")
        print(f"{'='*80}\n")

        # Fill initial batch
        self.fill_empty_slots()

        while self.has_active_work():
            # Build batch of prompts
            prompts = []
            active_slots = []

            for slot in self.slots:
                if not slot.is_empty():
                    prompt = build_prompt(slot.state)
                    prompts.append(prompt)
                    active_slots.append(slot)

            if not prompts:
                break  # No active work

            print(f"\n--- Batch {self.total_batch_calls + 1} ({len(prompts)} prompts) ---")

            # Single batch inference call
            responses = mock_batch_inference(prompts)
            self.total_batch_calls += 1

            # Process responses
            for slot, response in zip(active_slots, responses):
                process_response(slot.state, response)

                # If done, save result and clear slot
                if slot.state.is_done:
                    self.completed_results.append({
                        'problem_id': slot.state.problem_id,
                        'problem': slot.state.problem,
                        'ground_truth': slot.state.ground_truth,
                        'success': slot.state.success,
                        'iterations_data': slot.state.iterations_data,
                        'total_iterations': len(slot.state.iterations_data)
                    })
                    slot.clear()

            # Fill empty slots with new problems
            self.fill_empty_slots()

        print(f"\n{'='*80}")
        print("EVALUATION COMPLETE")
        print(f"{'='*80}")
        print(f"Total batch calls: {self.total_batch_calls}")
        print(f"Completed problems: {len(self.completed_results)}")
        successful = sum(1 for r in self.completed_results if r['success'])
        print(f"Success rate: {successful}/{len(self.completed_results)} = {successful/len(self.completed_results)*100:.1f}%")

        return self.completed_results


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    random.seed(42)

    # Create test problems
    test_problems = [
        {'id': f'P{i:02d}', 'problem': f'Problem {i}', 'answer': '7'}
        for i in range(1, 21)  # 20 problems
    ]

    # Run with batch size 5
    scheduler = BatchScheduler(
        batch_size=5,
        max_iterations=5,
        autonomy_level=3
    )
    scheduler.add_problems(test_problems)
    results = scheduler.run()

    # Show some results
    print("\nSample Results:")
    for result in results[:3]:
        print(f"\n{result['problem_id']}: {'SUCCESS' if result['success'] else 'FAILED'}")
        print(f"  Iterations: {result['total_iterations']}")
        for iter_data in result['iterations_data']:
            print(f"    Iter {iter_data['iteration']}: answer={iter_data['answer']}, correct={iter_data['correct']}")
