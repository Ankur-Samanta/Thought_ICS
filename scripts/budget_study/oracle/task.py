"""Print the complete task text for one judge on one batch (paste into any agent/CLI).

    python scripts/budget_study/oracle/task.py <oracle_dir> <judge_name> <batch_k>

judge_name names the label file: <oracle_dir>/labels_<judge_name>_<k>.json. Use a name without
underscores that identifies the judge model, e.g. codex-gpt5, opus, sonnet, fable.
"""
import sys
from pathlib import Path

odir, judge, k = Path(sys.argv[1]).resolve(), sys.argv[2], sys.argv[3]
assert "_" not in judge, "judge name must not contain '_' (it is the file-name separator)"
prompt = (Path(__file__).parent / "judge_prompt.md").read_text()
print(prompt)
print(f"\n## Your files\n\n- Input batch: `{odir / f'batch_{k}.json'}`\n"
      f"- Output (write exactly this file): `{odir / f'labels_{judge}_{k}.json'}`\n")
