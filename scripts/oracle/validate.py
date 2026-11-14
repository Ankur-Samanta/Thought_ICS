"""Check every labels_<judge>_<k>.json against its batch: all ids present once, steps in range.

    python scripts/oracle/validate.py <oracle_dir>        # exit 1 if any problem
"""
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

odir = Path(sys.argv[1])
batches = {Path(f).stem.split("_")[1]: json.load(open(f)) for f in glob.glob(str(odir / "batch_*.json"))}
status, ok_all = defaultdict(dict), True
for f in sorted(glob.glob(str(odir / "labels_*_*.json"))):
    _, judge, k = Path(f).stem.split("_")
    problems = []
    try:
        labels = json.load(open(f))
    except Exception as e:  # noqa: BLE001
        problems.append(f"not valid JSON ({e})")
        labels = []
    items = {x["id"]: len(x["steps"]) for x in batches.get(k, [])}
    seen = defaultdict(int)
    for x in labels:
        i, s = x.get("id"), x.get("first_error_step")
        seen[i] += 1
        if i not in items:
            problems.append(f"unknown id {i}")
        elif not isinstance(s, int) or not 0 <= s <= items[i]:
            problems.append(f"{i}: first_error_step {s!r} not an int in [0, {items[i]}]")
        if x.get("confidence") not in ("high", "medium", "low"):
            problems.append(f"{i}: confidence {x.get('confidence')!r}")
    problems += [f"missing {i}" for i in items if seen[i] == 0]
    problems += [f"duplicate {i}" for i, c in seen.items() if c > 1]
    status[judge][k] = "OK" if not problems else f"{len(problems)} problem(s)"
    if problems:
        ok_all = False
        print(f"{Path(f).name}: " + "; ".join(problems[:5]))
for judge, ks in sorted(status.items()):
    print(f"{judge:16s} " + "  ".join(f"batch_{k}:{v}" for k, v in sorted(ks.items()))
          + ("" if len(ks) == len(batches) else f"  (missing {len(batches) - len(ks)} batch(es))"))
sys.exit(0 if ok_all else 1)
