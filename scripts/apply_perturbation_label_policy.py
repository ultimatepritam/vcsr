"""Fix rare perturbation+label=1 rows in verifier JSONL without re-running neggen.

Usage:
  python scripts/apply_perturbation_label_policy.py \\
    results/neggen/pilot/verifier_train.jsonl \\
    results/neggen/pilot/verifier_train.relabeled.jsonl \\
    --policy relabel

Policies:
  relabel — set label to 0 (default; perturbations are synthetic negatives)
  drop    — omit those lines entirely
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument(
        "--policy",
        choices=("relabel", "drop"),
        default="relabel",
    )
    args = p.parse_args()

    n_in = n_touch = 0
    out_lines: list[str] = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            d = json.loads(line)
            if d.get("source") == "perturbation" and int(d.get("label", -1)) == 1:
                n_touch += 1
                if args.policy == "drop":
                    continue
                d["label"] = 0
                d["perturbation_positive_policy"] = "relabel"
            out_lines.append(json.dumps(d, ensure_ascii=False))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for line in out_lines:
            f.write(line + "\n")

    print(
        f"Read {n_in} lines; perturbation+positive: {n_touch} "
        f"({args.policy}); wrote {len(out_lines)} lines to {args.output}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
