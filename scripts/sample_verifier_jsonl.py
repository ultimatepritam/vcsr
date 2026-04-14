"""Quick stratified sample from verifier_train.jsonl for sanity review."""
import argparse
import json
import random
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        default="results/neggen/pilot/verifier_train.jsonl",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-stratum", type=int, default=3)
    args = parser.parse_args()

    random.seed(args.seed)
    by_key: dict[tuple[str, int], list[dict]] = defaultdict(list)
    with open(args.path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            key = (d.get("source", "?"), int(d.get("label", -1)))
            by_key[key].append(d)

    print("Stratum counts (source, label) -> n")
    for k in sorted(by_key.keys()):
        print(f"  {k}: {len(by_key[k])}")
    print()

    n = args.per_stratum
    for key in sorted(by_key.keys()):
        rows = by_key[key]
        sample = random.sample(rows, min(n, len(rows)))
        print("=" * 72)
        print(f"STRATUM source={key[0]!r} label={key[1]}  ({len(sample)} of {len(rows)})")
        print("=" * 72)
        for i, d in enumerate(sample, 1):
            nl = (d.get("nl") or "")[:400].replace("\n", " ")
            pddl = (d.get("pddl") or "")[:500].replace("\n", " ")
            name = (d.get("planetarium_name") or "")[:100]
            print(f"--- sample {i} | name={name}")
            print(f"nl (trunc): {nl}...")
            print(f"pddl (trunc): {pddl}...")
            if d.get("perturbation_type"):
                print(f"perturbation_type: {d['perturbation_type']}")
            if d.get("source_model"):
                print(f"source_model: {d['source_model']}")
            print()


if __name__ == "__main__":
    main()
