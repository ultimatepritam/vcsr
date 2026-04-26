"""Export paper-facing VCSR tables and figure specs from frozen artifacts.

This script is intentionally read-only with respect to experiment outputs. It
aggregates existing JSON/JSONL/Markdown artifacts and writes paper-prep files
under results/paper/final_vcsr by default.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


POLICY_LABELS = {
    "greedy_first": "Greedy first",
    "random_parseable": "Random parseable",
    "verifier_ranked": "Verifier-ranked",
    "verifier_ranked_repair": "VCSR repair-augmented",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fmt(value: float | int | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.4f}"


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate_main_tables(final_dir: Path) -> dict[str, Any]:
    seeds = sorted(
        int(p.name.split("_")[-1])
        for p in final_dir.iterdir()
        if p.is_dir() and p.name.startswith("seed_")
    )
    if not seeds:
        raise FileNotFoundError(f"No seed_* directories found under {final_dir}")

    by_k_policy: dict[tuple[int, str], list[dict[str, float]]] = defaultdict(list)
    by_slice: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    per_seed: list[dict[str, Any]] = []
    repair_outcomes = Counter()
    repair_by_domain: dict[str, Counter[str]] = defaultdict(Counter)
    repair_by_style: dict[str, Counter[str]] = defaultdict(Counter)

    for seed in seeds:
        seed_dir = final_dir / f"seed_{seed}"
        aggregate = read_json(seed_dir / "aggregate_metrics.json")
        for k_str, k_payload in aggregate["comparisons"].items():
            k = int(k_str)
            for policy, policy_payload in k_payload["policies"].items():
                metrics = policy_payload["metrics"]
                by_k_policy[(k, policy)].append(
                    {
                        "parse_rate": metrics["parse_rate"],
                        "equiv_rate": metrics["equiv_rate"],
                        "equiv_given_parse": metrics["equiv_given_parse"],
                    }
                )

                if k == 8:
                    for slice_name in (
                        "domain=blocksworld",
                        "domain=gripper",
                        "style=abstract/abstract",
                        "style=explicit/explicit",
                    ):
                        slice_metrics = policy_payload["stratified"].get(slice_name)
                        if slice_metrics:
                            key = (policy, slice_name)
                            by_slice[key]["total"] += int(slice_metrics["total"])
                            by_slice[key]["parse"] += int(slice_metrics["parse_count"])
                            by_slice[key]["equiv"] += int(slice_metrics["equiv_count"])

        summary = read_json(final_dir / "final_repair_gate_summary.json")
        seed_report = next(r for r in summary["seed_reports"] if r["seed"] == seed)
        per_seed.append(
            {
                "seed": seed,
                "baseline_k8": seed_report["baseline_k8_equiv_rate"],
                "repair_k8": seed_report["repair_augmented_k8_equiv_rate"],
                "delta": seed_report["delta_k8"],
                "helped": seed_report["repair_metrics"]["helped"],
                "hurt": seed_report["repair_metrics"]["hurt"],
                "tied": seed_report["repair_metrics"]["tied"],
                "repair_parse_rate": seed_report["repair_metrics"][
                    "repair_parse_rate"
                ],
            }
        )

        for row in read_jsonl(seed_dir / "repair_outputs.jsonl"):
            outcome = row.get("outcome", "unknown")
            domain = row.get("domain", "unknown")
            style = row.get("style", "unknown")
            repair_outcomes[outcome] += 1
            repair_by_domain[domain][outcome] += 1
            repair_by_style[style][outcome] += 1

    table_1 = []
    for (k, policy), rows in sorted(by_k_policy.items()):
        table_1.append(
            {
                "k": k,
                "policy": policy,
                "label": POLICY_LABELS.get(policy, policy),
                "mean_parse": mean([r["parse_rate"] for r in rows]),
                "mean_equiv": mean([r["equiv_rate"] for r in rows]),
                "mean_equiv_given_parse": mean(
                    [r["equiv_given_parse"] for r in rows]
                ),
            }
        )

    table_2 = []
    for (policy, slice_name), counts in sorted(by_slice.items()):
        total = counts["total"]
        parse = counts["parse"]
        equiv = counts["equiv"]
        table_2.append(
            {
                "policy": policy,
                "label": POLICY_LABELS.get(policy, policy),
                "slice": slice_name,
                "equiv_count": equiv,
                "total": total,
                "parse_rate": parse / total if total else 0.0,
                "equiv_rate": equiv / total if total else 0.0,
            }
        )

    final_summary = read_json(final_dir / "final_repair_gate_summary.json")
    return {
        "seeds": seeds,
        "table_1_main_metrics": table_1,
        "table_2_k8_slices": table_2,
        "per_seed_final_gate": per_seed,
        "repair_outcomes": dict(repair_outcomes),
        "repair_by_domain": {k: dict(v) for k, v in repair_by_domain.items()},
        "repair_by_style": {k: dict(v) for k, v in repair_by_style.items()},
        "final_gate_mean_metrics": final_summary["mean_metrics"],
        "accepted": final_summary["accepted"],
    }


def read_supporting_summaries(args: argparse.Namespace) -> dict[str, Any]:
    data: dict[str, Any] = {}
    paths = {
        "search_ablation": Path(args.search_ablation),
        "fixed_pool_round7": Path(args.fixed_pool_round7),
        "repair_pilot": Path(args.repair_pilot),
        "domainaware_repair": Path(args.domainaware_repair),
        "guarded_repair": Path(args.guarded_repair),
    }
    for key, path in paths.items():
        data[key] = {
            "path": str(path),
            "text": path.read_text(encoding="utf-8") if path.exists() else "",
        }
    return data


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def write_markdown(artifacts: dict[str, Any], out_path: Path) -> None:
    table_1_rows = [
        [
            str(r["k"]),
            r["label"],
            fmt(r["mean_parse"]),
            fmt(r["mean_equiv"]),
            fmt(r["mean_equiv_given_parse"]),
        ]
        for r in artifacts["table_1_main_metrics"]
    ]
    table_2_rows = [
        [
            r["slice"],
            r["label"],
            f"{r['equiv_count']}/{r['total']}",
            fmt(r["equiv_rate"]),
            fmt(r["parse_rate"]),
        ]
        for r in artifacts["table_2_k8_slices"]
    ]
    seed_rows = [
        [
            str(r["seed"]),
            fmt(r["baseline_k8"]),
            fmt(r["repair_k8"]),
            f"{r['delta']:+.4f}",
            str(r["helped"]),
            str(r["hurt"]),
            str(r["tied"]),
            fmt(r["repair_parse_rate"]),
        ]
        for r in artifacts["per_seed_final_gate"]
    ]

    mean_metrics = artifacts["final_gate_mean_metrics"]
    content = f"""# Paper Tables: Final VCSR Evidence

Source: `results/vcsr/final_repair_gate_round4`

## Main Result

- Final seeds: `{artifacts['seeds']}`
- Accepted: `{artifacts['accepted']}`
- Mean K=8 verifier-ranked: `{fmt(mean_metrics['baseline_k8_equiv_rate'])}`
- Mean K=8 repair-augmented VCSR: `{fmt(mean_metrics['repair_augmented_k8_equiv_rate'])}`
- Mean K=8 delta: `{mean_metrics['mean_delta_k8']:+.4f}`
- Repair parse rate: `{fmt(mean_metrics['repair_parse_rate_mean'])}`
- Helped / hurt: `{mean_metrics['total_helped']} / {mean_metrics['total_hurt']}`

## Table 1: Aggregate Metrics by Policy and K

{markdown_table(['K', 'Policy', 'Parse', 'Equiv', 'Equiv / Parse'], table_1_rows)}

## Table 2: K=8 Domain and Style Breakdown

{markdown_table(['Slice', 'Policy', 'Equiv Count', 'Equiv Rate', 'Parse Rate'], table_2_rows)}

## Table 3: Final Gate Per-Seed Outcomes

{markdown_table(['Seed', 'Verifier K=8', 'Repair K=8', 'Delta', 'Helped', 'Hurt', 'Tied', 'Repair Parse'], seed_rows)}

## Repair Outcome Counts

```json
{json.dumps(artifacts['repair_outcomes'], indent=2)}
```
"""
    out_path.write_text(content, encoding="utf-8")


def write_claims(out_path: Path) -> None:
    content = """# Paper Claim Sheet

## Recommended Main Claim

VCSR improves semantic equivalence for text-to-PDDL generation by combining
verifier-ranked best-of-K search with one-step domain-aware repair. The strongest
evidence is the final untouched seed gate (`51-55`) at `K=8`, where repair
raises semantic equivalence from `0.4200` to `0.7720`.

## What To Claim

- Semantic equivalence, not parseability or solvability, is the primary metric.
- Verifier-ranked search is useful as the scaffold for repair.
- Planner/solvability-only policies are weak semantic selectors.
- Repair-augmented VCSR is the paper-facing system.

## What Not To Claim

- Do not claim repair is uniformly harmless.
- Do not claim final results on `floor-tile` or unseen domains.
- Do not claim abstention is the final system mechanism.
- Do not promote guarded repair; it did not reduce blocksworld hurts.
- Do not tune anything on seeds `51-55`.
"""
    out_path.write_text(content, encoding="utf-8")


def write_figures(out_path: Path, artifacts: dict[str, Any]) -> None:
    mean_metrics = artifacts["final_gate_mean_metrics"]
    repair_outcomes = artifacts["repair_outcomes"]
    helped = repair_outcomes.get("repair_helped", 0)
    hurt = repair_outcomes.get("repair_hurt", 0)
    both_success = repair_outcomes.get("both_success", 0)
    both_fail = repair_outcomes.get("both_fail", 0)

    content = f"""# Paper Figure Specs

## Figure 1: VCSR Pipeline

```mermaid
flowchart LR
  A["Natural-language task"] --> B["Generate K PDDL candidates"]
  B --> C["Parse candidates"]
  C --> D["Score parseable candidates with frozen semantic verifier"]
  D --> E["Select verifier-ranked candidate"]
  E --> F["One-step domain-aware repair at K=8"]
  F --> G["Evaluate semantic equivalence"]
```

## Figure 2: Evidence Funnel

```mermaid
flowchart TD
  A["Verifier training and calibration"] --> B["Round 4 frozen verifier"]
  B --> C["Verifier-ranked K=8: {fmt(mean_metrics['baseline_k8_equiv_rate'])}"]
  C --> D["Planner/search ablation: no reliable gain"]
  D --> E["Domain-aware repair"]
  E --> F["Repair-augmented K=8: {fmt(mean_metrics['repair_augmented_k8_equiv_rate'])}"]
```

## Figure 3: Repair Help/Hurt Outcomes

```mermaid
pie title Final repair outcomes on seeds 51-55
  "Helped" : {helped}
  "Hurt" : {hurt}
  "Both success" : {both_success}
  "Both fail" : {both_fail}
```
"""
    out_path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--final_dir",
        default="results/vcsr/final_repair_gate_round4",
        help="Frozen final repair gate directory.",
    )
    parser.add_argument(
        "--output_dir",
        default="results/paper/final_vcsr",
        help="Directory for derived paper-prep artifacts.",
    )
    parser.add_argument(
        "--search_ablation",
        default="results/vcsr/search_ablation_round4/search_ablation_summary.md",
    )
    parser.add_argument(
        "--fixed_pool_round7",
        default="results/vcsr/fixed_pool_round7_compare/comparison_summary.md",
    )
    parser.add_argument(
        "--repair_pilot",
        default="results/vcsr/repair_pilot_round4/repair_summary.md",
    )
    parser.add_argument(
        "--domainaware_repair",
        default=(
            "results/vcsr/fresh_repair_gate_round4_domainaware/"
            "fresh_repair_gate_summary.md"
        ),
    )
    parser.add_argument(
        "--guarded_repair",
        default=(
            "results/vcsr/final_guarded_repair_gate_round4/"
            "final_repair_gate_summary.md"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    final_dir = Path(args.final_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = aggregate_main_tables(final_dir)
    artifacts["supporting_summaries"] = read_supporting_summaries(args)

    (output_dir / "paper_tables.json").write_text(
        json.dumps(artifacts, indent=2), encoding="utf-8"
    )
    write_markdown(artifacts, output_dir / "paper_tables.md")
    write_claims(output_dir / "paper_claims.md")
    write_figures(output_dir / "figure_specs.md", artifacts)

    print(f"Wrote paper artifacts to {output_dir}")


if __name__ == "__main__":
    main()
