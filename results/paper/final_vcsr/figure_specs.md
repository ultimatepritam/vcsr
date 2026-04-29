# Paper Figure Specs

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
  B --> C["Verifier-ranked K=8: 0.4200"]
  C --> D["Planner/search ablation: no reliable gain"]
  D --> E["Domain-aware repair"]
  E --> F["Repair-augmented K=8: 0.7720"]
```

## Figure 3: Repair Help/Hurt Outcomes

```mermaid
pie title Final repair outcomes on seeds 51-55
  "Helped" : 104
  "Hurt" : 16
  "Both success" : 89
  "Both fail" : 41
```

## Appendix Figure: Claude-Family Robustness

```mermaid
xychart-beta
  title "K=8 Semantic Equivalence by Claude Generator"
  x-axis ["Haiku 4.5", "Sonnet 4.5", "Opus 4.6"]
  y-axis "Equivalence" 0 --> 1
  bar "Prompt K=1" [0.4000, 0.5000, 0.3667]
  bar "Verifier K=8" [0.4667, 0.5333, 0.4333]
  bar "VCSR Repair K=8" [0.9000, 0.9333, 0.9000]
```
