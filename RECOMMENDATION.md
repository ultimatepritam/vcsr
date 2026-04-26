# Recommendation

This document captures the current project-level recommendation for VCSR after
the first verifier-ranked best-of-K pilot, fixed-pool replay evaluation, four
ranking-aligned verifier training rounds, repeated fresh held-out comparisons,
fixed-round-4 selector analysis, focused pointwise round 7, planner/search
ablations, and the final repair-augmented VCSR gate.

## Goal

The project goal is not just to train a verifier with decent offline metrics.
The real goal is to use that verifier to improve downstream NL-to-PDDL
decisions:

- rank multiple generated PDDL candidates for the same natural-language input
- choose semantically equivalent candidates more reliably than greedy decoding
  or simple non-verifier baselines
- use targeted repair to recover from verifier-selected but semantically wrong
  candidates without exposing gold PDDL to the generator
- later support abstention with trustworthy confidence estimates

## What We Now Know

- The end-to-end VCSR pipeline works.
- The generator can produce meaningful candidate diversity.
- The verifier has real offline signal and can be trained reproducibly.
- Fixed-pool replay is now implemented, so we can compare verifiers on the exact
  same candidate sets.
- Generic verifier improvements on validation AUC alone do not guarantee better
  downstream ranking.
- Ranking-aligned verifier training is the right modeling direction.
- Round 3 was the strongest replay-backed verifier across the existing cached
  pools and remains the key baseline in the experiment record.
- The held-out failure-analysis gate was useful:
  it found concentrated `blocksworld`, mostly `abstract/abstract`,
  within-pool misranking errors rather than diffuse failure.
- Focused round 4 improved the verifier over round 3 on replay and on the fresh
  held-out run.
- The repeated fresh held-out comparison makes the round-4 case materially
  stronger:
  the cleanest gain is now at `K=8`, while `K=4` remains closer to a tie than
  to a decisive win.
- Pairwise/ranking follow-up training did not improve the promoted round-4
  selector:
  round 5 and round 6 both failed replay against round 4.
- Simple fixed-model selector policies also did not improve round 4:
  margin fallback, top-gap fallback, agreement fallback, score normalization,
  and index-penalized ranking all tied or regressed on cached replay.
- The corrected "improve round 4" direction is now supported:
  a larger round-4-style pointwise round 7 passed cached replay against round 4
  at `K=8` while tying at `K=4`.
- The fresh multiseed gate did not justify promotion:
  round 7 improved mean `K=4`, but tied mean `K=8`, which is the main
  best-of-K operating point.
- The row-level fresh-gate analysis explains the tie:
  at `K=8`, round 7 helped `11` rows and hurt `11` rows across `150` rows.
  On seed `56`, where round 7 regressed by `-0.0800`, `6` of `7` hurt rows
  still had an equivalent candidate available in the round-7 pool, meaning the
  loss was mostly within-pool selection rather than only generation variance.
- The identical-pool fresh gate gives the cleanest round-7 read:
  round 7 gains only `+0.0067` at `K=8` and regresses `-0.0133` at `K=4`.
  That is a real but too-small `K=8` selector gain, not a promotion case.
- The first Phase 3 search ablation is complete:
  simple Planetarium-oracle solvability signals did not beat round-4
  `verifier_ranked` on cached replay. Solvability is useful diagnostics, but
  not a strong semantic selector.
- Domain-aware repair is the strongest system result so far.
- The final fresh gate on untouched seeds `51-55` passed strongly:
  repair-augmented VCSR improved mean `K=8` equivalence from `0.4200` to
  `0.7720`.

## Key Evidence

Round 4 versus round 3 on the replay-controlled held-out candidate pool:

- `K=4`
  - round 3 `verifier_ranked`: `0.4200`
  - round 4 `verifier_ranked`: `0.5000`
- `K=8`
  - round 3 `verifier_ranked`: `0.4400`
  - round 4 `verifier_ranked`: `0.4600`

Fresh end-to-end held-out comparison:

- Round 3 fresh held-out run:
  - `K=4` `greedy_first`: `0.4400`
  - `K=4` `verifier_ranked`: `0.4200`
  - `K=8` `greedy_first`: `0.4400`
  - `K=8` `random_parseable`: `0.3800`
  - `K=8` `verifier_ranked`: `0.4600`
- Round 4 fresh held-out run:
  - `K=4` `greedy_first`: `0.4600`
  - `K=4` `verifier_ranked`: `0.4400`
  - `K=8` `greedy_first`: `0.4600`
  - `K=8` `random_parseable`: `0.5000`
  - `K=8` `verifier_ranked`: `0.4800`

Repeated fresh held-out comparison across seeds `48`, `49`, and `50` from
[results/vcsr/multiseed_holdout_compare/comparison_summary.md](/E:/Engineering/vcsr/results/vcsr/multiseed_holdout_compare/comparison_summary.md):

- Round 3 mean `verifier_ranked`
  - `K=4`: `0.4000`
  - `K=8`: `0.4000`
- Round 4 mean `verifier_ranked`
  - `K=4`: `0.4000`
  - `K=8`: `0.4267`
- Head-to-head by seed
  - `K=4`: round 4 win / loss / tie = `1 / 1 / 1`
  - `K=8`: round 4 win / loss / tie = `2 / 0 / 1`

This tells us three important things:

- round 4 is a real model improvement over round 3
- the strongest and most repeatable downstream gain is now at `K=8`
- `K=4` is still not a clean promotion story; it looks approximately tied on
  the current multi-seed gate
- replay is still the right checkpoint-selection tool, but we now also have a
  more credible fresh held-out signal for promotion decisions

## Main Conclusion

Round 4 remains the frozen promoted verifier, but the paper-facing system is no
longer verifier ranking alone. The strongest current system is:

1. generate best-of-`K` candidates
2. select with the frozen round-4 verifier
3. apply one domain-aware repair call to the selected parseable `K=8` candidate
4. evaluate/report the repair-augmented outcome

My current judgment is:

- **round 4 is now the promoted `best_current` verifier in the repo**
- **round 3 remains the strongest historical replay-backed baseline**
- **repair-augmented VCSR is now the paper-facing system result**
- **plain verifier-ranked selection alone should be framed as an important
  baseline, not the final system**

Why:

- round 4 improved the verifier at both `K=4` and `K=8` relative to round 3
- the repeated held-out comparison now shows a clear mean gain at `K=8`
- the repeated held-out comparison does **not** show a clear mean gain at `K=4`
- so round 4 looks promotion-worthy if our project emphasis is best-of-`8`
  ranking, but not yet as a claim that the selector is uniformly dominant

So the project bottleneck has shifted again. It is no longer "can the verifier
help?" or "can repair work on hand-picked cached failures?". The final gate now
shows that domain-aware repair survives untouched fresh seeds. The remaining
work is paper framing, artifact hygiene, and careful caveat analysis.

## Recommendation

### Highest-Priority Next Step

Write the paper around the final repair-augmented result as the main
paper-facing system result:

- round 4 remains the default verifier
- `verifier_ranked` remains the plain best-of-K baseline
- `verifier_ranked_repair` is the new system policy at the main `K=8`
  operating point
- seeds `51-55` are final evidence and must not be reused for prompt tuning,
  checkpoint selection, or policy design
- use `PAPER_PLAN.md` plus `results/paper/final_vcsr/` for the claim sheet,
  table values, and figure specs

### Final Gate Result

Final fresh repair gate:

- output:
  [results/vcsr/final_repair_gate_round4](/E:/Engineering/vcsr/results/vcsr/final_repair_gate_round4)
- seeds: `51`, `52`, `53`, `54`, `55`
- rows per seed: `50`
- mean `K=8` plain round-4 `verifier_ranked`: `0.4200`
- mean `K=8` repair-augmented `verifier_ranked_repair`: `0.7720`
- mean delta: `+0.3520`
- repair parse rate: `0.9840`
- helped / hurt / tied: `104 / 16 / 130`

This passes the pre-set acceptance gate by a wide margin.

### Modeling Recommendation

Do **not** train another verifier before writing the current result.

The sequence of negative and positive evidence is now clear:

- pairwise round 5 did not beat round 4
- conservative ranking round 6 did not beat round 4
- focused pointwise round 7 was safe but not promotion-worthy
- simple planner/search policies did not move the needle
- domain-aware repair produced the first large final-gate improvement

That means the next research direction should be paper assembly and careful
analysis, not another blind modeling run.

If there is time after the paper-facing result is frozen, the most useful
follow-up is a better repair acceptance policy, because the final gate also
showed an important caveat:

- gripper: `96` helped, `0` hurt
- blocksworld: `8` helped, `16` hurt
- unconditional repair is a huge net win, but can damage already-correct
  blocksworld selections

We tested the first domain-agnostic version of that idea as a verifier-score
guard:

- development-selected margin: `0.05`
- fresh seeds: `67-71`
- plain round-4 `verifier_ranked` mean `K=8`: `0.4360`
- guarded repair mean `K=8`: `0.7960`
- unconditional repair on the same seeds: `0.7960`
- guarded hurt rows: `14`
- unconditional hurt rows: `14`
- guarded blocksworld hurts: `14`
- unconditional blocksworld hurts: `14`

This means the score guard replicated the repair benefit, but did **not** fix
the blocksworld harm. Do not promote the current guarded policy as a stronger
paper-facing system.

The right paper framing is therefore:

- main result: repair-augmented VCSR substantially improves semantic
  equivalence at `K=8` over greedy, random parseable best-of-K,
  planner/solvability search, and verifier-only search
- mechanism: domain-aware repair recovers many verifier-selected failures,
  especially gripper
- caveat: unconditional repair may hurt some already-correct blocksworld
  selections
- future work: stronger repair acceptance checks, structural diff checks,
  confidence-gated repair, and abstention

### Paper Artifact Package

- Plan: [PAPER_PLAN.md](/E:/Engineering/vcsr/PAPER_PLAN.md)
- Exporter: [scripts/export_paper_artifacts.py](/E:/Engineering/vcsr/scripts/export_paper_artifacts.py)
- Generated tables: [results/paper/final_vcsr/paper_tables.md](/E:/Engineering/vcsr/results/paper/final_vcsr/paper_tables.md)
- Claim sheet: [results/paper/final_vcsr/paper_claims.md](/E:/Engineering/vcsr/results/paper/final_vcsr/paper_claims.md)
- Figure specs: [results/paper/final_vcsr/figure_specs.md](/E:/Engineering/vcsr/results/paper/final_vcsr/figure_specs.md)

## What We Should Not Over-Prioritize Right Now

- more verifier training before writing up the final repair result
- changing the generator, prompt, verifier checkpoint, or selection policy on
  seeds `51-55`
- claiming repair is uniformly harmless across domains
- treating same-pool repair results as the main evidence now that the final
  fresh gate exists
- overfitting a new blocksworld repair gate to remove the `16` hurt rows before
  documenting the current result

## Bottom Line

This is no longer depressing. This is the first genuinely strong endgame result.

The paper-facing claim should be:

> Verifier-ranked best-of-K alone is helpful but not uniformly dominant; adding
> domain-aware repair to the verifier-selected candidate substantially improves
> semantic equivalence on in-domain Planetarium tasks at the main `K=8`
> operating point.

Round 4 remains the stable paper-facing verifier. Repair-augmented selection is
now the paper-facing VCSR system.
