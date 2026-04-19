# Recommendation

This document captures the current project-level recommendation for VCSR after
the first verifier-ranked best-of-K pilot, fixed-pool replay evaluation, and
four ranking-aligned verifier training rounds.

## Goal

The project goal is not just to train a verifier with decent offline metrics.
The real goal is to use that verifier to improve downstream NL-to-PDDL
decisions:

- rank multiple generated PDDL candidates for the same natural-language input
- choose semantically equivalent candidates more reliably than greedy decoding
  or simple non-verifier baselines
- later support abstention and repair with trustworthy confidence estimates

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
  pools and remains the frozen `best_current` checkpoint in the repo.
- The held-out failure-analysis gate was useful:
  it found concentrated `blocksworld`, mostly `abstract/abstract`,
  within-pool misranking errors rather than diffuse failure.
- Focused round 4 improved the verifier over round 3 on replay and on the fresh
  held-out run, but not enough to clearly dominate simple baselines.

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

This tells us three important things:

- round 4 is a real model improvement over round 3
- the improvement is still small enough that simple baselines can trade places
  with it on one 50-row held-out sample
- replay is still the right checkpoint-selection tool, but fresh held-out runs
  are now close enough that we need repeated held-out evidence before promotion

## Main Conclusion

Round 4 is promising, but not decisive.

My current judgment is:

- **do not promote round 4 to `best_current` yet**
- **treat round 4 as the leading provisional candidate**
- **run a small repeated fresh held-out evaluation before changing the official
  baseline**

Why:

- round 4 improved the verifier at both `K=4` and `K=8` relative to round 3
- but on the fresh held-out run it still lost to `greedy_first` at `K=4`
- and it also lost to `random_parseable` at `K=8`
- both gaps are small, but that is exactly the point:
  if the selector were already robust, it should not still be this close to
  those baselines

So the project bottleneck is no longer "can we make the verifier better?".
We already did.
The bottleneck is now "is the gain stable enough across fresh pools to justify
promotion and paper-facing claims?".

## Recommendation

### Highest-Priority Next Step

Run a **small multi-seed fresh held-out comparison** before promoting round 4.

Compare on new held-out runs:

- `greedy_first`
- `random_parseable`
- round 3 `verifier_ranked`
- round 4 `verifier_ranked`

Use the same generation budget and evaluation protocol as the current held-out
run, but repeat over several seeds instead of relying on one 50-row sample.

### Promotion Rule

Promote round 4 only if repeated fresh held-out evaluation shows:

- round 4 beats round 3 on average at `K=8`
- round 4 does not continue to lose materially to `greedy_first` at `K=4`
- round 4 is at least competitive with, and ideally clearly above,
  `random_parseable` at `K=8`

If those conditions hold, then promotion is justified.
If not, round 4 should remain a useful intermediate result rather than the new
official baseline.

### Modeling Recommendation

Do **not** change backbone yet.

DeBERTa is still a reasonable verifier backbone for this phase because:

- it is a cross-encoder and can jointly read NL plus candidate PDDL
- it is fast enough for many iterations on available hardware
- the current evidence still points more strongly to ranking robustness than to
  a hard capacity ceiling

If repeated fresh held-out checks still show the verifier hovering near
`greedy_first` and `random_parseable`, then the next real escalation should be
an objective change, not an architecture change:

- pairwise ranking loss
- listwise ranking supervision
- or another explicitly within-pool ranking objective

## What We Should Not Over-Prioritize Right Now

- promoting round 4 purely because it improved over round 3 once
- replacing replay with offline AUC as the main checkpoint-selection criterion
- treating a 1-row difference on a 50-row held-out sample as decisive
- generic extra epochs on the same data without stronger evaluation
- architecture changes before we establish whether the current gain is stable

## Bottom Line

My view is cautiously optimistic.

Round 4 is not a failure.
It moved the verifier in the right direction:

- better replay performance
- better offline validation AUC
- better fresh held-out `verifier_ranked` than round 3 at both `K=4` and `K=8`

But it still did not clear the bar of being obviously better than simple
baselines on the fresh held-out run.

So the clearest path from here is:

- keep round 3 as the official frozen `best_current`
- treat round 4 as the provisional best candidate
- run repeated fresh held-out evaluation to decide promotion
- if the gain is not stable, move next to a stronger ranking objective

That is the most defensible next step for the project and the paper.
