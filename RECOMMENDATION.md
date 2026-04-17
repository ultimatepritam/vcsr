# Recommendation

This document captures the current project-level recommendation for VCSR after
the first verifier-ranked best-of-K pilot, fixed-pool replay evaluation, and
the first ranking-aligned verifier training round.

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
- Ranking-aligned verifier training is now validated as the right direction, but
  the first round was not yet strong enough to solve the ranking problem.

That last point is the key update.
We are no longer asking whether ranking-aligned training is worth trying.
We now know it helps, but we also know the current scale of that intervention is
not enough.

## Key Evidence

On the cached `results/vcsr/bestofk_pilot/candidate_dump.jsonl` pool, fixed-pool
replay now gives:

- At `K=8`, `random_parseable` reaches `0.5000` equivalence.
- At `K=8`, the original best-current verifier (`lr_5em05`) reaches `0.4333`.
- At `K=8`, the capacity-push verifier (`lr_2p0em05`) reaches `0.4000`.
- At `K=8`, ranking-aligned round 1 reaches `0.4667`.
- At `K=4`, ranking-aligned round 1 does not improve over the earlier replay
  result and remains at `0.4000`.

This tells us two important things:

- fixed-pool replay was the right diagnostic tool
- ranking-aligned supervision is improving the downstream task we care about,
  but not yet enough to beat the simple non-verifier baseline

## Main Conclusion

The next crucial project step is:

- **ranking-aligned verifier training round 2**

Not because the first round failed completely, but because it produced a
meaningful partial improvement and therefore justified doubling down in a more
serious way.

The project bottleneck still looks like training-signal alignment rather than
backbone choice or pure optimization budget.

## Why This Is The Right Next Step

The current verifier is still being asked to perform a harder deployment-time
decision than its original training data emphasized:

- training mostly teaches pointwise good-vs-bad discrimination
- deployment requires choosing the best candidate within a pool of plausible,
  parseable options for the same NL input

The first ranking-aligned round moved the replay result in the correct
direction:

- more pool-based positives
- more parseable near-miss negatives
- warm-starting from the best ranking-oriented checkpoint

That is exactly the kind of improvement signal we wanted to see.
It means the approach is not misguided; it is just not yet strong enough.

## Recommendation

### Highest-Priority Modeling Task

Run a stronger **ranking-aligned verifier round 2** built from larger real
candidate pools.

That round should:

- mine substantially more best-of-K candidate pools than the 30-row pilot
- include more rows with equivalent candidates and more rows with parseable-only
  negative pools
- keep multiple equivalent positives per row when available
- keep high-scoring parseable non-equivalent negatives from the same pool
- continue to emphasize verifier-selected wrong candidates and near-miss
  outranking negatives
- likely repeat or otherwise upweight the pool-mined data relative to the base
  neggen set

This is still the most justified next modeling bet.

### Acceptance Criterion

Judge the next verifier primarily by **fixed-pool replay**, not by validation AUC
alone.

Success should mean:

- verifier-ranked beats greedy and random-parseable on the same cached candidate
  pool
- especially at `K=8`, where the first ranking-aligned round already showed
  some gain
- with enough margin and stability to justify a fresh held-out downstream rerun

### Architecture Recommendation

Do **not** change backbone yet.

DeBERTa is still a reasonable verifier backbone for this phase because:

- it is a cross-encoder and can jointly read NL plus candidate PDDL
- it is fast enough for many iterations on available hardware
- the current evidence still points more strongly to supervision mismatch than
  to a hard capacity ceiling

Right now the bigger issue is training-signal alignment, not backbone choice.

## What We Should Not Over-Prioritize Right Now

- more generic epochs on the same current training data
- larger-batch sweeps without more ranking-focused supervision
- fresh regenerated best-of-K comparisons as the main decision criterion
- abstention policy work before the ranker itself is useful
- architecture changes before we run a stronger ranking-aligned round

These may become important later, but they are not the central unknown today.

## If Ranking-Aligned Round 2 Still Fails

If a stronger second round still does not produce a replay win, then we should
consider a deeper shift:

- explicit pairwise or listwise ranking objectives
- larger verifier backbones
- structure-aware scoring features
- hybrid ranking approaches beyond plain pointwise classification

That would be a justified escalation only after we give ranking-aligned training
at a larger scale a fair test.

## Bottom Line

The project direction still looks viable, and the most important update is that
we now have evidence of partial downstream progress from ranking-aligned
training.

Earlier, the main recommendation was:

- move toward ranking-aligned supervision

The updated recommendation is:

- stay with the current verifier architecture family
- run a stronger ranking-aligned round 2 on larger real candidate pools
- keep fixed-pool replay as the main downstream acceptance test

That is the clearest path from our current state to the actual VCSR project
goal.
