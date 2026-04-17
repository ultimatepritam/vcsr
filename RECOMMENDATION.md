# Recommendation

This document captures the current project-level recommendation for VCSR after
the first verifier-ranked best-of-K pilot, fixed-pool replay evaluation, and
two ranking-aligned verifier training rounds.

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
- Ranking-aligned verifier training is now validated as the right direction.
- Round 2 is the first checkpoint that produces a real downstream replay win
  over the simple `random_parseable` baseline on the fixed `K=8` pool.
- A later fresh end-to-end pool run with that verifier was weaker, but that
  fresh run has **not** yet been analyzed through the same replay-controlled
  comparison across checkpoints.
- So the current concern is not that round 2 has been disproven, but that we do
  not yet have enough controlled evidence to claim cross-pool robustness.

That last point is the key update.
We are no longer asking whether ranking-aligned training is worth trying.
We now know it can work downstream, but we also know the current gains are not
yet stable enough across pools.

## Key Evidence

On the cached `results/vcsr/bestofk_pilot/candidate_dump.jsonl` pool, fixed-pool
replay now gives:

- At `K=8`, `random_parseable` reaches `0.5000` equivalence.
- At `K=8`, the original best-current verifier (`lr_5em05`) reaches `0.4333`.
- At `K=8`, the capacity-push verifier (`lr_2p0em05`) reaches `0.4000`.
- At `K=8`, ranking-aligned round 1 reaches `0.4667`.
- At `K=8`, ranking-aligned round 2 reaches `0.5333`.
- At `K=4`, ranking-aligned round 2 reaches `0.4667`.

On the later fresh `50`-row round-2 pool rerun, the same round-2 verifier does
not dominate in the direct end-to-end run:

- At `K=4`, `verifier_ranked` is `0.4800`, tied with `random_parseable`.
- At `K=8`, `verifier_ranked` falls to `0.4200` versus `0.4400` for
  `random_parseable`.

This tells us two important things:

- fixed-pool replay was the right diagnostic tool
- ranking-aligned supervision is improving the downstream task we care about
- we still need replay-style controlled analysis on newer pools before making
  strong cross-pool claims
- robustness across pools is still the central unsolved problem

## Main Conclusion

The next crucial project step is:

- **robustness-focused ranking-aligned verifier round 3**

Not because round 2 failed, but because it succeeded in the right place: it
produced our best replay result so far and therefore justified doubling down in
a more controlled, generalization-oriented way.

The project bottleneck still looks like training-signal alignment rather than
backbone choice or pure optimization budget, but now specifically in the form of
cross-pool robustness.

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

Run a stronger **ranking-aligned verifier round 3** built from multiple real
candidate pools.

That round should:

- mine substantially more best-of-K candidate pools than any single earlier
  round
- combine examples from multiple independently generated pools rather than
  trusting one pool's failure pattern
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
- especially at `K=8`, where round 2 already produced a modest win
- verifier improvements also survive replay-style evaluation on at least one
  newer generated pool, not just the original pilot pool
- with enough margin and stability to justify a fresh held-out downstream rerun

### Architecture Recommendation

Do **not** change backbone yet.

DeBERTa is still a reasonable verifier backbone for this phase because:

- it is a cross-encoder and can jointly read NL plus candidate PDDL
- it is fast enough for many iterations on available hardware
- the current evidence still points more strongly to supervision mismatch than
  to a hard capacity ceiling

Right now the bigger issue is robustness of the learned ranking signal, not
backbone choice.

## What We Should Not Over-Prioritize Right Now

- more generic epochs on the same current training data
- larger-batch sweeps without more ranking-focused supervision
- fresh regenerated best-of-K comparisons alone as the main decision criterion
- abstention policy work before the ranker itself is useful
- architecture changes before we run a stronger robustness-focused round

These may become important later, but they are not the central unknown today.

## If Round 3 Still Fails

If a stronger robustness-focused round still does not produce a stable replay
win, then we should
consider a deeper shift:

- explicit pairwise or listwise ranking objectives
- larger verifier backbones
- structure-aware scoring features
- hybrid ranking approaches beyond plain pointwise classification

That would be a justified escalation only after we give ranking-aligned training
at a larger scale a fair test.

## Bottom Line

The project direction still looks viable, and the most important update is that
we now have evidence of a real downstream replay win from ranking-aligned
training on the fixed evaluation pool.

Earlier, the main recommendation was:

- move toward ranking-aligned supervision

The updated recommendation is:

- keep the current verifier architecture family
- treat ranking-aligned round 2 as the current best downstream checkpoint
- run a robustness-focused round 3 on multiple real candidate pools
- keep replay-controlled evaluation as the main downstream acceptance test

That is the clearest path from our current state to the actual VCSR project
goal.
