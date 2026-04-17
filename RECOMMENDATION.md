# Recommendation

This document captures the current project-level recommendation for VCSR after
the first verifier-ranked best-of-K pilot, the verifier retraining rounds, and
the fixed-pool replay evaluation.

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
- Hard-negative mining and capacity-push sweeps improved verifier validation
  metrics.
- Fixed-pool replay is now implemented, so we can compare verifiers on the exact
  same candidate sets.
- On that fixed candidate pool, verifier-ranked selection still did not beat the
  simple baselines we care about.

That last point is the important one. It means the project is no longer blocked
by infrastructure or by lack of a clean evaluation protocol. It is now blocked
by the verifier's inability to rank real candidate pools in the way the project
needs.

## Key Evidence

On the cached `results/vcsr/bestofk_pilot/candidate_dump.jsonl` pool, replay
comparison gave:

- At `K=8`, `random_parseable` reached `0.5000` equivalence.
- At `K=8`, `verifier_ranked` reached `0.4333` with `lr_5em05`.
- At `K=8`, `verifier_ranked` reached `0.4000` with `lr_2p0em05`.
- At `K=4`, verifier-ranked was also below greedy/random.
- At `K=1`, all policies matched, which confirms the replay setup is behaving as
  expected.

This is strong evidence that the problem is not just regeneration noise.
The verifier is not yet making the within-pool ranking decisions we want.

## Main Conclusion

The next crucial project step is **not** another generic verifier retrain on the
same style of data.

The next crucial step is:

- **ranking-aligned verifier training**

More specifically, we should keep the current DeBERTa cross-encoder family for
now, but change the supervision so it matches the actual downstream decision:
among multiple parseable candidates for the same NL input, score the truly
equivalent one above subtle near-miss negatives.

## Why This Is The Right Next Step

The current verifier is trained mostly like a pointwise binary classifier:

- given `(NL, candidate PDDL)`, predict good vs bad

But our actual use case is:

- given one NL input and a pool of parseable candidates, choose the best one

Those are related, but not identical, problems. A model can have decent AUC/F1
while still failing to order hard, plausible candidates correctly inside a
single pool.

That is exactly what the replay results suggest is happening.

## Recommendation

### Highest-Priority Modeling Task

Train a **ranking-aligned verifier** using candidate pools from real generation
runs.

The first practical version does not need a new architecture or a fancy ranking
loss. It should:

- mine larger candidate pools from held-out best-of-K runs
- keep semantically equivalent candidates as positives
- add parseable-but-non-equivalent candidates from the same pool as hard
  negatives
- upweight or increase the share of these within-pool near-miss negatives during
  training

This is the most direct way to teach the verifier the distinction it currently
fails to make.

### Acceptance Criterion

Judge the next verifier primarily by **fixed-pool replay**, not by validation AUC
alone.

Success should mean:

- verifier-ranked beats greedy and random-parseable on the same cached candidate
  pool
- especially at `K=4` and `K=8`
- with results stable enough to justify a fresh held-out rerun

### Architecture Recommendation

Do **not** change backbone yet.

DeBERTa is still a reasonable verifier backbone for this phase because:

- it is a cross-encoder and can jointly read NL plus candidate PDDL
- it is fast enough for many iterations on available hardware
- we do not yet have evidence that raw capacity is the main bottleneck

Right now the bigger issue is training-signal alignment, not backbone choice.

## What We Should Not Over-Prioritize Right Now

- more generic epochs on the same current training data
- larger-batch sweeps without new ranking-focused supervision
- fresh regenerated best-of-K comparisons as the main decision criterion
- abstention policy work before the ranker itself is useful
- architecture changes before we properly test ranking-aligned training

These may become important later, but they are not the central unknown today.

## If Ranking-Aligned Training Still Fails

If one or two serious rounds of ranking-aligned retraining still do not produce
a replay win, then we should consider a deeper shift:

- explicit pairwise or listwise ranking objectives
- larger verifier backbones
- structure-aware scoring features
- hybrid ranking approaches beyond plain pointwise classification

But that is a second-order decision. We have not exhausted the most justified
next move yet.

## Bottom Line

The project direction still looks viable, but the decision frontier has changed.

Earlier, the main recommendation was to build fixed-pool replay.
That is now done.

The updated recommendation is:

- keep the current verifier architecture family
- move training toward ranking-aligned hard-negative supervision
- use fixed-pool replay as the main downstream acceptance test

That is the clearest path from our current state to the actual VCSR project
goal.
