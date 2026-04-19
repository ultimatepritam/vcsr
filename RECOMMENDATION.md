# Recommendation

This document captures the current project-level recommendation for VCSR after
the first verifier-ranked best-of-K pilot, fixed-pool replay evaluation, and
three ranking-aligned verifier training rounds.

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
- Round 3 now beats the frozen round-2 verifier on the stronger 50-row replay
  pool at both `K=4` and `K=8`.
- Round 3 also improves `K=4` and matches the prior best `K=8` result on the
  original 30-row fixed pilot pool.
- The most important lesson from this round is that downstream replay improved
  even though the old offline verifier validation AUC did not.
- So the project now has a better downstream verifier checkpoint and a much more
  trustworthy model-selection principle: replay beats offline AUC.

That last point is the key update.
We are no longer asking whether ranking-aligned training can produce a useful
verifier. We now have evidence that it can, and that the new round-3 checkpoint
is the right baseline for the next real VCSR phase.

## Key Evidence

On the cached `results/vcsr/bestofk_pilot/candidate_dump.jsonl` pool, fixed-pool
replay now gives:

- At `K=8`, `random_parseable` reaches `0.5000` equivalence.
- At `K=8`, ranking-aligned round 2 reaches `0.5333`.
- At `K=8`, ranking-aligned round 3 also reaches `0.5333`.
- At `K=4`, ranking-aligned round 2 reaches `0.4667`.
- At `K=4`, ranking-aligned round 3 reaches `0.5000`.

On the newer cached round-2 pool under replay-controlled comparison:

- At `K=4`, round 2 reaches `0.4800`.
- At `K=4`, round 3 reaches `0.5200`.
- At `K=8`, round 2 reaches `0.4600`.
- At `K=8`, round 3 reaches `0.5400`.
- At `K=8`, `random_parseable` is only `0.4400`.
- Oracle on that newer pool is `0.6200`, so there is still substantial room to
  improve ranking quality.

This tells us two important things:

- fixed-pool replay was the right diagnostic tool
- ranking-aligned supervision is improving the downstream task we care about
- round 3 is now the current best verifier on both replay-tested pools
- there is still headroom to oracle, but the project now has a genuine
  downstream-positive baseline rather than only a directionally promising one

## Main Conclusion

The next crucial project step is:

- **freeze round 3 as the default verifier and run a fresh held-out end-to-end
  VCSR evaluation**

Not because verifier research is done, but because the replay evidence is now
good enough that the best way to learn more is to use this stronger checkpoint
in the actual downstream VCSR loop.

The project bottleneck is no longer "can we build a useful verifier?".
It is now "how much end-to-end gain do we get from this better verifier, and
what gap to oracle remains after we use it properly?"

## Why This Is The Right Next Step

The current verifier is finally strong enough that the next question should be
about end-to-end leverage, not just verifier-only iteration.

The verifier is still being asked to perform a harder deployment-time decision
than its original training data emphasized:

- training mostly teaches pointwise good-vs-bad discrimination
- deployment requires choosing the best candidate within a pool of plausible,
  parseable options for the same NL input

Round 3 shows the approach is not only directionally correct.
It is now producing the best downstream replay results we have seen so far.

## Recommendation

### Highest-Priority Modeling Task

Run a fresh **held-out end-to-end verifier-ranked best-of-K evaluation** using
the frozen round-3 checkpoint as the default verifier.

That evaluation should:

- keep round 3 fixed as the selected verifier
- use a fresh held-out pool or held-out end-to-end generation slice
- report the same policy comparisons:
  `greedy_first`, `random_parseable`, `verifier_ranked`
- focus on whether the replay gains translate into a cleaner downstream win
- preserve artifact discipline so the run can become a paper-facing result later

This is now the highest-value next experiment.

### Acceptance Criterion

Judge the next end-to-end milestone primarily by **downstream equivalence rate**
with round 3 frozen, while continuing to use fixed-pool replay as the model
selection guardrail.

Success should mean:

- verifier-ranked beats greedy and simple non-verifier baselines on a fresh
  held-out downstream run
- the round-3 replay gains are not just replay artifacts but actually carry into
  end-to-end use
- remaining failures are concentrated enough to justify either:
  more pool mining, or a stronger ranking objective, rather than broad confusion

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
- abstention policy work before the ranker itself is useful
- architecture changes before we run a stronger robustness-focused round
- replacing replay with offline AUC as the main selection criterion

These may become important later, but they are not the central unknown today.

## If The Fresh End-to-End Run Is Still Weak

If the fresh end-to-end run still underwhelms despite round 3 being the best
replay checkpoint, then we should consider a deeper shift:

- explicit pairwise or listwise ranking objectives
- larger verifier backbones
- structure-aware scoring features
- hybrid ranking approaches beyond plain pointwise classification

That would then be a justified escalation, because we will have already shown
that multi-pool ranking-aligned pointwise training has taken us meaningfully,
but not all the way, toward the project goal.

## Bottom Line

The project direction looks genuinely promising now.
The most important update is that ranking-aligned round 3 is the best verifier
on both replay-tested pools, and it won the stronger 50-row pool by a
meaningful margin.

Earlier, the main recommendation was:

- move toward ranking-aligned supervision and test it with replay

The updated recommendation is:

- keep the current verifier architecture family
- treat ranking-aligned round 3 as the current best downstream checkpoint
- freeze it as `best_current`
- run a fresh held-out end-to-end VCSR evaluation with this verifier
- keep replay-controlled evaluation as the main checkpoint-selection test

That is the clearest path from our current state to the actual VCSR project
goal.
