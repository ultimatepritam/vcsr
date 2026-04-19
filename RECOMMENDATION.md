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

- **analyze the remaining held-out selection failures and decide whether to do a
  focused round 4 or move to a more explicit ranking objective**

Not because round 3 failed, but because we have now completed the fresh
held-out end-to-end run and it gave us the answer we needed:
the verifier helps at `K=8`, but not yet reliably at `K=4`.

The project bottleneck is no longer "can we build a useful verifier?".
It is now "how do we remove the remaining within-pool misrankings so the
downstream gain becomes stronger and more stable across `K` settings?"

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

Do a focused **error-analysis pass on the fresh held-out run**, then use that to
choose the lightest next modeling step that can plausibly improve ranking.

What we now know from the held-out run:

- at `K=8`, `verifier_ranked` reached `0.4600`
- at `K=8`, `greedy_first` was `0.4400`
- at `K=8`, `random_parseable` was `0.3800`
- at `K=4`, `verifier_ranked` was only `0.4200`, below both baselines
- oracle remained above the verifier at both `K=4` and `K=8`

So the highest-value immediate work is not another blind large run.
It is understanding the residual miss pattern well enough to decide whether:

- one more pool-mined round is likely enough, or
- the supervision objective itself now needs to change

### Acceptance Criterion

Judge the next modeling step primarily by **whether it improves the held-out
downstream selector at `K=8` without harming the simpler regimes as much**,
while continuing to use fixed-pool replay as the checkpoint-selection guardrail.

Success should mean:

- verifier-ranked continues to beat both simple baselines at `K=8`
- verifier-ranked no longer regresses at `K=4`, or at least narrows that gap
- remaining failures look concentrated and interpretable rather than noisy and
  diffuse

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
- treating the fresh held-out `K=8` win as if the problem were already solved

These may become important later, but they are not the central unknown today.

## If The Fresh End-to-End Run Is Still Weak

The fresh end-to-end run is not a failure, but it is also not a complete win.
If the next targeted iteration still leaves us with the same pattern, then we
should consider a deeper shift:

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

The fresh held-out end-to-end run is the final piece that makes this feel real:

- at `K=8`, the verifier beat both baselines on a fresh run
- at `K=4`, it still regressed

So the current state is neither "done" nor "still only a hypothesis."
It is a credible positive direction with a clear remaining weakness.

Earlier, the main recommendation was:

- move toward ranking-aligned supervision and test it with replay

The updated recommendation is:

- keep the current verifier architecture family
- treat ranking-aligned round 3 as the current best downstream checkpoint
- freeze it as `best_current`
- use the held-out result to drive targeted error analysis next
- only then decide between a focused round 4 and a more explicit ranking loss
- keep replay-controlled evaluation as the main checkpoint-selection test

That is the clearest path from our current state to the actual VCSR project
goal.
