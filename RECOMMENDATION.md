# Recommendation

This document captures the current project-level recommendation for VCSR after
the first verifier-ranked best-of-K pilot, fixed-pool replay evaluation, four
ranking-aligned verifier training rounds, the first repeated fresh held-out
multi-seed comparison, a fixed-round-4 selector analysis, the focused
pointwise round-7 replay gate, and the fresh round-4-vs-round-7 multiseed gate.

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

Round 4 is now the stronger end-to-end candidate, but the promotion story is
still asymmetric across `K`.

My current judgment is:

- **round 4 is now the promoted `best_current` verifier in the repo**
- **round 3 remains the strongest historical replay-backed baseline**
- **interpret the promotion case as strongest for `K=8`, not as a universal
  across-the-board win**

Why:

- round 4 improved the verifier at both `K=4` and `K=8` relative to round 3
- the repeated held-out comparison now shows a clear mean gain at `K=8`
- the repeated held-out comparison does **not** show a clear mean gain at `K=4`
- so round 4 looks promotion-worthy if our project emphasis is best-of-`8`
  ranking, but not yet as a claim that the selector is uniformly dominant

So the project bottleneck is no longer "can we make the verifier better?".
We already did.
The bottleneck is now "how narrowly and honestly do we frame the gain in the
paper and the repo decision?".

## Recommendation

### Highest-Priority Next Step

Now that round 4 has been promoted, keep it as the default verifier and treat
the first pairwise round-5 run as a completed but non-promoted experiment:

- round 4 is the default verifier
- the strongest repeated fresh held-out evidence is at `K=8`
- `K=4` remains mixed enough that it should not be oversold
- hybrid pairwise round 5 is implemented and trained, but did not beat round 4
  on replay

### Promotion Rule

The promotion decision has now been made.

The rule we are implicitly adopting is:

- we are willing to promote on the basis of the stronger repeated held-out
  improvement at `K=8`
- we still require the documentation to say plainly that `K=4` is not a clean
  win

### Modeling Recommendation

Do **not** change backbone yet.

DeBERTa is still a reasonable verifier backbone for this phase because:

- it is a cross-encoder and can jointly read NL plus candidate PDDL
- it is fast enough for many iterations on available hardware
- the current evidence still points more strongly to ranking robustness than to
  a hard capacity ceiling

If future fresh held-out checks still show the verifier hovering near
`greedy_first` and `random_parseable`, especially at `K=4`, then the next real
escalation should be an objective change, not an architecture change:

- pairwise ranking loss
- listwise ranking supervision
- or another explicitly within-pool ranking objective

The first hybrid pairwise attempt was the right class of experiment, but this
specific recipe should **not** be promoted:

- it tied round 4 at `K=4` and regressed at `K=8` on
  `bestofk_round4_holdout_eval_clean`
- it regressed at both `K=4` and `K=8` on `bestofk_round3_holdout_eval`
- its pairwise validation split was small (`38` examples)
- the mined pairs were entirely `blocksworld`, with heavy
  `abstract/abstract` concentration

So the next modeling move should be a better-controlled ranking objective, not
a blind rerun of round 5.

We tried that better-controlled ranking experiment as round 6:

- warm start from round 4
- larger cached-pool ranking set
- explicit pairwise dev file
- pointwise-dominant hybrid loss
- replay gate before fresh generation

Round 6 also failed the replay gate:

- mean replay `K=4`: round 4 `0.5000`, round 6 `0.4733`
- mean replay `K=8`: round 4 `0.5156`, round 6 `0.4978`

That changed the recommendation, so we ran a fixed-round-4 selector analysis
instead of another training run.

Fixed-round-4 selector analysis tested:

- margin fallback to greedy
- top-score gap fallback to greedy
- round-3/round-4 agreement fallback
- row-wise score normalization
- index-penalized ranking

The result was also negative:

- mean cached replay at `K=4`: round-4 `verifier_ranked` stayed best at
  `0.5050`
- mean cached replay at `K=8`: round-4 `verifier_ranked` stayed best or tied at
  `0.5167`
- the strongest changed policies either regressed or tied with no useful
  helped-over-hurt pattern
- oracle-positive misses often had tiny top-score gaps, but simple fallback
  rules hurt more correct selections than they rescued

This is not a project failure. It is a useful boundary:

- round 4 is not trivially improvable by simple score-use heuristics
- another small pairwise/listwise retrain is unlikely to be the right next move
- the next improvement probably needs genuinely new signal, not just a
  different way to squeeze the same scalar score

We then tested the corrected interpretation: do what made round 4 work, but
larger and cleaner.

Focused pointwise round 7:

- warm-started from promoted round 4
- used pure pointwise BCE, not pairwise/listwise loss
- mined `788` parseable cached-pool examples across `K=4` and `K=8`
- kept the original verifier validation split fixed

Replay gate result:

- mean replay `K=4`: round 4 `0.5050`, round 7 `0.5050`
- mean replay `K=8`: round 4 `0.5167`, round 7 `0.5283`
- row-level `K=8`: `4` helped, `1` hurt
- acceptance: round 7 passed cached replay but is not promoted yet

Fresh multiseed round-4-vs-round-7 gate:

- mean fresh `K=4`: round 4 `0.4000`, round 7 `0.4133`
- mean fresh `K=8`: round 4 `0.4200`, round 7 `0.4200`
- seed-wise at `K=8`: round 7 won seeds `57` and `58`, but regressed seed
  `56` enough to erase the mean gain

Round-7 fresh row-level analysis:

- `K=4`: round 7 helped `7` rows, hurt `5`, tied `138`
- `K=8`: round 7 helped `11` rows, hurt `11`, tied `128`
- seed `56`, `K=8`: round 7 helped `3`, hurt `7`
- seed `56`, `K=8`: `6` of `7` hurt rows were selector losses with an
  equivalent candidate still available in the round-7 pool
- round 7 improved `blocksworld` but regressed sparse `gripper` successes

Fresh identical-pool round-4-vs-round-7 gate:

- seeds: `59`, `60`, `61`
- `K=4`: round 4 `0.4067`, round 7 `0.3933`, delta `-0.0133`
- `K=8`: round 4 `0.4400`, round 7 `0.4467`, delta `+0.0067`
- `K=4` head-to-head: round 7 wins `0`, losses `1`, ties `2`
- `K=8` head-to-head: round 7 wins `1`, losses `0`, ties `2`

This changes the recommendation again:

- the round-4-style pointwise direction is worth continuing
- round 7 is a useful provisional/diagnostic result
- round 4 remains `best_current` because round 7 did not improve the fresh
  `K=8` mean
- another blind focused-pointwise scale-up is not justified without a cleaner
  identical-pool evaluation design or genuinely new signal
- the cleaner identical-pool evaluation is now done, and it still does not
  justify promotion

## What We Should Not Over-Prioritize Right Now

- promoting round 4 while claiming the selector is now uniformly robust
- replacing replay with offline AUC as the main checkpoint-selection criterion
- treating a 1-row difference on a 50-row held-out sample as decisive
- generic extra epochs on the same data without stronger evaluation
- architecture changes before we establish whether the current gain is stable
- promoting pairwise round 5 just because it matches the hypothesized failure
  mode
- running another pairwise/listwise retrain before understanding why rounds 5
  and 6 both failed replay
- adding margin/fallback selector policies unless they pass cached replay first
- promoting round 7 after the fresh gate tied `K=8`

## Bottom Line

My view is now cautiously positive.

Round 4 is not a failure.
It moved the verifier in the right direction:

- better replay performance
- better offline validation AUC
- better fresh held-out `verifier_ranked` than round 3 at both `K=4` and `K=8`
- a repeated multi-seed held-out win pattern at `K=8`

But it still did not clear the bar of being obviously better than simple
baselines at every `K` we care about.

So the clearest path from here is:

- use round 4 as the promoted default verifier
- keep describing the result as strongest at `K=8`
- treat hybrid pairwise round 5 as a useful negative result
- treat conservative ranking round 6 as a second negative result
- treat fixed-round-4 selector heuristics as a third useful negative result
- treat focused pointwise round 7 as a useful but non-promoted diagnostic whose
  fresh row analysis showed equal helped/hurt counts at `K=8`
- treat the identical-pool round-7 result as a small positive `K=8` signal that
  is below the promotion threshold because `K=4` regressed
- do not promote round 7 and do not blindly scale the same recipe again

Round 4 remains the stable paper-facing verifier. If we continue improving the
verifier, the next step should bring in genuinely new signal or improve
candidate-pool diversity. The identical-pool gate has already separated model
error from generation variance, and round 7's clean gain is too small to be the
next paper-facing checkpoint.
