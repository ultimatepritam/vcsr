# Recommendation

This document captures the current project-level recommendation for VCSR based on
the completed verifier and best-of-K experiments.

## Goal

The project goal is not just to train a verifier with good offline metrics.
The real goal is to use a verifier to improve downstream NL-to-PDDL outcomes:

- rank multiple generated PDDL candidates
- choose semantically equivalent candidates more reliably than greedy or simple baselines
- later support abstention and repair in a trustworthy way

## Current Situation

What we now know:

- The end-to-end pipeline works.
- The generator can produce useful candidate diversity.
- The verifier has real offline signal.
- Hard-negative mining and capacity-push training improved verifier metrics offline.
- But verifier improvements have **not yet translated into a robust downstream best-of-K selection win**.

This means the project is no longer blocked on infrastructure.
It is blocked on **cleanly measuring and improving ranker usefulness downstream**.

## Main Conclusion

The next crucial project step is **not** another generic verifier retrain.
It is:

- **build a fixed candidate-pool replay evaluator**

Why:

- Current best-of-K reruns regenerate candidates, so generator randomness is mixed
  together with verifier quality.
- That prevents us from answering the key project question:
  given the same candidate pool, does a better verifier actually choose better candidates?

Until we answer that, more verifier training is informative but not decisive.

## What We Should Do Next

### Highest Priority

1. Implement fixed-pool replay evaluation for best-of-K candidate dumps.
2. Compare multiple verifier checkpoints on the exact same cached candidates.
3. Decide which verifier is best downstream based on replay, not regenerated runs.

### Next After That

4. Move verifier training toward ranking-aligned supervision.
   The current verifier is trained mainly as a pointwise classifier, but we use it as a ranker.
5. Keep mining hard negatives from real candidate pools.
   The most valuable negatives are subtle, parseable, semantically wrong candidates.
6. Only then run fresh downstream held-out evaluations for stronger claims.

## What We Should Not Over-Prioritize Right Now

- more generic epochs on the same data
- more large-batch sweeps without a downstream decision criterion
- more regenerated best-of-K comparisons
- abstention policy work before we know the ranker is useful
- planner filtering as a substitute for semantic ranking

These may still matter later, but they are not the central unknown right now.

## Should We Change Approach?

Not yet.

The overall VCSR direction still looks viable:

- generator produces candidate pools with recoverable good options
- verifier learns useful signal

But the project needs to shift from:

- "train a better verifier classifier"

to:

- "train and evaluate a verifier that actually wins as a ranker on fixed candidate pools"

## Practical Recommendation

If we want the highest-value next implementation task, it is:

- **fixed-pool replay evaluator for best-of-K**

If we want the highest-value next modeling task after that, it is:

- **ranking-aligned verifier training**

That is the clearest path from our current results to the actual project goal.
