# Held-Out Failure Analysis

Source pool: `results\vcsr\bestofk_round3_holdout_eval\candidate_dump.jsonl`

## Validation

- Aggregate metric recomputation validated: `True`

## Top-Line Counts

- Total rows: `50`
- Oracle-positive rows at `K=4`: `26`
- Oracle-positive rows at `K=8`: `27`
- Oracle-positive verifier misses at `K=4`: `5`
- Oracle-positive verifier misses at `K=8`: `4`

## Domain Breakdown (`K=8` verifier)

| Domain | Rows | Oracle+ | Verifier Hits | Verifier Oracle Misses |
| --- | --- | --- | --- | --- |
| blocksworld | 29 | 27 | 23 | 4 |
| gripper | 21 | 0 | 0 | 0 |


## Style Breakdown (`K=8` verifier)

| Style | Rows | Oracle+ | Verifier Hits | Verifier Oracle Misses |
| --- | --- | --- | --- | --- |
| abstract/abstract | 28 | 13 | 10 | 3 |
| explicit/explicit | 22 | 14 | 13 | 1 |


## Highest-Value Miss Rows

| Row | K | Domain | Style | Miss Type | Equiv Cands | Selected | Best Equiv | Gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| blocksworld_swap_to_swap_blocks_list_16_2 | 8 | blocksworld | abstract/abstract | near_tie_misranking | 5 | 1 | 0 | 0.0298 |
| blocksworld_invert_to_invert_blocks_list_1_1_1_1_2_2_2_2_3_5 | 8 | blocksworld | abstract/abstract | equivalent_in_pool_but_misranked | 2 | 4 | 2 | 0.0688 |
| blocksworld_invert_to_invert_blocks_list_1_2_2_5_6_6 | 8 | blocksworld | abstract/abstract | equivalent_in_pool_but_misranked | 1 | 6 | 0 | 0.0706 |
| blocksworld_invert_to_invert_blocks_list_1_4_4_4_4 | 8 | blocksworld | explicit/explicit | equivalent_in_pool_but_misranked | 6 | 3 | 4 | 0.0317 |
| blocksworld_swap_to_swap_blocks_list_16_2 | 4 | blocksworld | abstract/abstract | near_tie_misranking | 2 | 1 | 0 | 0.0298 |
| blocksworld_invert_to_invert_blocks_list_1_1_1_1_2_2_2_2_3_5 | 4 | blocksworld | abstract/abstract | near_tie_misranking | 1 | 3 | 2 | 0.0002 |
| blocksworld_invert_to_invert_blocks_list_1_2_2_4_6 | 4 | blocksworld | abstract/abstract | near_tie_misranking | 1 | 0 | 2 | 0.0105 |
| blocksworld_invert_to_invert_blocks_list_1_2_2_5_6_6 | 4 | blocksworld | abstract/abstract | equivalent_in_pool_but_misranked | 1 | 2 | 0 | 0.0591 |


## `K=8` Verifier Wins Over Greedy

| Row | Domain | Style | Gain Reason | Selected | Best Equiv | Extra K Only |
| --- | --- | --- | --- | --- | --- | --- |
| blocksworld_invert_to_invert_blocks_list_1_2_2_4_6 | blocksworld | abstract/abstract | within_pool_reranking | 4 | 4 | no |
| blocksworld_invert_to_invert_blocks_list_1_3_4_8 | blocksworld | abstract/abstract | within_pool_reranking | 1 | 1 | no |
| blocksworld_invert_to_invert_blocks_list_1_1_1_2_3_3_7 | blocksworld | abstract/abstract | requires_extra_k_candidates | 6 | 6 | yes |
| blocksworld_invert_to_invert_blocks_list_1_1_1_1_2_2_2_4_4 | blocksworld | abstract/abstract | dominated_by_unparseable_first_greedy_behavior | 2 | 2 | no |


## Recommendation

- Recommended path: `focused_round4`
- Rationale: Most residual oracle-positive misses are still within-pool misrankings concentrated in blocksworld, especially abstract/abstract rows, with score gaps small enough to justify a targeted next mining round rather than an immediate objective change.
- Evidence: blocksworld ratio `1.00`, abstract/abstract ratio `0.78`, moderate-gap ratio `1.00`

## Comparison-Pool Recurrence (`K=8` miss slices)

| Pool | Oracle Misses | Blocksworld | Gripper | Abstract/Abstract | Explicit/Explicit |
| --- | --- | --- | --- | --- | --- |
| results\vcsr\bestofk_pilot\candidate_dump.jsonl | 4 | 4 | 0 | 3 | 1 |
| results\vcsr\bestofk_ranking_round2_pool\candidate_dump.jsonl | 10 | 10 | 0 | 8 | 2 |
