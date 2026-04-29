# Round 5 Regression Analysis

Baseline verifier: `retrain_from_round3_focused`
Candidate verifier: `retrain_from_round4_hybrid_pairwise`

## Top Line

- Replay dumps analyzed: `2`
- Changed outcome rows: `6`
- Round 5 helped rows: `1`
- Round 5 hurt rows: `5`

## Breakdown

- By K: `{'4': 4, '8': 2}`
- By domain: `{'blocksworld': 6}`
- By style: `{'explicit/explicit': 6}`

## Highest-Value Changed Rows

| Direction | K | Row | Domain | Style | Baseline idx | Candidate idx |
|---|---:|---|---|---|---:|---:|
| round5_hurt | 4 | `blocksworld_invert_to_invert_blocks_list_1_1_1_1_2_2_2_2_3_5` | blocksworld | explicit/explicit | 2 | 1 |
| round5_hurt | 4 | `blocksworld_invert_to_invert_blocks_list_1_2_2_4_4` | blocksworld | explicit/explicit | 3 | 2 |
| round5_hurt | 4 | `blocksworld_invert_to_invert_blocks_list_1_4_4_4_4` | blocksworld | explicit/explicit | 0 | 3 |
| round5_hurt | 8 | `blocksworld_invert_to_invert_blocks_list_1_2_2_4_4` | blocksworld | explicit/explicit | 3 | 2 |
| round5_hurt | 8 | `blocksworld_invert_to_invert_blocks_list_1_4_4_4_4` | blocksworld | explicit/explicit | 4 | 3 |
| round5_helped | 4 | `blocksworld_invert_to_invert_blocks_list_1_4_4_4_4` | blocksworld | explicit/explicit | 0 | 2 |

## Interpretation

Round 5 should be treated as diagnostic signal, not a promoted checkpoint. Rows where round 5 hurt round 4 are priority negatives for round 6, while rows where it helped show what a ranking objective should preserve.
