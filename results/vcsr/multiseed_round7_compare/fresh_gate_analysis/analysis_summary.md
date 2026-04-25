# Round 7 Fresh Gate Analysis

This is a fixed-artifact analysis only. It does not generate, train, or rescore.

Important caveat: the fresh multi-seed gate used separate generation runs for round 4 and round 7, so changed rows combine selector behavior with candidate-pool variance. Oracle-availability changes are therefore tracked separately from within-pool selection losses.

## Top Line

| K | Rows | Round 4 Eq | Round 7 Eq | Delta | Round 7 Helped | Round 7 Hurt | Ties |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 150 | 0.4000 | 0.4133 | +0.0133 | 7 | 5 | 138 |
| 8 | 150 | 0.4200 | 0.4200 | +0.0000 | 11 | 11 | 128 |

## Per-Seed K=8

| Seed | Round 4 Eq | Round 7 Eq | Delta | Helped | Hurt | Main Cause |
|---:|---:|---:|---:|---:|---:|---|
| 56 | 0.4800 | 0.4000 | -0.0800 | 3 | 7 | both_no_oracle |
| 57 | 0.2600 | 0.2800 | +0.0200 | 3 | 2 | both_no_oracle |
| 58 | 0.5200 | 0.5800 | +0.0600 | 5 | 2 | both_oracle_available_same_outcome |

## Cause Counts

### K=4

| Category | Count |
|---|---:|
| both_no_oracle | 72 |
| both_oracle_available_same_outcome | 59 |
| oracle_availability_loss | 8 |
| round7_selector_gain_with_oracle | 5 |
| round7_selector_loss_with_oracle | 3 |
| oracle_availability_gain | 3 |

### K=8

| Category | Count |
|---|---:|
| both_no_oracle | 67 |
| both_oracle_available_same_outcome | 59 |
| round7_selector_loss_with_oracle | 7 |
| round7_selector_gain_with_oracle | 7 |
| oracle_availability_loss | 5 |
| oracle_availability_gain | 5 |

## Seed 56 K=8 Hurt Rows

| Row | Domain | Style | Cause | R4 Eq Cnt | R7 Eq Cnt | R4 Sel | R7 Sel | R7 Wrong - Best Eq |
|---|---|---|---|---:|---:|---:|---:|---:|
| blocksworld_invert_to_invert_blocks_list_1_1_5_10 | blocksworld | abstract/abstract | round7_selector_loss_with_oracle | 1 | 1 | 3 | 2 | +0.0024 |
| blocksworld_invert_to_invert_blocks_list_1_1_1_1_2_2_2_4_4 | blocksworld | explicit/explicit | round7_selector_loss_with_oracle | 6 | 4 | 0 | 0 | +0.0000 |
| blocksworld_invert_to_invert_blocks_list_1_1_1_2_2_2_4_6 | blocksworld | abstract/abstract | round7_selector_loss_with_oracle | 2 | 1 | 6 | 0 | +0.0923 |
| blocksworld_invert_to_invert_blocks_list_1_4_15 | blocksworld | abstract/abstract | round7_selector_loss_with_oracle | 4 | 4 | 2 | 0 | +0.0010 |
| blocksworld_invert_to_invert_blocks_list_1_2_3_3_3_4_4 | blocksworld | abstract/abstract | round7_selector_loss_with_oracle | 2 | 2 | 5 | 0 | +0.0073 |
| gripper_juggle_to_juggle_balls_in_grippers_1_1_0_0_0_0_0_0_0_0_balls_in_rooms_7 | gripper | abstract/abstract | oracle_availability_loss | 1 | 0 | 0 | 0 |  |
| blocksworld_swap_to_swap_blocks_list_11_16 | blocksworld | abstract/abstract | round7_selector_loss_with_oracle | 8 | 5 | 3 | 6 | +0.0024 |

## Interpretation

- Round 7 tied round 4 at K=8 overall on the fresh gate, so it is not promotion-worthy despite replay gains.
- The seed-56 K=8 drop is mixed: it includes both candidate-pool/oracle-availability effects and within-pool round-7 selector losses.
- Round 7 is not uniformly worse; it helped and hurt different rows, which supports keeping it as evidence but not as best_current.
- Round 7 improved K=4 slightly on fresh seeds, but the project operating point is K=8 and the K=8 gate did not move.

## Recommendation

Keep round 4 as the official `best_current`. Treat round 7 as a useful focused-pointwise diagnostic, not a promoted model. The next step should not be another blind retrain; either design a fixed-pool fresh comparison to isolate verifier effects, or shift attention to candidate-pool/generator diversity because fresh-pool variance is now large enough to hide small verifier gains.
