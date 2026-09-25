# Q2 DS High-Compute v2.1 Correction Run Report

- Work directory: `/data2/hy/q2_runs/q2_ds_hc_v2_1_20260925`
- Baseline replay: `passed`
- Selected member count: `5`
- Selector gate: `True`
- Confirmation: `candidate_rejected_baseline_retained`
- Frozen choice: `baseline`

| Confirmation partition | R_MAE | R_F1 | Worst MAE | Clean MAE | Clean F1 |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.550052 | 0.594693 | 0.576508 | 0.537906 | 0.604744 |
| Candidate | 0.548435 | 0.595562 | 0.579566 | 0.532002 | 0.611626 |

| Full valid, descriptive | R_MAE | R_F1 | Worst MAE | Clean MAE | Clean F1 |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.598691 | 0.579696 | 0.637334 | 0.579967 | 0.599707 |
| Candidate | 0.593489 | 0.584305 | 0.635518 | 0.570395 | 0.613101 |

## Compute

- Completed training jobs: `84`
- Sum of per-job training time: `0.991 GPU-hours`
- Maximum recorded allocated GPU memory: `1.306 GiB`

## Freeze

- Status: `baseline_retained_without_new_test`
- Model version: `M3-ensemble9-retained`
- Compact member bytes: `12709084`
- Projected combined bytes: `41724828`
- Test status at freeze: `not_run_for_this_version`

## Post-Freeze Publication

- Test Accuracy: `0.635488`
- Test Macro-F1: `0.596337`
- Test MAE: `0.643349`
- Test Pearson: `0.657517`
- Attachment 3 CSV SHA-256: `aa25e119480aabd278d3249e1388522c20528520595e9e6985777e80b77470a4`
- Test and attachment 3 did not change the frozen selection.
