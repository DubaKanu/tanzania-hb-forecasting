# Stage 2 results on the development women

- Run ID: `run-c6b409a2f7`
- Partition: development (402 women)
- Python 3.9.25, scikit-learn 1.6.1, pandas 2.3.3, numpy 2.0.2
- Stage 1 pooled out-of-fold MAE on the training women: 0.760
- Training median first Hb used by the simple rule: 11.00 g/dL
- Demo cutoff for the API (development-fitted, not the evaluation rule): 0.915 g/dL

Lower scores mean more reliable forecasts. Errors are in g/dL. MAE is mean absolute error.

## Stage 1 forecasts on these women

MAE 0.794, RMSE 1.097, mean signed error -0.027 (positive means the forecast ran high).

## Primary comparison (declared in advance): learned minus tree disagreement at 80% coverage

On the development women, the learned score minus tree disagreement at 80% coverage is -0.015 g/dL (95% interval -0.058 to +0.025). The interval crosses zero. This shows neither superiority nor equivalence: the study reports the estimated difference and its uncertainty and does not establish equivalence.

At 80% coverage the exact-coverage rule keeps 322 of 402 women. Mean absolute error of the kept forecasts:

- Learned Stage 2 score: 0.698 g/dL
- Tree disagreement: 0.713 g/dL
- Simple rule: 0.796 g/dL
- Oracle (reference only): 0.482 g/dL
- Random selection: 0.794 g/dL (2.5th to 97.5th percentile of 200 repeats: 0.759 to 0.835)
- Keeping every forecast (100% coverage): 0.794 g/dL

## Every selector at every coverage level

| selector | coverage | kept | achieved coverage | MAE | RMSE | mean signed error | random 2.5th to 97.5th pct of MAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Learned Stage 2 score | 100% | 402 | 100.0% | 0.794 | 1.097 | -0.027 |  |
| Learned Stage 2 score | 90% | 362 | 90.0% | 0.753 | 1.036 | -0.007 |  |
| Learned Stage 2 score | 80% | 322 | 80.1% | 0.698 | 0.967 | 0.014 |  |
| Learned Stage 2 score | 70% | 281 | 69.9% | 0.693 | 0.961 | 0.011 |  |
| Tree disagreement | 100% | 402 | 100.0% | 0.794 | 1.097 | -0.027 |  |
| Tree disagreement | 90% | 362 | 90.0% | 0.764 | 1.057 | -0.001 |  |
| Tree disagreement | 80% | 322 | 80.1% | 0.713 | 0.987 | 0.035 |  |
| Tree disagreement | 70% | 281 | 69.9% | 0.671 | 0.924 | -0.002 |  |
| Simple rule | 100% | 402 | 100.0% | 0.794 | 1.097 | -0.027 |  |
| Simple rule | 90% | 362 | 90.0% | 0.759 | 1.048 | 0.004 |  |
| Simple rule | 80% | 322 | 80.1% | 0.796 | 1.090 | -0.003 |  |
| Simple rule | 70% | 281 | 69.9% | 0.771 | 1.053 | 0.003 |  |
| Oracle (reference only) | 100% | 402 | 100.0% | 0.794 | 1.097 | -0.027 |  |
| Oracle (reference only) | 90% | 362 | 90.0% | 0.598 | 0.755 | -0.011 |  |
| Oracle (reference only) | 80% | 322 | 80.1% | 0.482 | 0.589 | -0.003 |  |
| Oracle (reference only) | 70% | 281 | 69.9% | 0.394 | 0.473 | 0.034 |  |
| Random selection | 100% | 402 | 100.0% | 0.794 | 1.097 | -0.027 | 0.794 to 0.794 |
| Random selection | 90% | 362 | 90.0% | 0.794 | 1.097 | -0.026 | 0.771 to 0.817 |
| Random selection | 80% | 322 | 80.1% | 0.794 | 1.098 | -0.026 | 0.759 to 0.835 |
| Random selection | 70% | 281 | 69.9% | 0.793 | 1.096 | -0.029 | 0.746 to 0.840 |

Random selection is repeated 200 times with fixed seeds. The oracle keeps the forecasts that really were closest; it cannot be used in practice and only marks how much room any selector has.

## All paired comparisons (paired bootstrap, 5000 replicates)

A negative 'A - B' means A kept forecasts with the lower average error. 'keep all' is the average error of all forecasts at 100% coverage. Only the 80% learned-minus-tree-disagreement row was declared as primary; every other row is secondary.

| comparison | coverage | difference (g/dL) | 95% low | 95% high | bootstrap SE |
| --- | --- | --- | --- | --- | --- |
| learned - tree disagreement | 90% | -0.011 | -0.046 | 0.017 | 0.016 |
| learned - tree disagreement | 80% | -0.015 | -0.058 | 0.025 | 0.021 |
| learned - tree disagreement | 70% | 0.022 | -0.032 | 0.060 | 0.023 |
| learned - simple rule | 90% | -0.006 | -0.060 | 0.027 | 0.022 |
| learned - simple rule | 80% | -0.099 | -0.149 | -0.042 | 0.028 |
| learned - simple rule | 70% | -0.078 | -0.147 | -0.016 | 0.033 |
| tree disagreement - simple rule | 90% | 0.005 | -0.039 | 0.034 | 0.018 |
| tree disagreement - simple rule | 80% | -0.084 | -0.136 | -0.022 | 0.030 |
| tree disagreement - simple rule | 70% | -0.100 | -0.159 | -0.031 | 0.033 |
| learned - keep all | 90% | -0.041 | -0.079 | -0.014 | 0.017 |
| learned - keep all | 80% | -0.096 | -0.140 | -0.052 | 0.022 |
| learned - keep all | 70% | -0.101 | -0.158 | -0.048 | 0.028 |
| tree disagreement - keep all | 90% | -0.030 | -0.062 | -0.002 | 0.015 |
| tree disagreement - keep all | 80% | -0.081 | -0.124 | -0.038 | 0.022 |
| tree disagreement - keep all | 70% | -0.123 | -0.173 | -0.061 | 0.028 |
| simple rule - keep all | 90% | -0.035 | -0.060 | 0.001 | 0.016 |
| simple rule - keep all | 80% | 0.003 | -0.044 | 0.040 | 0.021 |
| simple rule - keep all | 70% | -0.023 | -0.080 | 0.031 | 0.029 |

## Who is kept at 80% coverage? (descriptive audit only)

These groups are never used to choose which forecasts to keep.

| selector | grouping | group | women | kept | share kept | MAE all women in group | MAE kept women in group |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Learned Stage 2 score | First haemoglobin (g/dL) | <10 | 68 | 31 | 45.6% | 1.116 | 0.874 |
| Learned Stage 2 score | First haemoglobin (g/dL) | 10 to <11 | 108 | 107 | 99.1% | 0.600 | 0.602 |
| Learned Stage 2 score | First haemoglobin (g/dL) | >=11 | 226 | 184 | 81.4% | 0.789 | 0.724 |
| Learned Stage 2 score | First-visit inputs | complete | 315 | 254 | 80.6% | 0.801 | 0.721 |
| Learned Stage 2 score | First-visit inputs | incomplete | 87 | 68 | 78.2% | 0.768 | 0.612 |
| Learned Stage 2 score | Follow-up gap | <=2 wk | 17 | 14 | 82.4% | 0.560 | 0.513 |
| Learned Stage 2 score | Follow-up gap | 3-4 wk | 192 | 161 | 83.9% | 0.842 | 0.747 |
| Learned Stage 2 score | Follow-up gap | 5-8 wk | 132 | 100 | 75.8% | 0.807 | 0.710 |
| Learned Stage 2 score | Follow-up gap | >8 wk | 61 | 47 | 77.0% | 0.677 | 0.558 |
| Learned Stage 2 score | Observed follow-up haemoglobin (g/dL) | <10 | 50 | 27 | 54.0% | 1.358 | 1.529 |
| Learned Stage 2 score | Observed follow-up haemoglobin (g/dL) | 10 to <11 | 119 | 106 | 89.1% | 0.620 | 0.534 |
| Learned Stage 2 score | Observed follow-up haemoglobin (g/dL) | >=11 | 233 | 189 | 81.1% | 0.761 | 0.671 |
| Tree disagreement | First haemoglobin (g/dL) | <10 | 68 | 37 | 54.4% | 1.116 | 0.883 |
| Tree disagreement | First haemoglobin (g/dL) | 10 to <11 | 108 | 108 | 100.0% | 0.600 | 0.600 |
| Tree disagreement | First haemoglobin (g/dL) | >=11 | 226 | 177 | 78.3% | 0.789 | 0.746 |
| Tree disagreement | First-visit inputs | complete | 315 | 252 | 80.0% | 0.801 | 0.727 |
| Tree disagreement | First-visit inputs | incomplete | 87 | 70 | 80.5% | 0.768 | 0.662 |
| Tree disagreement | Follow-up gap | <=2 wk | 17 | 12 | 70.6% | 0.560 | 0.537 |
| Tree disagreement | Follow-up gap | 3-4 wk | 192 | 157 | 81.8% | 0.842 | 0.772 |
| Tree disagreement | Follow-up gap | 5-8 wk | 132 | 101 | 76.5% | 0.807 | 0.708 |
| Tree disagreement | Follow-up gap | >8 wk | 61 | 52 | 85.2% | 0.677 | 0.583 |
| Tree disagreement | Observed follow-up haemoglobin (g/dL) | <10 | 50 | 33 | 66.0% | 1.358 | 1.499 |
| Tree disagreement | Observed follow-up haemoglobin (g/dL) | 10 to <11 | 119 | 107 | 89.9% | 0.620 | 0.531 |
| Tree disagreement | Observed follow-up haemoglobin (g/dL) | >=11 | 233 | 182 | 78.1% | 0.761 | 0.678 |
| Simple rule | First haemoglobin (g/dL) | <10 | 68 | 58 | 85.3% | 1.116 | 1.175 |
| Simple rule | First haemoglobin (g/dL) | 10 to <11 | 108 | 74 | 68.5% | 0.600 | 0.674 |
| Simple rule | First haemoglobin (g/dL) | >=11 | 226 | 190 | 84.1% | 0.789 | 0.729 |
| Simple rule | First-visit inputs | complete | 315 | 315 | 100.0% | 0.801 | 0.801 |
| Simple rule | First-visit inputs | incomplete | 87 | 7 | 8.0% | 0.768 | 0.610 |
| Simple rule | Follow-up gap | <=2 wk | 17 | 15 | 88.2% | 0.560 | 0.583 |
| Simple rule | Follow-up gap | 3-4 wk | 192 | 169 | 88.0% | 0.842 | 0.817 |
| Simple rule | Follow-up gap | 5-8 wk | 132 | 92 | 69.7% | 0.807 | 0.830 |
| Simple rule | Follow-up gap | >8 wk | 61 | 46 | 75.4% | 0.677 | 0.724 |
| Simple rule | Observed follow-up haemoglobin (g/dL) | <10 | 50 | 42 | 84.0% | 1.358 | 1.327 |
| Simple rule | Observed follow-up haemoglobin (g/dL) | 10 to <11 | 119 | 82 | 68.9% | 0.620 | 0.731 |
| Simple rule | Observed follow-up haemoglobin (g/dL) | >=11 | 233 | 198 | 85.0% | 0.761 | 0.711 |

## Does a higher score go with a larger real error?

Spearman correlation between each score and the real absolute error: learned stage 2 score 0.252, tree disagreement 0.243, simple rule 0.098.

| fifth of Stage 2 score | women | mean Stage 2 estimate | mean real error |
| --- | --- | --- | --- |
| 1 | 81 | 0.474 | 0.613 |
| 2 | 80 | 0.584 | 0.649 |
| 3 | 80 | 0.708 | 0.682 |
| 4 | 80 | 0.853 | 0.836 |
| 5 | 81 | 1.095 | 1.185 |

This table checks the ordering. It does not make the score a calibrated confidence interval.

## Complete-case sensitivity (women with all ten inputs)

The whole pipeline is repeated on the complete-case women only: 1412 training and 315 development women (of 1884 and 402). Folds, the out-of-fold table, Stage 1, Stage 2 and the comparison are all refitted on these women, with the same fixed settings.

On the development complete-case women, the learned score minus tree disagreement at 80% coverage is -0.043 g/dL (95% interval -0.088 to +0.005). The interval crosses zero. This shows neither superiority nor equivalence: the study reports the estimated difference and its uncertainty and does not establish equivalence.

| selector | coverage | kept | achieved coverage | MAE | RMSE | mean signed error | random 2.5th to 97.5th pct of MAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Learned Stage 2 score | 100% | 315 | 100.0% | 0.798 | 1.092 | -0.008 |  |
| Learned Stage 2 score | 90% | 284 | 90.2% | 0.738 | 1.015 | -0.014 |  |
| Learned Stage 2 score | 80% | 252 | 80.0% | 0.676 | 0.936 | 0.030 |  |
| Learned Stage 2 score | 70% | 221 | 70.2% | 0.677 | 0.937 | 0.034 |  |
| Tree disagreement | 100% | 315 | 100.0% | 0.798 | 1.092 | -0.008 |  |
| Tree disagreement | 90% | 284 | 90.2% | 0.769 | 1.051 | 0.026 |  |
| Tree disagreement | 80% | 252 | 80.0% | 0.719 | 0.993 | 0.051 |  |
| Tree disagreement | 70% | 221 | 70.2% | 0.681 | 0.945 | 0.055 |  |
| Simple rule | 100% | 315 | 100.0% | 0.798 | 1.092 | -0.008 |  |
| Simple rule | 90% | 284 | 90.2% | 0.772 | 1.051 | -0.008 |  |
| Simple rule | 80% | 252 | 80.0% | 0.741 | 1.013 | -0.010 |  |
| Simple rule | 70% | 221 | 70.2% | 0.698 | 0.961 | 0.027 |  |
| Oracle (reference only) | 100% | 315 | 100.0% | 0.798 | 1.092 | -0.008 |  |
| Oracle (reference only) | 90% | 284 | 90.2% | 0.605 | 0.757 | 0.013 |  |
| Oracle (reference only) | 80% | 252 | 80.0% | 0.492 | 0.597 | 0.021 |  |
| Oracle (reference only) | 70% | 221 | 70.2% | 0.405 | 0.481 | 0.027 |  |
| Random selection | 100% | 315 | 100.0% | 0.798 | 1.092 | -0.008 | 0.798 to 0.798 |
| Random selection | 90% | 284 | 90.2% | 0.798 | 1.091 | -0.009 | 0.770 to 0.821 |
| Random selection | 80% | 252 | 80.0% | 0.800 | 1.094 | -0.009 | 0.760 to 0.850 |
| Random selection | 70% | 221 | 70.2% | 0.797 | 1.090 | -0.004 | 0.749 to 0.851 |

## Limits that apply to every number above

- This is internal validation on one public dataset (five Tanzanian districts, not nationally representative). It is not temporal or external validation.
- There is no verified patient identifier. Keeping rows with an identical ten-input vector together is only a proxy for the same woman and cannot guarantee independence between partitions.
- The target is haemoglobin at the next recorded visit. The gap between visits runs from 1 to 28 weeks (median 4), so this is not a fixed-interval forecast.
- The out-of-fold forests that make the Stage 2 training table are fitted on two thirds of the training women, so the forecasts Stage 2 later receives come from a slightly stronger forest than the one that produced its training examples.
- The Stage 2 score is used to rank forecasts. It is not a calibrated confidence interval.
- Eligibility requires a repeat test, which selects women who were tested twice.
- 'Set aside' means a research output is not counted in this evaluation. It does not mean withholding care or testing, and clinical usefulness has not been verified.
