# NAB Robustness Analysis

Phase: 5  
Analysis run: `experiments/anomaly/nab/robustness/run_phase5_nab_robustness_final`  
SMD profile run: `experiments/smd/profile/run_phase5_smd_profile_final`

This phase asks why Phase 4 detectors have low median per-series robustness and whether more NAB tuning is worth doing before moving to SMD.

## Phase 4 Baseline Metrics

| Model | Precision | Recall | F1 | PR-AUC | FPR | Detected windows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Rolling z-score | 0.0660 | 0.0142 | 0.0233 | 0.1175 | 0.0239 | 37 / 42 |
| Isolation Forest | 0.2511 | 0.1397 | 0.1795 | 0.2093 | 0.0498 | 36 / 42 |
| Dense Autoencoder | 0.2964 | 0.2718 | 0.2835 | 0.1796 | 0.0771 | 35 / 42 |
| LSTM Autoencoder | 0.2523 | 0.2245 | 0.2376 | 0.1913 | 0.0795 | 35 / 42 |

## Robustness Summary

| Model | Nonzero F1 series | Nonzero F1 on positive series | Mean F1 | Median F1 | F1 variance | Best F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Rolling z-score | 27 / 58 | 90.0% | 0.0163 | 0.0000 | 0.0008 | 0.1818 |
| Isolation Forest | 26 / 58 | 86.7% | 0.0834 | 0.0000 | 0.0183 | 0.6019 |
| Dense Autoencoder | 26 / 58 | 86.7% | 0.1521 | 0.0000 | 0.0472 | 0.8245 |
| LSTM Autoencoder | 26 / 58 | 86.7% | 0.1475 | 0.0000 | 0.0423 | 0.7505 |

The low median is not because every positive series is missed. It is caused by a mix of no-anomaly series where false positives force F1 to 0, heterogeneous labeled series, and a few positive-test series where every model misses the relevant window.

## Threshold Sensitivity

Thresholds were selected without test labels. The precision-recall tradeoff CSV is diagnostic and uses test labels only to visualize the operating tradeoff.

| Model | Threshold strategy | Precision | Recall | F1 | FPR | Detected windows |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Dense AE | train-MAD | 0.1891 | 0.3621 | 0.2485 | 0.1854 | 38 / 42 |
| Dense AE | train-p99 | 0.2964 | 0.2718 | 0.2835 | 0.0771 | 35 / 42 |
| Dense AE | validation-p99 | 0.3425 | 0.2901 | 0.3141 | 0.0665 | 35 / 42 |
| IF | train-MAD | 0.1096 | 0.0750 | 0.0891 | 0.0728 | 31 / 42 |
| IF | train-p99 | 0.2511 | 0.1397 | 0.1795 | 0.0498 | 36 / 42 |
| IF | validation-p99 | 0.2803 | 0.1243 | 0.1722 | 0.0381 | 36 / 42 |

Recommendation: use validation-p99 for Dense AE in the next NAB-style evaluation. Keep train-p99 as the IF operating point unless the application strongly prioritizes precision over recall.

## Window Sensitivity

This was a bounded diagnostic on six labeled NAB series with two training epochs per neural model.

| Model | Window | Precision | Recall | F1 | FPR | Detected windows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense AE | 16 | 0.8943 | 0.1645 | 0.2779 | 0.0076 | 3 / 7 |
| LSTM AE | 16 | 0.8956 | 0.1625 | 0.2751 | 0.0074 | 3 / 7 |
| Dense AE | 32 | 0.8612 | 0.1869 | 0.3072 | 0.0113 | 3 / 7 |
| LSTM AE | 32 | 0.8842 | 0.1864 | 0.3079 | 0.0091 | 3 / 7 |
| Dense AE | 64 | 0.8208 | 0.2724 | 0.4090 | 0.0205 | 3 / 7 |
| LSTM AE | 64 | 0.8437 | 0.2690 | 0.4079 | 0.0172 | 3 / 7 |

Window 64 improved F1 and recall on this subset, but it did not increase detected-window count. Treat it as a promising default for SMD sequence modeling, not as proof that NAB is solved.

## IF vs Dense Complementarity

Point-level overlap:

- Both models fired on 4,815 points.
- IF-only detections: 1,589 points.
- Dense-only detections: 5,742 points.
- Detection-union Jaccard: 0.3964.

Window-level overlap:

| Window overlap | Count |
| --- | ---: |
| both | 35 |
| IF only | 1 |
| Dense only | 0 |
| neither | 6 |

Dense AE adds many true-positive points inside already-detected windows, especially on `cpu_utilization_asg_misconfiguration`, `ec2_cpu_utilization_ac20cd`, and `nyc_taxi`. It does not add separate window coverage over IF in this operating-point analysis.

Simple train-calibrated IF/Dense max-percentile diagnostic:

| Precision | Recall | F1 | FPR | Detected windows |
| ---: | ---: | ---: | ---: | ---: |
| 0.2899 | 0.3048 | 0.2972 | 0.0892 | 36 / 42 |

The ensemble improves recall over Dense train-p99 and IF, but it is worse than Dense validation-p99 F1 and has higher FPR. A production ensemble is not justified yet.

## Official NAB Scoring

Exact official NAB scoring is not reported. The local NAB mirror contains `data`, `labels`, `README.md`, and `LICENSE.txt`, but not the official scorer/application-profile implementation required to reproduce normalized NAB scores exactly. Phase 5 therefore reports generic point metrics, detection delay, and window-hit counts only.

## Final Recommendation

Best baseline: rolling z-score remains useful as a cheap smoke-test detector, but its recall and F1 are too weak for the portfolio-grade incident engine.

Best classical method: Isolation Forest with train-p99. It has the best PR-AUC among Phase 4 models and lower FPR than the neural models, but its operating-point recall is limited.

Best neural method: Dense Autoencoder with validation-p99. It has the strongest measured F1 in Phase 5, 0.3141, and improves both recall and FPR versus the original train-p99 Dense run.

Best window strategy: window 64 is the most promising diagnostic setting for neural models, but this should be validated on SMD rather than over-optimized on NAB.

Ensemble decision: do not promote the simple IF/Dense ensemble. The overlap analysis shows limited window-level complementarity, and the diagnostic ensemble does not beat Dense validation-p99.

Further NAB tuning: stop major NAB optimization. The benchmark has served its purpose: it exposed threshold sensitivity, heterogeneous univariate behavior, noisy normals, and delayed detection. The next scientific step is multivariate SMD modeling, where cross-metric dependencies can be tested directly.
