# Dynamic Weighted Logistic Regression — Publication Core v1.4.0

Standalone implementation and user-facing runner for:

- `LR`: LBFGS + L2 logistic-regression baseline
- `LR_SGD`: unweighted SGD logistic-regression comparator
- `LR_sen`
- `LR_spe`
- `LR_BA`
- `LR_sen_spe_corridor`
- `LR_sen_spe_tails`

`LR_SGD` deliberately uses unit sample weights and no temperature calibration (`T=1`) so that the SGD optimizer can be compared with the five dynamic-weighted variants.

## Three-layer evaluation workflow

A single run can use both an internal holdout and an external independent dataset:

```text
intra-dataset
    -> stratified outer split
         -> development_training
              -> inner stratified CV
              -> best-epoch selection
              -> final refit
         -> internal_test
external dataset
    -> external_test
```

The internal and external test layers do not participate in fitting, temperature estimation, threshold derivation, or best-epoch selection.

## Input

Use either a combined CSV/TSV containing sample ID, binary label and feature columns, or separate feature/label files joined by sample ID. External features are reordered to the development feature order.

Example:

```bash
python dynamic_weighted_logistic_regression.py \
  --development-data intra_dataset.csv \
  --external-data external_dataset.csv \
  --sample-id-col sample_id \
  --label-col label \
  --internal-test-fraction 0.20 \
  --split-random-state 731 \
  --models LR LR_SGD LR_BA LR_sen LR_spe LR_sen_spe_corridor LR_sen_spe_tails \
  --k-folds 10 \
  --standardize-features \
  --output-dir results
```

JSON configuration is supported:

```bash
python dynamic_weighted_logistic_regression.py --config dwlr_example_config.json
```

## Diagnostic outputs in v1.4.0

The runner now preserves training diagnostics by default for iterative models.

```text
results/
├── metrics.csv
├── training_summary.csv
├── run_manifest.json
├── predictions/
├── history/
├── cv_fold_history/
├── final_refit_history/
├── epoch_probabilities/
├── epoch_probability_plots/
└── models/
```

`training_summary.csv` is the quickest place to review:

- `best_epoch`, `best_epoch_metric`
- best CV score/log loss/balanced accuracy
- epochs completed and maximum requested epochs
- early-stop flag and stop reason
- final temperature and final classification threshold
- requested/actual inner-CV folds
- final dynamic thresholds when applicable

`metrics.csv` carries the same frozen training metadata alongside every development-training, internal-test and external-test performance row.

## Epoch probability trajectories

For `LR_SGD` and the five dynamic variants, the final-development refit can save:

- `epoch_probabilities/<model>__epoch_raw_probabilities.csv`
- `epoch_probabilities/<model>__epoch_calibrated_probabilities.csv`
- `epoch_probabilities/<model>__epoch_temperatures.csv`

The first physical row is sample ID, the second is `y_true`, and subsequent rows are `epoch_1`, `epoch_2`, ... . The sample order is fixed across epochs.

Two 3D figures are generated from those exact two probability matrices:

- `epoch_probability_plots/<model>__raw_probability_3D.png`
- `epoch_probability_plots/<model>__calibrated_probability_3D.png`

Axes: X = fixed sample position, Y = final-refit epoch, Z = predicted probability. `<model>__plot_sample_order.csv` maps sample positions back to IDs and labels.

For `LR_SGD`, raw and calibrated trajectories are intentionally identical because calibration is disabled (`T=1`). For dynamic variants, the difference between the two surfaces shows the effect of temperature scaling.

## Training history

- `history/`: one row per epoch with cross-fold mean/SD selection diagnostics;
- `cv_fold_history/`: one row per fold per epoch, including thresholds, temperatures, active samples and weight summaries where applicable;
- `final_refit_history/`: per-epoch diagnostics during the final refit on all development-training samples.

History tables identify the selected best epoch, making it straightforward to check whether `n_epochs` is acting as a harmless ceiling or truncating model selection.

## Predictions and metrics

Sample-level predictions save raw probability, calibrated probability, predicted label, threshold, temperature, sample ID, true label and split.

Performance metrics include confusion counts, prevalence, accuracy, balanced accuracy, sensitivity, specificity, PPV, NPV, FPR/FNR/FDR, F1/F2, MCC, kappa, Youden's J, ROC AUC, average precision, PR AUC, Brier score and log loss.

A genuine integrated Brier score is not derived from the current single-horizon binary probability; it requires time-to-event/censoring information and survival-probability curves.
