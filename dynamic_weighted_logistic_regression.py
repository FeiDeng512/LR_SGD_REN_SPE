"""Dynamic weighted logistic regression algorithms.

Standalone implementation of an LBFGS logistic-regression baseline, an unweighted
SGD logistic-regression comparator, and five dynamic SGD variants: LR_sen, LR_spe, LR_BA, LR_sen_spe_corridor, and
LR_sen_spe_tails. The module includes temperature scaling, target-oriented
thresholds, dynamic sample weighting, configurable cross-validation, optional
feature standardization, Brier/log-loss evaluation, common-gene alignment,
and optional epoch-wise probability export.

This module intentionally excludes study-specific cohort preprocessing and legacy
comparison models.  A generic command-line interface is provided so users can
load their own sample-by-feature tables, configure model/training parameters,
fit the baseline and/or dynamic variants, split an intra-dataset into
development-training and internal-test subsets, optionally evaluate an independent
external dataset, and save predictions, histories, models, and epoch-wise
probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_curve
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

__version__ = "1.4.0"

__all__ = [
    "DYNAMIC_VARIANTS",
    "RUNNER_DEFAULTS",
    "DynamicFitResult",
    "fit_baseline_lr",
    "fit_sgd_lr",
    "fit_dynamic_lr",
    "predict_probability",
    "predict_label",
    "evaluate_binary_predictions",
    "align_common_gene_tables",
    "build_epoch_probability_table",
    "save_epoch_probability_tables",
    "compute_integrated_brier_score",
    "load_supervised_data",
    "split_development_internal_test",
    "run_models_from_config",
    "main",
]

EPS = 1e-7

DYNAMIC_VARIANTS = {
    "LR_sen",
    "LR_spe",
    "LR_BA",
    "LR_sen_spe_corridor",
    "LR_sen_spe_tails",
}


def _as_binary_xy(X, y):
    Xn = X.values if hasattr(X, "values") else np.asarray(X)
    yn = y.values.ravel() if hasattr(y, "values") else np.asarray(y).ravel()
    yn = yn.astype(int)
    u = np.unique(yn)
    if not set(u.tolist()).issubset({0, 1}) or len(u) != 2:
        raise ValueError(f"This implementation requires binary labels 0/1; got {u}.")
    if len(Xn) != len(yn):
        raise ValueError("X and y have different numbers of samples.")
    return np.asarray(Xn, dtype=float), yn


def _fit_feature_transformer(X_fit, *, impute_missing=True, standardize_features=False):
    """Fit median imputation and optional standardization on model-fitting data only."""
    X_fit = np.asarray(X_fit, dtype=float)
    if X_fit.ndim != 2:
        raise ValueError("X must be a 2D feature matrix.")
    all_missing = np.isnan(X_fit).all(axis=0)
    if np.any(all_missing):
        bad = np.where(all_missing)[0].tolist()
        raise ValueError(f"Model-fitting subset contains all-missing feature columns at indices {bad[:10]}.")
    imputer = None
    Xt = X_fit
    if bool(impute_missing):
        imputer = SimpleImputer(strategy="median")
        Xt = imputer.fit_transform(Xt)
    elif np.isnan(Xt).any():
        raise ValueError("Missing values are present but impute_missing=False.")
    scaler = None
    if bool(standardize_features):
        scaler = StandardScaler()
        Xt = scaler.fit_transform(Xt)
    return np.asarray(Xt, dtype=float), imputer, scaler


def _apply_feature_transformer(X, imputer=None, scaler=None):
    Xt = np.asarray(X, dtype=float)
    if imputer is not None:
        Xt = imputer.transform(Xt)
    elif np.isnan(Xt).any():
        raise ValueError("Missing values are present but the fitted model has no imputer.")
    if scaler is not None:
        Xt = scaler.transform(Xt)
    return np.asarray(Xt, dtype=float)


# ============================================================================
# USER-EDITABLE CORE HYPERPARAMETERS
# Keep these assignments explicit so they can be changed later without
# rewriting the model functions.
#
# IMPORTANT learning-rate note:
#   SGD_LEARNING_RATE_SCHEDULE = "optimal"  -> sklearn ignores SGD_ETA0.
#   SGD_LEARNING_RATE_SCHEDULE = "constant" -> eta_t = SGD_ETA0 at every update.
#   SGD_LEARNING_RATE_SCHEDULE = "invscaling" -> eta_t = SGD_ETA0 / t**SGD_POWER_T.
# Current default uses a constant learning rate (eta0 = 0.003). Change the schedule to "optimal" later if desired; the option is retained explicitly.
# ============================================================================
BASELINE_LR_C = 1.0
DYNAMIC_GAMMA = 1.0
MIN_ACTIVE_SAMPLES = 3
SGD_ETA0 = 0.003
SGD_LEARNING_RATE_SCHEDULE = "constant"
SGD_POWER_T = 0.5
SGD_ALPHA = 2e-3
SGD_L1_RATIO = 0.05
SGD_PENALTY = "elasticnet"
SGD_AVERAGE = True
WEIGHT_HOLD_EPOCHS = 3
WEIGHT_RAMP_EPOCHS = 3
BEST_EPOCH_METRIC = "log_loss"  # options: "log_loss" or "balanced_accuracy"
DEFAULT_K_FOLDS = 10  # current default; callers may set 5, 10, or another valid integer
STANDARDIZE_FEATURES = True  # current default; set False to disable
ALLOW_SMALL_SAMPLE_CV_REDUCTION = False



def clip_prob(p, eps: float = EPS):
    return np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)


def logit_from_prob(p, eps: float = 1e-12):
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def apply_temperature(p_raw, T: float, eps: float = EPS):
    z = logit_from_prob(p_raw)
    z = np.clip(z / max(float(T), 1e-8), -50.0, 50.0)
    return clip_prob(1.0 / (1.0 + np.exp(-z)), eps=eps)


def learn_temperature_from_holdout(p_raw_cal, y_cal,
                                   bounds: Tuple[float, float] = (0.05, 20.0),
                                   eps: float = EPS):
    y_cal = np.asarray(y_cal, dtype=int).ravel()
    p_raw_cal = clip_prob(p_raw_cal, eps)
    if len(np.unique(y_cal)) < 2:
        return 1.0

    lo, hi = map(float, bounds)
    if not (0 < lo < hi):
        raise ValueError("temp_bounds must satisfy 0 < low < high.")

    def objective(log_t_arr):
        T = float(np.exp(log_t_arr[0]))
        q = apply_temperature(p_raw_cal, T, eps=eps)
        return float(-np.mean(y_cal * np.log(q) + (1 - y_cal) * np.log(1 - q)))

    res = minimize(
        objective,
        x0=np.array([0.0]),
        bounds=[(np.log(lo), np.log(hi))],
        method="L-BFGS-B",
    )
    if not res.success or not np.isfinite(res.fun):
        return 1.0
    return float(np.clip(np.exp(res.x[0]), lo, hi))


def proba_of_pos(model, X, pos_label=1):
    classes = list(model.classes_)
    if pos_label not in classes:
        raise ValueError(f"Positive label {pos_label} is absent from model.classes_={classes}.")
    return model.predict_proba(X)[:, classes.index(pos_label)]


def _candidate_thresholds(p):
    p = clip_prob(p)
    # Include 0 and 1 so all empirical classification regimes are available.
    return np.unique(np.r_[0.0, p, 1.0])


def threshold_for_target_sensitivity(y_true, p, target_sens: float):
    """Empirical sup{tau: TPR(tau) >= target_sens}, using prediction rule p >= tau."""
    y = np.asarray(y_true, dtype=int).ravel()
    p = clip_prob(p)
    fpr, tpr, thresholds = roc_curve(y, p)
    finite = np.isfinite(thresholds)
    feasible = finite & (tpr >= float(target_sens))
    if np.any(feasible):
        return float(np.max(thresholds[feasible]))
    # Fallback: smallest finite cutoff gives maximal sensitivity.
    th = thresholds[finite]
    return float(np.min(th)) if len(th) else 0.5


def threshold_for_target_specificity(y_true, p, target_spec: float):
    """Empirical inf{tau: TNR(tau) >= target_spec}, using prediction rule p >= tau."""
    y = np.asarray(y_true, dtype=int).ravel()
    p = clip_prob(p)
    fpr, tpr, thresholds = roc_curve(y, p)
    spec = 1.0 - fpr
    finite = np.isfinite(thresholds)
    feasible = finite & (spec >= float(target_spec))
    if np.any(feasible):
        return float(np.min(thresholds[feasible]))
    th = thresholds[finite]
    return float(np.max(th)) if len(th) else 0.5


def _ba_at_threshold(y, p, tau):
    pred = (p >= tau).astype(int)
    return float(balanced_accuracy_score(y, pred))


def threshold_max_ba(y_true, p):
    y = np.asarray(y_true, dtype=int).ravel()
    p = clip_prob(p)
    thresholds = _candidate_thresholds(p)
    bas = np.asarray([_ba_at_threshold(y, p, t) for t in thresholds])
    best = np.max(bas)
    idx = np.where(np.isclose(bas, best, atol=1e-12))[0]
    # deterministic tie-break toward 0.5
    j = idx[np.argmin(np.abs(thresholds[idx] - 0.5))]
    return float(thresholds[j]), float(best)


def normalized_left_distance(p, tau, eps: float = 1e-12):
    tau = float(np.clip(tau, EPS, 1.0 - EPS))
    return np.maximum(tau - np.asarray(p), 0.0) / (tau + eps)


def normalized_right_distance(p, tau, eps: float = 1e-12):
    tau = float(np.clip(tau, EPS, 1.0 - EPS))
    return np.maximum(np.asarray(p) - tau, 0.0) / (1.0 - tau + eps)


def _shape(d, gamma: float):
    if not (1.0 <= float(gamma) <= 2.0):
        raise ValueError("gamma must be in [1, 2].")
    return np.power(np.clip(np.asarray(d, dtype=float), 0.0, 1.0), float(gamma))


def _active_count_from_threshold(y, p, variant, taus):
    if variant == "LR_sen":
        return int(np.sum((y == 1) & (p < taus["sen"])))
    if variant == "LR_spe":
        return int(np.sum((y == 0) & (p > taus["spe"])))
    if variant == "LR_BA":
        t = taus["ba"]
        return int(np.sum(((y == 1) & (p < t)) | ((y == 0) & (p > t))))
    if variant == "LR_sen_spe_corridor":
        lo, hi = taus["low"], taus["high"]
        return int(np.sum((p > lo) & (p < hi)))
    if variant == "LR_sen_spe_tails":
        return int(np.sum(((y == 1) & (p < taus["sen"])) | ((y == 0) & (p > taus["spe"]))))
    raise ValueError(variant)


def _ensure_min_active(y, p, variant, taus: Dict[str, float], min_active: int):
    """Apply the minimum-active-sample training safeguard with audit metadata.

    The requested minimum is capped by the number of samples that are eligible to
    receive nonzero intensity for one-sided variants (positives for LR_sen and
    negatives for LR_spe). For the other variants, every sample can in principle
    become active after threshold relaxation.
    """
    taus = dict(taus)
    requested = 0 if min_active is None else max(0, int(min_active))

    if variant == "LR_sen":
        eligible = int(np.sum(np.asarray(y) == 1))
    elif variant == "LR_spe":
        eligible = int(np.sum(np.asarray(y) == 0))
    else:
        eligible = int(len(y))

    effective = min(requested, eligible)
    before = _active_count_from_threshold(y, p, variant, taus)
    taus["safeguard_requested_min"] = int(requested)
    taus["safeguard_effective_min"] = int(effective)
    taus["safeguard_active_before"] = int(before)
    taus["safeguard_shifted"] = False

    if effective <= 0 or before >= effective:
        taus["safeguard_active_after"] = int(before)
        return taus

    original = dict(taus)

    if variant == "LR_sen":
        vals = np.sort(p[y == 1])
        if len(vals):
            k = min(effective, len(vals))
            taus["sen"] = float(np.clip(np.nextafter(vals[k - 1], 1.0), EPS, 1.0 - EPS))

    elif variant == "LR_spe":
        vals = np.sort(p[y == 0])[::-1]
        if len(vals):
            k = min(effective, len(vals))
            taus["spe"] = float(np.clip(np.nextafter(vals[k - 1], 0.0), EPS, 1.0 - EPS))

    elif variant == "LR_BA":
        # Base design: tau_BA = (tau_sen + tau_spe) / 2.
        # The safeguard is separate and may minimally shift only the operative
        # tau_BA; ba_base preserves the unshifted midpoint for traceability.
        base = float(taus["ba"])
        taus.setdefault("ba_base", base)
        candidates = _candidate_thresholds(p)
        feasible = []
        for t in candidates:
            n = np.sum(((y == 1) & (p < t)) | ((y == 0) & (p > t)))
            if n >= effective:
                feasible.append(float(t))
        if feasible:
            taus["ba"] = min(feasible, key=lambda t: (abs(t - base), abs(t - 0.5), t))
        taus["ba_safeguard_shifted"] = bool(
            not np.isclose(taus["ba"], base, atol=1e-15, rtol=0.0)
        )

    elif variant == "LR_sen_spe_corridor":
        lo, hi = taus["low"], taus["high"]
        center = 0.5 * (lo + hi)
        # Expand symmetrically just enough to contain the effective minimum
        # closest probabilities. nextafter makes boundary observations active
        # because the triangular intensity is zero exactly at the boundaries.
        dist = np.sort(np.abs(p - center))
        if len(dist):
            radius = float(dist[min(effective, len(dist)) - 1])
            taus["low"] = float(np.clip(np.nextafter(center - radius, 0.0), EPS, 1.0 - EPS))
            taus["high"] = float(np.clip(np.nextafter(center + radius, 1.0), EPS, 1.0 - EPS))

    elif variant == "LR_sen_spe_tails":
        # Relax the nearest class-specific tail boundary/boundaries.
        need = effective - before
        events = []
        sen = taus["sen"]
        spe = taus["spe"]
        for val in p[(y == 1) & (p >= sen)]:
            events.append((float(val - sen), "sen", float(val)))
        for val in p[(y == 0) & (p <= spe)]:
            events.append((float(spe - val), "spe", float(val)))
        events.sort(key=lambda z: (z[0], z[1], z[2]))
        chosen = events[:max(0, need)]
        chosen_pos = [v for _, side, v in chosen if side == "sen"]
        chosen_neg = [v for _, side, v in chosen if side == "spe"]
        if chosen_pos:
            taus["sen"] = float(np.clip(np.nextafter(max(chosen_pos), 1.0), EPS, 1.0 - EPS))
        if chosen_neg:
            taus["spe"] = float(np.clip(np.nextafter(min(chosen_neg), 0.0), EPS, 1.0 - EPS))

    else:
        raise ValueError(variant)

    after = _active_count_from_threshold(y, p, variant, taus)
    taus["safeguard_active_after"] = int(after)
    # Generic flag for all variants; LR_BA also keeps its more specific flag.
    keys_to_compare = {
        "LR_sen": ("sen",),
        "LR_spe": ("spe",),
        "LR_BA": ("ba",),
        "LR_sen_spe_corridor": ("low", "high"),
        "LR_sen_spe_tails": ("sen", "spe"),
    }[variant]
    taus["safeguard_shifted"] = any(
        not np.isclose(float(taus[k]), float(original[k]), atol=1e-15, rtol=0.0)
        for k in keys_to_compare
    )
    return taus


def compute_dynamic_intensity(y, p, variant: str, taus: Dict[str, float],
                              gamma: float = DYNAMIC_GAMMA,
                              lambda_fn: Optional[float] = None,
                              lambda_fp: Optional[float] = None,
):
    y = np.asarray(y, dtype=int).ravel()
    p = clip_prob(p)

    if variant == "LR_sen":
        t = taus["sen"]
        s = ((y == 1) & (p < t)).astype(float) * _shape(normalized_left_distance(p, t), gamma)

    elif variant == "LR_spe":
        t = taus["spe"]
        s = ((y == 0) & (p > t)).astype(float) * _shape(normalized_right_distance(p, t), gamma)

    elif variant == "LR_BA":
        # Current design: tau_BA is the midpoint of the sensitivity- and
        # specificity-derived thresholds.  It is the single gate for both
        # potential FN and potential FP samples.
        t = float(taus["ba"])
        gate = ((y == 1) & (p < t)) | ((y == 0) & (p > t))
        s = gate.astype(float) * _shape(np.abs(p - y), gamma)

    elif variant == "LR_sen_spe_corridor":
        lo, hi = float(taus["low"]), float(taus["high"])
        if hi <= lo + 1e-12:
            s = np.zeros_like(p)
        else:
            center = 0.5 * (lo + hi)
            halfw = 0.5 * (hi - lo)
            s = np.maximum(0.0, 1.0 - np.abs(p - center) / (halfw + 1e-12))

    elif variant == "LR_sen_spe_tails":
        sen, spe = float(taus["sen"]), float(taus["spe"])
        s_left = (y == 1).astype(float) * _shape(normalized_left_distance(p, sen), gamma)
        s_right = (y == 0).astype(float) * _shape(normalized_right_distance(p, spe), gamma)
        if lambda_fn is None:
            lambda_fn = 1.0 / max(np.mean(y == 1), 1e-12)
        if lambda_fp is None:
            lambda_fp = 1.0 / max(np.mean(y == 0), 1e-12)
        s = np.minimum(1.0, float(lambda_fn) * s_left + float(lambda_fp) * s_right)

    else:
        raise ValueError(f"Unknown dynamic variant: {variant}")

    return np.clip(s, 0.0, 1.0)


def _derive_thresholds(y_ref, p_ref, target_sens, target_spec):
    tau_sen = threshold_for_target_sensitivity(y_ref, p_ref, target_sens)
    tau_spe = threshold_for_target_specificity(y_ref, p_ref, target_spec)
    # Current model design:
    #     tau_BA = (tau_sen + tau_spe) / 2
    # This is a threshold midpoint, not a threshold obtained by solving
    # |BA(tau) - Target_BA| -> min.
    tau_ba = 0.5 * (float(tau_sen) + float(tau_spe))
    target_ba = 0.5 * (float(target_sens) + float(target_spec))
    low, high = sorted([tau_sen, tau_spe])
    return {
        "sen": float(tau_sen),
        "spe": float(tau_spe),
        "ba": float(tau_ba),
        "ba_base": float(tau_ba),
        "low": float(low),
        "high": float(high),
        "target_ba": float(target_ba),
        "ba_safeguard_shifted": False,
    }


def _make_sgd(random_state, learning_rate, sgd_schedule, sgd_alpha,
              sgd_l1_ratio, sgd_penalty, sgd_average, sgd_power_t):
    allowed_schedules = {"constant", "optimal", "invscaling", "adaptive"}
    if sgd_schedule not in allowed_schedules:
        raise ValueError(f"sgd_schedule must be one of {sorted(allowed_schedules)}; got {sgd_schedule!r}")
    return SGDClassifier(
        loss="log_loss",
        penalty=sgd_penalty,
        alpha=float(sgd_alpha),
        l1_ratio=float(sgd_l1_ratio) if sgd_penalty == "elasticnet" else 0.0,
        fit_intercept=True,
        learning_rate=sgd_schedule,
        eta0=float(learning_rate),
        power_t=float(sgd_power_t),
        max_iter=1,
        tol=None,
        shuffle=True,
        average=sgd_average,
        random_state=int(random_state),
    )


def _split_fit_cal(X, y, frac, random_state):
    if frac is None or frac <= 0 or frac >= 0.5:
        idx = np.arange(len(y))
        return idx, idx
    # Need at least one calibration observation per class.
    counts = np.bincount(y, minlength=2)
    if counts.min() < 2:
        idx = np.arange(len(y))
        return idx, idx
    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=float(frac), random_state=int(random_state))
        fit_idx, cal_idx = next(sss.split(X, y))
        return fit_idx, cal_idx
    except ValueError:
        idx = np.arange(len(y))
        return idx, idx


def _weight_from_intensity(
    s, emphasis_factor, epoch, hold_epochs, ramp_epochs, w_clip
):
    """Map intensity to weights with a configurable hold-then-ramp schedule.

    Epoch numbering is 1-based conceptually while ``epoch`` is 0-based internally.
    - first ``hold_epochs`` epochs: rho_t = 1
    - next ``ramp_epochs`` epochs: linearly ramp rho_t from 1 toward emphasis_factor
    - afterward: rho_t = emphasis_factor

    Examples
    --------
    hold=3, ramp=3 -> rho is 1 for epochs 1-3, then ramps on epochs 4-6.
    hold=0, ramp=3 -> reproduces the earlier immediate-ramp behavior.
    hold=0, ramp=0 -> full emphasis from epoch 1.
    """
    hold = max(0, int(hold_epochs or 0))
    ramp_n = max(0, int(ramp_epochs or 0))
    epoch_num = int(epoch) + 1

    if epoch_num <= hold:
        progress = 0.0
    elif ramp_n <= 0:
        progress = 1.0
    else:
        ramp_step = epoch_num - hold
        progress = min(1.0, float(ramp_step) / float(ramp_n))

    rho_t = 1.0 + (float(emphasis_factor) - 1.0) * progress
    w = 1.0 + (rho_t - 1.0) * np.asarray(s, dtype=float)
    if w_clip is not None:
        w = np.clip(w, float(w_clip[0]), float(w_clip[1]))
    return w / (np.mean(w) + 1e-12), rho_t, progress


def _selection_metric_value(avg_loss, avg_ba, metric):
    metric = str(metric).lower().strip()
    if metric == "log_loss":
        return float(avg_loss), False  # lower is better
    if metric == "balanced_accuracy":
        return float(avg_ba), True  # higher is better
    raise ValueError('best_epoch_metric must be "log_loss" or "balanced_accuracy".')


def _is_better_score(score, best_score, higher_is_better, tol=1e-12):
    if best_score is None:
        return True
    if higher_is_better:
        return score > best_score + tol
    return score < best_score - tol


@dataclass
class DynamicFitResult:
    model: SGDClassifier
    final_threshold: float
    final_temperature: float
    best_epoch: int
    best_epoch_metric: str
    best_cv_score: float
    best_cv_log_loss: float
    best_cv_balanced_accuracy: float
    history: List[dict]
    cv_fold_history: List[dict]
    final_refit_history: List[dict]
    epochs_completed: int
    max_epochs_requested: int
    early_stopped: bool
    stop_reason: str
    final_thresholds: Dict[str, float]
    variant: str
    epoch_sample_ids: Optional[List[str]] = None
    epoch_labels: Optional[np.ndarray] = None
    epoch_raw_probabilities: Optional[np.ndarray] = None
    epoch_calibrated_probabilities: Optional[np.ndarray] = None
    epoch_temperatures: Optional[List[float]] = None


@dataclass
class SGDFitResult:
    """Result for the unweighted LR_SGD comparator.

    LR_SGD uses the same SGD/regularization family as the dynamic variants but
    deliberately disables temperature calibration and dynamic sample weighting.
    Inner CV selects the number of unweighted SGD epochs; the final decision
    threshold remains fixed at 0.5.
    """
    model: SGDClassifier
    final_threshold: float
    best_epoch: int
    best_epoch_metric: str
    best_cv_score: float
    best_cv_log_loss: float
    best_cv_balanced_accuracy: float
    history: List[dict]
    cv_fold_history: List[dict]
    final_refit_history: List[dict]
    epochs_completed: int
    max_epochs_requested: int
    early_stopped: bool
    stop_reason: str
    epoch_sample_ids: Optional[List[str]] = None
    epoch_labels: Optional[np.ndarray] = None
    epoch_raw_probabilities: Optional[np.ndarray] = None
    epoch_calibrated_probabilities: Optional[np.ndarray] = None
    epoch_temperatures: Optional[List[float]] = None


def fit_sgd_lr(
    X, y,
    n_epochs: int = 50,
    k_folds: int = DEFAULT_K_FOLDS,
    learning_rate: float = SGD_ETA0,
    sgd_schedule: str = SGD_LEARNING_RATE_SCHEDULE,
    sgd_alpha: float = SGD_ALPHA,
    sgd_l1_ratio: float = SGD_L1_RATIO,
    sgd_penalty: str = SGD_PENALTY,
    sgd_average: bool = SGD_AVERAGE,
    sgd_power_t: float = SGD_POWER_T,
    best_epoch_metric: str = BEST_EPOCH_METRIC,
    early_stopping_rounds: Optional[int] = 5,
    random_state: int = 42,
    pos_label: int = 1,
    verbose: bool = False,
    capture_epoch_probabilities: bool = False,
    sample_ids: Optional[List[str]] = None,
    impute_missing: bool = True,
    standardize_features: bool = STANDARDIZE_FEATURES,
    allow_small_sample_cv_reduction: bool = ALLOW_SMALL_SAMPLE_CV_REDUCTION,
) -> SGDFitResult:
    """Fit an unweighted SGD logistic-regression comparator.

    Besides the fitted model, this implementation records an epoch-level CV
    summary, a fold-by-epoch CV audit table, and a final-refit history.  These
    diagnostics do not alter model fitting; they make convergence and epoch
    selection inspectable after the run.
    """
    X, y = _as_binary_xy(X, y)
    n = len(y)
    if sample_ids is None:
        sample_ids = [str(i) for i in range(n)]
    else:
        sample_ids = [str(x) for x in sample_ids]
        if len(sample_ids) != n:
            raise ValueError("sample_ids length must match X/y.")

    class_counts = np.bincount(y, minlength=2)
    max_k = int(class_counts.min())
    if max_k < 2:
        raise ValueError("At least two samples are needed in each class for stratified CV.")
    requested_k = int(k_folds)
    if requested_k < 2:
        raise ValueError("k_folds must be at least 2.")
    if requested_k > max_k:
        if bool(allow_small_sample_cv_reduction):
            k = max_k
        else:
            raise ValueError(
                f"Requested k_folds={requested_k}, but the minority class has only {max_k} samples."
            )
    else:
        k = requested_k
    n_epochs = max(1, int(n_epochs))
    max_epochs_requested = int(n_epochs)

    metric_name = str(best_epoch_metric).lower().strip()
    _, higher_is_better = _selection_metric_value(0.0, 0.0, metric_name)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=int(random_state))
    fold_states = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, imp, scaler = _fit_feature_transformer(
            X[train_idx], impute_missing=impute_missing, standardize_features=standardize_features
        )
        X_val = _apply_feature_transformer(X[val_idx], imp, scaler)
        fold_states.append({
            "fold": int(fold_idx + 1),
            "model": _make_sgd(
                random_state + fold_idx, learning_rate, sgd_schedule, sgd_alpha,
                sgd_l1_ratio, sgd_penalty, sgd_average, sgd_power_t,
            ),
            "X_train": X_train, "y_train": y[train_idx],
            "X_val": X_val, "y_val": y[val_idx],
        })

    history: List[dict] = []
    cv_fold_history: List[dict] = []
    best_score = None
    best_epoch = 0
    best_cv_log_loss = np.nan
    best_cv_ba = np.nan
    no_improve = 0
    early_stopped = False
    stop_reason = "max_epochs_reached"

    for epoch in range(n_epochs):
        losses, bas = [], []
        for state in fold_states:
            model = state["model"]
            if epoch == 0:
                model.partial_fit(state["X_train"], state["y_train"], classes=np.array([0, 1]))
            else:
                model.partial_fit(state["X_train"], state["y_train"])
            p_val = clip_prob(proba_of_pos(model, state["X_val"], pos_label))
            fold_loss = float(log_loss(state["y_val"], p_val, labels=[0, 1]))
            fold_ba = float(_ba_at_threshold(state["y_val"], p_val, 0.5))
            losses.append(fold_loss)
            bas.append(fold_ba)
            cv_fold_history.append({
                "epoch": int(epoch + 1),
                "fold": int(state["fold"]),
                "n_fit": int(len(state["y_train"])),
                "n_validation": int(len(state["y_val"])),
                "cv_log_loss": fold_loss,
                "cv_balanced_accuracy": fold_ba,
                "evaluation_threshold": 0.5,
                "temperature": 1.0,
                "active_samples": int(len(state["y_train"])),
                "active_fraction": 1.0,
                "weight_mean": 1.0,
                "weight_std": 0.0,
                "weight_min": 1.0,
                "weight_max": 1.0,
            })

        avg_loss = float(np.mean(losses))
        avg_ba = float(np.mean(bas))
        sd_loss = float(np.std(losses, ddof=1)) if len(losses) > 1 else 0.0
        sd_ba = float(np.std(bas, ddof=1)) if len(bas) > 1 else 0.0
        history.append({
            "epoch": int(epoch + 1),
            "cv_log_loss": avg_loss,
            "cv_log_loss_sd": sd_loss,
            "cv_balanced_accuracy": avg_ba,
            "cv_balanced_accuracy_sd": sd_ba,
            "n_folds": int(k),
            "weight_multiplier": 1.0,
            "weight_progress": 1.0,
            "mean_active_samples": float(np.mean([len(st["y_train"]) for st in fold_states])),
        })
        if verbose:
            print(f"[LR_SGD] epoch={epoch+1:03d} CV logloss={avg_loss:.5f} BA={avg_ba:.4f}")

        current_score, higher_is_better = _selection_metric_value(avg_loss, avg_ba, metric_name)
        if _is_better_score(current_score, best_score, higher_is_better):
            best_score = float(current_score)
            best_epoch = epoch + 1
            best_cv_log_loss = avg_loss
            best_cv_ba = avg_ba
            no_improve = 0
        else:
            no_improve += 1
            if early_stopping_rounds is not None and no_improve >= int(early_stopping_rounds):
                early_stopped = True
                stop_reason = "early_stopping"
                break

    epochs_completed = int(len(history))
    if best_epoch <= 0:
        raise RuntimeError("LR_SGD did not complete a selectable epoch.")

    X_final, final_imputer, final_scaler = _fit_feature_transformer(
        X, impute_missing=impute_missing, standardize_features=standardize_features
    )
    final_model = _make_sgd(
        random_state + 9999, learning_rate, sgd_schedule, sgd_alpha,
        sgd_l1_ratio, sgd_penalty, sgd_average, sgd_power_t,
    )

    epoch_raw_rows = [] if capture_epoch_probabilities else None
    final_refit_history: List[dict] = []
    for epoch in range(best_epoch):
        if epoch == 0:
            final_model.partial_fit(X_final, y, classes=np.array([0, 1]))
        else:
            final_model.partial_fit(X_final, y)
        p_after_raw = clip_prob(proba_of_pos(final_model, X_final, pos_label))
        if capture_epoch_probabilities:
            epoch_raw_rows.append(np.asarray(p_after_raw, dtype=float))
        coef = np.asarray(final_model.coef_, dtype=float)
        final_refit_history.append({
            "epoch": int(epoch + 1),
            "temperature": 1.0,
            "decision_threshold": 0.5,
            "raw_probability_mean": float(np.mean(p_after_raw)),
            "raw_probability_std": float(np.std(p_after_raw)),
            "raw_probability_min": float(np.min(p_after_raw)),
            "raw_probability_max": float(np.max(p_after_raw)),
            "calibrated_probability_mean": float(np.mean(p_after_raw)),
            "calibrated_probability_std": float(np.std(p_after_raw)),
            "predicted_positive_rate": float(np.mean(p_after_raw >= 0.5)),
            "coef_l2_norm": float(np.linalg.norm(coef)),
            "intercept": float(np.asarray(final_model.intercept_).ravel()[0]),
            "active_samples": int(n),
            "active_fraction": 1.0,
            "weight_mean": 1.0,
            "weight_std": 0.0,
            "weight_min": 1.0,
            "weight_max": 1.0,
        })

    final_model.dw_variant_ = "LR_SGD"
    final_model.dw_temperature_ = 1.0
    final_model.dw_threshold_ = 0.5
    final_model.dw_imputer_ = final_imputer
    final_model.dw_scaler_ = final_scaler
    final_model.dw_impute_missing_ = bool(impute_missing)
    final_model.dw_standardize_features_ = bool(standardize_features)
    final_model.dw_requested_k_folds_ = int(requested_k)
    final_model.dw_actual_k_folds_ = int(k)
    final_model.dw_best_epoch_ = int(best_epoch)
    final_model.dw_best_epoch_metric_ = metric_name
    final_model.dw_best_cv_score_ = float(best_score)
    final_model.dw_best_cv_log_loss_ = float(best_cv_log_loss)
    final_model.dw_best_cv_balanced_accuracy_ = float(best_cv_ba)
    final_model.dw_learning_rate_eta0_ = float(learning_rate)
    final_model.dw_sgd_schedule_ = str(sgd_schedule)
    final_model.dw_sgd_alpha_ = float(sgd_alpha)
    final_model.dw_sgd_l1_ratio_ = float(sgd_l1_ratio)
    final_model.dw_sgd_penalty_ = str(sgd_penalty)
    final_model.dw_sgd_average_ = bool(sgd_average)
    final_model.dw_sgd_power_t_ = float(sgd_power_t)
    final_model.dw_epochs_completed_ = int(epochs_completed)
    final_model.dw_max_epochs_requested_ = int(max_epochs_requested)
    final_model.dw_early_stopped_ = bool(early_stopped)
    final_model.dw_stop_reason_ = str(stop_reason)

    raw_mat = np.vstack(epoch_raw_rows) if capture_epoch_probabilities and epoch_raw_rows else None
    cal_mat = None if raw_mat is None else raw_mat.copy()
    temps = None if raw_mat is None else [1.0] * raw_mat.shape[0]

    return SGDFitResult(
        model=final_model,
        final_threshold=0.5,
        best_epoch=int(best_epoch),
        best_epoch_metric=metric_name,
        best_cv_score=float(best_score),
        best_cv_log_loss=float(best_cv_log_loss),
        best_cv_balanced_accuracy=float(best_cv_ba),
        history=history,
        cv_fold_history=cv_fold_history,
        final_refit_history=final_refit_history,
        epochs_completed=int(epochs_completed),
        max_epochs_requested=int(max_epochs_requested),
        early_stopped=bool(early_stopped),
        stop_reason=str(stop_reason),
        epoch_sample_ids=list(sample_ids) if capture_epoch_probabilities else None,
        epoch_labels=np.asarray(y, dtype=int).copy() if capture_epoch_probabilities else None,
        epoch_raw_probabilities=raw_mat,
        epoch_calibrated_probabilities=cal_mat,
        epoch_temperatures=temps,
    )

def fit_dynamic_lr(
    X, y,
    variant: str,
    n_epochs: int = 50,
    k_folds: int = DEFAULT_K_FOLDS,
    target_sens: float = 0.70,
    target_spec: float = 0.70,
    emphasis_factor: float = 2.0,
    gamma: float = DYNAMIC_GAMMA,
    learning_rate: float = SGD_ETA0,
    sgd_schedule: str = SGD_LEARNING_RATE_SCHEDULE,
    sgd_alpha: float = SGD_ALPHA,
    sgd_l1_ratio: float = SGD_L1_RATIO,
    sgd_penalty: str = SGD_PENALTY,
    sgd_average: bool = SGD_AVERAGE,
    sgd_power_t: float = SGD_POWER_T,
    calibrate: str = "temperature",
    calib_holdout: float = 0.20,
    temp_bounds: Tuple[float, float] = (0.05, 20.0),
    weight_hold_epochs: int = WEIGHT_HOLD_EPOCHS,
    weight_ramp_epochs: int = WEIGHT_RAMP_EPOCHS,
    w_clip: Tuple[float, float] = (0.67, 3.0),
    min_active_samples: int = MIN_ACTIVE_SAMPLES,
    best_epoch_metric: str = BEST_EPOCH_METRIC,
    early_stopping_rounds: Optional[int] = 5,
    random_state: int = 42,
    pos_label: int = 1,
    verbose: bool = False,
    capture_epoch_probabilities: bool = False,
    sample_ids: Optional[List[str]] = None,
    impute_missing: bool = True,
    standardize_features: bool = STANDARDIZE_FEATURES,
    allow_small_sample_cv_reduction: bool = ALLOW_SMALL_SAMPLE_CV_REDUCTION,
) -> DynamicFitResult:
    """Fit one dynamic-weighted SGD logistic model with detailed audit output.

    The numerical fitting algorithm is unchanged: fold-specific preprocessing,
    calibration and thresholds drive dynamic sample weighting; inner CV selects
    the epoch count; the final model is then refit on the full development set.
    Additional histories record fold-level and final-refit diagnostics so that
    convergence, thresholds, temperatures, active samples and weights can be
    inspected after training.
    """
    if variant not in DYNAMIC_VARIANTS:
        raise ValueError(f"variant must be one of {sorted(DYNAMIC_VARIANTS)}; got {variant!r}")
    if not (0 < target_sens <= 1 and 0 < target_spec <= 1):
        raise ValueError("target_sens and target_spec must be in (0, 1].")

    X, y = _as_binary_xy(X, y)
    n = len(y)
    if sample_ids is None:
        sample_ids = [str(i) for i in range(n)]
    else:
        sample_ids = [str(x) for x in sample_ids]
        if len(sample_ids) != n:
            raise ValueError("sample_ids length must match X/y for epoch-probability capture.")

    class_counts = np.bincount(y, minlength=2)
    max_k = int(class_counts.min())
    if max_k < 2:
        raise ValueError("At least two samples are needed in each class for stratified CV.")
    requested_k = int(k_folds)
    if requested_k < 2:
        raise ValueError("k_folds must be at least 2.")
    if requested_k > max_k:
        if bool(allow_small_sample_cv_reduction):
            k = max_k
        else:
            raise ValueError(
                f"Requested k_folds={requested_k}, but the minority class has only {max_k} samples. "
                "Reduce N_SPLITS or enable ALLOW_SMALL_SAMPLE_CV_REDUCTION only for small tests."
            )
    else:
        k = requested_k
    n_epochs = max(1, int(n_epochs))
    max_epochs_requested = int(n_epochs)

    lambda_fn = 1.0 / max(np.mean(y == 1), 1e-12)
    lambda_fp = 1.0 / max(np.mean(y == 0), 1e-12)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=int(random_state))
    fold_states = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        fit_local, cal_local = _split_fit_cal(
            X_tr, y_tr, calib_holdout, random_state + 1000 + fold_idx
        )
        X_fit_raw, y_fit = X_tr[fit_local], y_tr[fit_local]
        X_cal_raw, y_cal = X_tr[cal_local], y_tr[cal_local]
        X_val_raw, y_val = X[val_idx], y[val_idx]
        X_fit, fold_imputer, fold_scaler = _fit_feature_transformer(
            X_fit_raw, impute_missing=impute_missing, standardize_features=standardize_features
        )
        X_cal = _apply_feature_transformer(X_cal_raw, fold_imputer, fold_scaler)
        X_val = _apply_feature_transformer(X_val_raw, fold_imputer, fold_scaler)
        state = {
            "fold": int(fold_idx + 1),
            "train_idx": train_idx,
            "val_idx": val_idx,
            "X_fit": X_fit,
            "y_fit": y_fit,
            "X_cal": X_cal,
            "y_cal": y_cal,
            "X_val": X_val,
            "y_val": y_val,
            "imputer": fold_imputer,
            "scaler": fold_scaler,
            "model": _make_sgd(
                random_state + fold_idx, learning_rate, sgd_schedule,
                sgd_alpha, sgd_l1_ratio, sgd_penalty, sgd_average, sgd_power_t,
            ),
        }
        # Neutral initialization gives the first probability distribution from
        # which dynamic thresholds/intensities can be derived.
        state["model"].partial_fit(
            state["X_fit"], state["y_fit"], classes=np.array([0, 1]),
            sample_weight=np.ones(len(state["y_fit"]), dtype=float),
        )
        fold_states.append(state)

    metric_name = str(best_epoch_metric).lower().strip()
    _, higher_is_better = _selection_metric_value(0.0, 0.0, metric_name)

    history: List[dict] = []
    cv_fold_history: List[dict] = []
    best_score = None
    best_epoch = 0
    best_cv_log_loss = np.nan
    best_cv_balanced_accuracy = np.nan
    no_improve = 0
    early_stopped = False
    stop_reason = "max_epochs_reached"

    for epoch in range(n_epochs):
        losses, bas, active_counts = [], [], []
        temperatures_pre, temperatures_eval = [], []
        tau_sens, tau_spes, tau_bas, tau_lows, tau_highs = [], [], [], [], []
        weight_means, weight_stds, weight_mins, weight_maxs = [], [], [], []
        intensity_means, intensity_maxs, safeguard_shifted = [], [], []
        rho_epoch = None
        progress_epoch = None

        for state in fold_states:
            model = state["model"]
            X_fit, y_fit = state["X_fit"], state["y_fit"]
            X_cal, y_cal = state["X_cal"], state["y_cal"]
            X_val, y_val = state["X_val"], state["y_val"]

            p_cal_raw = clip_prob(proba_of_pos(model, X_cal, pos_label))
            if str(calibrate).lower() == "temperature":
                T = learn_temperature_from_holdout(p_cal_raw, y_cal, temp_bounds)
            else:
                T = 1.0
            p_cal = apply_temperature(p_cal_raw, T)
            taus = _derive_thresholds(y_cal, p_cal, target_sens, target_spec)

            p_fit_raw = clip_prob(proba_of_pos(model, X_fit, pos_label))
            p_fit = apply_temperature(p_fit_raw, T)
            taus = _ensure_min_active(y_fit, p_fit, variant, taus, min_active_samples)
            if variant != "LR_sen_spe_corridor":
                taus["low"], taus["high"] = sorted([taus["sen"], taus["spe"]])

            intensity = compute_dynamic_intensity(
                y_fit, p_fit, variant, taus, gamma,
                lambda_fn=lambda_fn, lambda_fp=lambda_fp,
            )
            w, rho_t, weight_progress = _weight_from_intensity(
                intensity, emphasis_factor, epoch, weight_hold_epochs, weight_ramp_epochs, w_clip
            )
            rho_epoch = float(rho_t)
            progress_epoch = float(weight_progress)
            model.partial_fit(X_fit, y_fit, sample_weight=w)

            # Re-estimate calibration after the weighted update. Validation is
            # evaluated only with parameters obtained on the training side.
            p_cal2_raw = clip_prob(proba_of_pos(model, X_cal, pos_label))
            if str(calibrate).lower() == "temperature":
                T_eval = learn_temperature_from_holdout(p_cal2_raw, y_cal, temp_bounds)
            else:
                T_eval = 1.0
            p_cal_eval = apply_temperature(p_cal2_raw, T_eval)
            eval_threshold, _ = threshold_max_ba(y_cal, p_cal_eval)
            p_val = apply_temperature(clip_prob(proba_of_pos(model, X_val, pos_label)), T_eval)
            loss = float(log_loss(y_val, p_val, labels=[0, 1]))
            ba = float(_ba_at_threshold(y_val, p_val, eval_threshold))
            active = int(np.sum(intensity > 0))

            losses.append(loss)
            bas.append(ba)
            active_counts.append(active)
            temperatures_pre.append(float(T))
            temperatures_eval.append(float(T_eval))
            tau_sens.append(float(taus["sen"]))
            tau_spes.append(float(taus["spe"]))
            tau_bas.append(float(taus["ba"]))
            tau_lows.append(float(taus["low"]))
            tau_highs.append(float(taus["high"]))
            weight_means.append(float(np.mean(w)))
            weight_stds.append(float(np.std(w)))
            weight_mins.append(float(np.min(w)))
            weight_maxs.append(float(np.max(w)))
            intensity_means.append(float(np.mean(intensity)))
            intensity_maxs.append(float(np.max(intensity)))
            safeguard_shifted.append(bool(taus.get("safeguard_shifted", False)))

            cv_fold_history.append({
                "epoch": int(epoch + 1),
                "fold": int(state["fold"]),
                "n_fit": int(len(y_fit)),
                "n_calibration": int(len(y_cal)),
                "n_validation": int(len(y_val)),
                "cv_log_loss": loss,
                "cv_balanced_accuracy": ba,
                "evaluation_threshold": float(eval_threshold),
                "temperature_before_update": float(T),
                "temperature_after_update": float(T_eval),
                "tau_sen": float(taus["sen"]),
                "tau_spe": float(taus["spe"]),
                "tau_ba": float(taus["ba"]),
                "tau_low": float(taus["low"]),
                "tau_high": float(taus["high"]),
                "safeguard_requested_min": int(taus.get("safeguard_requested_min", min_active_samples)),
                "safeguard_effective_min": int(taus.get("safeguard_effective_min", min_active_samples)),
                "safeguard_active_before": int(taus.get("safeguard_active_before", active)),
                "safeguard_active_after": int(taus.get("safeguard_active_after", active)),
                "safeguard_shifted": bool(taus.get("safeguard_shifted", False)),
                "active_samples": active,
                "active_fraction": float(active / max(len(y_fit), 1)),
                "intensity_mean": float(np.mean(intensity)),
                "intensity_max": float(np.max(intensity)),
                "weight_mean": float(np.mean(w)),
                "weight_std": float(np.std(w)),
                "weight_min": float(np.min(w)),
                "weight_max": float(np.max(w)),
                "weight_multiplier": float(rho_t),
                "weight_progress": float(weight_progress),
            })

        avg_loss = float(np.mean(losses))
        avg_ba = float(np.mean(bas))
        row = {
            "epoch": int(epoch + 1),
            "cv_log_loss": avg_loss,
            "cv_log_loss_sd": float(np.std(losses, ddof=1)) if len(losses) > 1 else 0.0,
            "cv_balanced_accuracy": avg_ba,
            "cv_balanced_accuracy_sd": float(np.std(bas, ddof=1)) if len(bas) > 1 else 0.0,
            "mean_active_samples": float(np.mean(active_counts)),
            "sd_active_samples": float(np.std(active_counts, ddof=1)) if len(active_counts) > 1 else 0.0,
            "mean_temperature_before_update": float(np.mean(temperatures_pre)),
            "mean_temperature_after_update": float(np.mean(temperatures_eval)),
            "mean_tau_sen": float(np.mean(tau_sens)),
            "mean_tau_spe": float(np.mean(tau_spes)),
            "mean_tau_ba": float(np.mean(tau_bas)),
            "mean_tau_low": float(np.mean(tau_lows)),
            "mean_tau_high": float(np.mean(tau_highs)),
            "mean_intensity": float(np.mean(intensity_means)),
            "mean_max_intensity": float(np.mean(intensity_maxs)),
            "mean_weight": float(np.mean(weight_means)),
            "mean_weight_std": float(np.mean(weight_stds)),
            "mean_weight_min": float(np.mean(weight_mins)),
            "mean_weight_max": float(np.mean(weight_maxs)),
            "safeguard_shift_fraction": float(np.mean(safeguard_shifted)),
            "weight_multiplier": float(rho_epoch),
            "weight_progress": float(progress_epoch),
            "n_folds": int(k),
        }
        history.append(row)
        if verbose:
            print(
                f"[{variant}] epoch={epoch+1:03d} "
                f"CV logloss={avg_loss:.5f} BA={avg_ba:.4f} "
                f"active={row['mean_active_samples']:.1f}"
            )

        full_emphasis = bool(np.isclose(rho_epoch, float(emphasis_factor), atol=1e-12, rtol=0.0))
        if full_emphasis:
            current_score, higher_is_better = _selection_metric_value(avg_loss, avg_ba, metric_name)
            if _is_better_score(current_score, best_score, higher_is_better):
                best_score = float(current_score)
                best_epoch = epoch + 1
                best_cv_log_loss = float(avg_loss)
                best_cv_balanced_accuracy = float(avg_ba)
                no_improve = 0
            else:
                no_improve += 1
                if early_stopping_rounds is not None and no_improve >= int(early_stopping_rounds):
                    early_stopped = True
                    stop_reason = "early_stopping"
                    break

    epochs_completed = int(len(history))
    if best_epoch <= 0:
        if not history:
            raise RuntimeError("No training epochs were completed.")
        chosen = history[-1]
        best_epoch = int(chosen["epoch"])
        best_cv_log_loss = float(chosen["cv_log_loss"])
        best_cv_balanced_accuracy = float(chosen["cv_balanced_accuracy"])
        best_score = float(
            chosen["cv_log_loss"] if metric_name == "log_loss" else chosen["cv_balanced_accuracy"]
        )
        if epochs_completed < (int(weight_hold_epochs) + int(weight_ramp_epochs)):
            stop_reason = "max_epochs_before_full_emphasis"

    # ---------------- Final refit on the full development set ----------------
    X_final, final_imputer, final_scaler = _fit_feature_transformer(
        X, impute_missing=impute_missing, standardize_features=standardize_features
    )
    final_model = _make_sgd(
        random_state + 9999, learning_rate, sgd_schedule,
        sgd_alpha, sgd_l1_ratio, sgd_penalty, sgd_average, sgd_power_t,
    )
    final_model.partial_fit(
        X_final, y, classes=np.array([0, 1]), sample_weight=np.ones(n, dtype=float)
    )

    epoch_raw_rows = [] if capture_epoch_probabilities else None
    epoch_cal_rows = [] if capture_epoch_probabilities else None
    epoch_temperatures = [] if capture_epoch_probabilities else None
    final_refit_history: List[dict] = []

    final_taus = None
    for epoch in range(max(1, best_epoch)):
        p_raw = clip_prob(proba_of_pos(final_model, X_final, pos_label))
        if str(calibrate).lower() == "temperature":
            T_epoch = learn_temperature_from_holdout(p_raw, y, temp_bounds)
        else:
            T_epoch = 1.0
        p_cal = apply_temperature(p_raw, T_epoch)
        taus = _derive_thresholds(y, p_cal, target_sens, target_spec)
        taus = _ensure_min_active(y, p_cal, variant, taus, min_active_samples)
        if variant != "LR_sen_spe_corridor":
            taus["low"], taus["high"] = sorted([taus["sen"], taus["spe"]])
        intensity = compute_dynamic_intensity(
            y, p_cal, variant, taus, gamma,
            lambda_fn=lambda_fn, lambda_fp=lambda_fp,
        )
        w, rho_t, weight_progress = _weight_from_intensity(
            intensity, emphasis_factor, epoch, weight_hold_epochs, weight_ramp_epochs, w_clip
        )
        final_model.partial_fit(X_final, y, sample_weight=w)

        p_after_raw = clip_prob(proba_of_pos(final_model, X_final, pos_label))
        if str(calibrate).lower() == "temperature":
            T_after = learn_temperature_from_holdout(p_after_raw, y, temp_bounds)
        else:
            T_after = 1.0
        p_after_cal = apply_temperature(p_after_raw, T_after)
        if capture_epoch_probabilities:
            epoch_raw_rows.append(np.asarray(p_after_raw, dtype=float))
            epoch_cal_rows.append(np.asarray(p_after_cal, dtype=float))
            epoch_temperatures.append(float(T_after))

        coef = np.asarray(final_model.coef_, dtype=float)
        active = int(np.sum(intensity > 0))
        final_refit_history.append({
            "epoch": int(epoch + 1),
            "temperature_before_update": float(T_epoch),
            "temperature_after_update": float(T_after),
            "tau_sen": float(taus["sen"]),
            "tau_spe": float(taus["spe"]),
            "tau_ba": float(taus["ba"]),
            "tau_low": float(taus["low"]),
            "tau_high": float(taus["high"]),
            "safeguard_requested_min": int(taus.get("safeguard_requested_min", min_active_samples)),
            "safeguard_effective_min": int(taus.get("safeguard_effective_min", min_active_samples)),
            "safeguard_active_before": int(taus.get("safeguard_active_before", active)),
            "safeguard_active_after": int(taus.get("safeguard_active_after", active)),
            "safeguard_shifted": bool(taus.get("safeguard_shifted", False)),
            "active_samples": active,
            "active_fraction": float(active / max(n, 1)),
            "intensity_mean": float(np.mean(intensity)),
            "intensity_max": float(np.max(intensity)),
            "weight_mean": float(np.mean(w)),
            "weight_std": float(np.std(w)),
            "weight_min": float(np.min(w)),
            "weight_max": float(np.max(w)),
            "weight_multiplier": float(rho_t),
            "weight_progress": float(weight_progress),
            "raw_probability_mean": float(np.mean(p_after_raw)),
            "raw_probability_std": float(np.std(p_after_raw)),
            "raw_probability_min": float(np.min(p_after_raw)),
            "raw_probability_max": float(np.max(p_after_raw)),
            "calibrated_probability_mean": float(np.mean(p_after_cal)),
            "calibrated_probability_std": float(np.std(p_after_cal)),
            "calibrated_probability_min": float(np.min(p_after_cal)),
            "calibrated_probability_max": float(np.max(p_after_cal)),
            "predicted_positive_rate_at_0_5": float(np.mean(p_after_cal >= 0.5)),
            "coef_l2_norm": float(np.linalg.norm(coef)),
            "intercept": float(np.asarray(final_model.intercept_).ravel()[0]),
        })
        final_taus = taus

    p_full_raw = clip_prob(proba_of_pos(final_model, X_final, pos_label))
    if str(calibrate).lower() == "temperature":
        T_final = learn_temperature_from_holdout(p_full_raw, y, temp_bounds)
    else:
        T_final = 1.0
    p_full = apply_temperature(p_full_raw, T_final)
    final_threshold, final_ba = threshold_max_ba(y, p_full)

    final_model.dw_variant_ = variant
    final_model.dw_temperature_ = float(T_final)
    final_model.dw_threshold_ = float(final_threshold)
    final_model.dw_target_sens_ = float(target_sens)
    final_model.dw_target_spec_ = float(target_spec)
    final_model.dw_target_ba_ = float(0.5 * (target_sens + target_spec))
    final_model.dw_gamma_ = float(gamma)
    final_model.dw_emphasis_factor_ = float(emphasis_factor)
    final_model.dw_min_active_samples_ = int(min_active_samples)
    final_model.dw_best_epoch_ = int(best_epoch)
    final_model.dw_best_epoch_metric_ = metric_name
    final_model.dw_best_cv_score_ = float(best_score)
    final_model.dw_best_cv_log_loss_ = float(best_cv_log_loss)
    final_model.dw_best_cv_balanced_accuracy_ = float(best_cv_balanced_accuracy)
    final_model.dw_final_train_ba_ = float(final_ba)
    final_model.dw_thresholds_ = dict(final_taus or {})
    final_model.dw_tau_ba_definition_ = "(tau_sen + tau_spe) / 2"
    final_model.dw_learning_rate_eta0_ = float(learning_rate)
    final_model.dw_sgd_schedule_ = str(sgd_schedule)
    final_model.dw_sgd_alpha_ = float(sgd_alpha)
    final_model.dw_sgd_l1_ratio_ = float(sgd_l1_ratio)
    final_model.dw_sgd_penalty_ = str(sgd_penalty)
    final_model.dw_sgd_average_ = bool(sgd_average)
    final_model.dw_sgd_power_t_ = float(sgd_power_t)
    final_model.dw_weight_hold_epochs_ = int(weight_hold_epochs)
    final_model.dw_weight_ramp_epochs_ = int(weight_ramp_epochs)
    final_model.dw_weight_clip_ = tuple(w_clip) if w_clip is not None else None
    final_model.dw_calibration_ = str(calibrate)
    final_model.dw_calib_holdout_ = float(calib_holdout)
    final_model.dw_imputer_ = final_imputer
    final_model.dw_scaler_ = final_scaler
    final_model.dw_impute_missing_ = bool(impute_missing)
    final_model.dw_standardize_features_ = bool(standardize_features)
    final_model.dw_requested_k_folds_ = int(requested_k)
    final_model.dw_actual_k_folds_ = int(k)
    final_model.dw_epochs_completed_ = int(epochs_completed)
    final_model.dw_max_epochs_requested_ = int(max_epochs_requested)
    final_model.dw_early_stopped_ = bool(early_stopped)
    final_model.dw_stop_reason_ = str(stop_reason)

    return DynamicFitResult(
        model=final_model,
        final_threshold=float(final_threshold),
        final_temperature=float(T_final),
        best_epoch=int(best_epoch),
        best_epoch_metric=metric_name,
        best_cv_score=float(best_score),
        best_cv_log_loss=float(best_cv_log_loss),
        best_cv_balanced_accuracy=float(best_cv_balanced_accuracy),
        history=history,
        cv_fold_history=cv_fold_history,
        final_refit_history=final_refit_history,
        epochs_completed=int(epochs_completed),
        max_epochs_requested=int(max_epochs_requested),
        early_stopped=bool(early_stopped),
        stop_reason=str(stop_reason),
        final_thresholds=dict(final_taus or {}),
        variant=variant,
        epoch_sample_ids=list(sample_ids) if capture_epoch_probabilities else None,
        epoch_labels=np.asarray(y, dtype=int).copy() if capture_epoch_probabilities else None,
        epoch_raw_probabilities=(np.vstack(epoch_raw_rows) if capture_epoch_probabilities and epoch_raw_rows else None),
        epoch_calibrated_probabilities=(np.vstack(epoch_cal_rows) if capture_epoch_probabilities and epoch_cal_rows else None),
        epoch_temperatures=(list(epoch_temperatures) if capture_epoch_probabilities else None),
    )

def fit_baseline_lr(
    X, y, C: float = BASELINE_LR_C, random_state: int = 42,
    impute_missing: bool = True, standardize_features: bool = STANDARDIZE_FEATURES,
):
    """Fit the LBFGS + L2 baseline with cutoff 0.5 and no calibration.

    Median imputation and optional feature standardization are fitted on the
    supplied training data and stored on the model for identical transformation
    at inference time.
    """
    X, y = _as_binary_xy(X, y)
    X_fit, imputer, scaler = _fit_feature_transformer(
        X, impute_missing=impute_missing, standardize_features=standardize_features
    )
    model = LogisticRegression(
        solver="lbfgs",
        C=float(C),
        class_weight=None,
        max_iter=2000,
        random_state=int(random_state),
    )
    model.fit(X_fit, y)
    model.dw_variant_ = "LR"
    model.dw_temperature_ = 1.0
    model.dw_threshold_ = 0.5
    model.dw_imputer_ = imputer
    model.dw_scaler_ = scaler
    model.dw_impute_missing_ = bool(impute_missing)
    model.dw_standardize_features_ = bool(standardize_features)
    return model


def predict_probability(model, X, pos_label: int = 1):
    Xv = X.values if hasattr(X, "values") else np.asarray(X)
    if hasattr(model, "dw_imputer_") or hasattr(model, "dw_scaler_"):
        Xv = _apply_feature_transformer(
            Xv, getattr(model, "dw_imputer_", None), getattr(model, "dw_scaler_", None)
        )
    p = proba_of_pos(model, Xv, pos_label=pos_label)
    T = float(getattr(model, "dw_temperature_", 1.0))
    return apply_temperature(p, T) if not np.isclose(T, 1.0) else clip_prob(p)


def predict_label(model, X, pos_label: int = 1):
    p = predict_probability(model, X, pos_label=pos_label)
    threshold = float(getattr(model, "dw_threshold_", 0.5))
    return (p >= threshold).astype(int)


# ==========================================================================
# Compact evaluation and audit utilities
# ==========================================================================

def specificity_score(y_true, y_pred) -> float:
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(np.asarray(y_true).ravel(), np.asarray(y_pred).ravel(), labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return float(tn / (tn + fp)) if (tn + fp) else np.nan


def evaluate_binary_predictions(y_true, probability, threshold, reference_threshold=0.5):
    """Comprehensive binary classification and probability evaluation."""
    import warnings
    from sklearn.metrics import (
        accuracy_score, auc, average_precision_score, balanced_accuracy_score,
        brier_score_loss, cohen_kappa_score, confusion_matrix, f1_score,
        fbeta_score, matthews_corrcoef, precision_recall_curve, precision_score,
        recall_score, roc_auc_score, log_loss as _log_loss,
    )

    def safe_div(a, b):
        return float(a / b) if b else np.nan

    y = np.asarray(y_true).ravel().astype(int)
    p = clip_prob(np.asarray(probability).ravel().astype(float))
    if len(y) != len(p):
        raise ValueError("y_true and probability have different lengths.")
    pred = (p >= float(threshold)).astype(int)
    ref_pred = (p >= float(reference_threshold)).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    n = int(len(y))
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    prevalence = safe_div(n_pos, n)
    predicted_positive_rate = safe_div(int(np.sum(pred == 1)), n)
    sensitivity = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    ppv = safe_div(tp, tp + fp)
    npv = safe_div(tn, tn + fn)
    fpr = safe_div(fp, fp + tn)
    fnr = safe_div(fn, fn + tp)
    fdr = safe_div(fp, fp + tp)
    false_omission = safe_div(fn, fn + tn)

    if len(np.unique(y)) == 2:
        try:
            roc_auc = float(roc_auc_score(y, p))
        except ValueError:
            roc_auc = np.nan
        try:
            avg_precision = float(average_precision_score(y, p))
            pc, rc, _ = precision_recall_curve(y, p)
            pr_auc = float(auc(rc[::-1], pc[::-1]))
        except ValueError:
            avg_precision = pr_auc = np.nan
    else:
        roc_auc = avg_precision = pr_auc = np.nan

    brier = float(brier_score_loss(y, p))
    try:
        nll = float(_log_loss(y, np.column_stack([1.0 - p, p]), labels=[0, 1]))
    except ValueError:
        nll = np.nan

    precision_pos = float(precision_score(y, pred, average="binary", zero_division=0))
    precision_weighted = float(precision_score(y, pred, average="weighted", zero_division=0))
    recall_pos = float(recall_score(y, pred, average="binary", zero_division=0))
    recall_weighted = float(recall_score(y, pred, average="weighted", zero_division=0))
    f1_pos = float(f1_score(y, pred, average="binary", zero_division=0))
    f1_weighted = float(f1_score(y, pred, average="weighted", zero_division=0))
    f2_pos = float(fbeta_score(y, pred, beta=2.0, average="binary", zero_division=0))
    ba = float(balanced_accuracy_score(y, pred))
    ba05 = float(balanced_accuracy_score(y, ref_pred))
    mcc = float(matthews_corrcoef(y, pred))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        kappa = float(cohen_kappa_score(y, pred))
    youden = float(sensitivity + specificity - 1.0) if np.isfinite(sensitivity) and np.isfinite(specificity) else np.nan

    return {
        "n_samples": n, "n_positive": n_pos, "n_negative": n_neg,
        "prevalence": prevalence, "threshold": float(threshold),
        "reference_threshold": float(reference_threshold),
        "predicted_positive_rate": predicted_positive_rate,
        "cm": cm, "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "kappa": kappa, "MCC": mcc, "Youden_J": youden,
        "balanced_accuracy_at_0_5": ba05,
        "threshold_balanced_accuracy": ba05,
        "balanced_accuracy": ba, "accuracy": float(accuracy_score(y, pred)),
        "sensitivity": sensitivity, "recall": recall_pos,
        "recall_weighted": recall_weighted, "TPR": sensitivity,
        "specificity": specificity, "TNR": specificity,
        "precision_positive": precision_pos, "PPV": ppv,
        "precision_weighted": precision_weighted, "precision": precision_weighted,
        "NPV": npv, "FPR": fpr, "FNR": fnr, "FDR": fdr,
        "false_omission_rate": false_omission,
        "f1_positive": f1_pos, "f1_weighted": f1_weighted, "f1": f1_weighted,
        "F2_positive": f2_pos,
        "AUC": roc_auc, "ROC_AUC": roc_auc,
        "average_precision": avg_precision, "PR_AUC": pr_auc,
        "Brier_score": brier, "log_loss": nll,
        "predict_proba": p, "predicted_label": pred, "label": y,
    }


def compute_integrated_brier_score(
    train_time, train_event, test_time, test_event, survival_probabilities, times
):
    """Compute IPCW IBS only when genuine survival outputs are available.

    The current binary classifier produces one probability per sample, so IBS
    cannot be derived from that output alone.  This wrapper is available for a
    future time-to-event workflow with censoring information and a survival
    probability matrix over multiple time points.
    """
    try:
        from sksurv.metrics import integrated_brier_score as _ibs
        from sksurv.util import Surv
    except ImportError as exc:
        raise ImportError(
            "IBS requires the optional 'scikit-survival' package, event/censoring "
            "times, and survival-probability curves over multiple time points."
        ) from exc

    train_time = np.asarray(train_time, dtype=float).ravel()
    train_event = np.asarray(train_event, dtype=bool).ravel()
    test_time = np.asarray(test_time, dtype=float).ravel()
    test_event = np.asarray(test_event, dtype=bool).ravel()
    times = np.asarray(times, dtype=float).ravel()
    surv = np.asarray(survival_probabilities, dtype=float)
    if surv.shape != (len(test_time), len(times)):
        raise ValueError("survival_probabilities must have shape (n_test_samples, n_times).")
    y_train = Surv.from_arrays(event=train_event, time=train_time)
    y_test = Surv.from_arrays(event=test_event, time=test_time)
    return float(_ibs(y_train, y_test, surv, times))


def align_common_gene_tables(train_expression, external_expression, requested_genes=None, gene_col="gene"):
    """Align two gene-by-sample tables on shared genes in a deterministic order.

    The raw tables may contain different total gene sets and different row order.
    Before normalization/modeling this utility keeps only genes available in both
    cohorts and forces identical order in both outputs.  If ``requested_genes``
    is supplied, its de-duplicated order defines the final order; otherwise the
    training-table order of shared genes is used.

    Returns
    -------
    train_aligned, external_aligned, audit_dict
    """
    import pandas as pd

    train = train_expression.copy()
    external = external_expression.copy()
    if gene_col not in train.columns or gene_col not in external.columns:
        raise KeyError(f"Both expression tables must contain {gene_col!r}.")

    # Drop blank/NA gene identifiers before checking uniqueness.
    for df in (train, external):
        df[gene_col] = df[gene_col].astype("string").str.strip()
        df.dropna(subset=[gene_col], inplace=True)
        df.drop(df.index[df[gene_col].eq("")], inplace=True)

    if not train[gene_col].is_unique:
        dup = train.loc[train[gene_col].duplicated(keep=False), gene_col].astype(str).unique().tolist()
        raise ValueError(f"Training expression contains duplicate gene IDs: {dup[:10]}")
    if not external[gene_col].is_unique:
        dup = external.loc[external[gene_col].duplicated(keep=False), gene_col].astype(str).unique().tolist()
        raise ValueError(f"External expression contains duplicate gene IDs: {dup[:10]}")

    train_genes = train[gene_col].astype(str).tolist()
    external_genes = external[gene_col].astype(str).tolist()
    train_set, external_set = set(train_genes), set(external_genes)
    common = train_set & external_set

    if requested_genes is None:
        used = [g for g in train_genes if g in common]
        requested = None
    else:
        cleaned = []
        for g in requested_genes:
            s = str(g).strip()
            if s and s.lower() != "nan":
                cleaned.append(s)
        requested = list(dict.fromkeys(cleaned))
        used = [g for g in requested if g in common]

    if not used:
        raise ValueError("No genes remain after common-gene alignment.")

    train_aligned = train.set_index(gene_col, drop=False).loc[used].reset_index(drop=True)
    external_aligned = external.set_index(gene_col, drop=False).loc[used].reset_index(drop=True)
    if train_aligned[gene_col].astype(str).tolist() != external_aligned[gene_col].astype(str).tolist():
        raise RuntimeError("Aligned training/external gene order is not identical.")

    audit = {
        "train_gene_count": len(train_set),
        "external_gene_count": len(external_set),
        "common_gene_count": len(common),
        "train_only_gene_count": len(train_set - external_set),
        "external_only_gene_count": len(external_set - train_set),
        "requested_gene_count": None if requested is None else len(requested),
        "selected_common_gene_count": len(used),
        "used_genes": list(used),
        "missing_train": [] if requested is None else [g for g in requested if g not in train_set],
        "missing_external": [] if requested is None else [g for g in requested if g not in external_set],
        "gene_order_verified": True,
    }
    return train_aligned, external_aligned, audit


def build_epoch_probability_table(sample_ids, y_true, epoch_matrix):
    """Build the requested row-oriented probability trajectory table.

    Row 1: sample_id, ID1, ID2, ...
    Row 2: y_true,    y1,  y2,  ...
    Row 3+: epoch_1, probabilities...
    """
    import pandas as pd
    if epoch_matrix is None:
        return None
    ids = [str(x) for x in sample_ids]
    y = np.asarray(y_true).ravel().astype(int)
    mat = np.asarray(epoch_matrix, dtype=float)
    if mat.ndim != 2 or mat.shape[1] != len(ids) or len(y) != len(ids):
        raise ValueError("Epoch-probability matrix/sample IDs/labels have inconsistent dimensions.")
    rows = [["sample_id", *ids], ["y_true", *[int(x) for x in y]]]
    for i, vals in enumerate(mat, start=1):
        rows.append([f"epoch_{i}", *[float(x) for x in vals]])
    return pd.DataFrame(rows)


def _plot_epoch_probability_surface(epoch_matrix, output_path, title, dpi=180):
    """Save a 3D surface: sample position x epoch x predicted probability."""
    from pathlib import Path
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    mat = np.asarray(epoch_matrix, dtype=float)
    if mat.ndim != 2 or mat.size == 0:
        return None
    sample_pos = np.arange(1, mat.shape[1] + 1, dtype=float)
    epochs = np.arange(1, mat.shape[0] + 1, dtype=float)
    xx, yy = np.meshgrid(sample_pos, epochs)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(xx, yy, mat, linewidth=0, antialiased=True)
    ax.set_xlabel("Sample position (same order as probability table)")
    ax.set_ylabel("Final-refit epoch")
    ax.set_zlabel("Predicted probability")
    ax.set_zlim(0.0, 1.0)
    ax.set_title(title)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_epoch_probability_tables(fit_result, output_prefix, *, plot_dir=None, save_plots=False, dpi=180):
    """Save raw/calibrated epoch trajectories, temperatures, and optional 3D plots."""
    from pathlib import Path
    import pandas as pd
    if fit_result.epoch_raw_probabilities is None or fit_result.epoch_calibrated_probabilities is None:
        raise ValueError("fit_result does not contain epoch probabilities; fit with capture_epoch_probabilities=True.")
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    raw_path = prefix.parent / f"{prefix.name}__epoch_raw_probabilities.csv"
    cal_path = prefix.parent / f"{prefix.name}__epoch_calibrated_probabilities.csv"
    raw_df = build_epoch_probability_table(
        fit_result.epoch_sample_ids, fit_result.epoch_labels, fit_result.epoch_raw_probabilities
    )
    cal_df = build_epoch_probability_table(
        fit_result.epoch_sample_ids, fit_result.epoch_labels, fit_result.epoch_calibrated_probabilities
    )
    raw_df.to_csv(raw_path, index=False, header=False)
    cal_df.to_csv(cal_path, index=False, header=False)
    if fit_result.epoch_temperatures is not None:
        pd.DataFrame({
            "epoch": np.arange(1, len(fit_result.epoch_temperatures) + 1, dtype=int),
            "temperature": np.asarray(fit_result.epoch_temperatures, dtype=float),
        }).to_csv(prefix.parent / f"{prefix.name}__epoch_temperatures.csv", index=False)

    if save_plots:
        pdir = Path(plot_dir) if plot_dir is not None else prefix.parent / "plots"
        pdir.mkdir(parents=True, exist_ok=True)
        _plot_epoch_probability_surface(
            fit_result.epoch_raw_probabilities,
            pdir / f"{prefix.name}__raw_probability_3D.png",
            f"{prefix.name}: raw probabilities across final-refit epochs",
            dpi=dpi,
        )
        _plot_epoch_probability_surface(
            fit_result.epoch_calibrated_probabilities,
            pdir / f"{prefix.name}__calibrated_probability_3D.png",
            f"{prefix.name}: calibrated probabilities across final-refit epochs",
            dpi=dpi,
        )
        pd.DataFrame({
            "sample_position": np.arange(1, len(fit_result.epoch_sample_ids) + 1, dtype=int),
            "sample_id": [str(x) for x in fit_result.epoch_sample_ids],
            "y_true": np.asarray(fit_result.epoch_labels, dtype=int),
        }).to_csv(pdir / f"{prefix.name}__plot_sample_order.csv", index=False)
    return raw_path, cal_path



# ==========================================================================
# Generic file input + command-line runner
# ==========================================================================

RUNNER_DEFAULTS = {
    "baseline_lr_c": BASELINE_LR_C,
    "sample_id_col": "sample_id",
    "label_col": "label",
    "models": ["LR", "LR_SGD", "LR_sen", "LR_spe", "LR_BA", "LR_sen_spe_corridor", "LR_sen_spe_tails"],
    # Outer split of the supplied intra-dataset.  Set to 0.0 to disable.
    "internal_test_fraction": 0.20,
    "split_random_state": 42,
    "n_epochs": 50,
    "k_folds": DEFAULT_K_FOLDS,
    "target_sens": 0.70,
    "target_spec": 0.70,
    "emphasis_factor": 2.0,
    "gamma": DYNAMIC_GAMMA,
    "learning_rate": SGD_ETA0,
    "sgd_schedule": SGD_LEARNING_RATE_SCHEDULE,
    "sgd_alpha": SGD_ALPHA,
    "sgd_l1_ratio": SGD_L1_RATIO,
    "sgd_penalty": SGD_PENALTY,
    "sgd_average": SGD_AVERAGE,
    "sgd_power_t": SGD_POWER_T,
    "calibrate": "temperature",
    "calib_holdout": 0.20,
    "temp_low": 0.05,
    "temp_high": 20.0,
    "weight_hold_epochs": WEIGHT_HOLD_EPOCHS,
    "weight_ramp_epochs": WEIGHT_RAMP_EPOCHS,
    "w_clip_low": 0.67,
    "w_clip_high": 3.0,
    "min_active_samples": MIN_ACTIVE_SAMPLES,
    "best_epoch_metric": BEST_EPOCH_METRIC,
    "early_stopping_rounds": 5,
    "random_state": 42,
    "impute_missing": True,
    "standardize_features": STANDARDIZE_FEATURES,
    "allow_small_sample_cv_reduction": ALLOW_SMALL_SAMPLE_CV_REDUCTION,
    "capture_epoch_probabilities": True,
    "save_epoch_probability_plots": True,
    "save_cv_history": True,
    "save_cv_fold_history": True,
    "save_final_refit_history": True,
    "epoch_probability_plot_dpi": 180,
    "save_models": True,
    "verbose": False,
    "output_dir": "dwlr_output",
}


def _read_delimited_table(path):
    """Read CSV/TSV/TXT into a pandas DataFrame."""
    from pathlib import Path
    import pandas as pd

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    raise ValueError(f"Unsupported table format {suffix!r}; use .csv, .tsv, or .txt.")


def _validate_sample_ids(values, *, name="sample_id"):
    import pandas as pd

    ids = pd.Series(values, dtype="string").str.strip()
    if ids.isna().any() or ids.eq("").any():
        raise ValueError(f"{name} contains blank or missing values.")
    if ids.duplicated().any():
        dup = ids[ids.duplicated(keep=False)].unique().tolist()
        raise ValueError(f"{name} contains duplicate IDs: {dup[:10]}")
    return ids.astype(str).tolist()


def _coerce_binary_labels(values, *, label_col="label"):
    import pandas as pd

    y = pd.to_numeric(pd.Series(values), errors="raise").astype(int).to_numpy()
    u = np.unique(y)
    if set(u.tolist()) != {0, 1}:
        raise ValueError(f"{label_col!r} must contain both binary labels 0 and 1; got {u.tolist()}.")
    return y


def _coerce_feature_frame(df, *, excluded_cols=()):
    """Return a numeric sample-by-feature frame while preserving column names/order."""
    feature_cols = [c for c in df.columns if c not in set(excluded_cols)]
    if not feature_cols:
        raise ValueError("No feature columns remain after excluding ID/label columns.")
    if len(feature_cols) != len(set(feature_cols)):
        raise ValueError("Feature column names must be unique.")
    out = df.loc[:, feature_cols].copy()
    import pandas as pd
    for c in feature_cols:
        out[c] = pd.to_numeric(out[c], errors="raise")
    return out


def load_supervised_data(
    *,
    data_path=None,
    features_path=None,
    labels_path=None,
    sample_id_col="sample_id",
    label_col="label",
):
    """Load one labeled sample-by-feature dataset.

    Supported layouts
    -----------------
    1. Combined table: one file contains sample ID, binary label, and feature columns.
    2. Separate tables: a feature file contains sample ID + features and a label file
       contains sample ID + label.  Labels are aligned by sample ID, never by row position.
    """
    use_combined = data_path is not None
    use_separate = features_path is not None or labels_path is not None
    if use_combined and use_separate:
        raise ValueError("Use either data_path OR features_path+labels_path, not both.")
    if not use_combined and not (features_path is not None and labels_path is not None):
        raise ValueError("Provide data_path, or provide both features_path and labels_path.")

    if use_combined:
        df = _read_delimited_table(data_path)
        missing = [c for c in (sample_id_col, label_col) if c not in df.columns]
        if missing:
            raise KeyError(f"Combined data are missing required columns: {missing}")
        ids = _validate_sample_ids(df[sample_id_col], name=sample_id_col)
        y = _coerce_binary_labels(df[label_col], label_col=label_col)
        X = _coerce_feature_frame(df, excluded_cols=(sample_id_col, label_col))
        X.index = ids
        return X, y, ids

    feat = _read_delimited_table(features_path)
    lab = _read_delimited_table(labels_path)
    if sample_id_col not in feat.columns:
        raise KeyError(f"Feature table is missing {sample_id_col!r}.")
    if sample_id_col not in lab.columns or label_col not in lab.columns:
        raise KeyError(f"Label table must contain {sample_id_col!r} and {label_col!r}.")

    feat_ids = _validate_sample_ids(feat[sample_id_col], name=f"features.{sample_id_col}")
    lab_ids = _validate_sample_ids(lab[sample_id_col], name=f"labels.{sample_id_col}")
    feat_set, lab_set = set(feat_ids), set(lab_ids)
    if feat_set != lab_set:
        missing_labels = [x for x in feat_ids if x not in lab_set]
        extra_labels = [x for x in lab_ids if x not in feat_set]
        raise ValueError(
            "Feature and label sample-ID sets differ. "
            f"Missing labels for {missing_labels[:10]}; extra label IDs {extra_labels[:10]}."
        )

    X = _coerce_feature_frame(feat, excluded_cols=(sample_id_col,))
    X.index = feat_ids
    lab_indexed = lab.set_index(sample_id_col, drop=False)
    y = _coerce_binary_labels(lab_indexed.loc[feat_ids, label_col].to_numpy(), label_col=label_col)
    return X, y, feat_ids


def _align_feature_columns(reference_X, other_X, *, other_name="external"):
    """Require all reference features in another set, then reorder to reference order."""
    reference_cols = list(reference_X.columns)
    other_cols = list(other_X.columns)
    missing = [c for c in reference_cols if c not in other_cols]
    if missing:
        raise ValueError(f"{other_name} data are missing reference features: {missing[:20]}")
    return other_X.loc[:, reference_cols].copy()


def split_development_internal_test(X, y, sample_ids, fraction=0.20, random_state=42):
    """Stratify an intra-dataset into development-training and internal-test sets.

    The internal-test subset is never used for inner cross-validation, calibration,
    threshold derivation, best-epoch selection, or final refitting.
    """
    from sklearn.model_selection import train_test_split

    frac = float(fraction)
    if not (0.0 <= frac < 0.5):
        raise ValueError("internal_test_fraction must be in [0, 0.5).")
    y = np.asarray(y, dtype=int).ravel()
    ids = [str(x) for x in sample_ids]
    if len(ids) != len(y) or len(X) != len(y):
        raise ValueError("X, y, and sample_ids have inconsistent lengths.")
    if frac == 0.0:
        return X.copy(), y.copy(), list(ids), None, None, None

    idx = np.arange(len(y))
    try:
        dev_idx, internal_idx = train_test_split(
            idx,
            test_size=frac,
            stratify=y,
            random_state=int(random_state),
        )
    except ValueError as exc:
        raise ValueError(
            "Unable to create the requested stratified internal-test split. "
            "Check internal_test_fraction and class counts."
        ) from exc

    if hasattr(X, "iloc"):
        X_dev = X.iloc[dev_idx].copy()
        X_internal = X.iloc[internal_idx].copy()
    else:
        Xa = np.asarray(X)
        X_dev = Xa[dev_idx].copy()
        X_internal = Xa[internal_idx].copy()
    y_dev, y_internal = y[dev_idx].copy(), y[internal_idx].copy()
    dev_ids = [ids[i] for i in dev_idx]
    internal_ids = [ids[i] for i in internal_idx]
    return X_dev, y_dev, dev_ids, X_internal, y_internal, internal_ids


def _load_json(path):
    import json
    from pathlib import Path

    if path is None:
        return {}
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("Configuration JSON must contain one top-level object/dictionary.")
    return obj


def _normalize_legacy_config_keys(cfg):
    """Accept 1.1.0 key names while preferring the clearer 1.2.0 terminology."""
    out = dict(cfg)
    aliases = {
        "train_data": "development_data",
        "train_features": "development_features",
        "train_labels": "development_labels",
        "test_data": "external_data",
        "test_features": "external_features",
        "test_labels": "external_labels",
        "holdout_fraction": "internal_test_fraction",
    }
    for old, new in aliases.items():
        if old in out:
            if new in out and out[new] != out[old]:
                raise ValueError(f"Configuration contains conflicting keys {old!r} and {new!r}.")
            out[new] = out.pop(old)
    return out


def _jsonable(value):
    from pathlib import Path
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonable(x) for x in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def run_models_from_config(
    X_development,
    y_development,
    development_ids,
    *,
    X_internal_test=None,
    y_internal_test=None,
    internal_test_ids=None,
    X_external=None,
    y_external=None,
    external_ids=None,
    config=None,
):
    """Fit on development-training data and evaluate frozen models on two test layers.

    Parameters
    ----------
    X_development, y_development, development_ids
        Data used for inner CV, best-epoch selection, and final refit.
    X_internal_test, y_internal_test, internal_test_ids
        Optional independent holdout carved from the intra-dataset before fitting.
    X_external, y_external, external_ids
        Optional independent external dataset.  It is never used during training.

    The same frozen fitted estimator, imputer/scaler, temperature and classification
    threshold are used for both internal and external evaluation.
    """
    from pathlib import Path
    import json
    import pandas as pd

    cfg = dict(RUNNER_DEFAULTS)
    if config:
        cfg.update(config)

    if hasattr(X_development, "columns"):
        feature_names = [str(c) for c in X_development.columns]
    else:
        feature_names = [f"feature_{i}" for i in range(np.asarray(X_development).shape[1])]

    if X_internal_test is not None and hasattr(X_development, "columns") and hasattr(X_internal_test, "columns"):
        X_internal_test = _align_feature_columns(X_development, X_internal_test, other_name="internal-test")
    if X_external is not None and hasattr(X_development, "columns") and hasattr(X_external, "columns"):
        X_external = _align_feature_columns(X_development, X_external, other_name="external")

    X_dev_arr = X_development.values if hasattr(X_development, "values") else np.asarray(X_development, dtype=float)
    y_dev = np.asarray(y_development, dtype=int).ravel()
    dev_ids = [str(x) for x in development_ids]
    if len(dev_ids) != len(y_dev) or len(X_dev_arr) != len(y_dev):
        raise ValueError("Development X/y/sample_ids have inconsistent lengths.")

    def prep_optional(X, y, ids, label):
        if X is None:
            if y is not None or ids is not None:
                raise ValueError(f"{label}: X is None but labels or IDs were supplied.")
            return None, None, None
        if y is None or ids is None:
            raise ValueError(f"{label}: X requires y and sample_ids.")
        Xa = X.values if hasattr(X, "values") else np.asarray(X, dtype=float)
        ya = np.asarray(y, dtype=int).ravel()
        ida = [str(x) for x in ids]
        if len(ida) != len(ya) or len(Xa) != len(ya):
            raise ValueError(f"{label} X/y/sample_ids have inconsistent lengths.")
        return Xa, ya, ida

    X_internal_arr, y_internal, internal_ids = prep_optional(
        X_internal_test, y_internal_test, internal_test_ids, "Internal test"
    )
    X_external_arr, y_ext, ext_ids = prep_optional(
        X_external, y_external, external_ids, "External test"
    )

    outdir = Path(cfg["output_dir"])
    pred_dir = outdir / "predictions"
    hist_dir = outdir / "history"
    fold_hist_dir = outdir / "cv_fold_history"
    refit_hist_dir = outdir / "final_refit_history"
    epoch_dir = outdir / "epoch_probabilities"
    epoch_plot_dir = outdir / "epoch_probability_plots"
    model_dir = outdir / "models"
    for d in (outdir, pred_dir, hist_dir, fold_hist_dir, refit_hist_dir):
        d.mkdir(parents=True, exist_ok=True)
    if bool(cfg.get("capture_epoch_probabilities", False)):
        epoch_dir.mkdir(parents=True, exist_ok=True)
    if bool(cfg.get("save_epoch_probability_plots", False)):
        epoch_plot_dir.mkdir(parents=True, exist_ok=True)
    if bool(cfg.get("save_models", True)):
        model_dir.mkdir(parents=True, exist_ok=True)

    models = list(cfg.get("models", RUNNER_DEFAULTS["models"]))
    allowed = {"LR", "LR_SGD", *DYNAMIC_VARIANTS}
    bad = [m for m in models if m not in allowed]
    if bad:
        raise ValueError(f"Unknown model(s): {bad}. Allowed: {sorted(allowed)}")

    metric_rows = []
    training_summary_rows = []
    fitted = {}

    def raw_probability(model, X):
        Xv = X.values if hasattr(X, "values") else np.asarray(X)
        if hasattr(model, "dw_imputer_") or hasattr(model, "dw_scaler_"):
            Xv = _apply_feature_transformer(
                Xv, getattr(model, "dw_imputer_", None), getattr(model, "dw_scaler_", None)
            )
        return proba_of_pos(model, Xv, pos_label=1)

    def record_predictions(model_name, dataset_name, ids, y, X, model, threshold, extra=None):
        p_raw = clip_prob(raw_probability(model, X))
        p = predict_probability(model, X)
        metrics = evaluate_binary_predictions(y, p, threshold=threshold)
        row = {
            "model": model_name,
            "dataset": dataset_name,
            **{k: v for k, v in metrics.items() if k not in {"cm", "predict_proba", "predicted_label", "label"}},
        }
        if extra:
            row.update(extra)
        metric_rows.append(row)
        pred = pd.DataFrame({
            "sample_id": [str(x) for x in ids],
            "y_true": np.asarray(y, dtype=int),
            "raw_probability": np.asarray(p_raw, dtype=float),
            "calibrated_probability": np.asarray(metrics["predict_proba"], dtype=float),
            "predicted_probability": np.asarray(metrics["predict_proba"], dtype=float),
            "predicted_label": np.asarray(metrics["predicted_label"], dtype=int),
            "threshold": float(threshold),
            "calibration_temperature": float(getattr(model, "dw_temperature_", 1.0)),
        })
        pred.to_csv(pred_dir / f"{model_name}__{dataset_name}.csv", index=False)
        return metrics


    for model_name in models:
        if bool(cfg.get("verbose", False)):
            print(f"\n=== Fitting {model_name} ===")

        if model_name == "LR":
            model = fit_baseline_lr(
                X_dev_arr,
                y_dev,
                C=float(cfg.get("baseline_lr_c", BASELINE_LR_C)),
                random_state=int(cfg["random_state"]),
                impute_missing=bool(cfg["impute_missing"]),
                standardize_features=bool(cfg["standardize_features"]),
            )
            fitted[model_name] = model
            extra = {
                "best_epoch": np.nan, "best_epoch_metric": "NA",
                "best_cv_score": np.nan, "best_cv_log_loss": np.nan,
                "best_cv_balanced_accuracy": np.nan, "epochs_completed": np.nan,
                "max_epochs_requested": np.nan, "early_stopped": False,
                "stop_reason": "not_iterative", "final_temperature": 1.0,
                "final_threshold": 0.5, "requested_k_folds": np.nan, "actual_k_folds": np.nan,
            }
            threshold = 0.5
        elif model_name == "LR_SGD":
            fit_result = fit_sgd_lr(
                X_dev_arr, y_dev,
                n_epochs=int(cfg["n_epochs"]), k_folds=int(cfg["k_folds"]),
                learning_rate=float(cfg["learning_rate"]), sgd_schedule=str(cfg["sgd_schedule"]),
                sgd_alpha=float(cfg["sgd_alpha"]), sgd_l1_ratio=float(cfg["sgd_l1_ratio"]),
                sgd_penalty=str(cfg["sgd_penalty"]), sgd_average=bool(cfg["sgd_average"]),
                sgd_power_t=float(cfg["sgd_power_t"]), best_epoch_metric=str(cfg["best_epoch_metric"]),
                early_stopping_rounds=(None if cfg.get("early_stopping_rounds") is None else int(cfg["early_stopping_rounds"])),
                random_state=int(cfg["random_state"]), verbose=bool(cfg["verbose"]),
                capture_epoch_probabilities=bool(cfg["capture_epoch_probabilities"]), sample_ids=dev_ids,
                impute_missing=bool(cfg["impute_missing"]), standardize_features=bool(cfg["standardize_features"]),
                allow_small_sample_cv_reduction=bool(cfg["allow_small_sample_cv_reduction"]),
            )
            model = fit_result.model
            fitted[model_name] = fit_result
            threshold = 0.5
            extra = {
                "best_epoch": int(fit_result.best_epoch),
                "best_epoch_metric": fit_result.best_epoch_metric,
                "best_cv_score": float(fit_result.best_cv_score),
                "best_cv_log_loss": float(fit_result.best_cv_log_loss),
                "best_cv_balanced_accuracy": float(fit_result.best_cv_balanced_accuracy),
                "epochs_completed": int(fit_result.epochs_completed),
                "max_epochs_requested": int(fit_result.max_epochs_requested),
                "early_stopped": bool(fit_result.early_stopped),
                "stop_reason": str(fit_result.stop_reason),
                "final_temperature": 1.0,
                "final_threshold": 0.5,
                "requested_k_folds": int(getattr(model, "dw_requested_k_folds_", cfg["k_folds"])),
                "actual_k_folds": int(getattr(model, "dw_actual_k_folds_", cfg["k_folds"])),
            }
            import pandas as pd
            if bool(cfg.get("save_cv_history", True)):
                h = pd.DataFrame(fit_result.history)
                if "epoch" in h.columns:
                    h["selected_best_epoch"] = int(fit_result.best_epoch)
                    h["is_best_epoch"] = h["epoch"].astype(int).eq(int(fit_result.best_epoch))
                h.to_csv(hist_dir / f"{model_name}__cv_history.csv", index=False)
            if bool(cfg.get("save_cv_fold_history", True)):
                h = pd.DataFrame(fit_result.cv_fold_history)
                if "epoch" in h.columns:
                    h["selected_best_epoch"] = int(fit_result.best_epoch)
                    h["is_best_epoch"] = h["epoch"].astype(int).eq(int(fit_result.best_epoch))
                h.to_csv(fold_hist_dir / f"{model_name}__cv_fold_history.csv", index=False)
            if bool(cfg.get("save_final_refit_history", True)):
                h = pd.DataFrame(fit_result.final_refit_history)
                h["selected_best_epoch"] = int(fit_result.best_epoch)
                if "epoch" in h.columns and len(h):
                    h["is_last_refit_epoch"] = h["epoch"].astype(int).eq(int(h["epoch"].max()))
                h.to_csv(refit_hist_dir / f"{model_name}__final_refit_history.csv", index=False)
            if bool(cfg.get("capture_epoch_probabilities", False)):
                save_epoch_probability_tables(
                    fit_result, epoch_dir / model_name, plot_dir=epoch_plot_dir,
                    save_plots=bool(cfg.get("save_epoch_probability_plots", False)),
                    dpi=int(cfg.get("epoch_probability_plot_dpi", 180)),
                )
        else:
            fit_result = fit_dynamic_lr(
                X_dev_arr,
                y_dev,
                variant=model_name,
                n_epochs=int(cfg["n_epochs"]),
                k_folds=int(cfg["k_folds"]),
                target_sens=float(cfg["target_sens"]),
                target_spec=float(cfg["target_spec"]),
                emphasis_factor=float(cfg["emphasis_factor"]),
                gamma=float(cfg["gamma"]),
                learning_rate=float(cfg["learning_rate"]),
                sgd_schedule=str(cfg["sgd_schedule"]),
                sgd_alpha=float(cfg["sgd_alpha"]),
                sgd_l1_ratio=float(cfg["sgd_l1_ratio"]),
                sgd_penalty=str(cfg["sgd_penalty"]),
                sgd_average=bool(cfg["sgd_average"]),
                sgd_power_t=float(cfg["sgd_power_t"]),
                calibrate=str(cfg["calibrate"]),
                calib_holdout=float(cfg["calib_holdout"]),
                temp_bounds=(float(cfg["temp_low"]), float(cfg["temp_high"])),
                weight_hold_epochs=int(cfg["weight_hold_epochs"]),
                weight_ramp_epochs=int(cfg["weight_ramp_epochs"]),
                w_clip=(float(cfg["w_clip_low"]), float(cfg["w_clip_high"])),
                min_active_samples=int(cfg["min_active_samples"]),
                best_epoch_metric=str(cfg["best_epoch_metric"]),
                early_stopping_rounds=(None if cfg.get("early_stopping_rounds") is None else int(cfg["early_stopping_rounds"])),
                random_state=int(cfg["random_state"]),
                verbose=bool(cfg["verbose"]),
                capture_epoch_probabilities=bool(cfg["capture_epoch_probabilities"]),
                sample_ids=dev_ids,
                impute_missing=bool(cfg["impute_missing"]),
                standardize_features=bool(cfg["standardize_features"]),
                allow_small_sample_cv_reduction=bool(cfg["allow_small_sample_cv_reduction"]),
            )
            model = fit_result.model
            fitted[model_name] = fit_result
            threshold = float(fit_result.final_threshold)
            extra = {
                "best_epoch": int(fit_result.best_epoch),
                "best_epoch_metric": fit_result.best_epoch_metric,
                "best_cv_score": float(fit_result.best_cv_score),
                "best_cv_log_loss": float(fit_result.best_cv_log_loss),
                "best_cv_balanced_accuracy": float(fit_result.best_cv_balanced_accuracy),
                "epochs_completed": int(fit_result.epochs_completed),
                "max_epochs_requested": int(fit_result.max_epochs_requested),
                "early_stopped": bool(fit_result.early_stopped),
                "stop_reason": str(fit_result.stop_reason),
                "final_temperature": float(fit_result.final_temperature),
                "final_threshold": float(fit_result.final_threshold),
                "requested_k_folds": int(getattr(model, "dw_requested_k_folds_", cfg["k_folds"])),
                "actual_k_folds": int(getattr(model, "dw_actual_k_folds_", cfg["k_folds"])),
                "tau_sen": float(fit_result.final_thresholds.get("sen", np.nan)),
                "tau_spe": float(fit_result.final_thresholds.get("spe", np.nan)),
                "tau_ba": float(fit_result.final_thresholds.get("ba", np.nan)),
            }
            import pandas as pd
            if bool(cfg.get("save_cv_history", True)):
                h = pd.DataFrame(fit_result.history)
                if "epoch" in h.columns:
                    h["selected_best_epoch"] = int(fit_result.best_epoch)
                    h["is_best_epoch"] = h["epoch"].astype(int).eq(int(fit_result.best_epoch))
                h.to_csv(hist_dir / f"{model_name}__cv_history.csv", index=False)
            if bool(cfg.get("save_cv_fold_history", True)):
                h = pd.DataFrame(fit_result.cv_fold_history)
                if "epoch" in h.columns:
                    h["selected_best_epoch"] = int(fit_result.best_epoch)
                    h["is_best_epoch"] = h["epoch"].astype(int).eq(int(fit_result.best_epoch))
                h.to_csv(fold_hist_dir / f"{model_name}__cv_fold_history.csv", index=False)
            if bool(cfg.get("save_final_refit_history", True)):
                h = pd.DataFrame(fit_result.final_refit_history)
                h["selected_best_epoch"] = int(fit_result.best_epoch)
                if "epoch" in h.columns and len(h):
                    h["is_last_refit_epoch"] = h["epoch"].astype(int).eq(int(h["epoch"].max()))
                h.to_csv(refit_hist_dir / f"{model_name}__final_refit_history.csv", index=False)
            if bool(cfg.get("capture_epoch_probabilities", False)):
                save_epoch_probability_tables(
                    fit_result, epoch_dir / model_name, plot_dir=epoch_plot_dir,
                    save_plots=bool(cfg.get("save_epoch_probability_plots", False)),
                    dpi=int(cfg.get("epoch_probability_plot_dpi", 180)),
                )

        training_summary_rows.append({"model": model_name, **extra})

        # The development-training data are shown for auditing; model selection and final
        # refitting use only this data, never the internal/external test layers.
        record_predictions(model_name, "development_training", dev_ids, y_dev, X_dev_arr, model, threshold, extra)
        if X_internal_arr is not None:
            record_predictions(model_name, "internal_test", internal_ids, y_internal, X_internal_arr, model, threshold, extra)
        if X_external_arr is not None:
            record_predictions(model_name, "external_test", ext_ids, y_ext, X_external_arr, model, threshold, extra)

        if bool(cfg.get("save_models", True)):
            import joblib
            estimator = fitted[model_name].model if isinstance(fitted[model_name], (DynamicFitResult, SGDFitResult)) else fitted[model_name]
            joblib.dump(estimator, model_dir / f"{model_name}.joblib")
            if isinstance(fitted[model_name], (DynamicFitResult, SGDFitResult)):
                joblib.dump({
                    "model": estimator, "training_summary": extra,
                    "cv_history": fitted[model_name].history,
                    "cv_fold_history": fitted[model_name].cv_fold_history,
                    "final_refit_history": fitted[model_name].final_refit_history,
                    "feature_names": feature_names,
                }, model_dir / f"{model_name}__artifact.joblib")

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(outdir / "metrics.csv", index=False)
    pd.DataFrame(training_summary_rows).to_csv(outdir / "training_summary.csv", index=False)

    manifest = {
        "n_intra_dataset_total_samples": int(len(y_dev) + (0 if y_internal is None else len(y_internal))),
        "n_development_training_samples": int(len(y_dev)),
        "development_training_class_counts": {"0": int(np.sum(y_dev == 0)), "1": int(np.sum(y_dev == 1))},
        "n_internal_test_samples": None if y_internal is None else int(len(y_internal)),
        "internal_test_class_counts": None if y_internal is None else {"0": int(np.sum(y_internal == 0)), "1": int(np.sum(y_internal == 1))},
        "n_external_test_samples": None if y_ext is None else int(len(y_ext)),
        "external_test_class_counts": None if y_ext is None else {"0": int(np.sum(y_ext == 0)), "1": int(np.sum(y_ext == 1))},
        "n_features": int(X_dev_arr.shape[1]),
        "feature_names": feature_names,
        "models": models,
        "config": {k: _jsonable(v) for k, v in cfg.items()},
    }
    with (outdir / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return metrics_df, fitted


def _build_arg_parser(defaults=None):
    import argparse

    d = dict(RUNNER_DEFAULTS)
    if defaults:
        d.update(defaults)

    p = argparse.ArgumentParser(
        prog="dynamic_weighted_logistic_regression.py",
        description=(
            "Fit baseline logistic regression and dynamic-weighted SGD variants on a user-supplied "
            "intra-dataset, optionally reserve an internal holdout, and optionally evaluate an independent external dataset."
        ),
    )
    p.add_argument("--config", default=None, help="Optional JSON configuration file. CLI arguments override JSON values.")

    in_grp = p.add_argument_group("Input data")
    in_grp.add_argument("--development-data", "--train-data", dest="development_data", default=d.get("development_data"),
                        help="Combined intra-dataset CSV/TSV with sample ID, label, and features. --train-data is a backward-compatible alias.")
    in_grp.add_argument("--development-features", "--train-features", dest="development_features", default=d.get("development_features"),
                        help="Intra-dataset feature CSV/TSV with sample ID + features.")
    in_grp.add_argument("--development-labels", "--train-labels", dest="development_labels", default=d.get("development_labels"),
                        help="Intra-dataset label CSV/TSV with sample ID + label.")
    in_grp.add_argument("--external-data", "--test-data", dest="external_data", default=d.get("external_data"),
                        help="Optional independent external combined CSV/TSV. --test-data is a backward-compatible alias.")
    in_grp.add_argument("--external-features", "--test-features", dest="external_features", default=d.get("external_features"),
                        help="Optional independent external feature CSV/TSV.")
    in_grp.add_argument("--external-labels", "--test-labels", dest="external_labels", default=d.get("external_labels"),
                        help="Optional independent external label CSV/TSV.")
    in_grp.add_argument("--sample-id-col", default=d["sample_id_col"], help="Sample ID column name.")
    in_grp.add_argument("--label-col", default=d["label_col"], help="Binary outcome column name; values must be 0/1.")
    in_grp.add_argument("--internal-test-fraction", type=float, default=float(d["internal_test_fraction"]),
                        help="Fraction of the supplied intra-dataset reserved as an independent stratified internal test. 0 disables it; default 0.20.")
    in_grp.add_argument("--split-random-state", type=int, default=int(d["split_random_state"]),
                        help="Random seed for the outer intra-dataset development/internal-test split.")

    m_grp = p.add_argument_group("Models and training")
    m_grp.add_argument("--models", nargs="+", default=d["models"], choices=["LR", "LR_SGD", *sorted(DYNAMIC_VARIANTS)], help="Models to fit.")
    m_grp.add_argument("--n-epochs", type=int, default=int(d["n_epochs"]))
    m_grp.add_argument("--k-folds", type=int, default=int(d["k_folds"]))
    m_grp.add_argument("--target-sens", type=float, default=float(d["target_sens"]))
    m_grp.add_argument("--target-spec", type=float, default=float(d["target_spec"]))
    m_grp.add_argument("--emphasis-factor", type=float, default=float(d["emphasis_factor"]))
    m_grp.add_argument("--gamma", type=float, default=float(d["gamma"]))
    m_grp.add_argument("--learning-rate", type=float, default=float(d["learning_rate"]))
    m_grp.add_argument("--sgd-schedule", choices=["constant", "optimal", "invscaling", "adaptive"], default=d["sgd_schedule"])
    m_grp.add_argument("--sgd-alpha", type=float, default=float(d["sgd_alpha"]))
    m_grp.add_argument("--sgd-l1-ratio", type=float, default=float(d["sgd_l1_ratio"]))
    m_grp.add_argument("--sgd-penalty", choices=["l2", "l1", "elasticnet"], default=d["sgd_penalty"])
    m_grp.add_argument("--sgd-power-t", type=float, default=float(d["sgd_power_t"]))
    m_grp.add_argument("--calibrate", choices=["temperature", "none"], default=d["calibrate"])
    m_grp.add_argument("--calib-holdout", type=float, default=float(d["calib_holdout"]))
    m_grp.add_argument("--temp-low", type=float, default=float(d["temp_low"]))
    m_grp.add_argument("--temp-high", type=float, default=float(d["temp_high"]))
    m_grp.add_argument("--weight-hold-epochs", type=int, default=int(d["weight_hold_epochs"]))
    m_grp.add_argument("--weight-ramp-epochs", type=int, default=int(d["weight_ramp_epochs"]))
    m_grp.add_argument("--w-clip-low", type=float, default=float(d["w_clip_low"]))
    m_grp.add_argument("--w-clip-high", type=float, default=float(d["w_clip_high"]))
    m_grp.add_argument("--min-active-samples", type=int, default=int(d["min_active_samples"]))
    m_grp.add_argument("--best-epoch-metric", choices=["log_loss", "balanced_accuracy"], default=d["best_epoch_metric"])
    m_grp.add_argument("--early-stopping-rounds", type=int, default=d["early_stopping_rounds"])
    m_grp.add_argument("--baseline-lr-c", type=float, default=float(d.get("baseline_lr_c", BASELINE_LR_C)))
    m_grp.add_argument("--random-state", type=int, default=int(d["random_state"]), help="Seed used inside model CV/training.")

    bool_action = argparse.BooleanOptionalAction
    m_grp.add_argument("--sgd-average", action=bool_action, default=bool(d["sgd_average"]))
    m_grp.add_argument("--impute-missing", action=bool_action, default=bool(d["impute_missing"]))
    m_grp.add_argument("--standardize-features", action=bool_action, default=bool(d["standardize_features"]))
    m_grp.add_argument("--allow-small-sample-cv-reduction", action=bool_action, default=bool(d["allow_small_sample_cv_reduction"]))

    out_grp = p.add_argument_group("Outputs")
    out_grp.add_argument("--output-dir", default=d["output_dir"])
    out_grp.add_argument("--capture-epoch-probabilities", action=bool_action, default=bool(d["capture_epoch_probabilities"]))
    out_grp.add_argument("--save-epoch-probability-plots", action=bool_action, default=bool(d["save_epoch_probability_plots"]))
    out_grp.add_argument("--save-cv-history", action=bool_action, default=bool(d["save_cv_history"]))
    out_grp.add_argument("--save-cv-fold-history", action=bool_action, default=bool(d["save_cv_fold_history"]))
    out_grp.add_argument("--save-final-refit-history", action=bool_action, default=bool(d["save_final_refit_history"]))
    out_grp.add_argument("--epoch-probability-plot-dpi", type=int, default=int(d["epoch_probability_plot_dpi"]))
    out_grp.add_argument("--save-models", action=bool_action, default=bool(d["save_models"]))
    out_grp.add_argument("--verbose", action=bool_action, default=bool(d["verbose"]))
    return p


def _validate_runner_args(args):
    if not (0.0 <= args.internal_test_fraction < 0.5):
        raise ValueError("internal_test_fraction must be in [0, 0.5).")
    if not (0 < args.target_sens <= 1 and 0 < args.target_spec <= 1):
        raise ValueError("target_sens and target_spec must be in (0, 1].")
    if not (1.0 <= args.gamma <= 2.0):
        raise ValueError("gamma must be in [1, 2].")
    if args.k_folds < 2:
        raise ValueError("k_folds must be >= 2.")
    if args.n_epochs < 1:
        raise ValueError("n_epochs must be >= 1.")
    if args.weight_hold_epochs < 0 or args.weight_ramp_epochs < 0:
        raise ValueError("weight hold/ramp epochs must be >= 0.")
    if args.w_clip_low <= 0 or args.w_clip_high < args.w_clip_low:
        raise ValueError("Require 0 < w_clip_low <= w_clip_high.")
    if args.temp_low <= 0 or args.temp_high <= args.temp_low:
        raise ValueError("Require 0 < temp_low < temp_high.")

    combined = args.development_data is not None
    separate_any = args.development_features is not None or args.development_labels is not None
    if combined and separate_any:
        raise ValueError("Development input: use --development-data OR --development-features + --development-labels.")
    if not combined and not (args.development_features and args.development_labels):
        raise ValueError("Development/intra-dataset input is required.")

    ext_combined = args.external_data is not None
    ext_separate_any = args.external_features is not None or args.external_labels is not None
    if ext_combined and ext_separate_any:
        raise ValueError("External input: use --external-data OR --external-features + --external-labels.")
    if ext_separate_any and not (args.external_features and args.external_labels):
        raise ValueError("Provide both --external-features and --external-labels.")
    # IMPORTANT: external data and an internal holdout are intentionally allowed together.


def main(argv=None):
    """Command-line entry point for an intra-dataset plus optional external dataset.

    Workflow
    --------
    1. Load the complete intra-dataset supplied by the user.
    2. Stratify it into development-training and independent internal-test subsets.
    3. Fit/CV/refit using only development-training data.
    4. Freeze preprocessing, model, temperature and final classification threshold.
    5. Evaluate that same frozen model on the internal test and, if supplied, the
       independent external dataset.
    """
    import argparse
    from pathlib import Path

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args(argv)
    json_cfg = _normalize_legacy_config_keys(_load_json(pre_args.config))

    accepted_extra = {
        "development_data", "development_features", "development_labels",
        "external_data", "external_features", "external_labels",
    }
    unknown_cfg = sorted(set(json_cfg) - (set(RUNNER_DEFAULTS) | accepted_extra))
    if unknown_cfg:
        raise ValueError(f"Unknown configuration key(s): {unknown_cfg}")

    parser = _build_arg_parser(json_cfg)
    args = parser.parse_args(argv)
    try:
        _validate_runner_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    X_all, y_all, all_ids = load_supervised_data(
        data_path=args.development_data,
        features_path=args.development_features,
        labels_path=args.development_labels,
        sample_id_col=args.sample_id_col,
        label_col=args.label_col,
    )

    X_dev, y_dev, dev_ids, X_internal, y_internal, internal_ids = split_development_internal_test(
        X_all,
        y_all,
        all_ids,
        fraction=args.internal_test_fraction,
        random_state=args.split_random_state,
    )

    explicit_external = args.external_data is not None or args.external_features is not None
    if explicit_external:
        X_external, y_external, external_ids = load_supervised_data(
            data_path=args.external_data,
            features_path=args.external_features,
            labels_path=args.external_labels,
            sample_id_col=args.sample_id_col,
            label_col=args.label_col,
        )
        # Feature meaning/order is anchored to the original intra-dataset columns.
        X_external = _align_feature_columns(X_all, X_external, other_name="external")
    else:
        X_external = y_external = external_ids = None

    cfg = vars(args).copy()
    cfg.pop("config", None)
    metrics_df, _ = run_models_from_config(
        X_dev,
        y_dev,
        dev_ids,
        X_internal_test=X_internal,
        y_internal_test=y_internal,
        internal_test_ids=internal_ids,
        X_external=X_external,
        y_external=y_external,
        external_ids=external_ids,
        config=cfg,
    )

    print("\nRun completed.")
    print(f"Output directory: {Path(args.output_dir).resolve()}")
    cols = [c for c in ["model", "dataset", "n_samples", "balanced_accuracy", "sensitivity", "specificity", "PPV", "NPV", "MCC", "AUC", "PR_AUC", "Brier_score", "log_loss"] if c in metrics_df.columns]
    print(metrics_df[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
