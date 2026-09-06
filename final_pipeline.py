"""
Credit Card Default Prediction - Final Pipeline
================================================

This is the final production pipeline for the credit card default
prediction task.

Core strategy:
1. Build a carefully engineered set of behavioral and financial features.
2. Train five diverse tree-based models:
   - XGBoost (depth 4)
   - XGBoost (depth 3)
   - ExtraTrees
   - LightGBM
   - CatBoost
3. Use repeated stratified 5-fold cross-validation to generate robust OOF
   predictions.
4. Optimize model blending weights directly against Log Loss.
5. Train a logistic-regression stacking model on the OOF predictions.
6. Apply isotonic calibration to improve probability quality.
7. Retrain the base models on the full training dataset.
8. Generate calibrated probabilities for the test dataset.
9. Save the final predictions to submission.csv.

The pipeline is optimized primarily for Log Loss rather than classification
accuracy or threshold-based metrics.
"""

# ============================================================================
# 1. IMPORTS AND GLOBAL CONFIGURATION
# ============================================================================
#
# This section imports all libraries required for:
# - Data processing
# - Feature engineering
# - Model training
# - Cross-validation
# - Probability calibration
# - Ensemble optimization
# - Evaluation
# - Feature importance visualization
# ============================================================================

import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.optimize import minimize

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from xgboost import XGBClassifier


# LightGBM is required for the ensemble.
try:
    from lightgbm import LGBMClassifier
except ImportError:
    raise ImportError(
        "Please install LightGBM first: pip install lightgbm"
    )


# CatBoost is required for the ensemble.
try:
    from catboost import CatBoostClassifier
except ImportError:
    raise ImportError(
        "Please install CatBoost first: pip install catboost"
    )


# ============================================================================
# 2. GLOBAL SETTINGS
# ============================================================================
#
# These parameters control the input files, reproducibility, and the number
# of cross-validation folds.
# ============================================================================

TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"

RANDOM_STATE = 42

N_FOLDS = 5

# Different random seeds are used for repeated cross-validation.
# This gives each training row multiple independent OOF predictions.
CV_SEEDS = [42, 137, 2026, 7, 99]


print("=" * 70)
print("Credit Card Default Prediction - Final Pipeline")
print("=" * 70)


# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================
#
# The feature engineering stage converts the raw monthly credit-card
# information into higher-level behavioral indicators.
#
# The features are grouped into:
# - Repayment behavior
# - Bill statistics
# - Payment intensity
# - Credit utilization
# - Payment coverage
# - Historical trajectories
# - Behavioral streaks
# - Risk interactions
# - Log-transformed financial variables
#
# No target information is used inside this function.
# ============================================================================


def make_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Build behavioral and financial features from the raw credit-card data.

    Parameters
    ----------
    data : pd.DataFrame
        Raw feature dataframe.

    Returns
    -------
    pd.DataFrame
        Feature-engineered dataframe.
    """

    df = data.copy()

    # Monthly repayment-status columns.
    pay_cols = [
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
    ]

    # Monthly bill amount columns.
    bill_cols = [
        f"BILL_AMT{i}"
        for i in range(1, 7)
    ]

    # Monthly payment amount columns.
    amount_cols = [
        f"PAY_AMT{i}"
        for i in range(1, 7)
    ]

    # ------------------------------------------------------------------------
    # 3.1 Basic categorical-value normalization
    # ------------------------------------------------------------------------
    #
    # Some datasets contain uncommon category codes that can be grouped into
    # broader categories to reduce noise.
    # ------------------------------------------------------------------------

    if "EDUCATION" in df.columns:
        df["EDUCATION"] = df["EDUCATION"].replace({
            0: 4,
            5: 4,
            6: 4,
        })

    if "MARRIAGE" in df.columns:
        df["MARRIAGE"] = df["MARRIAGE"].replace({
            0: 3,
        })

    # ------------------------------------------------------------------------
    # 3.2 Repayment behavior
    # ------------------------------------------------------------------------
    #
    # These features summarize repayment status across the six observed
    # months. They capture severity, frequency, recent changes, and whether
    # delinquency appears repeatedly.
    # ------------------------------------------------------------------------

    df["PAY_max"] = df[pay_cols].max(axis=1)

    df["PAY_mean"] = df[pay_cols].mean(axis=1)

    df["PAY_bad_count"] = (
        df[pay_cols] > 0
    ).sum(axis=1)

    df["PAY_clear_count"] = (
        df[pay_cols] < 0
    ).sum(axis=1)

    df["PAY_recent_change"] = (
        df["PAY_0"] - df["PAY_2"]
    )

    df["PAY_0_severe"] = (
        df["PAY_0"] >= 2
    ).astype(int)

    df["PAY_any_severe"] = (
        df[pay_cols] >= 2
    ).any(axis=1).astype(int)

    df["PAY_trend"] = (
        df["PAY_0"] - df["PAY_max"]
    )

    # ------------------------------------------------------------------------
    # Consecutive delinquency streak
    # ------------------------------------------------------------------------
    #
    # Measures how many consecutive recent months show a positive repayment
    # status. Persistent delinquency is generally more informative than a
    # single isolated bad month.
    # ------------------------------------------------------------------------

    pay_recent_to_old = df[pay_cols].to_numpy()

    is_bad = pay_recent_to_old > 0

    streak = np.zeros(
        len(df),
        dtype=int,
    )

    still_running = np.ones(
        len(df),
        dtype=bool,
    )

    for month_idx in range(
        is_bad.shape[1]
    ):
        still_running &= (
            is_bad[:, month_idx]
        )

        streak += (
            still_running.astype(int)
        )

    df["PAY_consecutive_bad"] = streak

    # Compare recent repayment status with older repayment status.
    recent_avg = (
        df[
            [
                "PAY_0",
                "PAY_2",
            ]
        ].mean(axis=1)
    )

    older_avg = (
        df[
            [
                "PAY_3",
                "PAY_4",
            ]
        ].mean(axis=1)
    )

    df["PAY_worsening"] = (
        recent_avg >= older_avg
    ).astype(int)

    # ------------------------------------------------------------------------
    # 3.3 Bill-level statistics
    # ------------------------------------------------------------------------
    #
    # These features describe the level, spread, volatility, and recent
    # movement of the customer's outstanding bills.
    # ------------------------------------------------------------------------

    df["BILL_mean"] = (
        df[bill_cols].mean(axis=1)
    )

    df["BILL_max"] = (
        df[bill_cols].max(axis=1)
    )

    df["BILL_min"] = (
        df[bill_cols].min(axis=1)
    )

    df["BILL_median"] = (
        df[bill_cols].median(axis=1)
    )

    df["BILL_std"] = (
        df[bill_cols].std(axis=1)
    )

    df["BILL_change_1_to_6"] = (
        df["BILL_AMT1"]
        -
        df["BILL_AMT6"]
    )

    df["BILL_last_minus_mean"] = (
        df["BILL_AMT1"]
        -
        df[bill_cols].mean(axis=1)
    )

    df["BILL_to_limit"] = (
        df[bill_cols].sum(axis=1)
        /
        (
            6
            *
            df["LIMIT_BAL"].clip(
                lower=1
            )
        )
    )

    df["BILL_volatile"] = (
        df["BILL_std"]
        /
        (
            df["BILL_mean"].abs()
            +
            1.0
        )
    )

    # Compare the most recent three months against the older three months.
    df["RECENT3_vs_OLD3_bill"] = (
        df[
            [
                "BILL_AMT1",
                "BILL_AMT2",
                "BILL_AMT3",
            ]
        ].mean(axis=1)
        -
        df[
            [
                "BILL_AMT4",
                "BILL_AMT5",
                "BILL_AMT6",
            ]
        ].mean(axis=1)
    )

    df["RECENT3_vs_OLD3_pay"] = (
        df[
            [
                "PAY_AMT1",
                "PAY_AMT2",
                "PAY_AMT3",
            ]
        ].mean(axis=1)
        -
        df[
            [
                "PAY_AMT4",
                "PAY_AMT5",
                "PAY_AMT6",
            ]
        ].mean(axis=1)
    )

    # ------------------------------------------------------------------------
    # 3.4 Payment intensity
    # ------------------------------------------------------------------------
    #
    # These features measure how much the customer pays relative to their
    # outstanding bills.
    # ------------------------------------------------------------------------

    df["PAYMENT_sum"] = (
        df[amount_cols].sum(axis=1)
    )

    df["PAYMENT_mean"] = (
        df[amount_cols].mean(axis=1)
    )

    df["PAYMENT_min"] = (
        df[amount_cols].min(axis=1)
    )

    df["PAYMENT_median"] = (
        df[amount_cols].median(axis=1)
    )

    df["PAY_zero_count"] = (
        df[amount_cols] == 0
    ).sum(axis=1)

    positive_bill_sum = (
        df[bill_cols]
        .clip(lower=0)
        .sum(axis=1)
    )

    df["PAYMENT_to_bill"] = np.where(
        positive_bill_sum > 0,
        df[amount_cols].sum(axis=1)
        /
        positive_bill_sum,
        np.nan,
    )

    df["RECENT_PAYMENT_to_bill"] = np.where(
        df["BILL_AMT1"] > 0,
        df["PAY_AMT1"]
        /
        df["BILL_AMT1"],
        np.nan,
    )

    # Approximate recent spending pressure using the relationship between
    # consecutive bills and the most recent payment.
    df["IMPLIED_SPEND_recent"] = (
        df["BILL_AMT1"]
        -
        df["BILL_AMT2"]
        +
        df["PAY_AMT1"]
    )

    # ------------------------------------------------------------------------
    # 3.5 Credit utilization and payment ratios
    # ------------------------------------------------------------------------
    #
    # Utilization measures how much of the available credit is being used.
    # Payment ratios measure the fraction of a bill that was paid.
    # ------------------------------------------------------------------------

    utilization_cols = []
    payment_ratio_cols = []

    for i in range(1, 7):

        util_col = f"UTIL_{i}"
        ratio_col = f"PAY_RATIO_{i}"

        bill_i = df[
            f"BILL_AMT{i}"
        ]

        pay_i = df[
            f"PAY_AMT{i}"
        ]

        limit_safe = (
            df["LIMIT_BAL"].clip(
                lower=1
            )
        )

        df[util_col] = (
            bill_i
            /
            limit_safe
        )

        df[ratio_col] = np.where(
            bill_i > 0,
            pay_i / bill_i,
            np.nan,
        )

        utilization_cols.append(
            util_col
        )

        payment_ratio_cols.append(
            ratio_col
        )

    df["UTIL_max"] = (
        df[utilization_cols].max(axis=1)
    )

    df["UTIL_std"] = (
        df[utilization_cols].std(axis=1)
    )

    df["PAY_RATIO_mean"] = (
        df[payment_ratio_cols].mean(axis=1)
    )

    df["PAY_RATIO_min"] = (
        df[payment_ratio_cols].min(axis=1)
    )

    # Count months where payments exceed positive bill amounts.
    positive_bill_matrix = (
        df[bill_cols]
        .clip(lower=0)
        .to_numpy()
    )

    df["PAID_OVER_BILL_COUNT"] = (
        df[amount_cols].to_numpy()
        >
        positive_bill_matrix
    ).sum(axis=1)

    df["BILL_POS_COUNT"] = (
        df[bill_cols] > 0
    ).sum(axis=1)

    # ------------------------------------------------------------------------
    # 3.6 Bill trajectory
    # ------------------------------------------------------------------------
    #
    # Captures whether outstanding balances have been increasing consistently
    # across consecutive months.
    # ------------------------------------------------------------------------

    bill_matrix = (
        df[bill_cols].to_numpy()
    )

    is_increasing = (
        bill_matrix[:, :-1]
        >
        bill_matrix[:, 1:]
    )

    bill_streak = np.zeros(
        len(df),
        dtype=int,
    )

    still_increasing = np.ones(
        len(df),
        dtype=bool,
    )

    for month_idx in range(
        is_increasing.shape[1]
    ):

        still_increasing &= (
            is_increasing[:, month_idx]
        )

        bill_streak += (
            still_increasing.astype(int)
        )

    df["BILL_increasing_streak"] = (
        bill_streak
    )

    # ------------------------------------------------------------------------
    # 3.7 Low-payment behavior
    # ------------------------------------------------------------------------
    #
    # A consistently low payment-to-bill ratio can indicate increasing
    # repayment pressure.
    # ------------------------------------------------------------------------

    ratio_matrix = (
        df[payment_ratio_cols].to_numpy()
    )

    df["PAYMENT_coverage_low_count"] = (
        ratio_matrix < 0.30
    ).sum(axis=1)

    is_low_ratio = (
        np.nan_to_num(
            ratio_matrix < 0.30,
            nan=False,
        ).astype(bool)
    )

    low_streak = np.zeros(
        len(df),
        dtype=int,
    )

    still_low = np.ones(
        len(df),
        dtype=bool,
    )

    for month_idx in range(
        is_low_ratio.shape[1]
    ):

        still_low &= (
            is_low_ratio[:, month_idx]
        )

        low_streak += (
            still_low.astype(int)
        )

    df["PAY_RATIO_low_streak"] = (
        low_streak
    )

    # ------------------------------------------------------------------------
    # 3.8 Payment trajectory
    # ------------------------------------------------------------------------
    #
    # Measures whether payment amounts have been consistently decreasing.
    # ------------------------------------------------------------------------

    payment_matrix = (
        df[amount_cols].to_numpy()
    )

    is_decreasing = (
        payment_matrix[:, :-1]
        <
        payment_matrix[:, 1:]
    )

    payment_streak = np.zeros(
        len(df),
        dtype=int,
    )

    still_decreasing = np.ones(
        len(df),
        dtype=bool,
    )

    for month_idx in range(
        is_decreasing.shape[1]
    ):

        still_decreasing &= (
            is_decreasing[:, month_idx]
        )

        payment_streak += (
            still_decreasing.astype(int)
        )

    df["PAYMENT_decreasing_streak"] = (
        payment_streak
    )

    # ------------------------------------------------------------------------
    # 3.9 Zero-payment behavior
    # ------------------------------------------------------------------------
    #
    # Captures persistent periods in which the customer made no payment.
    # ------------------------------------------------------------------------

    is_zero_pay = (
        payment_matrix == 0
    )

    zero_streak = np.zeros(
        len(df),
        dtype=int,
    )

    still_zero = np.ones(
        len(df),
        dtype=bool,
    )

    for month_idx in range(
        is_zero_pay.shape[1]
    ):

        still_zero &= (
            is_zero_pay[:, month_idx]
        )

        zero_streak += (
            still_zero.astype(int)
        )

    df["ZERO_PAY_streak"] = (
        zero_streak
    )

    # ------------------------------------------------------------------------
    # 3.10 High-utilization behavior
    # ------------------------------------------------------------------------
    #
    # Persistent high credit utilization is an important risk signal.
    # Two thresholds are used:
    # - 70%: high utilization
    # - 90%: extreme utilization
    # ------------------------------------------------------------------------

    util_matrix = (
        df[utilization_cols].to_numpy()
    )

    is_high_util = (
        util_matrix > 0.70
    )

    util_streak = np.zeros(
        len(df),
        dtype=int,
    )

    still_high = np.ones(
        len(df),
        dtype=bool,
    )

    for month_idx in range(
        is_high_util.shape[1]
    ):

        still_high &= (
            is_high_util[:, month_idx]
        )

        util_streak += (
            still_high.astype(int)
        )

    df["UTIL_high_streak"] = (
        util_streak
    )

    is_extreme = (
        util_matrix > 0.90
    )

    extreme_streak = np.zeros(
        len(df),
        dtype=int,
    )

    still_extreme = np.ones(
        len(df),
        dtype=bool,
    )

    for month_idx in range(
        is_extreme.shape[1]
    ):

        still_extreme &= (
            is_extreme[:, month_idx]
        )

        extreme_streak += (
            still_extreme.astype(int)
        )

    df["UTIL_extreme_streak"] = (
        extreme_streak
    )

    # ------------------------------------------------------------------------
    # 3.11 Risk interactions
    # ------------------------------------------------------------------------
    #
    # These features combine multiple risk signals. Tree-based models can
    # learn interactions automatically, but explicit high-value interactions
    # can make important relationships easier to learn.
    # ------------------------------------------------------------------------

    df["PAY_severe_recent3"] = (
        (
            df[
                [
                    "PAY_0",
                    "PAY_2",
                    "PAY_3",
                ]
            ] >= 2
        ).sum(axis=1)
    )

    df["HIGH_UTIL_LOW_COVER"] = (
        (
            df["UTIL_max"] > 0.70
        ).astype(int)
        *
        df["PAYMENT_coverage_low_count"]
    )

    df["BALANCE_REBOUND"] = np.where(
        df["PAY_AMT1"] > 0,

        (
            df["BILL_AMT1"]
            -
            (
                df["BILL_AMT2"]
                -
                df["PAY_AMT1"]
            )
        )
        /
        df["PAY_AMT1"].clip(
            lower=1
        ),

        0.0,
    )

    df["HIGH_UTIL_ZERO_PAY"] = (
        (
            df["UTIL_max"] > 0.70
        ).astype(int)
        *
        (
            df["PAY_AMT1"] == 0
        ).astype(int)
    )

    df["AGE_x_LIMIT"] = (
        df["AGE"]
        *
        df["LIMIT_BAL"]
    )

    df["BADCOUNT_x_UTILMAX"] = (
        df["PAY_bad_count"]
        *
        df["UTIL_max"]
    )

    df["PAYMAX_x_UTILMAX"] = (
        df["PAY_max"]
        *
        df["UTIL_max"]
    )

    df["SEVERE_x_ZEROSTREAK"] = (
        df["PAY_0_severe"]
        *
        df["ZERO_PAY_streak"]
    )

    # ------------------------------------------------------------------------
    # 3.12 Log transformations
    # ------------------------------------------------------------------------
    #
    # Financial variables can have highly skewed distributions.
    # Signed log transformations compress extreme values while preserving
    # whether the original value was positive or negative.
    # ------------------------------------------------------------------------

    for column in (
        ["LIMIT_BAL"]
        +
        bill_cols
        +
        amount_cols
    ):

        df[f"LOGABS_{column}"] = (
            np.sign(df[column])
            *
            np.log1p(
                df[column].abs()
            )
        )

    # ------------------------------------------------------------------------
    # 3.13 Final numerical cleanup
    # ------------------------------------------------------------------------
    #
    # Replace infinite values and remaining missing values so all downstream
    # models receive a clean numerical matrix.
    # ------------------------------------------------------------------------

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    df = df.fillna(0)

    return df


# ============================================================================
# 4. MODEL FACTORIES
# ============================================================================
#
# Each factory function returns one model configuration.
#
# The ensemble deliberately contains models with different inductive biases:
# - XGBoost depth 4: moderately expressive gradient boosting
# - XGBoost depth 3: more regularized gradient boosting
# - ExtraTrees: highly randomized tree ensemble
# - LightGBM: efficient histogram-based boosting
# - CatBoost: another gradient boosting implementation with different
#   optimization behavior
#
# Diversity between models is important because the final objective is
# probability quality after blending, not simply the score of one model.
# ============================================================================


def xgb_depth4(
    n_estimators=700,
    early_stopping_rounds=None,
) -> XGBClassifier:
    """
    Moderately deep XGBoost model.

    Depth 4 provides enough flexibility to capture nonlinear interactions
    while remaining more regularized than very deep trees.
    """

    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=4,
        learning_rate=0.019,
        min_child_weight=14,
        subsample=0.85,
        colsample_bytree=0.90,
        reg_lambda=28,
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        early_stopping_rounds=early_stopping_rounds,
    )


def xgb_depth3(
    n_estimators=1200,
    early_stopping_rounds=None,
) -> XGBClassifier:
    """
    More strongly regularized XGBoost model.

    Shallower trees reduce model complexity and provide prediction diversity
    relative to the depth-4 XGBoost model.
    """

    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=3,
        learning_rate=0.015,
        min_child_weight=12,
        subsample=0.80,
        colsample_bytree=0.90,
        reg_lambda=18,
        eval_metric="logloss",
        tree_method="hist",
        random_state=52,
        n_jobs=-1,
        early_stopping_rounds=early_stopping_rounds,
    )


def extra_trees() -> ExtraTreesClassifier:
    """
    Extremely randomized tree ensemble.

    ExtraTrees introduces substantial model diversity and can complement
    gradient boosting models in a probability ensemble.
    """

    return ExtraTreesClassifier(
        n_estimators=1200,
        max_features=0.75,
        min_samples_leaf=8,
        class_weight=None,
        random_state=21,
        n_jobs=-1,
    )


def lgbm_model(
    n_estimators=2000,
    early_stopping_rounds=None,
) -> LGBMClassifier:
    """
    LightGBM gradient boosting model.

    Early stopping is enabled during cross-validation so the model can
    determine an appropriate number of boosting iterations for each fold.
    """

    callbacks = None

    if early_stopping_rounds is not None:

        import lightgbm as lgb

        callbacks = [
            lgb.early_stopping(
                early_stopping_rounds,
                verbose=False,
            )
        ]

    model = LGBMClassifier(
        n_estimators=n_estimators,
        num_leaves=16,
        learning_rate=0.021,
        min_child_samples=25,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=18,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )

    # Store callbacks so the training loop can reuse them.
    model._early_stopping_callbacks = callbacks

    return model


def catboost_model(
    n_estimators=2000,
    early_stopping_rounds=None,
) -> CatBoostClassifier:
    """
    CatBoost gradient boosting model.

    A shallow tree depth is used to keep the model relatively conservative
    and complementary to the XGBoost and LightGBM models.
    """

    return CatBoostClassifier(
        iterations=n_estimators,
        depth=3,
        learning_rate=0.029,
        l2_leaf_reg=12,
        random_seed=RANDOM_STATE,
        eval_metric="Logloss",
        loss_function="Logloss",
        verbose=False,
        thread_count=-1,
        early_stopping_rounds=early_stopping_rounds,
        allow_writing_files=False,
    )


# ============================================================================
# 5. MODEL EVALUATION UTILITIES
# ============================================================================
#
# Because the competition metric is probability-based, we track:
#
# - Log Loss: primary metric
# - ROC AUC: ranking quality
# - Brier Score: probability calibration quality
#
# Log Loss is the main metric used for model selection and ensemble
# optimization.
# ============================================================================


def report_scores(
    name,
    y_true,
    probability,
):
    """
    Print the main probability-quality metrics.
    """

    probability = np.clip(
        probability,
        1e-6,
        1 - 1e-6,
    )

    print(
        f"{name:<26} "
        f"Log Loss="
        f"{log_loss(y_true, probability, labels=[0, 1]):.6f} | "
        f"AUC="
        f"{roc_auc_score(y_true, probability):.6f} | "
        f"Brier="
        f"{brier_score_loss(y_true, probability):.6f}"
    )


# ============================================================================
# 6. OOF BLEND WEIGHT OPTIMIZATION
# ============================================================================
#
# The ensemble weights are optimized directly against OOF Log Loss.
#
# Instead of assuming equal weights, the optimizer searches for a combination
# of the five models that minimizes the observed out-of-fold Log Loss.
#
# Absolute values are used before normalization so the optimizer does not
# need to satisfy an explicit positivity constraint.
# ============================================================================


def optimize_blend_weights(
    oof_probs,
    y_true,
):
    """
    Find non-negative blend weights that minimize OOF Log Loss.
    """

    def neg_score(raw_weights):

        weights = np.abs(
            raw_weights
        )

        weight_sum = weights.sum()

        if weight_sum <= 0:
            weights = np.ones(
                len(weights)
            )
            weight_sum = weights.sum()

        weights = (
            weights
            /
            weight_sum
        )

        blend = sum(
            weights[i]
            *
            oof_probs[i]
            for i in range(
                len(oof_probs)
            )
        )

        blend = np.clip(
            blend,
            1e-6,
            1 - 1e-6,
        )

        return log_loss(
            y_true,
            blend,
            labels=[0, 1],
        )

    n_models = len(
        oof_probs
    )

    # Start from an equal-weight ensemble.
    x0 = (
        np.ones(n_models)
        /
        n_models
    )

    result = minimize(
        neg_score,
        x0,
        method="Nelder-Mead",
    )

    weights = np.abs(
        result.x
    )

    if weights.sum() <= 0:
        weights = np.ones(
            n_models
        )

    weights = (
        weights
        /
        weights.sum()
    )

    return weights


# ============================================================================
# 7. LOAD DATA
# ============================================================================
#
# Read the raw training and test data.
#
# The target column "default" is used only for training.
# The client identifier is preserved separately for the final submission.
# ============================================================================


print("\n1. Loading data...")

train = pd.read_csv(
    TRAIN_PATH
)

test = pd.read_csv(
    TEST_PATH
)

print(
    f"   Training data: {train.shape}"
)

print(
    f"   Test data    : {test.shape}"
)


# ============================================================================
# 8. BUILD FEATURES
# ============================================================================
#
# Apply exactly the same feature engineering process to train and test.
# This is critical because both datasets must have identical model features.
# ============================================================================


print("\n2. Building features...")

y = train[
    "default"
].astype(int)

X = make_features(
    train.drop(
        columns=[
            "client_id",
            "default",
        ]
    )
)

X_test = make_features(
    test.drop(
        columns=[
            "client_id",
        ]
    )
)

print(
    f"   Default rate : {y.mean():.4f}"
)

print(
    f"   Feature count: {X.shape[1]}"
)


# ============================================================================
# 9. REPEATED STRATIFIED CROSS-VALIDATION
# ============================================================================
#
# We perform five independent 5-fold CV runs using different random seeds.
#
# This creates five independent OOF prediction sets for each model.
# Averaging them reduces variance in the estimated model probabilities.
#
# Each row receives exactly one OOF prediction per repeat.
# ============================================================================


print("\n3. Starting repeated 5-fold cross-validation...")

oof_depth4_repeats = []
oof_depth3_repeats = []
oof_extra_repeats = []
oof_lgbm_repeats = []
oof_cat_repeats = []

weights_by_repeat = []

best_iters_depth4 = []
best_iters_depth3 = []
best_iters_lgbm = []
best_iters_cat = []


for repeat_no, cv_seed in enumerate(
    CV_SEEDS,
    start=1,
):

    skf = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=cv_seed,
    )

    repeat_depth4 = np.zeros(
        len(X)
    )

    repeat_depth3 = np.zeros(
        len(X)
    )

    repeat_extra = np.zeros(
        len(X)
    )

    repeat_lgbm = np.zeros(
        len(X)
    )

    repeat_cat = np.zeros(
        len(X)
    )

    print(
        f"\n   Repeat "
        f"{repeat_no}/{len(CV_SEEDS)} "
        f"(seed={cv_seed})"
    )

    for fold, (
        train_idx,
        val_idx,
    ) in enumerate(
        skf.split(X, y),
        start=1,
    ):

        print(
            f"      Fold {fold}/{N_FOLDS}"
        )

        X_tr = X.iloc[
            train_idx
        ]

        X_va = X.iloc[
            val_idx
        ]

        y_tr = y.iloc[
            train_idx
        ]

        y_va = y.iloc[
            val_idx
        ]

        # --------------------------------------------------------------------
        # XGBoost depth 4
        # --------------------------------------------------------------------

        m_depth4 = xgb_depth4(
            n_estimators=3000,
            early_stopping_rounds=50,
        )

        m_depth4.fit(
            X_tr,
            y_tr,
            eval_set=[
                (
                    X_va,
                    y_va,
                )
            ],
            verbose=False,
        )

        # --------------------------------------------------------------------
        # XGBoost depth 3
        # --------------------------------------------------------------------

        m_depth3 = xgb_depth3(
            n_estimators=3000,
            early_stopping_rounds=50,
        )

        m_depth3.fit(
            X_tr,
            y_tr,
            eval_set=[
                (
                    X_va,
                    y_va,
                )
            ],
            verbose=False,
        )

        # --------------------------------------------------------------------
        # ExtraTrees
        # --------------------------------------------------------------------

        m_extra = (
            extra_trees()
            .fit(
                X_tr,
                y_tr,
            )
        )

        # --------------------------------------------------------------------
        # LightGBM
        # --------------------------------------------------------------------

        m_lgbm = lgbm_model(
            n_estimators=3000,
            early_stopping_rounds=50,
        )

        m_lgbm.fit(
            X_tr,
            y_tr,
            eval_set=[
                (
                    X_va,
                    y_va,
                )
            ],
            callbacks=(
                m_lgbm
                ._early_stopping_callbacks
            ),
        )

        # --------------------------------------------------------------------
        # CatBoost
        # --------------------------------------------------------------------

        m_cat = catboost_model(
            n_estimators=3000,
            early_stopping_rounds=50,
        )

        m_cat.fit(
            X_tr,
            y_tr,
            eval_set=(
                X_va,
                y_va,
            ),
            use_best_model=True,
        )

        # --------------------------------------------------------------------
        # Generate OOF predictions
        # --------------------------------------------------------------------
        #
        # OOF predictions are generated only from models that did not train
        # on the corresponding validation rows.
        # This prevents target leakage when the ensemble is trained later.
        # --------------------------------------------------------------------

        repeat_depth4[
            val_idx
        ] = (
            m_depth4
            .predict_proba(X_va)[:, 1]
        )

        repeat_depth3[
            val_idx
        ] = (
            m_depth3
            .predict_proba(X_va)[:, 1]
        )

        repeat_extra[
            val_idx
        ] = (
            m_extra
            .predict_proba(X_va)[:, 1]
        )

        repeat_lgbm[
            val_idx
        ] = (
            m_lgbm
            .predict_proba(X_va)[:, 1]
        )

        repeat_cat[
            val_idx
        ] = (
            m_cat
            .predict_proba(X_va)[:, 1]
        )

        # Save the best iteration count for later full-data training.
        best_iters_depth4.append(
            m_depth4.best_iteration
        )

        best_iters_depth3.append(
            m_depth3.best_iteration
        )

        best_iters_lgbm.append(
            m_lgbm.best_iteration_
        )

        best_iters_cat.append(
            m_cat.best_iteration_
        )

        print(
            f"         Best iterations: "
            f"d4={m_depth4.best_iteration}, "
            f"d3={m_depth3.best_iteration}, "
            f"lgbm={m_lgbm.best_iteration_}, "
            f"cat={m_cat.best_iteration_}"
        )

    # ------------------------------------------------------------------------
    # Optimize the ensemble weights for this CV repeat.
    # ------------------------------------------------------------------------

    repeat_weights = (
        optimize_blend_weights(
            [
                repeat_depth4,
                repeat_depth3,
                repeat_extra,
                repeat_lgbm,
                repeat_cat,
            ],
            y,
        )
    )

    repeat_blend = sum(
        repeat_weights[i]
        * prediction
        for i, prediction in enumerate(
            [
                repeat_depth4,
                repeat_depth3,
                repeat_extra,
                repeat_lgbm,
                repeat_cat,
            ]
        )
    )

    print(
        f"      Optimized blend "
        f"Log Loss="
        f"{log_loss(y, repeat_blend, labels=[0, 1]):.6f}"
    )

    # Store OOF predictions from this repeat.
    oof_depth4_repeats.append(
        repeat_depth4
    )

    oof_depth3_repeats.append(
        repeat_depth3
    )

    oof_extra_repeats.append(
        repeat_extra
    )

    oof_lgbm_repeats.append(
        repeat_lgbm
    )

    oof_cat_repeats.append(
        repeat_cat
    )

    weights_by_repeat.append(
        repeat_weights
    )


# ============================================================================
# 10. AVERAGE OOF PREDICTIONS ACROSS REPEATS
# ============================================================================
#
# Each model now has five independent OOF prediction vectors.
# Averaging them provides a more stable estimate of its generalization
# probability.
# ============================================================================


print("\n4. Averaging OOF predictions across repeats...")

oof_depth4 = np.mean(
    oof_depth4_repeats,
    axis=0,
)

oof_depth3 = np.mean(
    oof_depth3_repeats,
    axis=0,
)

oof_extra = np.mean(
    oof_extra_repeats,
    axis=0,
)

oof_lgbm = np.mean(
    oof_lgbm_repeats,
    axis=0,
)

oof_cat = np.mean(
    oof_cat_repeats,
    axis=0,
)


# Use the median of the optimized weights from all repeats.
# Median is more robust than simply averaging weights when one repeat
# produces an unusual optimization result.

weights = np.median(
    weights_by_repeat,
    axis=0,
)

weights = (
    weights
    /
    weights.sum()
)

(
    w_depth4,
    w_depth3,
    w_extra,
    w_lgbm,
    w_cat,
) = weights


# ============================================================================
# 11. EVALUATE INDIVIDUAL BASE MODELS
# ============================================================================
#
# This section shows how much predictive information each base learner
# contributes before stacking.
# ============================================================================


print("\n5. Base model performance:")

report_scores(
    "XGBoost depth=4",
    y,
    oof_depth4,
)

report_scores(
    "XGBoost depth=3",
    y,
    oof_depth3,
)

report_scores(
    "ExtraTrees",
    y,
    oof_extra,
)

report_scores(
    "LightGBM",
    y,
    oof_lgbm,
)

report_scores(
    "CatBoost",
    y,
    oof_cat,
)


# ============================================================================
# 12. OPTIMIZED CLASSIC BLEND
# ============================================================================
#
# Combine the five OOF predictions using the optimized weights.
# ============================================================================


print("\n6. Optimized blend weights:")

print(
    f"   XGBoost depth=4 : "
    f"{w_depth4:.4f}"
)

print(
    f"   XGBoost depth=3 : "
    f"{w_depth3:.4f}"
)

print(
    f"   ExtraTrees      : "
    f"{w_extra:.4f}"
)

print(
    f"   LightGBM        : "
    f"{w_lgbm:.4f}"
)

print(
    f"   CatBoost        : "
    f"{w_cat:.4f}"
)


oof_classic = (
    w_depth4 * oof_depth4
    +
    w_depth3 * oof_depth3
    +
    w_extra * oof_extra
    +
    w_lgbm * oof_lgbm
    +
    w_cat * oof_cat
)

report_scores(
    "Optimized classic blend",
    y,
    oof_classic,
)


# ============================================================================
# 13. SECOND-LEVEL STACKING
# ============================================================================
#
# The five base-model probabilities become features for a logistic-regression
# meta-model.
#
# A separate CV procedure is used for the meta-model to produce an honest
# estimate of stacking performance.
# ============================================================================


print("\n7. Training probability stacking model...")

oof_meta_X = np.column_stack(
    [
        oof_depth4,
        oof_depth3,
        oof_extra,
        oof_lgbm,
        oof_cat,
    ]
)

meta_oof = np.zeros(
    len(y)
)

meta_skf = StratifiedKFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE,
)


for (
    tr_idx,
    va_idx,
) in meta_skf.split(
    oof_meta_X,
    y,
):

    meta = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )

    meta.fit(
        oof_meta_X[tr_idx],
        y.iloc[tr_idx],
    )

    meta_oof[
        va_idx
    ] = (
        meta
        .predict_proba(
            oof_meta_X[va_idx]
        )[:, 1]
    )


# Train the final meta-model on all OOF predictions.
final_meta = LogisticRegression(
    C=1.0,
    max_iter=1000,
    random_state=RANDOM_STATE,
    solver="lbfgs",
)

final_meta.fit(
    oof_meta_X,
    y,
)


# ============================================================================
# 14. ISOTONIC PROBABILITY CALIBRATION
# ============================================================================
#
# Stacking produces a strong ranking model, but its raw probabilities may not
# be perfectly calibrated.
#
# Isotonic regression learns a monotonic mapping from the stacked OOF
# probabilities to observed default frequencies.
#
# Calibration is especially relevant because the primary evaluation metric
# is Log Loss, which directly rewards well-calibrated probabilities.
# ============================================================================


print("\n8. Applying isotonic probability calibration...")

iso = IsotonicRegression(
    out_of_bounds="clip"
)

iso.fit(
    meta_oof,
    y,
)

oof_stacked_cal = np.clip(
    iso.predict(meta_oof),
    1e-6,
    1 - 1e-6,
)


# ============================================================================
# 15. FINAL OOF RESULTS
# ============================================================================


print("\n9. Final OOF performance:")

report_scores(
    "Stacked + calibrated",
    y,
    oof_stacked_cal,
)


# ============================================================================
# 16. DETERMINE FINAL TREE COUNTS
# ============================================================================
#
# During CV, early stopping tells us how many boosting iterations were useful.
#
# For full-data training, we use the median best iteration across all folds
# and add a modest 20% safety margin because the final model sees more
# training data than any individual CV fold.
# ============================================================================


print("\n10. Preparing final full-data models...")

final_n_depth4 = (
    int(
        np.median(
            best_iters_depth4
        )
        *
        1.20
    )
    +
    1
)

final_n_depth3 = (
    int(
        np.median(
            best_iters_depth3
        )
        *
        1.20
    )
    +
    1
)

final_n_lgbm = (
    int(
        np.median(
            best_iters_lgbm
        )
        *
        1.20
    )
    +
    1
)

final_n_cat = (
    int(
        np.median(
            best_iters_cat
        )
        *
        1.20
    )
    +
    1
)

print(
    f"   Final tree counts: "
    f"d4={final_n_depth4}, "
    f"d3={final_n_depth3}, "
    f"lgbm={final_n_lgbm}, "
    f"cat={final_n_cat}"
)


# ============================================================================
# 17. TRAIN FINAL BASE MODELS ON ALL TRAINING DATA
# ============================================================================
#
# At this stage, no validation rows are held out.
#
# Each model is trained on all available labeled examples using the iteration
# counts estimated from cross-validation.
# ============================================================================


print("\n11. Training final models on the full dataset...")

final_depth4 = (
    xgb_depth4(
        n_estimators=final_n_depth4
    )
    .fit(
        X,
        y,
    )
)

final_depth3 = (
    xgb_depth3(
        n_estimators=final_n_depth3
    )
    .fit(
        X,
        y,
    )
)

final_extra = (
    extra_trees()
    .fit(
        X,
        y,
    )
)

final_lgbm = (
    lgbm_model(
        n_estimators=final_n_lgbm
    )
    .fit(
        X,
        y,
    )
)

final_cat = (
    catboost_model(
        n_estimators=final_n_cat
    )
    .fit(
        X,
        y,
    )
)


# ============================================================================
# 18. GENERATE TEST PREDICTIONS
# ============================================================================
#
# Generate one probability prediction from each final base model.
#
# These five probabilities are passed through the trained logistic stacking
# model and then through the isotonic calibration mapping.
# ============================================================================


print("\n12. Generating test predictions...")

p_d4 = (
    final_depth4
    .predict_proba(
        X_test
    )[:, 1]
)

p_d3 = (
    final_depth3
    .predict_proba(
        X_test
    )[:, 1]
)

p_extra = (
    final_extra
    .predict_proba(
        X_test
    )[:, 1]
)

p_lgbm = (
    final_lgbm
    .predict_proba(
        X_test
    )[:, 1]
)

p_cat = (
    final_cat
    .predict_proba(
        X_test
    )[:, 1]
)


# Construct the test meta-feature matrix.
test_meta_X = np.column_stack(
    [
        p_d4,
        p_d3,
        p_extra,
        p_lgbm,
        p_cat,
    ]
)


# First obtain the raw stacking probability.
final_prob_raw = (
    final_meta
    .predict_proba(
        test_meta_X
    )[:, 1]
)


# Then apply the isotonic calibration learned exclusively from OOF data.
final_prob = np.clip(
    iso.predict(
        final_prob_raw
    ),
    1e-6,
    1 - 1e-6,
)


# ============================================================================
# 19. CREATE FINAL SUBMISSION
# ============================================================================
#
# Preserve the original client IDs and attach the final calibrated default
# probability.
# ============================================================================


submission = pd.DataFrame({
    "client_id": test["client_id"],
    "default": final_prob,
})

submission.to_csv(
    "submission.csv",
    index=False,
)


print(
    "\nFinal submission saved to: "
    "submission.csv"
)

print(
    "\nFirst 10 predictions:"
)

print(
    submission.head(10)
)


# ============================================================================
# 20. FEATURE IMPORTANCE
# ============================================================================
#
# XGBoost depth-4 is used here as the representative model for feature
# importance visualization.
#
# The plot is intended for interpretation rather than model selection.
# ============================================================================


print(
    "\n13. Calculating feature importance..."
)

importance = (
    pd.Series(
        final_depth4.feature_importances_,
        index=X.columns,
    )
    .sort_values(
        ascending=False
    )
)

print(
    "\nTop 25 features:"
)

print(
    importance
    .head(25)
    .to_string()
)


# Create the feature-importance plot.
plt.figure(
    figsize=(9, 9)
)

(
    importance
    .head(25)
    .sort_values()
    .plot(
        kind="barh"
    )
)

plt.title(
    "Top 25 Features"
)

plt.xlabel(
    "XGBoost Feature Importance"
)

plt.tight_layout()

plt.savefig(
    "feature_importance.png",
    dpi=150,
)

plt.close()


print(
    "\nFeature importance plot saved to: "
    "feature_importance.png"
)


# ============================================================================
# 21. FINAL SUMMARY
# ============================================================================


print("\n" + "=" * 70)

print(
    "Final pipeline completed successfully."
)

print(
    "=" * 70
)

print(
    f"Training rows : {len(X)}"
)

print(
    f"Test rows     : {len(X_test)}"
)

print(
    f"Feature count : {X.shape[1]}"
)

print(
    f"Final OOF Log Loss: "
    f"{log_loss(y, oof_stacked_cal, labels=[0, 1]):.6f}"
)

print(
    "\nOutput files:"
)

print(
    "  - submission.csv"
)

print(
    "  - feature_importance.png"
)

print(
    "\nPipeline finished."
)