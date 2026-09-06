"""
Credit Card Default Prediction - Final Pipeline
================================================

A production-style tabular classification pipeline using:
1. Feature engineering for repayment, billing, payment, and utilization behavior.
2. AutoGluon TabularPredictor optimized for log loss.
3. Bagging and dynamic stacking for robust probability predictions.
4. Final probability clipping for numerical stability.
"""

import warnings
import numpy as np
import pandas as pd

from autogluon.tabular import TabularPredictor

warnings.filterwarnings("ignore")


# ============================================================================
# Configuration
# ============================================================================

TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"

RANDOM_STATE = 42

# ============================================================================
# Feature Engineering
# ============================================================================
# Builds behavioral and financial features from the original credit-card data.
# The goal is to capture repayment patterns, bill trends, payment coverage,
# credit utilization, and interactions between risk-related variables.
# ============================================================================

def make_features(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    pay_cols = [
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
    ]

    bill_cols = [
        f"BILL_AMT{i}"
        for i in range(1, 7)
    ]

    amount_cols = [
        f"PAY_AMT{i}"
        for i in range(1, 7)
    ]

    # Clean inconsistent categorical values.
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
    # Repayment behavior
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

    # Count consecutive months with a positive repayment-status value.
    pay_matrix = df[pay_cols].to_numpy()

    is_bad = pay_matrix > 0

    streak = np.zeros(len(df), dtype=int)
    still_running = np.ones(len(df), dtype=bool)

    for month_idx in range(is_bad.shape[1]):
        still_running &= is_bad[:, month_idx]
        streak += still_running.astype(int)

    df["PAY_consecutive_bad"] = streak

    recent_avg = df[
        ["PAY_0", "PAY_2"]
    ].mean(axis=1)

    older_avg = df[
        ["PAY_3", "PAY_4"]
    ].mean(axis=1)

    df["PAY_worsening"] = (
        recent_avg >= older_avg
    ).astype(int)

    # ------------------------------------------------------------------------
    # Bill amount statistics and utilization
    # ------------------------------------------------------------------------

    df["BILL_mean"] = df[bill_cols].mean(axis=1)
    df["BILL_max"] = df[bill_cols].max(axis=1)
    df["BILL_min"] = df[bill_cols].min(axis=1)
    df["BILL_median"] = df[bill_cols].median(axis=1)
    df["BILL_std"] = df[bill_cols].std(axis=1)

    df["BILL_change_1_to_6"] = (
        df["BILL_AMT1"] - df["BILL_AMT6"]
    )

    df["BILL_last_minus_mean"] = (
        df["BILL_AMT1"] - df[bill_cols].mean(axis=1)
    )

    df["BILL_to_limit"] = (
        df[bill_cols].sum(axis=1)
        / (6 * df["LIMIT_BAL"].clip(lower=1))
    )

    df["BILL_volatile"] = (
        df["BILL_std"]
        / (df["BILL_mean"].abs() + 1.0)
    )

    # Compare recent three months with the older three months.
    df["RECENT3_vs_OLD3_bill"] = (
        df[
            ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3"]
        ].mean(axis=1)
        -
        df[
            ["BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
        ].mean(axis=1)
    )

    df["RECENT3_vs_OLD3_pay"] = (
        df[
            ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3"]
        ].mean(axis=1)
        -
        df[
            ["PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]
        ].mean(axis=1)
    )

    # ------------------------------------------------------------------------
    # Payment intensity and coverage
    # ------------------------------------------------------------------------

    df["PAYMENT_sum"] = df[amount_cols].sum(axis=1)
    df["PAYMENT_mean"] = df[amount_cols].mean(axis=1)
    df["PAYMENT_min"] = df[amount_cols].min(axis=1)
    df["PAYMENT_median"] = df[amount_cols].median(axis=1)

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
        / positive_bill_sum,
        np.nan,
    )

    df["RECENT_PAYMENT_to_bill"] = np.where(
        df["BILL_AMT1"] > 0,
        df["PAY_AMT1"] / df["BILL_AMT1"],
        np.nan,
    )

    df["IMPLIED_SPEND_recent"] = (
        df["BILL_AMT1"]
        - df["BILL_AMT2"]
        + df["PAY_AMT1"]
    )

    # Monthly utilization and payment coverage ratios.
    utilization_cols = []
    payment_ratio_cols = []

    for i in range(1, 7):

        util_col = f"UTIL_{i}"
        ratio_col = f"PAY_RATIO_{i}"

        bill_i = df[f"BILL_AMT{i}"]
        pay_i = df[f"PAY_AMT{i}"]

        df[util_col] = (
            bill_i
            / df["LIMIT_BAL"].clip(lower=1)
        )

        df[ratio_col] = np.where(
            bill_i > 0,
            pay_i / bill_i,
            np.nan,
        )

        utilization_cols.append(util_col)
        payment_ratio_cols.append(ratio_col)

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

    positive_bill_matrix = (
        df[bill_cols]
        .clip(lower=0)
        .to_numpy()
    )

    df["PAID_OVER_BILL_COUNT"] = (
        df[amount_cols].to_numpy()
        > positive_bill_matrix
    ).sum(axis=1)

    df["BILL_POS_COUNT"] = (
        df[bill_cols] > 0
    ).sum(axis=1)

    # ------------------------------------------------------------------------
    # Behavioral trajectory features
    # ------------------------------------------------------------------------
    # These features capture persistent trends rather than only monthly values.
    # ------------------------------------------------------------------------

    bill_matrix = df[bill_cols].to_numpy()

    is_increasing = (
        bill_matrix[:, :-1]
        > bill_matrix[:, 1:]
    )

    bill_streak = np.zeros(len(df), dtype=int)
    still_increasing = np.ones(
        len(df),
        dtype=bool,
    )

    for month_idx in range(is_increasing.shape[1]):
        still_increasing &= (
            is_increasing[:, month_idx]
        )

        bill_streak += (
            still_increasing.astype(int)
        )

    df["BILL_increasing_streak"] = bill_streak

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
        )
        .astype(bool)
    )

    low_streak = np.zeros(len(df), dtype=int)
    still_low = np.ones(
        len(df),
        dtype=bool,
    )

    for month_idx in range(is_low_ratio.shape[1]):
        still_low &= (
            is_low_ratio[:, month_idx]
        )

        low_streak += (
            still_low.astype(int)
        )

    df["PAY_RATIO_low_streak"] = low_streak

    payment_matrix = (
        df[amount_cols].to_numpy()
    )

    is_decreasing = (
        payment_matrix[:, :-1]
        < payment_matrix[:, 1:]
    )

    payment_streak = np.zeros(
        len(df),
        dtype=int,
    )

    still_decreasing = np.ones(
        len(df),
        dtype=bool,
    )

    for month_idx in range(is_decreasing.shape[1]):
        still_decreasing &= (
            is_decreasing[:, month_idx]
        )

        payment_streak += (
            still_decreasing.astype(int)
        )

    df["PAYMENT_decreasing_streak"] = (
        payment_streak
    )

    # Consecutive zero-payment months.
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

    for month_idx in range(is_zero_pay.shape[1]):
        still_zero &= (
            is_zero_pay[:, month_idx]
        )

        zero_streak += (
            still_zero.astype(int)
        )

    df["ZERO_PAY_streak"] = zero_streak

    # Persistent high and extreme credit utilization.
    util_matrix = (
        df[utilization_cols].to_numpy()
    )

    is_high_util = (
        util_matrix > 0.7
    )

    util_streak = np.zeros(
        len(df),
        dtype=int,
    )

    still_high = np.ones(
        len(df),
        dtype=bool,
    )

    for month_idx in range(is_high_util.shape[1]):
        still_high &= (
            is_high_util[:, month_idx]
        )

        util_streak += (
            still_high.astype(int)
        )

    df["UTIL_high_streak"] = util_streak

    is_extreme = (
        util_matrix > 0.9
    )

    extreme_streak = np.zeros(
        len(df),
        dtype=int,
    )

    still_extreme = np.ones(
        len(df),
        dtype=bool,
    )

    for month_idx in range(is_extreme.shape[1]):
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
    # Risk interaction features
    # ------------------------------------------------------------------------

    df["PAY_severe_recent3"] = (
        df[
            ["PAY_0", "PAY_2", "PAY_3"]
        ] >= 2
    ).sum(axis=1)

    df["HIGH_UTIL_LOW_COVER"] = (
        (df["UTIL_max"] > 0.7).astype(int)
        * df["PAYMENT_coverage_low_count"]
    )

    df["BALANCE_REBOUND"] = np.where(
        df["PAY_AMT1"] > 0,
        (
            df["BILL_AMT1"]
            - (
                df["BILL_AMT2"]
                - df["PAY_AMT1"]
            )
        )
        / df["PAY_AMT1"].clip(lower=1),
        0.0,
    )

    df["HIGH_UTIL_ZERO_PAY"] = (
        (df["UTIL_max"] > 0.7).astype(int)
        *
        (df["PAY_AMT1"] == 0).astype(int)
    )

    df["AGE_x_LIMIT"] = (
        df["AGE"] * df["LIMIT_BAL"]
    )

    df["BADCOUNT_x_UTILMAX"] = (
        df["PAY_bad_count"]
        * df["UTIL_max"]
    )

    df["PAYMAX_x_UTILMAX"] = (
        df["PAY_max"]
        * df["UTIL_max"]
    )

    df["SEVERE_x_ZEROSTREAK"] = (
        df["PAY_0_severe"]
        * df["ZERO_PAY_streak"]
    )

    # Log-transformed financial variables help tree models capture
    # nonlinear relationships across different monetary scales.
    for column in (
        ["LIMIT_BAL"]
        + bill_cols
        + amount_cols
    ):
        df[f"LOGABS_{column}"] = (
            np.sign(df[column])
            * np.log1p(df[column].abs())
        )

    # Replace invalid numerical values before model training.
    return (
        df.replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
    )


# ============================================================================
# Main AutoGluon Training Pipeline
# ============================================================================
# AutoGluon handles model selection, bagging, stacking, and ensemble
# construction. Log loss is used because the competition evaluates
# probability quality rather than only classification accuracy.
# ============================================================================

print("=" * 80)
print("Credit Card Default Prediction - Final Pipeline")
print("=" * 80)

print("\n[1/4] Loading data...")

train_raw = pd.read_csv(TRAIN_PATH)
test_raw = pd.read_csv(TEST_PATH)

test_client_ids = test_raw["client_id"].copy()

print(f"Train shape: {train_raw.shape}")
print(f"Test shape : {test_raw.shape}")


# ============================================================================
# Feature Construction
# ============================================================================

print("\n[2/4] Building features...")

train_df = make_features(
    train_raw.drop(
        columns=["default"],
        errors="ignore",
    )
)

train_df["default"] = (
    train_raw["default"].astype(int)
)

test_df = make_features(
    test_raw.drop(
        columns=["client_id"],
        errors="ignore",
    )
)

print(f"Feature train shape: {train_df.shape}")
print(f"Feature test shape : {test_df.shape}")

print(
    f"Number of features: "
    f"{train_df.drop(columns=['default']).shape[1]}"
)

print(
    f"Default rate: "
    f"{train_df['default'].mean():.6f}"
)


# ============================================================================
# AutoGluon Training
# ============================================================================
# Bagging improves prediction stability, while dynamic stacking allows
# AutoGluon to determine an effective ensemble structure automatically.
# ============================================================================

print("\n[3/4] Training AutoGluon...")

predictor = TabularPredictor(
    label="default",
    eval_metric="log_loss",
    problem_type="binary",
    path="AutogluonModels/cc_default_final",
).fit(
    train_data=train_df.drop(
        columns=["client_id"],
        errors="ignore",
    ),

    # Increase this value if more training time is available.
    time_limit=1200,

    presets="best_quality",

    # Multiple folds provide more reliable out-of-fold predictions.
    num_bag_folds=10,

    # Automatically searches for useful stacking configurations.
    dynamic_stacking=True,
)

print("\n=== AutoGluon Leaderboard ===")

leaderboard = predictor.leaderboard(
    silent=True
)

print(leaderboard)


# ============================================================================
# Test Prediction and Submission
# ============================================================================
# Generate probability predictions for the positive class.
# Clipping prevents numerical issues when evaluating log loss.
# ============================================================================

print("\n[4/4] Generating submission...")

test_pred_probs = predictor.predict_proba(
    test_df.drop(
        columns=["client_id"],
        errors="ignore",
    )
)

if isinstance(test_pred_probs, pd.DataFrame):

    if 1 in test_pred_probs.columns:
        raw_test_prob = (
            test_pred_probs[1].values
        )
    else:
        raw_test_prob = (
            test_pred_probs.iloc[:, 1].values
        )

else:

    if test_pred_probs.ndim > 1:
        raw_test_prob = (
            test_pred_probs[:, 1]
        )
    else:
        raw_test_prob = test_pred_probs


# Keep probabilities away from exactly 0 and 1 for log-loss stability.
final_prob = np.clip(
    raw_test_prob,
    1e-6,
    1 - 1e-6,
)


submission = pd.DataFrame({
    "client_id": test_client_ids,
    "default": final_prob,
})

submission.to_csv(
    "submission.csv",
    index=False,
)

print("\nSubmission generated successfully!")
print("\nFirst five predictions:")
print(submission.head())

print("\nPrediction statistics:")
print(submission["default"].describe())

print("\n" + "=" * 80)
print("Final pipeline completed successfully.")
print("Output file: submission.csv")
print("=" * 80)