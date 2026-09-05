"""
Stream 1 - Credit Card Default Prediction v3
============================================

Place this file in the same folder as train.csv and test.csv, then run it.

Outputs:
    submission_v3.csv
    feature_importance_v3.png

Final probability blend:
    0.68 * XGBoost depth=4
  + 0.15 * XGBoost depth=2
  + 0.17 * ExtraTrees
"""

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
RANDOM_STATE = 42


def make_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add label-free repayment, bill, payment and utilization features."""
    df = data.copy()

    pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
    amount_cols = [f"PAY_AMT{i}" for i in range(1, 7)]

    # Repayment behavior.
    df["PAY_max"] = df[pay_cols].max(axis=1)
    df["PAY_mean"] = df[pay_cols].mean(axis=1)
    df["PAY_bad_count"] = (df[pay_cols] > 0).sum(axis=1)
    df["PAY_clear_count"] = (df[pay_cols] < 0).sum(axis=1)
    df["PAY_recent_change"] = df["PAY_0"] - df["PAY_2"]

    # Bill amount level, variation and credit-limit utilization.
    df["BILL_mean"] = df[bill_cols].mean(axis=1)
    df["BILL_max"] = df[bill_cols].max(axis=1)
    df["BILL_min"] = df[bill_cols].min(axis=1)
    df["BILL_median"] = df[bill_cols].median(axis=1)
    df["BILL_std"] = df[bill_cols].std(axis=1)
    df["BILL_change_1_to_6"] = df["BILL_AMT1"] - df["BILL_AMT6"]
    df["BILL_last_minus_mean"] = df["BILL_AMT1"] - df[bill_cols].mean(axis=1)
    df["BILL_to_limit"] = df[bill_cols].sum(axis=1) / (
        6 * df["LIMIT_BAL"].clip(lower=1)
    )

    # Payment amount level and repayment intensity.
    df["PAYMENT_sum"] = df[amount_cols].sum(axis=1)
    df["PAYMENT_mean"] = df[amount_cols].mean(axis=1)
    df["PAYMENT_min"] = df[amount_cols].min(axis=1)
    df["PAYMENT_median"] = df[amount_cols].median(axis=1)
    positive_bill_sum = df[bill_cols].clip(lower=0).sum(axis=1)
    df["PAYMENT_to_bill"] = df[amount_cols].sum(axis=1) / (positive_bill_sum + 1)
    df["RECENT_PAYMENT_to_bill"] = np.where(
    df["BILL_AMT1"] > 0,
    df["PAY_AMT1"] / (df["BILL_AMT1"] + 1),
    np.nan
)

    # Monthly utilization and payment ratios.
    utilization_cols = []
    payment_ratio_cols = []
    for i in range(1, 7):
        util_col = f"UTIL_{i}"
        ratio_col = f"PAY_RATIO_{i}"
        df[util_col] = df[f"BILL_AMT{i}"] / df["LIMIT_BAL"].clip(lower=1)
        df[ratio_col] = df[f"PAY_AMT{i}"] / (
            df[f"BILL_AMT{i}"].abs() + 1
        )
        utilization_cols.append(util_col)
        payment_ratio_cols.append(ratio_col)

    df["UTIL_max"] = df[utilization_cols].max(axis=1)
    df["UTIL_std"] = df[utilization_cols].std(axis=1)
    df["PAY_RATIO_mean"] = df[payment_ratio_cols].mean(axis=1)
    df["PAY_RATIO_min"] = df[payment_ratio_cols].min(axis=1)
    df["PAID_OVER_BILL_COUNT"] = (
        df[amount_cols].to_numpy() > df[bill_cols].abs().to_numpy()
    ).sum(axis=1)
    df["BILL_POS_COUNT"] = (df[bill_cols] > 0).sum(axis=1)

    # Log transforms help ExtraTrees handle the long-tailed amount variables.
    for column in ["LIMIT_BAL"] + bill_cols + amount_cols:
        df[f"LOGABS_{column}"] = np.sign(df[column]) * np.log1p(df[column].abs())

    return df.replace([np.inf, -np.inf], np.nan).fillna(0)


def xgb_depth4() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=700,
        max_depth=4,
        learning_rate=0.015,
        min_child_weight=20,
        subsample=0.85,
        colsample_bytree=0.90,
        reg_lambda=30,
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def xgb_depth2() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=1000,
        max_depth=2,
        learning_rate=0.015,
        min_child_weight=10,
        subsample=0.80,
        colsample_bytree=0.90,
        reg_lambda=20,
        eval_metric="logloss",
        tree_method="hist",
        random_state=52,
        n_jobs=-1,
    )


def extra_trees() -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=700,
        max_features=0.80,
        min_samples_leaf=15,
        class_weight=None,
        random_state=21,
        n_jobs=-1,
    )


def report_scores(name, y_true, probability):
    print(
        f"{name:<18} "
        f"Log Loss={log_loss(y_true, probability, labels=[0, 1]):.6f} | "
        f"AUC={roc_auc_score(y_true, probability):.6f} | "
        f"Brier={brier_score_loss(y_true, probability):.6f}"
    )


# 1. Read data and build identical features for train and test.
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

required_train = {"client_id", "default"}
if not required_train.issubset(train.columns) or "client_id" not in test.columns:
    raise ValueError("请确认 train.csv 含 client_id/default，test.csv 含 client_id。")

y = train["default"].astype(int)
X = make_features(train.drop(columns=["client_id", "default"]))
X_test = make_features(test.drop(columns=["client_id"]))

# 2. Holdout validation for a sanity check only.
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

val_depth4 = xgb_depth4().fit(X_train, y_train)
val_depth2 = xgb_depth2().fit(X_train, y_train)
val_extra = extra_trees().fit(X_train, y_train)

p_val_depth4 = val_depth4.predict_proba(X_val)[:, 1]
p_val_depth2 = val_depth2.predict_proba(X_val)[:, 1]
p_val_extra = val_extra.predict_proba(X_val)[:, 1]
p_val = (
    0.68 * p_val_depth4
    + 0.15 * p_val_depth2
    + 0.17 * p_val_extra
)

print("=== 验证集表现 ===")
report_scores("XGB depth=4", y_val, p_val_depth4)
report_scores("XGB depth=2", y_val, p_val_depth2)
report_scores("ExtraTrees", y_val, p_val_extra)
report_scores("Final blend", y_val, p_val)

# 3. Retrain all three models on the complete labelled dataset.
print("\n=== 使用全部训练数据重训最终模型 ===")
final_depth4 = xgb_depth4().fit(X, y)
final_depth2 = xgb_depth2().fit(X, y)
final_extra = extra_trees().fit(X, y)

# 4. The requested final probability formula.
p_xgb_depth4 = final_depth4.predict_proba(X_test)[:, 1]
p_xgb_depth2 = final_depth2.predict_proba(X_test)[:, 1]
p_extra_trees = final_extra.predict_proba(X_test)[:, 1]

final_prob = (
    0.68 * p_xgb_depth4
    + 0.15 * p_xgb_depth2
    + 0.17 * p_extra_trees
)

submission = pd.DataFrame(
    {
        "client_id": test["client_id"],
        "default": np.clip(final_prob, 1e-6, 1 - 1e-6),
    }
)
submission.to_csv("submission_v3.csv", index=False)

# 5. Feature importance from the strongest XGBoost component.
importance = pd.Series(
    final_depth4.feature_importances_, index=X.columns
).sort_values(ascending=False)

print("\n=== 特征重要性 Top 15 ===")
print(importance.head(15).to_string())

plt.figure(figsize=(9, 7))
importance.head(15).sort_values().plot(kind="barh", color="#2b6cb0")
plt.title("Top 15 Features Predicting Credit Card Default")
plt.xlabel("XGBoost feature importance")
plt.tight_layout()
plt.savefig("feature_importance_v3.png", dpi=150)

print(f"\n已生成 submission_v3.csv，共 {len(submission)} 行。")
print(submission.head())
