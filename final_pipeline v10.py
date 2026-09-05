"""
Stream 1 - Credit Card Default Prediction v10
==============================================

Place this file in the same folder as train.csv and test.csv, then run it.

Outputs:
    submission_v10.csv
    feature_importance_v10.png

Builds on v9. v9 confirmed LightGBM is a genuinely strong 4th model
(OOF Log Loss 0.4235, nearly tying depth4's 0.4231) but did NOT bring
real ensemble diversity (correlation with depth4 was 0.9964, actually
higher than the two XGBoost models have with each other). Net effect
on OOF was a tiny improvement (0.422874 vs v7/v8's ~0.4229), but the
public score moved the other way (0.41314 vs 0.41304) - most likely
just noise at this scale, not a real regression, but not a confirmed
win either. Ensemble-diversity avenues are now largely exhausted for
this feature set; model changes have plateaued around 0.4130 on the
public leaderboard.

New in v10: feature engineering focused on PAY_max and PAY_0, since
they are (by a wide margin) the top-2 features in every version's
importance ranking (PAY_max ~0.22, PAY_0 ~0.18, next feature ~0.07).
Added:
  - PAY_0_severe: binary flag, PAY_0 >= 2 (2+ months overdue right now)
  - PAY_any_severe: binary flag, ANY of the 6 months had PAY_x >= 2
  - PAY_trend: PAY_0 - PAY_max (how far the most recent month is from
    the worst month on record - near 0 means "currently at their worst",
    very negative means "currently much better than their worst")
  - PAY_consecutive_bad: longest streak of consecutive months (starting
    from the most recent, PAY_0, and walking backward) with PAY_x > 0 -
    captures sustained delinquency vs an isolated bad month, which a
    simple count (PAY_bad_count) cannot distinguish
  - PAY_worsening: 1 if the two most recent months (PAY_0, PAY_2) are
    both worse than or equal to the two months before that
    (mean(PAY_3, PAY_4)), else 0 - a coarse "getting worse recently"
    signal

Everything else (v4-style base features, 4-model blend with LightGBM,
calibration fix, OOF-optimized weights, 5-fold CV, early stopping) is
unchanged from v9.
"""

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

try:
    from lightgbm import LGBMClassifier
except ImportError as exc:
    raise ImportError(
        "此脚本需要 lightgbm 库，但当前环境未安装。\n"
        "请先在终端运行: pip install lightgbm\n"
        "安装完成后再重新运行本脚本。"
    ) from exc

warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
RANDOM_STATE = 42
N_FOLDS = 5


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def make_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add label-free repayment, bill, payment and utilization features."""
    df = data.copy()

    pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
    amount_cols = [f"PAY_AMT{i}" for i in range(1, 7)]

    # --- Clean up known garbage categories in EDUCATION / MARRIAGE. ---
    if "EDUCATION" in df.columns:
        df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    if "MARRIAGE" in df.columns:
        df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

    # Repayment behavior.
    df["PAY_max"] = df[pay_cols].max(axis=1)
    df["PAY_mean"] = df[pay_cols].mean(axis=1)
    df["PAY_bad_count"] = (df[pay_cols] > 0).sum(axis=1)
    df["PAY_clear_count"] = (df[pay_cols] < 0).sum(axis=1)
    df["PAY_recent_change"] = df["PAY_0"] - df["PAY_2"]

    # --- New in v10: features focused on PAY_max / PAY_0, the two
    # dominant features in importance rankings across every version. ---

    # Is the customer *currently* (most recent month) seriously overdue?
    df["PAY_0_severe"] = (df["PAY_0"] >= 2).astype(int)

    # Did the customer EVER hit a serious overdue status in the last
    # 6 months, even if they're not currently in one?
    df["PAY_any_severe"] = (df[pay_cols] >= 2).any(axis=1).astype(int)

    # How far is "right now" from "the worst month on record"?
    # ~0 means currently at their historical worst; very negative means
    # they've recovered a lot since their worst month.
    df["PAY_trend"] = df["PAY_0"] - df["PAY_max"]

    # Longest streak of consecutive overdue months, counting backward
    # from the most recent month (PAY_0). Distinguishes "overdue every
    # month for the last 3 months" from "overdue in 3 separate,
    # non-consecutive months", which PAY_bad_count cannot tell apart.
    pay_recent_to_old = df[pay_cols].to_numpy()  # columns already ordered PAY_0..PAY_6
    is_bad = pay_recent_to_old > 0
    streak = np.zeros(len(df), dtype=int)
    still_running = np.ones(len(df), dtype=bool)
    for month_idx in range(is_bad.shape[1]):
        still_running &= is_bad[:, month_idx]
        streak += still_running.astype(int)
    df["PAY_consecutive_bad"] = streak

    # Coarse "getting worse recently" signal: are the two most recent
    # months worse than (or equal to) the average of the two before that?
    recent_avg = df[["PAY_0", "PAY_2"]].mean(axis=1)
    older_avg = df[["PAY_3", "PAY_4"]].mean(axis=1)
    df["PAY_worsening"] = (recent_avg >= older_avg).astype(int)

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
    # NOTE: negative BILL_AMT means the bank owes the customer money (an
    # overpayment / refund), not a real debt. We treat those months as
    # having zero "bill to cover" rather than folding them in via abs(),
    # which would otherwise conflate the two very different situations.
    df["PAYMENT_sum"] = df[amount_cols].sum(axis=1)
    df["PAYMENT_mean"] = df[amount_cols].mean(axis=1)
    df["PAYMENT_min"] = df[amount_cols].min(axis=1)
    df["PAYMENT_median"] = df[amount_cols].median(axis=1)

    positive_bill_sum = df[bill_cols].clip(lower=0).sum(axis=1)
    df["PAYMENT_to_bill"] = np.where(
        positive_bill_sum > 0,
        df[amount_cols].sum(axis=1) / positive_bill_sum,
        np.nan,
    )
    df["RECENT_PAYMENT_to_bill"] = np.where(
        df["BILL_AMT1"] > 0,
        df["PAY_AMT1"] / df["BILL_AMT1"],
        np.nan,
    )

    # Monthly utilization and payment ratios.
    utilization_cols = []
    payment_ratio_cols = []
    for i in range(1, 7):
        util_col = f"UTIL_{i}"
        ratio_col = f"PAY_RATIO_{i}"
        bill_i = df[f"BILL_AMT{i}"]
        pay_i = df[f"PAY_AMT{i}"]

        df[util_col] = bill_i / df["LIMIT_BAL"].clip(lower=1)

        # Same fix as above: only meaningful when the customer actually
        # owed money that month.
        df[ratio_col] = np.where(bill_i > 0, pay_i / bill_i, np.nan)

        utilization_cols.append(util_col)
        payment_ratio_cols.append(ratio_col)

    df["UTIL_max"] = df[utilization_cols].max(axis=1)
    df["UTIL_std"] = df[utilization_cols].std(axis=1)
    df["PAY_RATIO_mean"] = df[payment_ratio_cols].mean(axis=1)
    df["PAY_RATIO_min"] = df[payment_ratio_cols].min(axis=1)

    # Only count "paid more than the bill" for months with a real
    # (positive) bill, consistent with the ratio features above.
    positive_bill_matrix = df[bill_cols].clip(lower=0).to_numpy()
    df["PAID_OVER_BILL_COUNT"] = (
        df[amount_cols].to_numpy() > positive_bill_matrix
    ).sum(axis=1)
    df["BILL_POS_COUNT"] = (df[bill_cols] > 0).sum(axis=1)

    # Business-motivated interaction features (kept from v4 - both showed
    # real importance / were already present there).
    df["AGE_x_LIMIT"] = df["AGE"] * df["LIMIT_BAL"]
    df["BADCOUNT_x_UTILMAX"] = df["PAY_bad_count"] * df["UTIL_max"]

    # NOTE: v5's extra features (LIMIT_per_bad_month, BILL_std_to_mean,
    # PAYMENT_to_LIMIT, AGE_bucket) are intentionally NOT included here -
    # this is the controlled-experiment feature set matching v4.

    # Log transforms help ExtraTrees/LogReg handle the long-tailed amount
    # variables.
    for column in ["LIMIT_BAL"] + bill_cols + amount_cols:
        df[f"LOGABS_{column}"] = np.sign(df[column]) * np.log1p(df[column].abs())

    return df.replace([np.inf, -np.inf], np.nan).fillna(0)


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------
def xgb_depth4(n_estimators=700, early_stopping_rounds=None) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=n_estimators,
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
        early_stopping_rounds=early_stopping_rounds,
    )


def xgb_depth2(n_estimators=1000, early_stopping_rounds=None) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=n_estimators,
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
        early_stopping_rounds=early_stopping_rounds,
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


def lgbm_model(n_estimators=2000, early_stopping_rounds=None) -> LGBMClassifier:
    """A gradient-boosted tree model with leaf-wise growth (unlike
    XGBoost's level-wise growth), aiming for real diversity in the
    ensemble rather than a linear model that turned out to be
    structurally incapable of capturing this dataset's signal."""
    callbacks = None
    if early_stopping_rounds is not None:
        import lightgbm as lgb
        callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=False)]

    model = LGBMClassifier(
        n_estimators=n_estimators,
        num_leaves=15,
        learning_rate=0.02,
        min_child_samples=30,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=20,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    model._early_stopping_callbacks = callbacks  # stashed for fit() calls below
    return model


def report_scores(name, y_true, probability):
    print(
        f"{name:<18} "
        f"Log Loss={log_loss(y_true, probability, labels=[0, 1]):.6f} | "
        f"AUC={roc_auc_score(y_true, probability):.6f} | "
        f"Brier={brier_score_loss(y_true, probability):.6f}"
    )


def optimize_blend_weights(oof_probs, y_true):
    """Find non-negative weights (summing to 1) that minimize log loss
    on the out-of-fold predictions."""

    def neg_score(raw_weights):
        w = np.abs(raw_weights)
        w = w / w.sum()
        blend = sum(w[i] * oof_probs[i] for i in range(len(oof_probs)))
        blend = np.clip(blend, 1e-6, 1 - 1e-6)
        return log_loss(y_true, blend, labels=[0, 1])

    n_models = len(oof_probs)
    x0 = np.ones(n_models) / n_models
    result = minimize(neg_score, x0, method="Nelder-Mead")
    weights = np.abs(result.x)
    weights = weights / weights.sum()
    return weights


# ---------------------------------------------------------------------------
# 1. Read data and build identical features for train and test.
# ---------------------------------------------------------------------------
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

required_train = {"client_id", "default"}
if not required_train.issubset(train.columns) or "client_id" not in test.columns:
    raise ValueError("请确认 train.csv 含 client_id/default，test.csv 含 client_id。")

y = train["default"].astype(int)
X = make_features(train.drop(columns=["client_id", "default"]))
X_test = make_features(test.drop(columns=["client_id"]))

pos_rate = y.mean()
print(f"违约比例: {pos_rate:.4f}")

# ---------------------------------------------------------------------------
# 2. 5-fold stratified CV to get honest out-of-fold (OOF) predictions and
#    a sensible number of boosting rounds via early stopping.
# ---------------------------------------------------------------------------
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

oof_depth4 = np.zeros(len(X))
oof_depth2 = np.zeros(len(X))
oof_extra = np.zeros(len(X))
oof_lgbm = np.zeros(len(X))

best_iters_depth4 = []
best_iters_depth2 = []
best_iters_lgbm = []

print("\n=== 5折交叉验证 ===")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
    X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

    m_depth4 = xgb_depth4(n_estimators=3000, early_stopping_rounds=50)
    m_depth4.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    m_depth2 = xgb_depth2(n_estimators=3000, early_stopping_rounds=50)
    m_depth2.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    m_extra = extra_trees().fit(X_tr, y_tr)

    m_lgbm = lgbm_model(n_estimators=3000, early_stopping_rounds=50)
    m_lgbm.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=m_lgbm._early_stopping_callbacks,
    )

    oof_depth4[val_idx] = m_depth4.predict_proba(X_va)[:, 1]
    oof_depth2[val_idx] = m_depth2.predict_proba(X_va)[:, 1]
    oof_extra[val_idx] = m_extra.predict_proba(X_va)[:, 1]
    oof_lgbm[val_idx] = m_lgbm.predict_proba(X_va)[:, 1]

    best_iters_depth4.append(m_depth4.best_iteration)
    best_iters_depth2.append(m_depth2.best_iteration)
    best_iters_lgbm.append(m_lgbm.best_iteration_)

    print(f"  fold {fold}: depth4 best_iter={m_depth4.best_iteration}, "
          f"depth2 best_iter={m_depth2.best_iteration}, "
          f"lgbm best_iter={m_lgbm.best_iteration_}")

print("\n=== OOF 表现（在完整训练集上，交叉验证得到） ===")
report_scores("XGB depth=4", y, oof_depth4)
report_scores("XGB depth=2", y, oof_depth2)
report_scores("ExtraTrees", y, oof_extra)
report_scores("LightGBM", y, oof_lgbm)

print("\n=== OOF 预测相关性矩阵 ===")
corr = np.corrcoef([oof_depth4, oof_depth2, oof_extra, oof_lgbm])
corr_df = pd.DataFrame(
    corr,
    index=["depth4", "depth2", "extra", "lgbm"],
    columns=["depth4", "depth2", "extra", "lgbm"],
)
print(corr_df.round(4))

# ---------------------------------------------------------------------------
# 3. Optimize blend weights against OOF predictions instead of hand-picking.
# ---------------------------------------------------------------------------
oof_list = [oof_depth4, oof_depth2, oof_extra, oof_lgbm]
weights = optimize_blend_weights(oof_list, y)
w_depth4, w_depth2, w_extra, w_lgbm = weights

print("\n=== 最优混合权重（基于OOF搜索得到） ===")
print(f"  XGB depth=4 : {w_depth4:.4f}")
print(f"  XGB depth=2 : {w_depth2:.4f}")
print(f"  ExtraTrees  : {w_extra:.4f}")
print(f"  LightGBM    : {w_lgbm:.4f}")

oof_blend = (
    w_depth4 * oof_depth4
    + w_depth2 * oof_depth2
    + w_extra * oof_extra
    + w_lgbm * oof_lgbm
)
report_scores("Final blend", y, oof_blend)

# ---------------------------------------------------------------------------
# 4. A single holdout split purely as an additional sanity check
#    (kept for continuity, not used for weight tuning).
# ---------------------------------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y,
)

sanity_depth4 = xgb_depth4().fit(X_train, y_train)
sanity_depth2 = xgb_depth2().fit(X_train, y_train)
sanity_extra = extra_trees().fit(X_train, y_train)
sanity_lgbm = lgbm_model(n_estimators=500).fit(X_train, y_train)  # fixed, small n for the sanity check only

p_sanity = (
    w_depth4 * sanity_depth4.predict_proba(X_val)[:, 1]
    + w_depth2 * sanity_depth2.predict_proba(X_val)[:, 1]
    + w_extra * sanity_extra.predict_proba(X_val)[:, 1]
    + w_lgbm * sanity_lgbm.predict_proba(X_val)[:, 1]
)
print("\n=== 单次80/20留出验证（sanity check，不用于调参） ===")
report_scores("Final blend", y_val, p_sanity)

# ---------------------------------------------------------------------------
# 5. Retrain all models on the complete labelled dataset.
#    XGBoost/LightGBM use the median best_iteration found across CV folds
#    (rounded up a bit) instead of a fixed guessed n_estimators.
# ---------------------------------------------------------------------------
final_n_depth4 = int(np.median(best_iters_depth4) * 1.1) + 1
final_n_depth2 = int(np.median(best_iters_depth2) * 1.1) + 1
final_n_lgbm = int(np.median(best_iters_lgbm) * 1.1) + 1

print(f"\n=== 使用全部训练数据重训最终模型 ===")
print(f"  depth4 最终树数量: {final_n_depth4} (CV median best_iter x1.1)")
print(f"  depth2 最终树数量: {final_n_depth2} (CV median best_iter x1.1)")
print(f"  lgbm   最终树数量: {final_n_lgbm} (CV median best_iter x1.1)")

final_depth4 = xgb_depth4(n_estimators=final_n_depth4).fit(X, y)
final_depth2 = xgb_depth2(n_estimators=final_n_depth2).fit(X, y)
final_extra = extra_trees().fit(X, y)
final_lgbm = lgbm_model(n_estimators=final_n_lgbm).fit(X, y)

# ---------------------------------------------------------------------------
# 6. Final blended probability using the OOF-optimized weights.
# ---------------------------------------------------------------------------
p_xgb_depth4 = final_depth4.predict_proba(X_test)[:, 1]
p_xgb_depth2 = final_depth2.predict_proba(X_test)[:, 1]
p_extra_trees = final_extra.predict_proba(X_test)[:, 1]
p_lgbm = final_lgbm.predict_proba(X_test)[:, 1]

final_prob = (
    w_depth4 * p_xgb_depth4
    + w_depth2 * p_xgb_depth2
    + w_extra * p_extra_trees
    + w_lgbm * p_lgbm
)

submission = pd.DataFrame(
    {
        "client_id": test["client_id"],
        "default": np.clip(final_prob, 1e-6, 1 - 1e-6),
    }
)
submission.to_csv("submission_v10.csv", index=False)

# ---------------------------------------------------------------------------
# 7. Feature importance from the strongest XGBoost component.
# ---------------------------------------------------------------------------
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
plt.savefig("feature_importance_v10.png", dpi=150)

print(f"\n已生成 submission_v10.csv，共 {len(submission)} 行。")
print(submission.head())