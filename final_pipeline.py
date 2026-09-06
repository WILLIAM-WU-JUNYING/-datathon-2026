"""
v42 - Continue fine-tuning based on v33 direction
Core strategy: Continue to slightly increase learning rate and slightly decrease complexity
"""

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
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

try:
    from lightgbm import LGBMClassifier
except ImportError:
    raise ImportError("Please install lightgbm: pip install lightgbm")

try:
    from catboost import CatBoostClassifier
except ImportError:
    raise ImportError("Please install catboost: pip install catboost")

TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
RANDOM_STATE = 42
N_FOLDS = 5

print("=" * 70)
print("v42 - Continue fine-tuning based on v33 direction")
print("=" * 70)


# ============================================================================
# Feature Engineering (Fully restored to v30)
# ============================================================================
def make_features(data: pd.DataFrame) -> pd.DataFrame:
    """v30 full feature engineering"""
    df = data.copy()

    pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
    amount_cols = [f"PAY_AMT{i}" for i in range(1, 7)]

    # Clean known garbage categories
    if "EDUCATION" in df.columns:
        df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    if "MARRIAGE" in df.columns:
        df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

    # ---------- Repayment behaviour ----------
    df["PAY_max"] = df[pay_cols].max(axis=1)
    df["PAY_mean"] = df[pay_cols].mean(axis=1)
    df["PAY_bad_count"] = (df[pay_cols] > 0).sum(axis=1)
    df["PAY_clear_count"] = (df[pay_cols] < 0).sum(axis=1)
    df["PAY_recent_change"] = df["PAY_0"] - df["PAY_2"]
    df["PAY_0_severe"] = (df["PAY_0"] >= 2).astype(int)
    df["PAY_any_severe"] = (df[pay_cols] >= 2).any(axis=1).astype(int)
    df["PAY_trend"] = df["PAY_0"] - df["PAY_max"]

    # Longest consecutive overdue
    pay_recent_to_old = df[pay_cols].to_numpy()
    is_bad = pay_recent_to_old > 0
    streak = np.zeros(len(df), dtype=int)
    still_running = np.ones(len(df), dtype=bool)
    for month_idx in range(is_bad.shape[1]):
        still_running &= is_bad[:, month_idx]
        streak += still_running.astype(int)
    df["PAY_consecutive_bad"] = streak

    recent_avg = df[["PAY_0", "PAY_2"]].mean(axis=1)
    older_avg = df[["PAY_3", "PAY_4"]].mean(axis=1)
    df["PAY_worsening"] = (recent_avg >= older_avg).astype(int)

    # ---------- Bill level / variation ----------
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
    df["BILL_volatile"] = df["BILL_std"] / (df["BILL_mean"].abs() + 1.0)

    # Recent 3 months vs older 3 months
    df["RECENT3_vs_OLD3_bill"] = (
        df[["BILL_AMT1", "BILL_AMT2", "BILL_AMT3"]].mean(axis=1)
        - df[["BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]].mean(axis=1)
    )
    df["RECENT3_vs_OLD3_pay"] = (
        df[["PAY_AMT1", "PAY_AMT2", "PAY_AMT3"]].mean(axis=1)
        - df[["PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]].mean(axis=1)
    )

    # ---------- Payment intensity ----------
    df["PAYMENT_sum"] = df[amount_cols].sum(axis=1)
    df["PAYMENT_mean"] = df[amount_cols].mean(axis=1)
    df["PAYMENT_min"] = df[amount_cols].min(axis=1)
    df["PAYMENT_median"] = df[amount_cols].median(axis=1)
    df["PAY_zero_count"] = (df[amount_cols] == 0).sum(axis=1)

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
    df["IMPLIED_SPEND_recent"] = (
        df["BILL_AMT1"] - df["BILL_AMT2"] + df["PAY_AMT1"]
    )

    # ---------- Utilization & payment ratios ----------
    utilization_cols = []
    payment_ratio_cols = []
    for i in range(1, 7):
        util_col = f"UTIL_{i}"
        ratio_col = f"PAY_RATIO_{i}"
        bill_i = df[f"BILL_AMT{i}"]
        pay_i = df[f"PAY_AMT{i}"]
        df[util_col] = bill_i / df["LIMIT_BAL"].clip(lower=1)
        df[ratio_col] = np.where(bill_i > 0, pay_i / bill_i, np.nan)
        utilization_cols.append(util_col)
        payment_ratio_cols.append(ratio_col)

    df["UTIL_max"] = df[utilization_cols].max(axis=1)
    df["UTIL_std"] = df[utilization_cols].std(axis=1)
    df["PAY_RATIO_mean"] = df[payment_ratio_cols].mean(axis=1)
    df["PAY_RATIO_min"] = df[payment_ratio_cols].min(axis=1)

    positive_bill_matrix = df[bill_cols].clip(lower=0).to_numpy()
    df["PAID_OVER_BILL_COUNT"] = (
        df[amount_cols].to_numpy() > positive_bill_matrix
    ).sum(axis=1)
    df["BILL_POS_COUNT"] = (df[bill_cols] > 0).sum(axis=1)

    # ---------- Trajectory features ----------
    bill_matrix = df[bill_cols].to_numpy()
    is_increasing = bill_matrix[:, :-1] > bill_matrix[:, 1:]
    bill_streak = np.zeros(len(df), dtype=int)
    still_increasing = np.ones(len(df), dtype=bool)
    for month_idx in range(is_increasing.shape[1]):
        still_increasing &= is_increasing[:, month_idx]
        bill_streak += still_increasing.astype(int)
    df["BILL_increasing_streak"] = bill_streak

    ratio_matrix = df[payment_ratio_cols].to_numpy()
    df["PAYMENT_coverage_low_count"] = (ratio_matrix < 0.30).sum(axis=1)

    is_low_ratio = np.nan_to_num(ratio_matrix < 0.30, nan=False).astype(bool)
    low_streak = np.zeros(len(df), dtype=int)
    still_low = np.ones(len(df), dtype=bool)
    for month_idx in range(is_low_ratio.shape[1]):
        still_low &= is_low_ratio[:, month_idx]
        low_streak += still_low.astype(int)
    df["PAY_RATIO_low_streak"] = low_streak

    payment_matrix = df[amount_cols].to_numpy()
    is_decreasing = payment_matrix[:, :-1] < payment_matrix[:, 1:]
    payment_streak = np.zeros(len(df), dtype=int)
    still_decreasing = np.ones(len(df), dtype=bool)
    for month_idx in range(is_decreasing.shape[1]):
        still_decreasing &= is_decreasing[:, month_idx]
        payment_streak += still_decreasing.astype(int)
    df["PAYMENT_decreasing_streak"] = payment_streak

    is_zero_pay = payment_matrix == 0
    zero_streak = np.zeros(len(df), dtype=int)
    still_zero = np.ones(len(df), dtype=bool)
    for month_idx in range(is_zero_pay.shape[1]):
        still_zero &= is_zero_pay[:, month_idx]
        zero_streak += still_zero.astype(int)
    df["ZERO_PAY_streak"] = zero_streak

    util_matrix = df[utilization_cols].to_numpy()
    is_high_util = util_matrix > 0.7
    util_streak = np.zeros(len(df), dtype=int)
    still_high = np.ones(len(df), dtype=bool)
    for month_idx in range(is_high_util.shape[1]):
        still_high &= is_high_util[:, month_idx]
        util_streak += still_high.astype(int)
    df["UTIL_high_streak"] = util_streak

    is_extreme = util_matrix > 0.9
    extreme_streak = np.zeros(len(df), dtype=int)
    still_extreme = np.ones(len(df), dtype=bool)
    for month_idx in range(is_extreme.shape[1]):
        still_extreme &= is_extreme[:, month_idx]
        extreme_streak += still_extreme.astype(int)
    df["UTIL_extreme_streak"] = extreme_streak

    # ---------- Interactions ----------
    df["PAY_severe_recent3"] = (
        (df[["PAY_0", "PAY_2", "PAY_3"]] >= 2).sum(axis=1)
    )
    df["HIGH_UTIL_LOW_COVER"] = (
        (df["UTIL_max"] > 0.7).astype(int) * df["PAYMENT_coverage_low_count"]
    )
    df["BALANCE_REBOUND"] = np.where(
        df["PAY_AMT1"] > 0,
        (df["BILL_AMT1"] - (df["BILL_AMT2"] - df["PAY_AMT1"]))
        / df["PAY_AMT1"].clip(lower=1),
        0.0,
    )
    df["HIGH_UTIL_ZERO_PAY"] = (
        (df["UTIL_max"] > 0.7).astype(int) * (df["PAY_AMT1"] == 0).astype(int)
    )

    df["AGE_x_LIMIT"] = df["AGE"] * df["LIMIT_BAL"]
    df["BADCOUNT_x_UTILMAX"] = df["PAY_bad_count"] * df["UTIL_max"]
    df["PAYMAX_x_UTILMAX"] = df["PAY_max"] * df["UTIL_max"]
    df["SEVERE_x_ZEROSTREAK"] = df["PAY_0_severe"] * df["ZERO_PAY_streak"]

    # ---------- Log transforms for long-tailed amounts ----------
    for column in ["LIMIT_BAL"] + bill_cols + amount_cols:
        df[f"LOGABS_{column}"] = np.sign(df[column]) * np.log1p(df[column].abs())

    return df.replace([np.inf, -np.inf], np.nan).fillna(0)


# ============================================================================
# v42 Model Factories - Continue fine-tuning based on v33
# ============================================================================
def xgb_depth4(n_estimators=700, early_stopping_rounds=None) -> XGBClassifier:
    """v42: lr 0.018 -> 0.019, min_child 16 -> 14"""
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=4,
        learning_rate=0.019,       # v33: 0.018 -> 0.019
        min_child_weight=14,       # v33: 16 -> 14
        subsample=0.85,
        colsample_bytree=0.90,
        reg_lambda=28,
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        early_stopping_rounds=early_stopping_rounds,
    )


def xgb_depth3(n_estimators=1200, early_stopping_rounds=None) -> XGBClassifier:
    """v42: lr 0.014 -> 0.015, reg_lambda 20 -> 18"""
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=3,
        learning_rate=0.015,       # v33: 0.014 -> 0.015
        min_child_weight=12,
        subsample=0.80,
        colsample_bytree=0.90,
        reg_lambda=18,             # v33: 20 -> 18
        eval_metric="logloss",
        tree_method="hist",
        random_state=52,
        n_jobs=-1,
        early_stopping_rounds=early_stopping_rounds,
    )


def extra_trees() -> ExtraTreesClassifier:
    """v42: n_estimators 1000 -> 1200, leaf 10 -> 8"""
    return ExtraTreesClassifier(
        n_estimators=1200,         # v33: 1000 -> 1200
        max_features=0.75,
        min_samples_leaf=8,        # v33: 10 -> 8
        class_weight=None,
        random_state=21,
        n_jobs=-1,
    )


def lgbm_model(n_estimators=2000, early_stopping_rounds=None) -> LGBMClassifier:
    """v42: lr 0.020 -> 0.021, num_leaves 18 -> 16"""
    callbacks = None
    if early_stopping_rounds is not None:
        import lightgbm as lgb
        callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=False)]

    model = LGBMClassifier(
        n_estimators=n_estimators,
        num_leaves=16,             # v33: 18 -> 16
        learning_rate=0.021,       # v33: 0.020 -> 0.021
        min_child_samples=25,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=18,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    model._early_stopping_callbacks = callbacks
    return model


def catboost_model(n_estimators=2000, early_stopping_rounds=None) -> CatBoostClassifier:
    """v42: lr 0.028 -> 0.029, depth 4 -> 3"""
    return CatBoostClassifier(
        iterations=n_estimators,
        depth=3,                   # v33: 4 -> 3
        learning_rate=0.029,       # v33: 0.028 -> 0.029
        l2_leaf_reg=12,
        random_seed=RANDOM_STATE,
        eval_metric="Logloss",
        loss_function="Logloss",
        verbose=False,
        thread_count=-1,
        early_stopping_rounds=early_stopping_rounds,
        allow_writing_files=False,
    )


def report_scores(name, y_true, probability):
    print(
        f"{name:<26} "
        f"Log Loss={log_loss(y_true, probability, labels=[0, 1]):.6f} | "
        f"AUC={roc_auc_score(y_true, probability):.6f} | "
        f"Brier={brier_score_loss(y_true, probability):.6f}"
    )


def optimize_blend_weights(oof_probs, y_true):
    """Non-negative weights summing to 1 that minimise log loss."""
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


# ============================================================================
# Main Program
# ============================================================================

print("\n1. Reading data...")
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

print(f"   Training set: {train.shape}")
print(f"   Test set: {test.shape}")

y = train["default"].astype(int)
X = make_features(train.drop(columns=["client_id", "default"]))
X_test = make_features(test.drop(columns=["client_id"]))

print(f"   Default rate: {y.mean():.4f}")
print(f"   Number of features: {X.shape[1]}")

# ============================================================================
# Cross Validation
# ============================================================================
print("\n2. Starting 5-fold cross validation...")
CV_SEEDS = [42, 137, 2026, 7, 99]

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

for repeat_no, cv_seed in enumerate(CV_SEEDS, start=1):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=cv_seed)
    
    repeat_depth4 = np.zeros(len(X))
    repeat_depth3 = np.zeros(len(X))
    repeat_extra = np.zeros(len(X))
    repeat_lgbm = np.zeros(len(X))
    repeat_cat = np.zeros(len(X))
    
    print(f"\n   Repeat {repeat_no}/{len(CV_SEEDS)} (seed={cv_seed})")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
        
        m_depth4 = xgb_depth4(n_estimators=3000, early_stopping_rounds=50)
        m_depth4.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        
        m_depth3 = xgb_depth3(n_estimators=3000, early_stopping_rounds=50)
        m_depth3.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        
        m_extra = extra_trees().fit(X_tr, y_tr)
        
        m_lgbm = lgbm_model(n_estimators=3000, early_stopping_rounds=50)
        m_lgbm.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=m_lgbm._early_stopping_callbacks)
        
        m_cat = catboost_model(n_estimators=3000, early_stopping_rounds=50)
        m_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
        
        repeat_depth4[val_idx] = m_depth4.predict_proba(X_va)[:, 1]
        repeat_depth3[val_idx] = m_depth3.predict_proba(X_va)[:, 1]
        repeat_extra[val_idx] = m_extra.predict_proba(X_va)[:, 1]
        repeat_lgbm[val_idx] = m_lgbm.predict_proba(X_va)[:, 1]
        repeat_cat[val_idx] = m_cat.predict_proba(X_va)[:, 1]
        
        best_iters_depth4.append(m_depth4.best_iteration)
        best_iters_depth3.append(m_depth3.best_iteration)
        best_iters_lgbm.append(m_lgbm.best_iteration_)
        best_iters_cat.append(m_cat.best_iteration_)
        
        print(f"      fold {fold}: d4={m_depth4.best_iteration}, d3={m_depth3.best_iteration}, lgbm={m_lgbm.best_iteration_}, cat={m_cat.best_iteration_}")
    
    repeat_weights = optimize_blend_weights(
        [repeat_depth4, repeat_depth3, repeat_extra, repeat_lgbm, repeat_cat], y
    )
    repeat_blend = sum(
        repeat_weights[i] * p
        for i, p in enumerate([repeat_depth4, repeat_depth3, repeat_extra, repeat_lgbm, repeat_cat])
    )
    print(f"      classic-blend Log Loss={log_loss(y, repeat_blend, labels=[0, 1]):.6f}")
    
    oof_depth4_repeats.append(repeat_depth4)
    oof_depth3_repeats.append(repeat_depth3)
    oof_extra_repeats.append(repeat_extra)
    oof_lgbm_repeats.append(repeat_lgbm)
    oof_cat_repeats.append(repeat_cat)
    weights_by_repeat.append(repeat_weights)

# Average OOF across repeats
oof_depth4 = np.mean(oof_depth4_repeats, axis=0)
oof_depth3 = np.mean(oof_depth3_repeats, axis=0)
oof_extra = np.mean(oof_extra_repeats, axis=0)
oof_lgbm = np.mean(oof_lgbm_repeats, axis=0)
oof_cat = np.mean(oof_cat_repeats, axis=0)

weights = np.median(weights_by_repeat, axis=0)
weights = weights / weights.sum()
w_depth4, w_depth3, w_extra, w_lgbm, w_cat = weights

print("\n3. Base model performance:")
report_scores("XGB depth=4", y, oof_depth4)
report_scores("XGB depth=3", y, oof_depth3)
report_scores("ExtraTrees", y, oof_extra)
report_scores("LightGBM", y, oof_lgbm)
report_scores("CatBoost", y, oof_cat)

print("\n4. Classic weighted average weights:")
print(f"   XGB depth=4 : {w_depth4:.4f}")
print(f"   XGB depth=3 : {w_depth3:.4f}")
print(f"   ExtraTrees  : {w_extra:.4f}")
print(f"   LightGBM    : {w_lgbm:.4f}")
print(f"   CatBoost    : {w_cat:.4f}")

oof_classic = (
    w_depth4 * oof_depth4
    + w_depth3 * oof_depth3
    + w_extra * oof_extra
    + w_lgbm * oof_lgbm
    + w_cat * oof_cat
)
report_scores("Classic blend", y, oof_classic)

# ============================================================================
# Stacking
# ============================================================================
print("\n5. Stacking...")
oof_meta_X = np.column_stack([oof_depth4, oof_depth3, oof_extra, oof_lgbm, oof_cat])

meta_oof = np.zeros(len(y))
meta_skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
for tr_idx, va_idx in meta_skf.split(oof_meta_X, y):
    meta = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE, solver="lbfgs")
    meta.fit(oof_meta_X[tr_idx], y.iloc[tr_idx])
    meta_oof[va_idx] = meta.predict_proba(oof_meta_X[va_idx])[:, 1]

final_meta = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE, solver="lbfgs")
final_meta.fit(oof_meta_X, y)

# ============================================================================
# Isotonic Calibration
# ============================================================================
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(meta_oof, y)
oof_stacked_cal = np.clip(iso.predict(meta_oof), 1e-6, 1 - 1e-6)

print("\n6. Final results:")
report_scores("Stacked (cal)", y, oof_stacked_cal)

# ============================================================================
# Final Prediction
# ============================================================================
print("\n7. Generating submission file...")
final_n_depth4 = int(np.median(best_iters_depth4) * 1.20) + 1
final_n_depth3 = int(np.median(best_iters_depth3) * 1.20) + 1
final_n_lgbm = int(np.median(best_iters_lgbm) * 1.20) + 1
final_n_cat = int(np.median(best_iters_cat) * 1.20) + 1

print(f"   Final tree counts: d4={final_n_depth4}, d3={final_n_depth3}, lgbm={final_n_lgbm}, cat={final_n_cat}")

final_depth4 = xgb_depth4(n_estimators=final_n_depth4).fit(X, y)
final_depth3 = xgb_depth3(n_estimators=final_n_depth3).fit(X, y)
final_extra = extra_trees().fit(X, y)
final_lgbm = lgbm_model(n_estimators=final_n_lgbm).fit(X, y)
final_cat = catboost_model(n_estimators=final_n_cat).fit(X, y)

p_d4 = final_depth4.predict_proba(X_test)[:, 1]
p_d3 = final_depth3.predict_proba(X_test)[:, 1]
p_extra = final_extra.predict_proba(X_test)[:, 1]
p_lgbm = final_lgbm.predict_proba(X_test)[:, 1]
p_cat = final_cat.predict_proba(X_test)[:, 1]

test_meta_X = np.column_stack([p_d4, p_d3, p_extra, p_lgbm, p_cat])
final_prob_raw = final_meta.predict_proba(test_meta_X)[:, 1]
final_prob = np.clip(iso.predict(final_prob_raw), 1e-6, 1 - 1e-6)

submission = pd.DataFrame({
    "client_id": test["client_id"],
    "default": final_prob
})
submission.to_csv("submission.csv", index=False)

print(f"\n✅ Done! Submission file: submission.csv")
print(submission.head())

# ============================================================================
# Feature Importance
# ============================================================================
importance = pd.Series(final_depth4.feature_importances_, index=X.columns).sort_values(ascending=False)

print("\n8. Top 25 Feature Importance:")
print(importance.head(25).to_string())

plt.figure(figsize=(9, 9))
importance.head(25).sort_values().plot(kind="barh", color="#2b6cb0")
plt.title("v42 - Top 25 Features")
plt.xlabel("XGBoost feature importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
print("\n   Feature importance plot saved: feature_importance.png")

print("\n" + "=" * 70)
print("execution complete!")
print("=" * 70)