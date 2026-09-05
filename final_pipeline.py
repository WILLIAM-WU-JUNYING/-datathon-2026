"""
Stream 1 - 信用卡违约预测 最终流程
=========================================
这是已经跑通、产出了提交文件的完整代码。
数据: train_...csv (24000行,带default标签) + Inter-uni_...csv (6000行,测试集)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss, confusion_matrix
from xgboost import XGBClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TRAIN_PATH = "train Inter-uni Datathon Stream 1 Credit card clients.csv"
TEST_PATH = "test Inter-uni Datathon Stream 1 Credit card clients.csv"


# ============================================================
# 第1步：读取数据
# ============================================================
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

X = train.drop(columns=["client_id", "default"])  # 特征
y = train["default"]                               # 答案（0=不违约, 1=违约）


# ============================================================
# 第2步：切出一份"内部验证集"，用来检验模型好不好
# ============================================================
# 注意：这跟最终的"测试集(test.csv)"不是一回事
# 训练集(24000行)内部再切一刀，一部分用来训练，一部分用来"自我检验"
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ============================================================
# 第3步：训练模型 —— XGBoost + 概率校准
# ============================================================
# 为什么用XGBoost：在这类表格数据上，通常比逻辑回归、随机森林效果更好
# 为什么要"校准"(CalibratedClassifierCV)：
#   XGBoost本身预测能力强，但如果直接用 class_weight/scale_pos_weight 处理类别不平衡，
#   会导致它输出的"概率"不准（比如说70%违约的一批人，实际违约率没那么高）
#   题目要求"输出概率"且强调"model calibration"，所以我们优先保证概率准确，
#   再用调整分类阈值的方式去应对业务上的取舍（而不是让模型本身的概率失真）

xgb_base = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    eval_metric="auc",
    random_state=42,
    n_jobs=-1,
)

# method="isotonic": 一种专门"矫正"概率输出的技术，让模型说的百分比更贴近真实情况
# cv=5: 用5折交叉验证来做校准，减少过拟合风险
calibrated_model = CalibratedClassifierCV(xgb_base, method="isotonic", cv=5)
calibrated_model.fit(X_train, y_train)


# ============================================================
# 第4步：在验证集上检查效果
# ============================================================
val_proba = calibrated_model.predict_proba(X_val)[:, 1]

print("=== 验证集表现 ===")
print(f"AUC (预测排序能力，越高越好，0.5=瞎猜，1.0=完美): {roc_auc_score(y_val, val_proba):.3f}")
print(f"Brier Score (概率准不准，越低越好): {brier_score_loss(y_val, val_proba):.3f}")


# ============================================================
# 第5步：代价敏感分析 —— 不同阈值下的业务影响
# ============================================================
# FN(漏报) = 真实会违约，但模型说不会 → 银行放贷给了会违约的人，面临财务损失
# FP(误报) = 真实不会违约，但模型说会 → 银行错误限制了好客户，损失业务机会
print("\n=== 不同分类阈值下的业务权衡 ===")
print(f"{'阈值':>6} {'漏报FN':>10} {'误报FP':>10}")
for thresh in [0.2, 0.3, 0.4, 0.5, 0.6]:
    pred = (val_proba >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, pred).ravel()
    print(f"{thresh:>6.1f} {fn:>10} {fp:>10}")
# 展示时可以讲：银行越怕坏账，就该把阈值设低一点（哪怕误伤更多好客户）


# ============================================================
# 第6步：用全部训练数据重新训练"最终模型"
# ============================================================
# 前面为了验证效果，只用了80%数据训练；现在正式预测时，用100%数据训练，
# 这样能让最终模型学到更多规律
final_model = CalibratedClassifierCV(
    XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, eval_metric="auc", random_state=42, n_jobs=-1),
    method="isotonic", cv=5
)
final_model.fit(X, y)


# ============================================================
# 第7步：特征重要性 —— 哪些信息最能预测违约
# ============================================================
# 用一个普通的XGBoost（不带校准包装）单独算重要性，因为校准后的模型结构会变复杂
importance_model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, eval_metric="auc", random_state=42, n_jobs=-1)
importance_model.fit(X, y)
importances = pd.Series(importance_model.feature_importances_, index=X.columns).sort_values(ascending=False)

print("\n=== 特征重要性 Top 10 ===")
print(importances.head(10))

plt.figure(figsize=(8, 6))
importances.head(12).sort_values().plot(kind="barh", color="#2b6cb0")
plt.title("Top 12 Features Predicting Credit Card Default")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)


# ============================================================
# 第8步：对真正的测试集(6000行)做预测，生成提交文件
# ============================================================
X_test = test.drop(columns=["client_id"])
test_proba = final_model.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({
    "client_id": test["client_id"],
    "default": test_proba   # 注意：这里是概率(0~1之间的小数)，不是0/1的判断
})
submission.to_csv("submission.csv", index=False)

print(f"\n提交文件已生成，共{len(submission)}行")
print(submission.head())

# ============================================================
# 提醒：
# 1. 提交前请对照 sample_submission.csv 检查列名是否匹配
# 2. PAY_0是最重要的特征（说明"最近一次的还款状态"最能预测未来是否违约）
# 3. 展示时可以强调：模型没有让性别(SEX)、婚姻(MARRIAGE)成为重要特征，
#    这体现了"负责任地使用人口统计信息"——模型主要依赖还款行为，而非身份属性做判断
# ============================================================
