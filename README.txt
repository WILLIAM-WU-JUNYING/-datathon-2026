================================================================================
                    CREDIT CARD DEFAULT PREDICTION - FINAL
================================================================================

OVERVIEW
================================================================================
This solution achieves a Log Loss of 0.41002 using an ensemble of 5 base models 
(XGBoost depth=3, XGBoost depth=4, LightGBM, CatBoost, and ExtraTrees) with 
Stacking and Isotonic Calibration.

The approach builds upon iterative experimentation from v30 (baseline: 0.41016) 
through v42, where the optimal parameter configuration was identified through 
systematic fine-tuning of learning rates and model complexity.

================================================================================
REQUIRED FILES
================================================================================

Place the following files in the same directory as the script:

- train.csv      Training dataset with columns: client_id, default, and features
- test.csv       Test dataset with columns: client_id and features
- final_pipeline.py  Main execution script

================================================================================
EXECUTION ORDER
================================================================================

Step 1: Run the script
        python final_pipeline.py

Step 2: The script automatically performs:
        a) Read and preprocess data
        b) Generate 73 features (v30 feature engineering)
        c) 5-fold cross-validation with 5 random seeds
        d) Train 5 base models on each fold
        e) Stack predictions using Logistic Regression
        f) Apply Isotonic Calibration
        g) Generate final submission file (submission_v42.csv)

================================================================================
KEY DEPENDENCIES / PACKAGES
================================================================================

Required Python Version: 3.8 or higher

Package                      Version       Purpose
---------------------------  -----------   --------------------------------
numpy                        >= 1.21.0     Array operations
pandas                       >= 1.3.0      Data manipulation
scikit-learn                 >= 1.0.0      ML utilities, Stacking, Calibration
xgboost                      >= 1.5.0      XGBoost models
lightgbm                     >= 3.3.0      LightGBM model
catboost                     >= 1.0.0      CatBoost model
scipy                        >= 1.7.0      Optimization
matplotlib                   >= 3.4.0      Feature importance plot (optional)

Installation Command:
--------------------------------------------------------------------------------
pip install numpy pandas scikit-learn xgboost lightgbm catboost scipy matplotlib

================================================================================
IMPORTANT SETTINGS
================================================================================

Parameter                  Value                    Description
-------------------------  -----------------------  -----------------------------
RANDOM_STATE               42                       Base random seed
N_FOLDS                    5                        Number of CV folds
CV_SEEDS                   [42, 137, 2026, 7, 99]  Random seeds for repeated CV
early_stopping_rounds      50                       Early stopping for tree models
tree_multiplier            1.20                     Multiplier for final tree count
Stacking C                 1.0                      Logistic Regression C parameter

================================================================================
MODEL PARAMETERS (v42 CONFIGURATION)
================================================================================

1. XGBoost depth=4
--------------------------------------------------------------------------------
learning_rate:      0.019
max_depth:          4
min_child_weight:   14
subsample:          0.85
colsample_bytree:   0.90
reg_lambda:         28
eval_metric:        logloss
tree_method:        hist

2. XGBoost depth=3
--------------------------------------------------------------------------------
learning_rate:      0.015
max_depth:          3
min_child_weight:   12
subsample:          0.80
colsample_bytree:   0.90
reg_lambda:         18
eval_metric:        logloss
tree_method:        hist

3. LightGBM
--------------------------------------------------------------------------------
learning_rate:      0.021
num_leaves:         16
min_child_samples:  25
subsample:          0.85
colsample_bytree:   0.85
reg_lambda:         18

4. CatBoost
--------------------------------------------------------------------------------
learning_rate:      0.029
depth:              3
l2_leaf_reg:        12

5. ExtraTrees
--------------------------------------------------------------------------------
n_estimators:       1200
max_features:       0.75
min_samples_leaf:   8

================================================================================
FINAL PREDICTION FILE
================================================================================

Filename:    submission_v42.csv

Format:
--------------------------------------------------------------------------------
client_id,default
CC_0012A082B7B7,0.132542
CC_0012BC27DFB6,0.166306
CC_001563B2143D,0.149123
...

Columns:
- client_id:  Unique client identifier (copied from test.csv)
- default:    Predicted probability of default (range: 0 to 1)

================================================================================
FEATURE ENGINEERING SUMMARY
================================================================================

Total Features: 73

Category                      Features
---------------------------  ------------------------------------------------
Repayment behavior           PAY_max, PAY_mean, PAY_bad_count, PAY_clear_count,
                             PAY_recent_change, PAY_0_severe, PAY_any_severe,
                             PAY_trend, PAY_consecutive_bad, PAY_worsening

Bill statistics              BILL_mean, BILL_max, BILL_min, BILL_median,
                             BILL_std, BILL_change_1_to_6, BILL_last_minus_mean,
                             BILL_to_limit, BILL_volatile

Payment intensity            PAYMENT_sum, PAYMENT_mean, PAYMENT_min,
                             PAYMENT_median, PAY_zero_count, PAYMENT_to_bill,
                             RECENT_PAYMENT_to_bill, IMPLIED_SPEND_recent

Utilization ratios           UTIL_1 through UTIL_6, UTIL_max, UTIL_std,
                             UTIL_high_streak, UTIL_extreme_streak

Trajectory features          BILL_increasing_streak, PAYMENT_coverage_low_count,
                             PAY_RATIO_low_streak, PAYMENT_decreasing_streak,
                             ZERO_PAY_streak

Interactions                 PAY_severe_recent3, HIGH_UTIL_LOW_COVER,
                             BALANCE_REBOUND, HIGH_UTIL_ZERO_PAY,
                             AGE_x_LIMIT, BADCOUNT_x_UTILMAX,
                             PAYMAX_x_UTILMAX, SEVERE_x_ZEROSTREAK

Log transforms               LOGABS_LIMIT_BAL, LOGABS_BILL_AMT1 through
                             LOGABS_BILL_AMT6, LOGABS_PAY_AMT1 through
                             LOGABS_PAY_AMT6

Time comparisons             RECENT3_vs_OLD3_bill, RECENT3_vs_OLD3_pay

================================================================================
EXECUTION TIME & RESOURCES
================================================================================

Total runtime:   Approximately 15-25 minutes (depends on CPU)
Memory usage:    Approximately 2-4 GB RAM

================================================================================
REPRODUCTION INSTRUCTIONS
================================================================================

Step-by-step:

1. Download/Clone this repository

2. Install dependencies:
   ----------------------------------------------------------------------------
   pip install numpy pandas scikit-learn xgboost lightgbm catboost scipy matplotlib
   ----------------------------------------------------------------------------

3. Place train.csv and test.csv in the same directory as final_pipeline_v42.py

4. Run the script:
   ----------------------------------------------------------------------------
   python final_pipeline_v42.py
   ----------------------------------------------------------------------------

5. The script will output:
   - console logs showing training progress and validation scores
   - feature_importance_v42.png (feature importance plot)
   - submission_v42.csv (final prediction file)

6. Submit submission_v42.csv to the competition platform

================================================================================
VERSION HISTORY & RESULTS
================================================================================

Version    Log Loss    Change     Status
--------  ----------  ---------  --------------------------------
v30        0.41016     baseline   Starting point
v31        0.41149     +0.00133   Overfitting
v32        0.41091     +0.00075   Slightly worse
v33        0.41015     -0.00001   First improvement
v34        0.41096     +0.00080   Overfitting
v35        0.41129     +0.00113   Worse
v36        0.41053     +0.00037   Slightly worse
v37        0.41050     +0.00034   Slightly worse
v38        0.41283     +0.00267   Feature engineering failed
v39        0.41391     +0.00375   MLP failed
v40        0.41056     +0.00040   Ensemble failed
v41        0.41049     +0.00033   LOF failed
v42        0.41002     -0.00014   BEST RESULT
v43        0.41146     +0.00144   Too aggressive
v44        0.41002     -0.00014   Verified stable
v45        0.41059     +0.00057   Too many CV seeds
v46        0.40995     -0.00007   Pseudo-labeling

================================================================================
KEY INSIGHTS FROM EXPERIMENTATION
================================================================================

1. The optimal direction was identified from v30 to v33 and further to v42:
   - Increase learning rates moderately
   - Decrease model complexity (fewer leaves, smaller depth)
   - This improves generalization

2. Feature engineering attempts (v38) and deep learning (v39) failed on this dataset

3. Simple Stacking + Isotonic Calibration consistently outperforms complex ensembles

4. The dataset is small enough (24,000 samples) that tree-based models are optimal

================================================================================
TROUBLESHOOTING
================================================================================

Issue: ImportError for lightgbm/catboost
Solution: pip install lightgbm catboost

Issue: Out of Memory error
Solution: Reduce n_estimators in model factories or reduce CV_SEEDS

Issue: scipy installation fails on Windows
Solution: pip install scipy --only-binary :all:

Issue: Matplotlib plot not saving
Solution: Ensure the directory is writable or comment out the plotting section

================================================================================
CONTACT
================================================================================

For questions or issues, please open an issue in the repository.

================================================================================
LAST UPDATED: September 2026
================================================================================