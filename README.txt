================================================================================
                    CREDIT CARD DEFAULT PREDICTION - FINAL
================================================================================

OVERVIEW
================================================================================
This solution uses AutoGluon with v33 feature engineering to achieve a 
competitive Log Loss score. The pipeline leverages AutoGluon's automated 
ensemble learning with 10-fold bagging and dynamic stacking for optimal 
performance.

================================================================================
REQUIRED FILES
================================================================================

Place the following files in the same directory as the script:

- train.csv          Training dataset with columns: client_id, default, and features
- test.csv           Test dataset with columns: client_id and features
- Final_Pipeline_v10(3).py   Main execution script

================================================================================
EXECUTION ORDER
================================================================================

Step 1: Run the script
        python Final_Pipeline_v10(3).py

Step 2: The script automatically performs:
        a) Read and preprocess data
        b) Generate 73 features (v33 feature engineering)
        c) Train AutoGluon with:
           - best_quality preset
           - 10-fold bagging
           - Dynamic stacking enabled
           - 600 seconds time limit
        d) Generate final submission file (submission_v55.csv)

================================================================================
KEY DEPENDENCIES / PACKAGES
================================================================================

Required Python Version: 3.10 - 3.13

Package                      Version       Purpose
---------------------------  -----------   --------------------------------
autogluon                    >= 1.0.0      Core AutoML framework
numpy                        >= 1.21.0     Array operations
pandas                       >= 1.3.0      Data manipulation
scikit-learn                 >= 1.0.0      ML utilities

Installation Command:
--------------------------------------------------------------------------------
pip install autogluon numpy pandas scikit-learn

Note: AutoGluon will automatically install its dependencies including:
- xgboost, lightgbm, catboost (tree models)
- torch (neural network models)
- transformers (for text features if applicable)
- ray (for distributed training)

================================================================================
IMPORTANT SETTINGS
================================================================================

Parameter                  Value                    Description
-------------------------  -----------------------  -----------------------------
RANDOM_STATE               42                       Base random seed
time_limit                 600 seconds (10 min)     Training time limit
presets                    best_quality             High quality ensemble mode
num_bag_folds              10                       10-fold cross-validation
dynamic_stacking           True                     Enable dynamic stacking
eval_metric                log_loss                 Optimization metric
problem_type               binary                   Binary classification

================================================================================
AUTOGLUON CONFIGURATION
================================================================================

AutoGluon automatically trains and ensembles multiple models including:

| Model Type          | Description                                      |
|---------------------|--------------------------------------------------|
| XGBoost             | Gradient boosting with tree models               |
| LightGBM            | Lightweight gradient boosting                    |
| CatBoost            | Gradient boosting with categorical support       |
| Random Forest       | Ensemble of decision trees                       |
| Extra Trees         | Extremely randomized trees                       |
| Neural Networks     | Multi-layer perceptron (PyTorch)                 |
| KNN                 | K-Nearest Neighbors                              |
| Linear Models       | Linear regression with regularization            |

Ensemble Strategy:
- Weighted ensemble of all models (stacking)
- 10-fold bagging for stability
- Dynamic stacking adds a second-level meta-model
- Best model selection based on validation log_loss

================================================================================
FEATURE ENGINEERING SUMMARY (v33)
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
FINAL PREDICTION FILE
================================================================================

Filename:    submission_v55.csv

Format:
--------------------------------------------------------------------------------
client_id,default
CC_0012A082B7B7,0.132542
CC_0012BC27DFB6,0.166306
CC_001563B2143D,0.149123
...

Columns:
- client_id:  Unique client identifier (copied from test.csv)
- default:    Predicted probability of default (clipped to 0.001-0.999 range)

================================================================================
EXECUTION TIME & RESOURCES
================================================================================

Total runtime:   Approximately 10-20 minutes (time_limit=600 seconds)
Memory usage:    Approximately 4-8 GB RAM (AutoGluon can be memory intensive)

================================================================================
REPRODUCTION INSTRUCTIONS
================================================================================

Step-by-step:

1. Download/Clone this repository

2. Install dependencies:
   ----------------------------------------------------------------------------
   pip install autogluon numpy pandas scikit-learn
   ----------------------------------------------------------------------------

3. Place train.csv and test.csv in the same directory as Final_Pipeline_v10(3).py

4. Run the script:
   ----------------------------------------------------------------------------
   python Final_Pipeline_v10(3).py
   ----------------------------------------------------------------------------

5. The script will output:
   - console logs showing AutoGluon training progress
   - AutoGluon model leaderboard
   - submission_v55.csv (final prediction file)

6. Submit submission_v55.csv to the competition platform

================================================================================
TROUBLESHOOTING
================================================================================

Issue: AutoGluon installation fails
Solution: 
   - Use Python 3.10 or 3.11 (most stable)
   - On Windows: Use conda environment for easier installation
   - Try: pip install autogluon --no-cache-dir

Issue: Out of Memory error
Solution:
   - Reduce time_limit or num_bag_folds
   - Use presets="medium_quality" instead of "best_quality"
   - Close other applications to free up RAM

Issue: Long training time
Solution:
   - Reduce time_limit to 300 seconds
   - Reduce num_bag_folds to 5
   - Use presets="medium_quality"

Issue: ValueError: The 'min_samples_leaf' parameter...
Solution:
   - This is an AutoGluon internal issue. Update to latest version:
     pip install --upgrade autogluon
   - Or use Python 3.11 with older version: pip install autogluon==1.0.0

================================================================================
VERSION HISTORY
================================================================================

Version    Method                    Status
--------  -------------------------  --------------------------------
v30        5 models + Stacking       0.41016 (baseline)
v33        Fine-tuned parameters     0.41015 (first improvement)
v42        Optimal parameters        0.41002 (best manual ensemble)
v46        Pseudo-labeling           0.40995 (best overall)
v55        AutoGluon Fixed Pipeline  Production version

================================================================================
NOTES
================================================================================

- This pipeline is optimized for the AutoGluon framework
- AutoGluon automatically handles:
  - Model selection and hyperparameter tuning
  - Feature preprocessing and scaling
  - Ensemble weighting and stacking
  - Cross-validation and bagging
- Probability clipping (0.001-0.999) prevents extreme predictions that can
  harm log_loss
- The v33 feature set was retained as it was the most stable and effective
  feature engineering approach

================================================================================
LAST UPDATED: September 2026
================================================================================