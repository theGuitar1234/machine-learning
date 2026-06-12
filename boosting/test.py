from boosting.catboost import CatBoost
from boosting.gradient_boosting import GradientBoosting
from boosting.xgboost import XGBoost
from calibration.data.load_streambox_churn_npz_example import load_stream_churn_npz

import sys

from trees.a_non_serious_decision_tree import ANonSeriousDecisionTree

number_of_classes = 2
seed = 42

(
    X,
    Y,
    X_train,
    Y_train,
    X_valid,
    Y_valid,
    X_calib,
    Y_calib,
    X_test,
    Y_test,
    feature_names,
    categorical_indices,
    metadata,
) = load_stream_churn_npz(verbose=True)

feature1 = 0
feature2 = 1

boosting_rounds = 20
max_depth = 6
minimum_population_size=50
sub_sample = 0.5
column_sub_sample = 0.6

gradient_boost = GradientBoosting(
    boosting_rounds=boosting_rounds,
    max_depth=max_depth,
    minimum_population_size=minimum_population_size,
    minimum_gain=0.001,
    information_gain=0.01,
    sub_sample=sub_sample,
    column_sub_sample=column_sub_sample,
    loss_type=GradientBoosting.LossType.BINARY_CROSS_ENTROPY,
    restore_best=True,
    validation=True,
    early_stopping=True,
)
gradient_boost.fit(
    X=X_train,
    y=Y_train,
    X_test=X_test,
    y_test=Y_test,
    X_val=X_valid,
    y_val=Y_valid,
    finalize=True,
    log=True,
)
model_name = "trained_gradient_boost_churn"
gradient_boost.save_model(model_name)

xgboost = XGBoost(
    boosting_rounds=boosting_rounds,
    max_depth=max_depth,
    minimum_population_size=minimum_population_size,
    sub_sample=sub_sample,
    column_sub_sample=column_sub_sample,
    loss_type=XGBoost.LossType.BINARY_CROSS_ENTROPY,
    restore_best=True,
    early_stopping=True,
    log=True,
    xgboost_split=ANonSeriousDecisionTree.XGBoostSplit.EXACT,
    proposal=ANonSeriousDecisionTree.XGBoostProposal.LOCAL,
    candidate_proposal=ANonSeriousDecisionTree.XGBoostCandidate.WEIGHTED_QUANTILE,
    missing_value=None,
    purpose_missing=False,
    vectorized=True,
    preprocess=True,
    batch_training=True,
)
xgboost.fit(
    X=X_train, 
    y=Y_train, 
    X_test=X_test,
    y_test=Y_test,
    X_val=X_valid, 
    y_val=Y_valid, 
    optimized=True,
    finalize=True,
)
model_name = "trained_xgboost_churn"
xgboost.save_model(model_name)

catboost = CatBoost(
    boosting_rounds=boosting_rounds,
    max_depth=max_depth,
    minimum_population_size=minimum_population_size,
    minimum_gain=0.001,
    loss_type=CatBoost.LossType.BINARY_CROSS_ENTROPY,
    restore_best=True,
    validation=True,
    early_stopping=True,
    sub_sample=sub_sample,
    column_sub_sample=column_sub_sample,
    symmetrical=False,
    boosting_type=CatBoost.BoostingType.PLAIN,
    categorical_features=list(categorical_indices),
)
catboost.fit(
    X=X_train, 
    y=Y_train,
    X_test=X_test,
    y_test=Y_test, 
    X_val=X_valid, 
    y_val=Y_valid,
    finalize=True,
    log=True,
)
model_name = "trained_catboost_churn"
catboost.save_model(model_name)
