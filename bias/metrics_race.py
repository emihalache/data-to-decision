import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from fairlearn.metrics import (
    equalized_odds_difference,  # To measure equalized odds
    demographic_parity_difference, demographic_parity_ratio,
)

def compute_demographic_parity_difference(y_true, y_pred, sensitive_features):
    # return demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features)
    # Define the privileged and unprivileged groups
    privileged_group = y_pred[sensitive_features == 0]
    unprivileged_group = y_pred[sensitive_features == 1]

    # Calculate the rate of favorable outcomes for each group
    privileged_rate = sum(privileged_group) / len(privileged_group)
    unprivileged_rate = sum(unprivileged_group) / len(unprivileged_group)

    # Calculate the Demographic Parity Difference
    demographic_parity_diff = unprivileged_rate - privileged_rate

    return demographic_parity_diff

def compute_disparate_impact_ratio(y_true, y_pred, sensitive_features):
    # Define the privileged and unprivileged groups
    privileged_group = y_pred[sensitive_features == 0]
    unprivileged_group = y_pred[sensitive_features == 1]

    # Calculate the rate of favorable outcomes for each group
    privileged_rate = sum(privileged_group) / len(privileged_group)
    unprivileged_rate = sum(unprivileged_group) / len(unprivileged_group)

    # Calculate the Disparate Impact
    disparate_impact = unprivileged_rate / privileged_rate

    return disparate_impact

def compute_equalized_odds_difference(y_true, y_pred, sensitive_features):
    return equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_features)

def compute_equal_opportunity_difference(y_true, y_pred, sensitive_features):
    groups = np.unique(sensitive_features)
    tpr_list = []
    for group in groups:
        group_mask = (sensitive_features == group)
        tn, fp, fn, tp = confusion_matrix(y_true[group_mask], y_pred[group_mask]).ravel()
        tpr = tp / (tp + fn) if (tp + fn) != 0 else 0
        tpr_list.append(tpr)
    tpr_diff = np.max(tpr_list) - np.min(tpr_list)
    return tpr_diff


# Define metrics for computation
metrics = ['Demographic Parity Difference', 'Disparate Impact Ratio', 'Equalized Odds Difference', 'Equal Opportunity Difference']

# Define function to compute all metrics
def compute_metrics(y_true, y_pred, sensitive_features):
    return {
        'Demographic Parity Difference': compute_demographic_parity_difference(y_true, y_pred, sensitive_features),
        'Disparate Impact Ratio': compute_disparate_impact_ratio(y_true, y_pred, sensitive_features),
        'Equal Opportunity Difference': compute_equal_opportunity_difference(y_true, y_pred, sensitive_features),
        'Equalized Odds Difference': compute_equalized_odds_difference(y_true, y_pred, sensitive_features)
    }

# Data preparation
train_data = pd.read_csv('../preprocess/adult_preprocessed_train.csv')
test_data = pd.read_csv('../preprocess/adult_preprocessed_test.csv')
test_results = pd.read_csv('../algorithms/adult_trained_test_results.csv')

# classifiers = ['pred_log_reg', 'pred_dec_tree', 'pred_rand_forest']
classifiers = ['pred_log_reg']
races = ['race_Amer-Indian-Eskimo', 'race_Asian-Pac-Islander', 'race_Black', 'race_Other', 'race_White']
overall_results = {}
