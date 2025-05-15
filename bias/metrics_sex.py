#%%
from fairlearn.metrics import (
    equalized_odds_difference,  # To measure equalized odds
    demographic_parity_difference, MetricFrame, demographic_parity_ratio,
    equalized_odds_ratio,  # To measure demographic parity
)
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix
import numpy as np


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


def main():
    # Load the data
    data = pd.read_csv('C:/Users/40766/Desktop/research_project/preprocess/adult_preprocessed_test.csv')
    trained_data = pd.read_csv('C:/Users/40766/Desktop/research_project/algorithms/adult_trained_test_results.csv')

    y_true = data['income']

    sensitive_attribute = data['sex_Female']

    # Iterate over each algorithm's predictions that are called 'pred_{algorithm}'
    algorithms = trained_data.columns[90:]
    print("Algorithms:", algorithms)
    for alg in algorithms:
        print(f"Processing {alg}...")
        y_pred = trained_data[alg]

        # Calculate metrics
        dpd = compute_demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_attribute)
        eod = compute_equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_attribute)
        di = compute_disparate_impact_ratio(y_true, y_pred, sensitive_features=sensitive_attribute)
        eopd = compute_equal_opportunity_difference(y_true, y_pred, sensitive_features=sensitive_attribute)

        # Print results for each algorithm
        print(f"Results for {alg}:")
        print("Demographic Parity Difference:", dpd)
        print("Equalized Odds Difference:", eod)
        print("Disparate Impact Ratio:", di)
        print("Equal Opportunity Difference:", eopd)
        print("\n")



if __name__ == "__main__":
    main()