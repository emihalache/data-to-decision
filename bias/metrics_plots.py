# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from fairlearn.metrics import (
    equalized_odds_difference,  # To measure equalized odds
)


# Define metric computation functions
def compute_demographic_parity_difference(y_true, y_pred, sensitive_features):
    privileged_group = y_pred[sensitive_features == 0]
    unprivileged_group = y_pred[sensitive_features == 1]
    privileged_rate = sum(privileged_group) / len(privileged_group)
    unprivileged_rate = sum(unprivileged_group) / len(unprivileged_group)
    demographic_parity_diff = unprivileged_rate - privileged_rate
    return demographic_parity_diff


def compute_disparate_impact_ratio(y_true, y_pred, sensitive_features):
    privileged_group = y_pred[sensitive_features == 0]
    unprivileged_group = y_pred[sensitive_features == 1]
    privileged_rate = sum(privileged_group) / len(privileged_group)
    unprivileged_rate = sum(unprivileged_group) / len(unprivileged_group)
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
metrics = ['Demographic Parity Difference', 'Disparate Impact Ratio', 'Equalized Odds Difference',
           'Equal Opportunity Difference']


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

classifiers = ['pred_log_reg', 'pred_dec_tree', 'pred_rand_forest']
races = ['race_Amer-Indian-Eskimo', 'race_Asian-Pac-Islander', 'race_Black', 'race_Other', 'race_White']
overall_results = {clf: {} for clf in classifiers}

# Process the data
y_true = test_data['income']  # Assuming 'income' is the target column

# Iterate over all classifiers and combinations of races, calculating metrics for each direction
for clf in classifiers:
    y_pred = test_results[clf]  # Update to use current classifier's predictions
    for i in range(len(races)):
        for j in range(len(races)):
            if i != j:  # Ensure we're not comparing the same race to itself
                # Prepare sensitive features for the current pair of races
                sensitive_features = np.where(
                    test_data[races[i]] == 1, 0,
                    np.where(test_data[races[j]] == 1, 1, np.nan)
                )

                # Filter out the rows not belonging to either of the two races
                mask = ~np.isnan(sensitive_features)
                filtered_y_true = y_true[mask]
                filtered_y_pred = y_pred[mask]
                filtered_sensitive_features = sensitive_features[mask]

                # Compute fairness metrics
                results = compute_metrics(filtered_y_true, filtered_y_pred, filtered_sensitive_features)

                # Store results, distinguishing the direction of comparison
                pair_key = f"{races[i]} vs {races[j]}"
                if pair_key not in overall_results[clf]:
                    overall_results[clf][pair_key] = {}
                overall_results[clf][pair_key] = results


# Define function to plot heatmaps for each classifier
def plot_heatmaps_for_classifier(classifier, results, metrics, races):
    # Adjust race labels by removing the "race_" prefix
    formatted_races = [race.replace("race_", "") for race in races]

    # Create a new figure for each classifier with a 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    axes = axes.flatten()  # Flatten the axes array for easy indexing

    # Initialize matrices to store the metric values for each pair
    metric_matrices = {metric: np.full((len(races), len(races)), np.nan) for metric in metrics}

    # Fill the matrices with the computed metric values, leaving diagonals as NaN
    for (i, race_i) in enumerate(races):
        for (j, race_j) in enumerate(races):
            if i != j:  # Skip diagonal entries
                results_key = f"{race_i} vs {race_j}"
                if results_key in results:
                    for metric in metrics:
                        metric_matrices[metric][i, j] = results[results_key][metric]

    # Plot the heatmaps for each metric
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        sns.heatmap(metric_matrices[metric], annot=True, fmt=".2f", ax=ax, cmap='coolwarm',
                    xticklabels=formatted_races, yticklabels=formatted_races, cbar_kws={'shrink': .82}, mask=np.isnan(metric_matrices[metric]))
        ax.set_title(f"{metric}")
        ax.set_xlabel('Unprivileged Group')
        # rotate x labels for better visibility
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_ylabel('Privileged Group')

    # Set the overall title with the classifier's name properly formatted
    if classifier == 'pred_log_reg':
        classifier_name = 'Logistic Regression'
    elif classifier == 'pred_dec_tree':
        classifier_name = 'Decision Tree'
    else:
        classifier_name = 'Random Forest'
    plt.suptitle(f"Comparison of Algorithm Fairness Metrics for {classifier_name} for Sensitive Attribute \'race\'", fontsize=15)
    plt.tight_layout()
    plt.show()

# Iterate over each classifier and plot the heatmaps
for clf in classifiers:
    plot_heatmaps_for_classifier(clf, overall_results[clf], metrics, races)
