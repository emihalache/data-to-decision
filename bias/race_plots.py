import itertools
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from fairlearn.metrics import equalized_odds_difference

# Define functions for fairness metrics
def compute_demographic_parity_difference(y_true, y_pred, sensitive_features):
    privileged_group = y_pred[sensitive_features == 0]
    unprivileged_group = y_pred[sensitive_features == 1]
    privileged_rate = sum(privileged_group) / len(privileged_group)
    unprivileged_rate = sum(unprivileged_group) / len(unprivileged_group)
    return unprivileged_rate - privileged_rate

def compute_disparate_impact_ratio(y_true, y_pred, sensitive_features):
    privileged_group = y_pred[sensitive_features == 0]
    unprivileged_group = y_pred[sensitive_features == 1]
    privileged_rate = sum(privileged_group) / len(privileged_group)
    unprivileged_rate = sum(unprivileged_group) / len(unprivileged_group)
    return unprivileged_rate / privileged_rate

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
    return np.max(tpr_list) - np.min(tpr_list)

# Define metrics for computation
metrics = ['Demographic Parity Difference', 'Disparate Impact Ratio', 'Equalized Odds Difference', 'Equal Opportunity Difference']

# Data preparation
train_data = pd.read_csv('../preprocess/adult_preprocessed_train.csv')
test_data = pd.read_csv('../preprocess/adult_preprocessed_test.csv')
test_results = pd.read_csv('../algorithms/adult_trained_test_results.csv')

classifiers = ['pred_log_reg', 'pred_dec_tree', 'pred_rand_forest']
classifier_names = {
    'pred_log_reg': 'Logistic Regression',
    'pred_dec_tree': 'Decision Tree',
    'pred_rand_forest': 'Random Forest'
}

races = ['race_Amer-Indian-Eskimo', 'race_Asian-Pac-Islander', 'race_Black', 'race_Other', 'race_White']
race_names = {
    'race_Amer-Indian-Eskimo': 'Amer-Indian-Eskimo',
    'race_Asian-Pac-Islander': 'Asian-Pac-Islander',
    'race_Black': 'Black',
    'race_Other': 'Other',
    'race_White': 'White'
}

# Function to map race to binary sensitive feature
def create_sensitive_feature(race_series, unprivileged_race):
    return (race_series == unprivileged_race).astype(int)

# Prepare a DataFrame to store results
results = []

# Iterate over classifiers and race pairs
for clf in classifiers:
    for (race1, race2) in itertools.combinations(races, 2):
        y_true = test_data['income']
        y_pred = test_results[clf]

        # Filter data to only include the two races being compared
        relevant_data_mask = (test_data[race1] == 1) | (test_data[race2] == 1)
        y_true_filtered = y_true[relevant_data_mask]
        y_pred_filtered = y_pred[relevant_data_mask]
        race_data_filtered = test_data[relevant_data_mask]

        # Compute metrics for race1 as unprivileged and race2 as privileged
        sensitive_features = create_sensitive_feature(race_data_filtered[race1], 1)
        metrics_values_race1 = {
            'Classifier': classifier_names[clf],
            'Unprivileged_Race': race_names[race1],
            'Privileged_Race': race_names[race2],
            'Demographic Parity Difference': compute_demographic_parity_difference(y_true_filtered, y_pred_filtered, sensitive_features),
            'Disparate Impact Ratio': compute_disparate_impact_ratio(y_true_filtered, y_pred_filtered, sensitive_features),
            'Equalized Odds Difference': compute_equalized_odds_difference(y_true_filtered, y_pred_filtered, sensitive_features),
            'Equal Opportunity Difference': compute_equal_opportunity_difference(y_true_filtered, y_pred_filtered, sensitive_features)
        }
        results.append(metrics_values_race1)

        # Compute metrics for race2 as unprivileged and race1 as privileged
        sensitive_features = create_sensitive_feature(race_data_filtered[race2], 1)
        metrics_values_race2 = {
            'Classifier': classifier_names[clf],
            'Unprivileged_Race': race_names[race2],
            'Privileged_Race': race_names[race1],
            'Demographic Parity Difference': compute_demographic_parity_difference(y_true_filtered, y_pred_filtered, sensitive_features),
            'Disparate Impact Ratio': compute_disparate_impact_ratio(y_true_filtered, y_pred_filtered, sensitive_features),
            'Equalized Odds Difference': compute_equalized_odds_difference(y_true_filtered, y_pred_filtered, sensitive_features),
            'Equal Opportunity Difference': compute_equal_opportunity_difference(y_true_filtered, y_pred_filtered, sensitive_features)
        }
        results.append(metrics_values_race2)

# Convert results to DataFrame for further analysis
results_df = pd.DataFrame(results)

# Create a new column for "Unprivileged vs Privileged Race"
results_df['Race Comparison'] = results_df['Unprivileged_Race'] + ' vs ' + results_df['Privileged_Race']

# Ensure the classifiers are in the desired order
results_df['Classifier'] = pd.Categorical(results_df['Classifier'], categories=['Logistic Regression', 'Decision Tree', 'Random Forest'], ordered=True)

# Set the style for the plots
sns.set(style="whitegrid")

# Create a pivot table for heatmap visualization
pivot_data = results_df.pivot_table(index='Race Comparison', columns='Classifier', values=metrics)

# Create heatmaps for each fairness metric
for metric in metrics:
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_data[metric], annot=True, cmap='coolwarm', center=0, linewidths=.5)
    plt.title(f'Heatmap of {metric}')
    plt.ylabel('Race Comparison')
    plt.xlabel('Classifier')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
