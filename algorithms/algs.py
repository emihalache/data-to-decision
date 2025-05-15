#%%
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the preprocessed training and test data
train_data = pd.read_csv('C:/Users/40766/Desktop/research_project/preprocess/adult_preprocessed_train.csv')
test_data = pd.read_csv('C:/Users/40766/Desktop/research_project/preprocess/adult_preprocessed_test.csv')

# Split the data into features (X) and target (y)
X_train = train_data.drop('income', axis=1)
y_train = train_data['income']
X_test = test_data.drop('income', axis=1)
y_test = test_data['income']

#automatically compare feature names to ensure they are the same
assert X_train.columns.equals(X_test.columns)


# Initialize algorithms
log_reg = LogisticRegression(max_iter=1000)
dec_tree = DecisionTreeClassifier(random_state=42)
rand_forest = RandomForestClassifier(random_state=42)

# Train and test Logistic Regression
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)
print('Logistic Regression Accuracy:', accuracy_score(y_test, y_pred_log_reg))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred_log_reg))
print('Classification Report:\n', classification_report(y_test, y_pred_log_reg))

# Train and test Decision Tree
dec_tree.fit(X_train, y_train)
y_pred_dec_tree = dec_tree.predict(X_test)
print('Decision Tree Accuracy:', accuracy_score(y_test, y_pred_dec_tree))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred_dec_tree))
print('Classification Report:\n', classification_report(y_test, y_pred_dec_tree))

# Train and test Random Forest
rand_forest.fit(X_train, y_train)
y_pred_rand_forest = rand_forest.predict(X_test)
print('Random Forest Accuracy:', accuracy_score(y_test, y_pred_rand_forest))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred_rand_forest))
print('Classification Report:\n', classification_report(y_test, y_pred_rand_forest))

# Collecting accuracy scores
accuracies = {
    'Logistic Regression': accuracy_score(y_test, y_pred_log_reg),
    'Decision Tree': accuracy_score(y_test, y_pred_dec_tree),
    'Random Forest': accuracy_score(y_test, y_pred_rand_forest)
}

#%%
# test_data['score_log_reg'] = log_reg.predict_proba(X_test)[:, 1]
test_data['pred_log_reg'] = y_pred_log_reg
test_data['pred_dec_tree'] = y_pred_dec_tree
test_data['pred_rand_forest'] = y_pred_rand_forest
test_data.to_csv('C:/Users/40766/Desktop/research_project/algorithms/adult_trained_test_results.csv', index=False)

#%%
# Bar Plot for Accuracy Comparison
plt.figure(figsize=(5, 6))
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()))
plt.title('Accuracy Comparison')
plt.ylabel('Accuracy Score')
plt.xlabel('Algorithm')
plt.ylim(0, 1)
plt.show()

#%%
# Filter columns for sex and race
sex_columns = [col for col in test_data.columns if col.startswith('sex_')]
race_columns = [col for col in test_data.columns if col.startswith('race_')]

# Prepare a dictionary to hold accuracies
accuracies_by_group = {}

# Calculate accuracies for each sex
for sex in sex_columns:
    group_data = test_data[test_data[sex] == 1]
    accuracies_by_group[sex.replace('sex_', '')] = {
        'Logistic Regression': accuracy_score(group_data['income'], group_data['pred_log_reg']),
        'Decision Tree': accuracy_score(group_data['income'], group_data['pred_dec_tree']),
        'Random Forest': accuracy_score(group_data['income'], group_data['pred_rand_forest'])
    }

# Calculate accuracies for each race
for race in race_columns:
    group_data = test_data[test_data[race] == 1]
    accuracies_by_group[race.replace('race_', '')] = {
        'Logistic Regression': accuracy_score(group_data['income'], group_data['pred_log_reg']),
        'Decision Tree': accuracy_score(group_data['income'], group_data['pred_dec_tree']),
        'Random Forest': accuracy_score(group_data['income'], group_data['pred_rand_forest'])
    }

# Convert accuracies_by_group to a DataFrame for easier plotting
acc_df = pd.DataFrame.from_dict(accuracies_by_group, orient='index')
acc_df = acc_df.reset_index().melt(id_vars="index", var_name="Model", value_name="Accuracy")
acc_df.rename(columns={'index': 'Group'}, inplace=True)

# Separate data for sexes and races
sex_df = acc_df[acc_df['Group'].isin(['Female', 'Male'])]
race_df = acc_df[~acc_df['Group'].isin(['Female', 'Male'])]

# Function to plot each group
# Function to plot each group
def plot_group(data, title):
    plt.figure(figsize=(10, 6))
    bar_plot = sns.barplot(x="Group", y="Accuracy", hue="Model", data=data)
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Group')
    plt.ylim(0, 1)
    # Add values on bars, avoid annotating zero values
    for p in bar_plot.patches:
        height = p.get_height()
        if height > 0:  # Ensures we only annotate non-zero values
            bar_plot.annotate(format(height, '.2f'),
                              (p.get_x() + p.get_width() / 2., height),
                              ha = 'center', va = 'center',
                              xytext = (0, 9),
                              textcoords = 'offset points')
    plt.legend(title='Classifier', loc='lower right')
    plt.show()


# Plot for sexes and races separately
plot_group(sex_df, 'Accuracy Comparison by Sex')
plot_group(race_df, 'Accuracy Comparison by Race')


#%%
# Confusion Matrix Heatmap Function
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix: {title}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

# Plotting Confusion Matrices
plot_confusion_matrix(confusion_matrix(y_test, y_pred_log_reg), 'Logistic Regression')
plot_confusion_matrix(confusion_matrix(y_test, y_pred_dec_tree), 'Decision Tree')
plot_confusion_matrix(confusion_matrix(y_test, y_pred_rand_forest), 'Random Forest')

# Display Classification Reports
print('Classification Report for Logistic Regression:\n', classification_report(y_test, y_pred_log_reg))
print('Classification Report for Decision Tree:\n', classification_report(y_test, y_pred_dec_tree))
print('Classification Report for Random Forest:\n', classification_report(y_test, y_pred_rand_forest))
