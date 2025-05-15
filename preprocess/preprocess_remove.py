#%%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns

# Load the dataset
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
           'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']
data = pd.read_csv('C:/Users/40766/Desktop/research_project/adult/adult.data', names=columns, sep=',\s*', engine='python')

data = data.drop_duplicates()

# Remove column "education" as it is redundant with "education_num"
data = data.drop(columns=['education'])

# Remove column "fnlwgt" as it is not useful for the analysis
data = data.drop(columns=['fnlwgt'])

data.loc[data['workclass'] == 'Never-worked', 'occupation'] = 'None'

# Replace 'Unknown' or '?' with NaN (if applicable)
data.replace('?', np.nan, inplace=True)

# Drop rows with missing values
data.dropna(inplace=True)


# Convert the 'income' column to a binary numeric variable
data['income'] = data['income'].apply(lambda x: 1 if x == '>50K' else 0)

# Plotting distributions for each numeric feature
data.hist(figsize=(20,15))
plt.show()

data_before = data.copy()

# Use one-hot encoding for categorical features
data = pd.get_dummies(data, columns=['workclass', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country'])

# Scale the numeric features
scaler = StandardScaler()
numeric_columns = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

# Save the preprocessed data to a new CSV file in current directory
# data.to_csv('adult_preprocessed_remove.csv', index=False)
data.to_csv('C:/Users/40766/Desktop/research_project/preprocess/adult_preprocessed_train.csv', index=False)

# Number of features
print("Number of features after preprocessing:", data.shape[1])
#print number of entries
print("Number of entries after preprocessing:", data.shape[0])


print("Preprocessing completed for train data!")

test_data = pd.read_csv('C:/Users/40766/Desktop/research_project/adult/adult.test', names=columns, sep=',\s*', engine='python', skiprows=1)

test_data = test_data.drop_duplicates()

# Remove column "education" as it is redundant with "education_num"
test_data = test_data.drop(columns=['education'])

# Remove column "fnlwgt" as it is not useful for the analysis
test_data = test_data.drop(columns=['fnlwgt'])

# Replace 'Unknown' or '?' with NaN (if applicable)
test_data.replace('?', np.nan, inplace=True)

# Drop rows with missing values
test_data.dropna(inplace=True)

# Convert the 'income' column to a binary numeric variable
test_data['income'] = test_data['income'].apply(lambda x: 1 if x == '>50K.' else 0)  # Note the '.' after '>50K' in the test file

test_data_before = test_data.copy()

# Use one-hot encoding for categorical features
test_data = pd.get_dummies(test_data, columns=['workclass', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country'])

# Ensure the test set has the same dummy variable columns as the training set
missing_cols = set(data.columns) - set(test_data.columns)
print("Missing columns in test data:", missing_cols)
for col in missing_cols:
    test_data[col] = 0
test_data = test_data[data.columns]

# Scale the numeric features using the same scaler as for training data
test_data[numeric_columns] = scaler.transform(test_data[numeric_columns])


# Save the preprocessed test data to a new CSV file
test_data.to_csv('C:/Users/40766/Desktop/research_project/preprocess/adult_preprocessed_test.csv', index=False)

print("Preprocessing completed for test data!")


#%% Calculate the percentage income distribution
income_counts = data_before['income'].value_counts(normalize=True) * 100

print("Percentage Income Distribution:")
print(income_counts)

#%%
# Calculate the percentages of males and females
sex_counts = data_before['sex'].value_counts(normalize=True) * 100

print("Percentage of Males Compared to Females:")
print(sex_counts)

#%%
# Calculate the percentages of each race
race_counts = data_before['race'].value_counts(normalize=True) * 100

print("Percentage of Each Race:")
print(race_counts)


#%%
# Histograms of numeric features
fig, ax = plt.subplots(2, 3, figsize=(15, 10))
ax = ax.ravel()
for i, col in enumerate(numeric_columns):
    bins = np.linspace(min(data_before[col].min(), data[col].min()), max(data_before[col].max(), data[col].max()), 30)
    ax[i].hist(data_before[col], bins=bins, alpha=0.5, color='blue', label='Before Scaling')
    ax[i].hist(data[col], bins=bins, alpha=0.5, color='red', label='After Scaling')
    ax[i].set_title(col)
    ax[i].legend()

plt.tight_layout()
plt.show()

#%%
#  Boxplots of numeric features
fig, ax = plt.subplots(2, 3, figsize=(15, 10))
ax = ax.ravel()
for i, col in enumerate(numeric_columns):
    ax[i].boxplot([data_before[col], data[col]], labels=['Before', 'After'])
    ax[i].set_title(col)

plt.tight_layout()
plt.show()

#%%
# Calculate correlation matrices
corr_before = data_before[numeric_columns].corr()
corr_after = data[numeric_columns].corr()
# Plot heatmaps of correlations
fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
sns.heatmap(corr_before, annot=True, fmt=".2f", cmap='coolwarm', ax=ax[0])
ax[0].set_title('Correlation Before Scaling')
ax[0].tick_params(axis='x', rotation=45)
sns.heatmap(corr_after, annot=True, fmt=".2f", cmap='coolwarm', ax=ax[1])
ax[1].set_title('Correlation After Scaling')
ax[1].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()

#%%
# Overall Income Distribution
plt.figure(figsize=(8, 4))
sns.countplot(x='income', data=data_before)
plt.title('Overall Income Distribution')
plt.xlabel('Income')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['<=50K', '>50K'])
plt.show()

#%%
# Race and Sex Distribution by Income
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
sns.countplot(x='race', hue='income', data=data_before, ax=axes[0])
axes[0].set_title('Race Distribution by Income')
axes[0].tick_params(axis='x', rotation=45)
axes[0].legend(title='Income')
axes[0].set_xlabel('Race')
axes[0].set_ylabel('Count')
sns.countplot(x='sex', hue='income', data=data_before, ax=axes[1])
axes[1].set_title('Sex Distribution by Income')
axes[1].tick_params(axis='x', rotation=45)
axes[1].legend(title='Income')
axes[1].set_xlabel('Sex')
axes[1].set_ylabel('Count')
axes[1].get_legend().get_texts()[0].set_text('<=50K')
axes[1].get_legend().get_texts()[1].set_text('>50K')
axes[0].get_legend().get_texts()[0].set_text('<=50K')
axes[0].get_legend().get_texts()[1].set_text('>50K')
plt.tight_layout()
plt.show()

# Calculate the percentages for Race and Income
race_income_percentage = data_before.groupby('race')['income'].value_counts(normalize=True).unstack().fillna(0) * 100
race_income_percentage.columns = ['<=50K', '>50K']

# Calculate the percentages for Sex and Income
sex_income_percentage = data_before.groupby('sex')['income'].value_counts(normalize=True).unstack().fillna(0) * 100
sex_income_percentage.columns = ['<=50K', '>50K']

# Print the percentage tables
print("Race and Income Distribution:")
print(race_income_percentage)
print("\nSex and Income Distribution:")
print(sex_income_percentage)



#%%
# Different Working Classes Distribution by Income
fig, ax = plt.subplots(figsize=(12, 6))
sns.countplot(x='workclass', hue='income', data=data_before)
ax.set_title('Income of Individuals of Different Working Classes')
ax.tick_params(axis='x', rotation=45)
ax.legend(title='Income')
ax.set_xlabel('Workclass')
ax.set_ylabel('Count')
ax.get_legend().get_texts()[0].set_text('<=50K')
ax.get_legend().get_texts()[1].set_text('>50K')
plt.tight_layout()
plt.show()

#%%
# Marital Status Distribution by Income
plt.figure(figsize=(12, 6))
sns.countplot(x='marital_status', hue='income', data=data_before)
plt.title('Income Distribution by Marital Status')
plt.xticks(rotation=45)
plt.legend(title='Income')
plt.xlabel('Marital Status')
plt.ylabel('Count')
plt.gca().get_legend().get_texts()[0].set_text('<=50K')
plt.gca().get_legend().get_texts()[1].set_text('>50K')
plt.tight_layout()
plt.show()

#%%
# # Education Level Distribution by Income
# plt.figure(figsize=(12, 6))
# sns.countplot(x='education', hue='income', data=data_before)
# plt.title('Income Distribution by Education Level')
# plt.xticks(rotation=45)
# plt.legend(title='Income')
# plt.xlabel('Education Level')
# plt.ylabel('Count')
# plt.tight_layout()
# plt.show()

#%%
# Education Num Distribution by Income
plt.figure(figsize=(12, 6))
sns.countplot(x='education_num', hue='income', data=data_before)
plt.title('Income Distribution by Education Num')
plt.xticks(rotation=45)
plt.legend(title='Income')
plt.xlabel('Education Num')
plt.ylabel('Count')
plt.gca().get_legend().get_texts()[0].set_text('<=50K')
plt.gca().get_legend().get_texts()[1].set_text('>50K')
plt.tight_layout()
plt.show()

#%%
# Income Distribution by Relationship Status
plt.figure(figsize=(12, 6))
sns.countplot(x='relationship', hue='income', data=data_before)
plt.title('Income Distribution by Relationship Status')
plt.xticks(rotation=45)
plt.legend(title='Income')
plt.xlabel('Relationship Status')
plt.ylabel('Count')
plt.gca().get_legend().get_texts()[0].set_text('<=50K')
plt.gca().get_legend().get_texts()[1].set_text('>50K')
plt.tight_layout()
plt.show()

#%%
# Work Hours per Week Distribution by Income
plt.figure(figsize=(12, 6))
bins_hours = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # Custom bins for work hours
data_before['hours_per_week_bins'] = pd.cut(data_before['hours_per_week'], bins=bins_hours)
sns.countplot(x='hours_per_week_bins', hue='income', data=data_before)
plt.title('Work Hours per Week Distribution by Income')
plt.xlabel('Hours per Week')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Income')
plt.gca().get_legend().get_texts()[0].set_text('<=50K')
plt.gca().get_legend().get_texts()[1].set_text('>50K')
plt.tight_layout()
plt.show()

#%%
# Age Group Distribution by Income
plt.figure(figsize=(12, 6))
bins_age = [0, 25, 45, 65, 100]  # Custom bins for age groups
labels_age = ['Young (17-25)', 'Adult (26-45)', 'Middle-aged (46-65)', 'Senior (66-99)']
data_before['age_groups'] = pd.cut(data_before['age'], bins=bins_age, labels=labels_age)
sns.countplot(x='age_groups', hue='income', data=data_before)
plt.title('Age Group Distribution by Income')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Income')
plt.gca().get_legend().get_texts()[0].set_text('<=50K')
plt.gca().get_legend().get_texts()[1].set_text('>50K')
plt.tight_layout()
plt.show()









#%%
# Histograms of numeric features
fig, ax = plt.subplots(2, 3, figsize=(15, 10))
ax = ax.ravel()
for i, col in enumerate(numeric_columns):
    bins = np.linspace(min(test_data_before[col].min(), test_data[col].min()), max(test_data_before[col].max(), test_data[col].max()), 30)
    ax[i].hist(test_data_before[col], bins=bins, alpha=0.5, color='blue', label='Before Scaling')
    ax[i].hist(test_data[col], bins=bins, alpha=0.5, color='red', label='After Scaling')
    ax[i].set_title(col)
    ax[i].legend()

plt.tight_layout()
plt.show()

#%%
#  Boxplots of numeric features
fig, ax = plt.subplots(2, 3, figsize=(15, 10))
ax = ax.ravel()
for i, col in enumerate(numeric_columns):
    ax[i].boxplot([test_data_before[col], test_data[col]], labels=['Before', 'After'])
    ax[i].set_title(col)

plt.tight_layout()
plt.show()

#%%
# Calculate correlation matrices
corr_before = test_data_before[numeric_columns].corr()
corr_after = test_data[numeric_columns].corr()
# Plot heatmaps of correlations
fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
sns.heatmap(corr_before, annot=True, fmt=".2f", cmap='coolwarm', ax=ax[0])
ax[0].set_title('Correlation Before Scaling')
ax[0].tick_params(axis='x', rotation=45)
sns.heatmap(corr_after, annot=True, fmt=".2f", cmap='coolwarm', ax=ax[1])
ax[1].set_title('Correlation After Scaling')
ax[1].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()

#%%
# Overall Income Distribution
plt.figure(figsize=(8, 4))
sns.countplot(x='income', data=test_data_before)
plt.title('Overall Income Distribution')
plt.xlabel('Income')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['<=50K', '>50K'])
plt.show()

#%%
# Race and Sex Distribution by Income
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
sns.countplot(x='race', hue='income', data=test_data_before, ax=axes[0])
axes[0].set_title('Race Distribution by Income')
axes[0].tick_params(axis='x', rotation=45)
axes[0].legend(title='Income')
axes[0].set_xlabel('Race')
axes[0].set_ylabel('Count')
sns.countplot(x='sex', hue='income', data=test_data_before, ax=axes[1])
axes[1].set_title('Sex Distribution by Income')
axes[1].tick_params(axis='x', rotation=45)
axes[1].legend(title='Income')
axes[1].set_xlabel('Race')
axes[1].set_ylabel('Count')
axes[1].get_legend().get_texts()[0].set_text('<=50K')
axes[1].get_legend().get_texts()[1].set_text('>50K')
axes[0].get_legend().get_texts()[0].set_text('<=50K')
axes[0].get_legend().get_texts()[1].set_text('>50K')
plt.tight_layout()
plt.show()


#%%
# Different Working Classes Distribution by Income
fig, ax = plt.subplots(figsize=(12, 6))
sns.countplot(x='workclass', hue='income', data=test_data_before)
ax.set_title('Income of Individuals of Different Working Classes')
ax.tick_params(axis='x', rotation=45)
ax.legend(title='Income')
ax.set_xlabel('Workclass')
ax.set_ylabel('Count')
ax.get_legend().get_texts()[0].set_text('<=50K')
ax.get_legend().get_texts()[1].set_text('>50K')
plt.tight_layout()
plt.show()

#%%
# Marital Status Distribution by Income
plt.figure(figsize=(12, 6))
sns.countplot(x='marital_status', hue='income', data=test_data_before)
plt.title('Income Distribution by Marital Status')
plt.xticks(rotation=45)
plt.legend(title='Income')
plt.xlabel('Marital Status')
plt.ylabel('Count')
plt.gca().get_legend().get_texts()[0].set_text('<=50K')
plt.gca().get_legend().get_texts()[1].set_text('>50K')
plt.tight_layout()
plt.show()

#%%
# # Education Level Distribution by Income
# plt.figure(figsize=(12, 6))
# sns.countplot(x='education', hue='income', data=data_before)
# plt.title('Income Distribution by Education Level')
# plt.xticks(rotation=45)
# plt.legend(title='Income')
# plt.xlabel('Education Level')
# plt.ylabel('Count')
# plt.tight_layout()
# plt.show()

#%%
# Education Num Distribution by Income
plt.figure(figsize=(12, 6))
sns.countplot(x='education_num', hue='income', data=test_data_before)
plt.title('Income Distribution by Education Years')
plt.xticks(rotation=45)
plt.legend(title='Income')
plt.xlabel('Education Years')
plt.ylabel('Count')
plt.gca().get_legend().get_texts()[0].set_text('<=50K')
plt.gca().get_legend().get_texts()[1].set_text('>50K')
plt.tight_layout()
plt.show()

#%%
# Income Distribution by Relationship Status
plt.figure(figsize=(12, 6))
sns.countplot(x='relationship', hue='income', data=test_data_before)
plt.title('Income Distribution by Relationship Status')
plt.xticks(rotation=45)
plt.legend(title='Income')
plt.xlabel('Relationship Status')
plt.ylabel('Count')
plt.gca().get_legend().get_texts()[0].set_text('<=50K')
plt.gca().get_legend().get_texts()[1].set_text('>50K')
plt.tight_layout()
plt.show()

#%%
# Work Hours per Week Distribution by Income
plt.figure(figsize=(12, 6))
bins_hours = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # Custom bins for work hours
test_data_before['hours_per_week_bins'] = pd.cut(test_data_before['hours_per_week'], bins=bins_hours)
sns.countplot(x='hours_per_week_bins', hue='income', data=test_data_before)
plt.title('Work Hours per Week Distribution by Income')
plt.xlabel('Hours per Week')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Income')
plt.gca().get_legend().get_texts()[0].set_text('<=50K')
plt.gca().get_legend().get_texts()[1].set_text('>50K')
plt.tight_layout()
plt.show()

#%%
# Age Group Distribution by Income
plt.figure(figsize=(12, 6))
bins_age = [0, 25, 45, 65, 100]  # Custom bins for age groups
labels_age = ['Young (17-25)', 'Adult (26-45)', 'Middle-aged (46-65)', 'Senior (66-99)']
test_data_before['age_groups'] = pd.cut(test_data_before['age'], bins=bins_age, labels=labels_age)
sns.countplot(x='age_groups', hue='income', data=test_data_before)
plt.title('Age Group Distribution by Income')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Income')
plt.gca().get_legend().get_texts()[0].set_text('<=50K')
plt.gca().get_legend().get_texts()[1].set_text('>50K')
plt.tight_layout()
plt.show()