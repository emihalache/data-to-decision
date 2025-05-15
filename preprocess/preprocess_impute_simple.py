#%%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns

# Adjust Pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Load the dataset
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
           'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']
data = pd.read_csv('../adult/adult.data', names=columns, sep=',\s*', engine='python')

data = data.drop_duplicates()

# Remove column "education" as it is redundant with "education_num"
data = data.drop(columns=['education'])

# Remove column "fnlwgt" as it is not useful for the analysis
data = data.drop(columns=['fnlwgt'])

data.replace('?', np.nan, inplace=True)

data.loc[data['workclass'] == 'Never-worked', 'occupation'] = 'None'

missing_value_columns = data.columns[data.isnull().any()].tolist()

imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
data[missing_value_columns] = imputer.fit_transform(data[missing_value_columns])

data.hist(figsize=(20,15))
plt.show()

data_before = data.copy()

# Save the data to a new CSV file before encoding and scaling
data_before.to_csv('adult_before_impute_simple.csv', index=False)

# Use one-hot encoding for categorical features
data = pd.get_dummies(data, columns=['workclass', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country', 'income'])

# Scale the numeric features
scaler = StandardScaler()
numeric_columns = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

# Number of features
print("Number of features after preprocessing:", data.shape[1])
#print number of entries
print("Number of entries after preprocessing:", data.shape[0])

# Save the preprocessed data to a new CSV file
data.to_csv('adult_preprocessed_impute_simple.csv', index=False)

print("Preprocessing completed!")

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
# Heatmaps for Correlation Changes
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
axes[1].set_xlabel('Race')
axes[1].set_ylabel('Count')
plt.tight_layout()
plt.show()


#%%
# Different Working Classes Distribution by Income
fig, ax = plt.subplots(figsize=(12, 6))
sns.countplot(x='workclass', hue='income', data=data_before)
ax.set_title('Income of Individuals of Different Working Classes')
ax.tick_params(axis='x', rotation=45)
ax.legend(title='Income')
ax.set_xlabel('Workclass')
ax.set_ylabel('Count')
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
plt.tight_layout()
plt.show()

#%%
# Education Level Distribution by Income
plt.figure(figsize=(12, 6))
sns.countplot(x='education', hue='income', data=data_before)
plt.title('Income Distribution by Education Level')
plt.xticks(rotation=45)
plt.legend(title='Income')
plt.xlabel('Education Level')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

#%%
# Education Num Distribution by Income
plt.figure(figsize=(12, 6))
sns.countplot(x='education_num', hue='income', data=data_before)
plt.title('Income Distribution by Education Num')
plt.xticks(rotation=45)
plt.legend(title='Income')
plt.xlabel('Education Num')
plt.ylabel('Count')
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
plt.tight_layout()
plt.show()