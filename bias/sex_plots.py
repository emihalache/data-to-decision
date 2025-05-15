import matplotlib.pyplot as plt
import pandas as pd

# Data preparation
data = {
    'Algorithm': ['Logistic Regression', 'Decision Tree', 'Random Forest'],
    'Demographic Parity': [-0.19, -0.17, -0.19],
    'Equalized Odds': [0.10, 0.09, 0.09],
    'Disparate Impact': [0.30, 0.41, 0.33],
    'Equal Opportunity': [0.10, 0.03, 0.07]
}
df = pd.DataFrame(data)

# Setting the positions and width for the bars
pos = list(range(len(df['Demographic Parity'])))
width = 0.2

# Plotting each metric in the desired order
fig, ax = plt.subplots(figsize=(10, 5))
colors = ['red', 'blue', 'yellow', 'green']  # Colors for the bars
labels = ['Demographic Parity', 'Disparate Impact', 'Equal Opportunity', 'Equalized Odds']  # Labels for the legend

bars1 = plt.bar(pos, df['Demographic Parity'], width, alpha=0.5, color=colors[0])
bars2 = plt.bar([p + width for p in pos], df['Disparate Impact'], width, alpha=0.5, color=colors[1])
bars3 = plt.bar([p + width*2 for p in pos], df['Equal Opportunity'], width, alpha=0.5, color=colors[2])
bars4 = plt.bar([p + width*3 for p in pos], df['Equalized Odds'], width, alpha=0.5, color=colors[3])

# Axis configuration
ax.set_ylabel('Metric Values')
ax.set_xlabel('Algorithm')  # Adding x-axis label
ax.set_title('Comparison of Algorithm Fairness Metrics for Sensitive Attribute \'sex\'')
ax.set_xticks([p + 1.5 * width for p in pos])
ax.set_xticklabels(df['Algorithm'])

# Adding labels to the bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height > 0 else -12),  # Adjusting vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)
add_labels(bars4)

# Fixing the legend to bottom right and adjusting the font size
plt.legend(labels, loc='lower right', fontsize='small')
plt.grid()
plt.show()
