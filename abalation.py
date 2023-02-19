#& Ours & 89.43\% \\
#Signals & EEG & 64.09\% \\
#& EMG & 64.79\% \\
#& Graph only & 52.35\% \\
#Graph & Fixed average graph & 71.67\% \\
#& Random graph & 63.24\% \\
#& Complete graph & 58.82\% \\
#Ensemble & Ensemble & 80.97\% \\
#& Exact prediction & 20.30\% \\
#& Same movement & 75.23\% \\

import matplotlib.pyplot as plt

fig, ax = plt.subplots()

fruits = ['Ours', 'EEG only', 'EMG only', 'Graph only', 'Fixed SPMI graph', 'Random graph', 'Complete graph', 'Ensemble', 'Same movement', 'Exact prediction']
counts = [88.89, 64.09, 64.79, 52.35, 71.67, 63.24, 58.82, 80.97, 75.23, 20.30]

bar_labels = ['Ours', 'Modals', '_Modals', '_Modals', 'Graphs', '_Graphs', '_Graphs', 'Ensemble', '_Ensemble', '_Ensemble']
bar_colors = ['tab:red', 'tab:blue', 'tab:blue', 'tab:blue', 'tab:orange', 'tab:orange', 'tab:orange', 'tab:green', 'tab:green', 'tab:green']

ax.bar(fruits, counts, label=bar_labels, color=bar_colors, width = 0.4)

plt.xticks(rotation=60)

plt.xticks(fontsize=12)
plt.yticks(fontsize=15)

for i in range(10):
	ax.annotate(f"{counts[i]:.2f}", (i, counts[i]), ha='center', fontsize=15)

ax.set_ylabel('Accuracy', fontsize=15)
ax.set_title('Ablation Experiments Results', fontsize=15)
ax.legend(title='Ablation Groups')

plt.show()