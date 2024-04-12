import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("cv_animosity.csv")

grouped_data = df.groupby('i').agg({
    'C': 'first',
    'accuracy_val': ['mean', 'std'],
    'accuracy_test': ['mean', 'std'],
    }).reset_index()
print(grouped_data)

df_mf = pd.read_csv("dummy_mf_animosity.csv")

plt.figure(figsize=(10, 6))
sns.lineplot(data=grouped_data, x=('C', 'first'), y=('accuracy_val', 'mean'), label="accuracy val", color="red")
plt.errorbar(grouped_data['C'], grouped_data['accuracy_val', 'mean'], yerr=grouped_data['accuracy_val', 'std'], fmt='o', capsize=5, color="red")

plt.axhline(y=df_mf['accuracy_val'].mean(), label="dummy val", color="red")
plt.axhline(y=df_mf['accuracy_val'].mean() + df_mf['accuracy_val'].std(), linestyle='--', lw=0.5, color="red")
plt.axhline(y=df_mf['accuracy_val'].mean() - df_mf['accuracy_val'].std(), linestyle='--', lw=0.5, color="red")

# sns.lineplot(data=grouped_data, x=('C', 'first'), y=('accuracy_test', 'mean'), label="accuracy test")
# plt.errorbar(grouped_data['C'], grouped_data['accuracy_test', 'mean'], yerr=grouped_data['accuracy_test', 'std'], fmt='o', capsize=5)
# 
# plt.axhline(y=df_mf['accuracy_test'].mean(), label="dummy test")
# plt.axhline(y=df_mf['accuracy_test'].mean() + df_mf['accuracy_test'].std(), linestyle='--', lw=0.5)
# plt.axhline(y=df_mf['accuracy_test'].mean() - df_mf['accuracy_test'].std(), linestyle='--', lw=0.5)

plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title('Mean/Std Accuracy of $\\texttt{animosity}$ classifier against C with 42-fold Cross-Validation')
plt.xscale('log')
plt.legend()
plt.savefig("fig/cv_animosity.pdf")
plt.show()
