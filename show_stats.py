import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Esempio di presentazione dei dati raccolti su Shopping List (5 items)

df = pd.read_csv("selections_shopping.csv")
print(df.head())
print(df.dtypes)
df_filtered = df[df['found'] == 1]

sns.scatterplot(x='evaluations', y='select_type', data=df_filtered)

plt.show()

sns.scatterplot(
    x='evaluations', 
    y='exec_time', 
    hue='select_type',
    data=df_filtered,
    sizes=(20, 200),
    alpha=0.6
)

plt.title('Exec Time vs Evaluations with Select Type')
plt.xlabel('Evaluations')
plt.ylabel('Exec Time')

plt.show()