
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt

df = pd.read_excel("dados/dados_frutas.xlsx")

arvore = tree.DecisionTreeClassifier(random_state=42)

resposta = df['Fruta']

caracteristicas = ["Arredondada", "Suculenta", "Vermelha", "Doce"]
X = df[caracteristicas]

arvore.fit(X, resposta)

answer = arvore.predict([[1, 1, 0,1]])
print(f"Fruta prevista: {answer}")

plt.figure(dpi=400)

tree.plot_tree(arvore, feature_names=caracteristicas, class_names=arvore.classes_, filled=True)


## plt.show()