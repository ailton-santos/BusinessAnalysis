import pandas as pd

# Importar dados de vendas e produção
df_vendas = pd.read_csv('dados_vendas.csv')
df_producao = pd.read_csv('dados_producao.csv')

# Renomear coluna "Data" para "Data_Venda"
df_vendas = df_vendas.rename(columns={"Data": "Data_Venda"})

# Juntar dataframes usando coluna "Data_Venda"
df = pd.merge(df_vendas, df_producao, on="Data_Venda")

import matplotlib.pyplot as plt

# Criar gráfico de vendas ao longo do tempo
plt.plot(df['Data_Venda'], df['Vendas'])
plt.xlabel('Data')
plt.ylabel('Vendas')
plt.title('Vendas ao longo do tempo')
plt.show()

from sklearn.linear_model import LinearRegression

# Separar dados de treino e teste
X = df['Producao'].values.reshape(-1,1)
y = df['Vendas'].values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Criar modelo de regressão linear
model = LinearRegression()
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Criar dashboard com gráfico de linhas e scatter plot
fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(go.Scatter(x=df['Data_Venda'], y=df['Producao'], name="Produção"),
              secondary_y=False)

fig.add_trace(go.Scatter(x=df['Data_Venda'], y=df['Vendas'], name="Vendas"),
              secondary_y=True)

fig.update_layout(title_text="Desempenho de vendas e produção")
fig.show()


