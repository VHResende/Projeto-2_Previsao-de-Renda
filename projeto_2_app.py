import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

# Configurar a página para o modo wide
st.set_page_config(layout="wide")

st.title('Projeto 2: Previsão de Renda')

st.markdown("__Curso: Cientista de Dados__")

st.markdown("__Aluno: Victor Resende__")

st.markdown("----")

st.write('# Análise exploratória da previsão de renda:')

renda = pd.read_csv('C:/Users/User/Documents/EBAC/Cientista de Dados/Profissão Cientista de Dados/Módulo 16_Métodos de Análise/2. Projeto/projeto 2/input/previsao_de_renda.csv')


# Função para carregar e preparar os dados
def load_data():
    renda = pd.read_csv('C:/Users/User/Documents/EBAC/Cientista de Dados/Profissão Cientista de Dados/Módulo 16_Métodos de Análise/2. Projeto/projeto 2/input/previsao_de_renda.csv')
    renda['data_ref'] = pd.to_datetime(renda['data_ref'], errors='coerce')
    selected_features = ['educacao', 'tipo_renda', 'estado_civil', 'tipo_residencia', 'tempo_emprego']
    X = renda[selected_features]
    y = renda['renda']
    X_encoded = pd.get_dummies(X, drop_first=True)
    imputer = SimpleImputer(strategy='mean')
    X_encoded_imputed = imputer.fit_transform(X_encoded)
    return renda, X_encoded_imputed, y, X_encoded.columns

# Função para treinar o modelo e obter importâncias das variáveis
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    importances = rf_model.feature_importances_
    return importances

# Função para criar gráficos
def plot_distribution_by_feature(df, feature, target, plot_type='box'):
    plt.figure(figsize=(10, 6))
    if plot_type == 'box':
        sns.boxplot(x=feature, y=target, data=df)
    elif plot_type == 'violin':
        sns.violinplot(x=feature, y=target, data=df)
    plt.title(f'Distribuição da Renda por {feature.capitalize()}')
    plt.xticks(rotation=45)
    st.pyplot(plt)

def plot_relationship(df, x_feature, y_feature):
    plt.figure(figsize=(10, 6))
    sns.regplot(x=x_feature, y=y_feature, data=df, scatter_kws={'s':50}, line_kws={'color':'red'})
    plt.title(f'Relação entre {x_feature.capitalize()} e {y_feature.capitalize()}')
    st.pyplot(plt)

def plot_feature_importances(importances, feature_names):
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(12, 8))
    sns.barplot(x=importances[indices], y=np.array(feature_names)[indices])
    plt.title('Importância das Variáveis no Random Forest')
    plt.xlabel('Importância')
    plt.ylabel('Variáveis')
    st.pyplot(plt)

# Aplicação Streamlit
st.title('Análise de Dados de Renda')

# Carregar e preparar os dados
renda, X, y, feature_names = load_data()

# Plotar gráficos de distribuição
st.header('Distribuição da Renda')
plot_distribution_by_feature(renda, 'educacao', 'renda', plot_type='box')
plot_distribution_by_feature(renda, 'posse_de_veiculo', 'renda', plot_type='violin')
plot_distribution_by_feature(renda, 'posse_de_imovel', 'renda', plot_type='violin')

# Plotar relação entre renda e tempo de emprego
st.header('Relação entre Renda e Tempo de Emprego')
plot_relationship(renda, 'tempo_emprego', 'renda')

# Treinar o modelo e obter importâncias das variáveis
importances = train_model(X, y)

# Plotar as importâncias das variáveis
st.header('Importância das Variáveis')
plot_feature_importances(importances, feature_names)

st.markdown("----")

# Carregar dados
@st.cache_data
def load_data():
    return pd.read_csv('C:/Users/User/Documents/EBAC/Cientista de Dados/Profissão Cientista de Dados/Módulo 16_Métodos de Análise/2. Projeto/projeto 2/input/previsao_de_renda.csv')

renda = load_data()

# Análise descritiva das variáveis numéricas
st.write("## Análise Descritiva das Variáveis Numéricas")
st.write(renda.describe())

# Distribuição da variável resposta (renda)
st.write("## Distribuição da Renda")
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(renda['renda'], kde=True, ax=ax)
ax.set_title('Distribuição da Renda')
st.pyplot(fig)

# Distribuição das variáveis categóricas
st.write("## Distribuição das Variáveis Categóricas")
categorical_cols = ['educacao', 'tipo_renda', 'estado_civil', 'tipo_residencia']  # Substitua pelos seus nomes de colunas categóricas
for col in categorical_cols:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x=col, data=renda, ax=ax)
    ax.set_title(f'Distribuição da variável {col}')
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Correlação entre variáveis numéricas
st.write("## Matriz de Correlação entre Variáveis Numéricas")
numerical_cols = ['tempo_emprego', 'renda']  # Substitua pelos seus nomes de colunas numéricas
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(renda[numerical_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
ax.set_title('Matriz de Correlação entre Variáveis Numéricas')
st.pyplot(fig)

st.markdown("----")

# Texto de interpretação e conclusões
st.write("## Interpretação e Conclusões")
st.markdown("""
Com base na análise e avaliação, podemos interpretar os resultados da seguinte forma:
> **Perfil de Renda:** O modelo pode ajudar a instituição financeira a identificar clientes com perfis de renda semelhantes sem a necessidade de documentação extra.

> **Variáveis Relevantes:** Variáveis como tempo de emprego, educação, posse de veículo e posse de imóveis podem ser determinantes na estimativa da renda, influenciando decisões como limite de crédito.

> **Recomendações:** A instituição pode considerar a automatização do processo de concessão de crédito com base nas variáveis identificadas, melhorando a experiência do cliente e a eficiência do processo.

**Conclusão:**
Este fluxo de trabalho fornece um caminho estruturado para análise e modelagem preditiva da renda dos clientes.
""")

st.markdown("----")