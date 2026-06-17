# 📊 Modelo de Predição de Inadimplência

## 📌 Visão Geral

Projeto de Ciência de Dados desenvolvido para prever a probabilidade de inadimplência de clientes utilizando Machine Learning.

A solução combina engenharia de atributos comportamentais, modelagem preditiva com LightGBM, calibração de probabilidades e um dashboard interativo em Streamlit para apoio à tomada de decisão.

---

## 🎯 Objetivo de Negócio

Antecipar clientes com maior probabilidade de inadimplência para:

* Priorizar ações de cobrança
* Reduzir custos operacionais
* Melhorar recuperação de crédito
* Apoiar decisões de risco

---

## 🏗️ Arquitetura da Solução

```text
Bases Brutas
      │
      ▼
Integração dos Dados
      │
      ▼
Feature Engineering
      │
      ▼
Split Temporal
      │
      ▼
LightGBM
      │
      ▼
Calibração de Probabilidades
      │
      ▼
Segmentação de Risco
      │
      ▼
Dashboard Streamlit
```

---

## 🔧 Engenharia de Features

### Variáveis Comportamentais

* target_lag_1
* target_lag_2
* target_lag_3
* inad_ult_3
* atraso_ult_3
* atraso_ult_6
* taxa_hist_inad
* atraso_hist_medio
* n_cobrancas_hist
* valor_medio_cliente

### Variáveis de Contexto

* tempo_cliente
* dias_prazo_cobranca
* mes_safra
* trimestre_safra
* flag_cliente_novo

### Cuidados contra Data Leakage

Todas as variáveis históricas utilizam:

```python
shift(1)
```

Garantindo que apenas informações disponíveis antes da cobrança sejam utilizadas.

---

## 🤖 Modelo

### Algoritmo

* LightGBM Classifier

### Estratégias Aplicadas

* Split temporal
* Scale Pos Weight
* Early Stopping
* Calibração via Platt Scaling (Sigmoid)

---

## 📈 Avaliação

Métricas utilizadas:

* AUC-ROC
* KS Statistic
* Average Precision
* Log Loss

Análises complementares:

* Curva ROC
* Precision-Recall Curve
* Gains Chart
* Matriz de Confusão
* Reliability Diagram

### Calibração

A calibração foi aplicada apenas após validação da melhoria do Log Loss.

Exemplo:

| Métrica  | Antes  | Depois |
| -------- | ------ | ------ |
| Log Loss | 0.1887 | 0.1194 |

Resultado: Probabilidades calibradas adotadas para produção.

---

## 🚦 Segmentação de Risco

Os clientes são classificados em:

| Faixa       | Descrição                            |
| ----------- | ------------------------------------ |
| Baixo Risco | Baixa probabilidade de inadimplência |
| Médio Risco | Atenção preventiva                   |
| Alto Risco  | Prioridade máxima de cobrança        |

---

## 📊 Dashboard

O projeto inclui dashboard interativo desenvolvido em Streamlit.

Funcionalidades:

* Ranking de clientes por risco
* Distribuição de probabilidades
* Segmentação de risco
* Métricas consolidadas
* Consulta individual de clientes

---

## 🗂️ Estrutura do Projeto

```text
Credit_Model_Teste/
│
├── app.py
├── requirements.txt
├── README.md
│
├── artefatos/
│   ├── model_final.pkl
│   ├── train_medians.pkl
│   ├── features.pkl
│   ├── cat_cols.pkl
│   └── cutoff.pkl
│
├── data/
│   ├── base_pagamentos_desenvolvimento.csv
│   ├── base_pagamentos_teste.csv
│   ├── base_info.csv
│   └── base_cadastral.csv
│
├── notebooks/
│   ├── eda.ipynb
│   ├── feature_engineering.ipynb
│   └── train_lgbm.ipynb
│
└── src/
    ├── features.py
    ├── predict.py
    └── train.py
```

---

## 🛠️ Tecnologias

* Python 3.11.15
* Pandas
* NumPy
* Scikit-Learn
* LightGBM
* SciPy
* Matplotlib
* Seaborn
* Streamlit
* Joblib

---

## ▶️ Como Executar

### 1. Criar ambiente

```bash
conda create -n credit_model python=3.11.15 -y
conda activate credit_model
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

### 3. Executar dashboard

```bash
streamlit run app.py
```

---

## 📌 Resultados

O modelo demonstrou forte capacidade de priorização de risco, permitindo concentrar ações de cobrança nos clientes com maior probabilidade de inadimplência.

A calibração das probabilidades melhorou significativamente o Log Loss, tornando as previsões mais confiáveis para uso operacional.

---

## 👨‍💻 Autor

Jonathas de Oliveira

Projeto desenvolvido como estudo prático de Ciência de Dados aplicado ao contexto de risco de crédito e inadimplência.
