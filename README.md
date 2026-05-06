# 📊 Case de Data Science — Predição de Inadimplência

## 🎯 Objetivo

Desenvolver um modelo de machine learning capaz de prever a probabilidade de inadimplência de clientes, permitindo priorização de ações de cobrança e otimização de recursos operacionais.

---

## 🧠 Abordagem

O projeto foi estruturado seguindo um pipeline completo de ciência de dados:

1. **Definição da target**

   * Inadimplência definida como atraso ≥ 5 dias
   * Problema tratado como classificação binária

2. **Construção da base analítica**

   * Integração de múltiplas fontes (cadastral, info e pagamentos)
   * Garantia de consistência temporal e integridade dos dados

3. **Feature Engineering**

   * Criação de variáveis comportamentais (lags, rolling, histórico)
   * Features de contexto (tempo de relacionamento, sazonalidade)
   * Uso rigoroso de `shift(1)` para evitar data leakage

4. **Split temporal**

   * Separação treino/validação respeitando ordem temporal
   * Simulação de cenário real de produção

5. **Modelagem**

   * Algoritmo: LightGBM
   * Tratamento de desbalanceamento com `scale_pos_weight`
   * Early stopping para evitar overfitting

6. **Avaliação**

   * Métricas utilizadas:

     * AUC-ROC
     * KS
     * Average Precision
     * Log Loss
   * Análises complementares:

     * Curva ROC e Precision-Recall
     * Gains Chart (priorização de risco)
     * Matriz de confusão

7. **Calibração de probabilidades**

   * Aplicação de Platt Scaling (sigmoid)
   * Validação com Log Loss e reliability curve
   * Uso condicionado à melhoria da métrica

8. **Regras de negócio**

   * Segmentação em níveis de risco:

     * Baixo risco
     * Médio risco
     * Alto risco
   * Permite aplicação prática do modelo em estratégias de cobrança

9. **Validações finais**

   * Sanity checks do modelo (range, NaN, consistência, distribuição)
   * Garantia de robustez antes da submissão

10. **Submissão**

* Geração de probabilidades para base de teste
* Exportação em formato CSV

---

## 🚀 Resultados

O modelo demonstrou boa capacidade de separação e priorização de risco:

* Identificação eficiente de clientes inadimplentes
* Capacidade de concentrar risco nos primeiros percentis (Gains)
* Probabilidades calibradas para uso em decisão

---

## 💼 Aplicação de Negócio

O modelo pode ser utilizado para:

* Priorizar ações de cobrança
* Reduzir custo operacional
* Aumentar taxa de recuperação
* Apoiar decisões de crédito

A segmentação por risco permite direcionar estratégias distintas para diferentes perfis de cliente.

---

## ⚠️ Cuidados Técnicos

* Prevenção rigorosa de **data leakage**
* Uso de **split temporal**
* Consistência total entre treino e teste
* Interpretação correta das métricas
* Separação entre **ranking e calibração**

---

## 🛠️ Tecnologias Utilizadas

* Python 3.10
* pandas, numpy
* scikit-learn
* LightGBM
* matplotlib, seaborn

---

## 📁 Estrutura do Projeto

```
📦 case-inadimplencia
 ┣ 📂 data
 ┣ 📂 notebooks
 ┣ 📄 requirements.txt
 ┣ 📄 submissao_case.csv
 ┗ 📄 README.md
```

---

## ▶️ Como Executar

```bash
# criar ambiente
conda create -n case_ds python=3.10 -y
conda activate case_ds

# instalar dependências
pip install -r requirements.txt

# rodar notebook
jupyter notebook
```

---

## 📌 Conclusão

O projeto entrega um pipeline completo de modelagem de crédito, desde a construção das variáveis até a aplicação prática em regras de negócio.

A solução é robusta, interpretável e pronta para uso em cenários reais de decisão.

---
