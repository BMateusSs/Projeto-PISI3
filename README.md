## Projeto-PSI3

# 🍷 Vinicoteca – Wine Analysis & Recommendation System

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Supervised%20%26%20Unsupervised-green)
![Status](https://img.shields.io/badge/status-em%20desenvolvimento-yellow)
![License](https://img.shields.io/badge/license-Academic-lightgrey)

Projeto acadêmico desenvolvido na disciplina **PISI3**, com foco na aplicação de **Machine Learning** e no desenvolvimento de um aplicativo mobile para recomendação de vinhos.

---

## Sobre o Projeto

O crescimento do consumo de vinhos trouxe também um desafio: **como escolher o vinho ideal diante de tantas opções?**

Este projeto propõe uma solução completa baseada em:

* **Predição da qualidade de vinhos**
* **Análise de padrões em descrições sensoriais**
* **Aplicativo mobile (Vinicoteca)**

A proposta é tornar a experiência com vinhos **mais acessível, personalizada e inteligente**.

---

## Objetivos

### Objetivo Geral

Desenvolver uma solução integrada utilizando **aprendizado de máquina** e um **aplicativo mobile** para recomendação de vinhos.

### 📌 Objetivos Específicos

* Realizar análise exploratória dos dados (EDA)
* Prever qualidade dos vinhos (boa vs. ruim)
* Identificar padrões em descrições textuais
* Desenvolver um app com recomendações personalizadas

---

## Tecnologias Utilizadas

### Data Science & ML

* Python
* Pandas & NumPy
* Scikit-learn
* XGBoost
* SHAP

### NLP (Processamento de Linguagem Natural)

* Sentence Transformers
* Embeddings
* UMAP

### Mobile

* React Native

---

## Modelos Utilizados

* Random Forest
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)
* Gaussian Naive Bayes
* Multi-Layer Perceptron (MLP)
* **XGBoost (melhor desempenho)**

---

## 📊 Metodologia

### 📁 Dados

* Wine Quality Dataset (UCI / Kaggle)
* Wine Reviews Dataset (Kaggle)

### ⚙️ Pré-processamento

* Normalização (StandardScaler)
* One-Hot Encoding
* Balanceamento:

  * SMOTE
  * ADASYN

---

### 📈 Aprendizado Supervisionado

* Classificação binária:

  * **Boa qualidade (≥ 6)**
  * **Má qualidade (< 6)**

---

### Aprendizado Não Supervisionado

* K-Means
* DBSCAN
* **Gaussian Mixture Model (GMM)**

---

## 📌 Resultados

### 📊 Classificação

* Melhor modelo: **XGBoost + ADASYN**
* Acurácia: **~76.5%**

Variáveis mais importantes:

* Teor alcoólico
* Acidez volátil
* Sulfatos
* Dióxido de enxofre

---

### Clusterização

Foi possível identificar padrões relevantes entre:

* País
* Região
* Variedade da uva

✔️ Forte correlação entre descrições sensoriais e características reais dos vinhos

---

## Aplicativo Vinicoteca

O app tem como objetivo melhorar a experiência do usuário com vinhos.

### Funcionalidades

* Cadastro de usuário
* Registro de preferências
* Recomendações personalizadas
* Cadastro de lojas e profissionais
* Mapa interativo de vinhos próximos

---

## Como Executar

```bash
# Clone o repositório
git clone https://github.com/BMateusSs/Projeto-PISI3

# Acesse a pasta
cd Projeto-PISI3

# Crie um ambiente virtual
python -m venv .venv

# Ative o ambiente
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate

# Instale as dependências
pip install -r requirements.txt
```

---

## 📂 Estrutura do Projeto

```
Projeto-PISI3/
│
├── data/              # Datasets
├── notebooks/         # Análises exploratórias
├── models/            # Modelos treinados
├── app/               # Aplicativo mobile
├── src/               # Código principal
└── README.md
```

---

## Equipe

* Bruno Rezende
* Bruno Mateus
* Gabriel Ferreira
* Ingrid Mylena
* Maria Clara
* Victor Yghor
* Gabriel Alves

---

## Considerações Finais

Este projeto demonstra que:

✔️ É possível prever qualidade de vinhos com boa precisão
✔️ Dados textuais possuem grande valor para análise
✔️ Aplicações práticas de IA podem melhorar a experiência do usuário

---

## Trabalhos Futuros

* Melhorar modelos de NLP
* Expandir base de dados
* Criar sistema de recomendação mais avançado
* Implementar interação entre usuários no app

---

## Licença

Projeto acadêmico para fins educacionais.
Repositório para a disciplina PSI3

---
