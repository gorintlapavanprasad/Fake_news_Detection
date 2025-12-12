# Explainable AI in Fake News Detection

A comprehensive end-to-end Fake News Detection system built using classical Machine Learning models and a custom Hybrid Deep Learning architecture.
This project demonstrates modern NLP preprocessing, topic modeling, embedding techniques, and model explainability.

## Overview
This repository implements and compares three different approaches to detecting fake news:

### Hybrid Deep Learning Model (CNN + BiLSTM + Attention + Word2Vec + LDA)

A powerful neural architecture that combines:  
    - Word2Vec embeddings  
    - LDA topic vectors  
    - Convolution layers for local pattern recognition  
    - BiLSTM for sequential context  
    - Custom Attention layer for focusing on key phrases  

### Random Forest Model

A classical ML baseline using:  
    - Average Word2Vec embeddings  
    - LDA topic vectors  
    - 110-dimensional engineered features  
    - Strong interpretability + fast inference  

### Decision Tree Model

A fully interpretable model using the same engineered features.
Useful for understanding feature importance and rule-based splitting.

## Dataset
Kindly download the dataset and store it in the same folder as the main file. Also modify the dataset file path accordingly

dataset:
<https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets>


## Problem Statement
Fake news creates misinformation, social mistrust, and real-world harm.
The goal of this project is to build models that can accurately classify news articles as FAKE or REAL using advanced NLP techniques.

## Methodology
### 1. Data Preprocessing

Every script uses a unified preprocessing pipeline:  
    ✔ Text cleaning  
        - Lowercasing  
        - Removing URLs  
        - Removing punctuation/numbers  
        - Stopword removal  
        - Tokenization  
    ✔ Train/Test Split  

Performed before any modeling to avoid data leakage.  


### 2. Feature Engineering
    
• Word2Vec  
     A custom Word2Vec model is trained only on the training set  
     → Documents represented as the average vector of all word embeddings (100-dim).  

• LDA Topic Modeling  
     - Captures global document semantics  
     - Trained only on training data  

## Evaluation Metrics

Each model reports:  
1. Accuracy  
    - Precision, Recall, F1-Score  
    - Confusion Matrix  
    - Example predictions  
    - LIME explanations  

2. Typical performance range:  
    - Decision Tree → 94%  
    - Random Forest → 97%  
    - Hybrid DL Model → 99.5%
  
## How to Run the Models
Install dependencies:
-     pip install numpy pandas nltk gensim scikit-learn tensorflow lime

Run any script:
-     python main.py                          # Hybrid Model  
-     python FND_using_random_forest.py       # Random Forest  
-     python FND_using_decision_tree.py       # Decision Tree

Dataset should contain two files:
-     Fake.csv
-     True.csv

## Key Insights

Hybrid models combining embeddings + topics + attention outperform classical ML

- Random Forests are great baselines and surprisingly strong.  
- Decision Trees offer interpretability with lower accuracy.  
- LDA topics significantly improve feature richness.  
- Word2Vec improves semantic understanding over TF-IDF.  
- LIME reveals which words influence model predictions.  


## Possible Enhancements

- Upgrade to BERT / DistilBERT embeddings.  
- Add ROC-AUC curves and more metrics.  
- Hyperparameter tuning via grid search / Optuna.  
- Deploy using FastAPI or Flask.  
- Streamlit dashboard for live predictions.  


## Contributors
1. Smirthya Somaskantha Iyer
2. Rahul Rubugunday
3. Pavan Gorintla
