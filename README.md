# Fake News Detection using BiLSTM with Attention

## Project Overview

This project implements a complete Fake News Detection system using a Bidirectional LSTM (BiLSTM) with an Attention mechanism.
The system reads news articles, learns linguistic patterns from text, and predicts whether an article is Real or Fake.

The project uses the ISOT Fake News Dataset, which contains two files:

- Fake.csv
- True.csv

Together, these files contain over 44,000 labelled articles.

The entire project is structured in a modular, research-friendly manner to allow easy upgrades, experiments, and documentation.

## Dataset Description

The ISOT Fake News dataset consists of two CSV files:

1. Fake.csv – approximately 23,000 fake news articles
2. True.csv – approximately 21,000 real news articles

Each file contains the following columns:

- title: headline of the article
- text: main content of the article
- subject: topic category
- date: publication date

For this project, the title and text fields are combined to form a single input text field.

## Project Structure

fake_news_detection_project/
│
├── main.py
├── config.yaml
├── requirements.txt
│
├── data/
│   └── raw/
│       ├── True.csv
│       └── Fake.csv
│
├── src/
│   ├── data/
│   │   ├── preprocess.py
│   │   └── dataset.py
│   ├── models/
│   │   └── bilstm_attention.py
│   ├── training/
│   │   ├── train.py
│   │   └── metrics.py
│   └── utils/
│       ├── seed.py
│       └── paths.py
│
├── outputs/
│   ├── checkpoints/
│   └── predictions/
│
├── results/
└── docs/
    └── project_documentation.md

## Project Overview

Fake news has become one of the most critical challenges in the digital age. This project compares multiple modeling strategies to determine which methods generalize best on a real-world fake-news dataset.

The project evaluates three major approaches:

1. Hybrid Deep Learning Model
    
    Techniques used:
    
    - Text Cleaning & Stopword Removal
    - Keras Tokenizer
    - Word2Vec embeddings (trained on training data only)
    - LDA Topic Modeling (10 topics)
    - CNN layer for n-gram detection
    - BiLSTM for long-range contextual information
    - Custom Attention Layer
    - Topic vectors concatenated with Attention output
    - EarlyStopping & ReduceLROnPlateau
    - LIME explainability
    
This is the most advanced pipeline, combining both semantic (topic-level) and sequential (word-level) information.
    
2. Random Forest Classifier

    File: FND_using_random_forest.py
    
    This classical ML model uses:
    
    - Average Word2Vec embeddings
    - LDA topic vectors
    - Combined 110-dim document feature vector
    - RandomForestClassifier
    - LIME explainability
    
It provides strong performance with low training cost.
    
3. Decision Tree Classifier

    File: FND_using_decision_tree.py

    A simple, fully interpretable baseline using:
    
    - Average Word2Vec vectors
    - LDA topic distributions
    - Depth-controlled DecisionTreeClassifier
    - LIME explainability
    
Useful for comparing interpretability vs model power.

## How to Run the Project

1. Create and activate a virtual environment (macOS/Linux):

python3 -m venv venv
source venv/bin/activate

2. Install required packages:

pip install -r requirements.txt

3. Ensure the dataset files exist in the correct folder:

data/raw/True.csv
data/raw/Fake.csv

4. Run the main script:

python3 main.py

python3 FND_using_random_forest.py        #Run separatly to find the Random Forest Approach 
python3 FND_using_decision_tree.py        #Run separatly to find the Decision Tree Approach

## Model Architecture

The model consists of the following components:

1. Embedding Layer
2. Bidirectional LSTM
3. Attention Layer
4. Fully Connected Layer
5. Sigmoid Activation

## Training Configuration

- Loss function: Binary Cross Entropy (BCELoss)
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 64
- Number of epochs: 10
- Device: CPU or GPU

## Evaluation and Outputs

Test metrics are saved in:

results/test_metrics.txt

Predictions:

outputs/predictions/test_predictions.csv

Best model:

outputs/checkpoints/best_model.pt

## Key Insights

- Random Forest performs extremely well with engineered features
- Decision Tree is interpretable but may overfit
- Hybrid deep learning model is powerful but requires regularization
- Word2Vec + LDA improves traditional ML performance
- LIME helps understand model decisions

## Future Extensions

- Add TF-IDF + Linear SVM
- Integrate BERT / DistilBERT
- Experiment with different LDA topic counts
- Add ROC-AUC curves
- Deploy via FastAPI or Flask

## Author

Pavan Prasad Gorintla
Smrithya Iyer
Rahul Rubugunday

