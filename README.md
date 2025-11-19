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

## Future Extensions

- Add CNN, GRU, or Transformer-based models
- Add explainability methods (SHAP, Integrated Gradients)
- Add visualizations (confusion matrix, ROC curve)
- Build a Gradio/Streamlit interface

## Author

Pavan Prasad Gorintla
Rahul 
Smrithya 
