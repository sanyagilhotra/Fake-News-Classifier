# Fake News Classifier

## Overview

The Fake News Classifier is a machine learning project aimed at automatically classifying news articles as either real or fake based on their content. The project utilizes a Decision Tree classifier trained on TF-IDF vectorized text data. TF-IDF (Term Frequency-Inverse Document Frequency) is used to convert text data into numerical vectors, capturing the importance of words in the documents.

## Dataset

The project employs two datasets:
- Genuine news articles dataset
- Fake news articles dataset

These datasets are used for training and testing the classification model. It is essential to maintain a balanced representation of both real and fake news articles to ensure the model's effectiveness.

## Methodology

1. **Data Preprocessing**: The text data is preprocessed by tokenizing, removing stopwords, and converting text into numerical vectors using TF-IDF vectorization.
2. **Model Training**: A Decision Tree classifier is trained on the TF-IDF vectorized text data using the genuine and fake news articles datasets.
3. **Model Evaluation**: The performance of the model is evaluated using metrics such as accuracy, confusion matrix, and classification report to assess its effectiveness in distinguishing between real and fake news articles.

## Usage

1. **Installation**: Clone the repository to your local machine.

```bash
git clone https://github.com/sanyagilhotra/Fake-News-Classifier.git
