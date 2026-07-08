# 📩 Spam Message Detection using Machine Learning

## Overview

This project is an end-to-end Machine Learning application that classifies SMS messages as **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP).

The application performs text preprocessing, converts text into numerical features using **TF-IDF**, trains and evaluates multiple machine learning models, selects the best-performing model, and provides predictions through a Flask web application.

---

## Features

* Text preprocessing using NLTK
* TF-IDF Vectorization
* Model comparison

  * Logistic Regression
  * Multinomial Naive Bayes
* Model evaluation using:

  * Accuracy
  * Precision
  * Recall
  * F1 Score
  * Confusion Matrix
* Flask Web Application
* Clean modular project structure
* Easy deployment

---

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* NLTK
* Flask
* HTML
* CSS

---

## Project Structure

```text
Spam-Detection/

│── app.py
│── requirements.txt
│── README.md
│── .gitignore
│
├── artifacts/
│
├── data/
│   └── spam.csv
│
├── models/
│   ├── model.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebooks/
│   └── Spam_Detection.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
│
├── templates/
│   └── index.html
│
└── static/
    └── style.css
```

---

## Dataset

* SMS Spam Collection Dataset
* Total Messages: 5,572
* Classes:

  * Ham
  * Spam

---

## Machine Learning Pipeline

1. Load Dataset
2. Data Cleaning
3. Feature Engineering
4. Text Preprocessing
5. TF-IDF Vectorization
6. Train-Test Split
7. Train Multiple Models
8. Model Evaluation
9. Save Best Model
10. Flask Deployment

---

## Text Preprocessing

The following preprocessing steps are applied:

* Convert text to lowercase
* Tokenization
* Remove non-alphanumeric characters
* Remove stopwords
* Apply Porter Stemming

---

## Models Compared

* Logistic Regression
* Multinomial Naive Bayes

The final model is selected based on evaluation metrics.

---

## Evaluation Metrics

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Classification Report

---

## Installation

Clone the repository:

```bash
git clone <your-repository-link>
```

Move into the project:

```bash
cd Spam-Detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Train the Model

```bash
python -m src.train
```

---

## Run the Application

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

---
### Live Demo:
https://spam-detection-r1rd.onrender.com

## Future Improvements

* Hyperparameter tuning
* Cross-validation
* Scikit-learn Pipeline
* Docker support
* Cloud deployment
* Transformer-based spam detection (BERT)

---

## Author

**Karthik Kulal**

Computer Science Engineering Student

Machine Learning & AI Enthusiast