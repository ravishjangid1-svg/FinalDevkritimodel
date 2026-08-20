# Fake News Detection using Machine Learning

A machine-learning based text classification project that classifies news articles as **Fake** or **Real** using Natural Language Processing (NLP), TF-IDF feature extraction, and multiple classical machine-learning algorithms.

- **Project status:** Academic
- **Task:** Binary text classification
- **Primary goal:** Compare traditional ML architectures for fake-news classification and build a reusable prediction pipeline.

---

## Overview

The project takes the textual content of news articles, cleans and normalizes the text, converts it into numerical TF-IDF features, and trains several machine-learning models to distinguish between:

- `0` → Fake News
- `1` → Real News

The current pipeline evaluates four machine-learning architectures:

1. Logistic Regression
2. Multinomial Naive Bayes
3. Passive Aggressive Classifier
4. Random Forest

The dataset is split into training and testing subsets **before** TF-IDF is fitted, preventing test-set text statistics from being used to learn the TF-IDF representation.

---

## Machine Learning Pipeline

```
True.csv / Fake.csv
        │
        ▼
Add class labels (Real = 1, Fake = 0)
        │
        ▼
Combine datasets & Remove duplicates
        │
        ▼
Exploratory Data Analysis (EDA) & Visualizations
        │
        ▼
Text preprocessing (Cleaning, Stopwords, Punctuation)
        │
        ▼
Train/Test Split (80% / 20%, stratified)
        │
        ├──────────────┐
        ▼              ▼
Training Text      Testing Text
        │              │
        ▼              ▼
TF-IDF fit_transform   TF-IDF transform
        │              │
        └──────┬───────┘
               ▼
Model Training & 5-Fold Cross-Validation
               │
               ▼
Evaluation (Hold-out Accuracy, Classification Report, Confusion Matrix)
```

---

## Text Preprocessing

Before machine learning, the raw article text passes through several preprocessing stages:

1. HTML/tag removal
2. Lowercasing
3. URL removal
4. Publisher/dateline cleaning
5. Punctuation removal
6. Stopword removal

---

## TF-IDF Feature Extraction

Machine-learning algorithms cannot directly process raw text, so the cleaned documents are converted into numerical vectors using Term Frequency–Inverse Document Frequency (TF-IDF).

The current configuration is:

```python
tfidf = TfidfVectorizer(max_features=5000)
```

This limits the vocabulary to the 5,000 selected features.

### Important: Data-Leakage Prevention

The dataset is split **before** TF-IDF is fitted:

```python
X_train_text, X_test_text, y_train, y_test = train_test_split(
    data['original_text'],
    data['NewsType'],
    test_size=0.2,
    random_state=42,
    stratify=data['NewsType']
)
```

TF-IDF is then fitted only on the training data:

```python
X_train = tfidf.fit_transform(X_train_text)
```

The test data is transformed using the already-fitted vectorizer:

```python
X_test = tfidf.transform(X_test_text)
```

This prevents the test set from influencing the vocabulary and IDF statistics learned during training.

---

## Model Architectures

The project compares four classical machine-learning approaches.

### 1. Logistic Regression
A linear classification algorithm that is often highly effective for sparse text representations such as TF-IDF.

```python
LogisticRegression(max_iter=1000)
```

### 2. Multinomial Naive Bayes
A probabilistic classifier commonly used for text classification.

```python
MultinomialNB()
```

### 3. Passive Aggressive Classifier
An online-learning algorithm designed to update aggressively when a prediction is incorrect.

```python
PassiveAggressiveClassifier(
    max_iter=50,
    random_state=42
)
```

### 4. Random Forest
An ensemble of decision trees that combines multiple tree predictions.

```python
RandomForestClassifier(random_state=42)
```

---

## Model Comparison

| Rank | Architecture                  | Accuracy (%) |
|------|-------------------------------|---------------|
| 1    | Passive Aggressive Classifier | 98.58         |
| 2    | Logistic Regression           | 98.28         |
| 3    | Random Forest                 | 98.20         |
| 4    | Multinomial Naive Bayes       | 93.54         |

---

## Evaluation

The project uses:

- `accuracy_score()`
- `classification_report()`

The classification report provides:

- Precision
- Recall
- F1-score
- Support
- Overall Accuracy

---

## Custom News Prediction

The project also supports testing an arbitrary news statement.

**Example:**

```python
my_news = "The Federal Reserve announced a quarter-point interest rate hike on Wednesday following the monthly committee meeting in Washington."
predict_custom_news(my_news)
```

**Output:**

```
Fake News Confidence: 31.00%
Real News Confidence: 69.00%
```

---

## Tech Stack

| Category                | Tools / Libraries |
|--------------------------|--------------------|
| Programming Language     | Python |
| Data Processing          | NumPy, Pandas |
| Natural Language Processing | NLTK |
| Regular Expressions      | `re` |
| Machine Learning         | Scikit-learn |
| Feature Engineering      | TF-IDF Vectorization |
| Development Environment  | Jupyter Notebook / JupyterLab |

**Models used:**
- Logistic Regression
- Multinomial Naive Bayes
- Passive Aggressive Classifier
- Random Forest

---

## 📦 Installation

Clone the repository:

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_PROJECT_DIRECTORY>
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install numpy pandas scikit-learn nltk jupyter
```

Download the NLTK stopword corpus:

```python
import nltk
nltk.download('stopwords')
```

---

## ▶️ Running the Project

Place the following files in the project directory:

```
True.csv
Fake.csv
```

Then launch Jupyter:

```bash
jupyter lab
```

Open the notebook and execute the cells from top to bottom.

---

## 📁 Suggested Project Structure

```
fake-news-detection/
│
├── data/
│   ├── True.csv
│   └── Fake.csv
│
├── notebooks/
│   └── model1.ipynb
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Author

**Ravish Jangid**

Machine Learning / NLP project focused on understanding practical text-classification pipelines and comparing classical machine-learning architectures.
