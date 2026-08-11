import os
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import scikit_posthoc as sp

from joblib import dump, load

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

from scipy.stats import kruskal

p_stemmer = PorterStemmer()
analyzer = CountVectorizer().build_analyzer()

def stemmed_words(doc):
    text = re.sub(r"[^\w\s]", "", doc.lower())
    return [p_stemmer.stem(w) for w in analyzer(text)]

def load_corpus(root_path):
    X, y = [], []

    for file in os.listdir(root_path):
        label = os.path.splitext(file)[0]
        file_path = os.path.join(root_path, file)

        with open(file_path, encoding='utf-8', errors='ignore', mode='r') as f:
            for line in f:
                content = line.strip()
                if content:
                    X.append(content)
                    y.append(label)

    return X, y

def train_test(root_path, dump_path):
    X, y = load_corpus(root_path)

    candidate_models = {
        "Logistic Regression": LogisticRegression(),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Linear Support Vector Machine": LinearSVC(),
    }

    best_name = None
    best_pipeline = None
    best_mean = 0.0
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    scores_dict = {}

    for name, model in candidate_models.items():
        pipeline = make_pipeline(TfidfVectorizer(tokenizer=stemmed_words, token_pattern=None, ngram_range=(1, 2), use_idf=True, sublinear_tf=True), model) #can refactor stemmed_words regex to token_pattern
        scores = cross_val_score(pipeline, X, y, cv=kfold, scoring='accuracy')
        mean = scores.mean()
        std = scores.std()

        scores_dict[name] = np.array(scores)

        print(f"{name}:")
        print(f"Cross-Validation Scores: {scores}")
        print(f"Mean Accuracy: {mean:.4f} (+/-{std:.4f})")

        if mean > best_mean:
            best_mean = mean
            best_name = name
            best_pipeline = pipeline


    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=1)
    best_pipeline.fit(X_train, y_train)
    predicted = best_pipeline.predict(X_test)

    print(f"\nBest Model: {best_name}")

    print(f"Confusion Matrix:")
    print(confusion_matrix(y_test, predicted))

    print(f"\nAccuracy: {accuracy_score(y_test, predicted):.4f}")
    print(f"F1 Score: {f1_score(y_test, predicted, average='weighted'):.4f}")

    print(f"\nClassification Report:")
    print(classification_report(y_test, predicted))
    print(f)

    best_pipeline.fit(X, y)
    dump(best_pipeline, dump_path)

    return scores_dict

def eval(scores_dict):
    stat, p = kruskal(*scores_dict.values())
    print(f"Kruskal-Wallis H-test: statistic={stat:.4f}, p-value={p:.4f}")

    if p < 0.05:
        print("There is a significant difference between the models' performance.")
        posthoc_results = sp.posthoc_dunn(scores_dict, p_adjust='bonferroni')
        print("\nPost-hoc Dunn's test results:")
        print(posthoc_results)
    else:
        print("There is no significant difference between the models' performance.")

    