import os
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import scikit_posthocs as sp

from handler import stemmed_words

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

def load_corpus(root_path):
    X, y = [], []

    for file in os.listdir(root_path):
        file_path = os.path.join(root_path, file)

        if not os.path.isfile(file_path):
            continue

        label = os.path.splitext(file)[0]

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
        "Random Forest": RandomForestClassifier(),
        "Support Vector Machine": SVC()
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

    best_pipeline.fit(X, y)

    os.makedirs(os.path.dirname(dump_path), exist_ok=True)
    dump(best_pipeline, dump_path)

    return scores_dict

def eval(scores_dict, output_path):
    os.makedirs(output_path, exist_ok=True)

    df = pd.DataFrame(scores_dict)

    df_long = df.melt(var_name='Classifier', value_name='Accuracy')

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Classifier', y='Accuracy', data=df_long)

    plt.title('Cross-Validation Accuracy of Classifiers')
    plt.xlabel('Classifier')
    plt.ylabel('Accuracy')
    plt.tight_layout()

    plot_path = os.path.join(output_path, 'classifier_accuracy_boxplot.png')
    plt.savefig(plot_path, dpi=300)

    stat, p = kruskal(*scores_dict.values())
    stat_path = os.path.join(output_path, 'kruskal_wallis_results.txt')

    with open(stat_path, 'w', encoding='utf-8') as f:
        f.write(f"Kruskal-Wallis H-test: statistic={stat:.4f}, p-value={p:.4f}\n")

        if p < 0.05:
            f.write("There is a significant difference between the models' performance.\n\n")

            posthoc_results = sp.posthoc_dunn(scores_dict, p_adjust='holm')
            csv_path = os.path.join(output_path, 'posthoc_dunn_results.csv')
            posthoc_results.to_csv(csv_path, index=True)

            f.write("\nPost-hoc Dunn's test results:\n")
            f.write(posthoc_results.to_string())
        else:
            f.write("There is no significant difference between the models' performance.")

if __name__ == "__main__":
    scores_dict = train_test(root_path="corpus/top_level_intent", dump_path="dumps/top_level_intent_pipeline.joblib")
    eval(scores_dict, output_path="evaluation")