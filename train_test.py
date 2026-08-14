import os
import re
import csv
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import scikit_posthocs as sp

from collections import defaultdict

from handler import stemmed_words, stemmed_stopped_words

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

# can see if faster with question as key instead of id but lab said faster with sorted id which is easier with a number
def load_csv_corpus(root_path, question_col='Question', answer_col='Answer'):
    corpus, question_exists = {}, {} # dont want to iterate through corpus to check if question exists scales quadratic so cache a question to alloted id
    next_id = 0

    with open(root_path, mode='r', newline='', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = row.get(question_col, '').strip()
            answer = row.get(answer_col, '').strip()

            if question:
                if question in question_exists:
                    existing_entries = corpus[question_exists[question]]
                    existing_answers = [a for a in existing_entries['answers'].split('; ') if a]

                    if answer and answer not in existing_answers:
                        existing_entries['answers'] += '; ' + answer
                else:
                    corpus[next_id] = {"question": question, "answers": answer}
                    question_exists[question] = next_id
                    next_id += 1
            
    return corpus

#should lemmatise when i have time w pos tagging
def build_inverted_index(corpus):
    inverted_index, doc_vectors, doc_norms = defaultdict(lambda: defaultdict(int)), defaultdict(dict), defaultdict(float)
    N = len(corpus)
    # vectors and norms are for cosine similarity
    # vectors is the angle, norms is the magnitude, better to precompute than calculate for every query

    for doc_id, entry in corpus.items():
        tokens = stemmed_stopped_words(entry['question'])
        for token in tokens:
            inverted_index[token][doc_id] += 1

    for term, postings in inverted_index.items():
        df = len(postings)
        idf = math.log(N / df) if df > 0 else 0.0

        for doc_id, tf in postings.items():
            weight = math.log(1 + tf) * idf
            doc_vectors[doc_id][term] = weight
            doc_norms[doc_id] += weight ** 2

    for doc_id in doc_norms:
            doc_norms[doc_id] = math.sqrt(doc_norms[doc_id])

    index_data = {
        "inverted_index": dict(inverted_index),
        "doc_vectors": dict(doc_vectors),
        "doc_norms": dict(doc_norms),
        "corpus": corpus,
        "N": N
    }
    return index_data

def load_text_corpus(root_path):
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
    X, y = load_text_corpus(root_path)

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

def train(root_path, dump_path):
    X, y = load_text_corpus(root_path)

    pipeline = make_pipeline(TfidfVectorizer(tokenizer=stemmed_words, token_pattern=None, ngram_range=(1,2), use_idf=True, sublinear_tf=True), LinearSVC())

    pipeline.fit(X, y)
    os.makedirs(os.path.dirname(dump_path), exist_ok=True)
    dump(pipeline, dump_path)

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
    # scores_dict = train_test(root_path="corpus/top_level_intent", dump_path="dumps/top_level_intent_pipeline.joblib")
    # eval(scores_dict, output_path="evaluation")
    
    #csv_path = 'corpus/question_answer/question_answer_dataset.csv'
    #dump_path = 'dumps/qa_inverted_index.joblib'

    #print("Loading corpus")
    #corpus = load_csv_corpus(csv_path, question_col='Question', answer_col='Answer')
    #print("Loaded corpus")

    #print("Building inverted index")
    #index = build_inverted_index(corpus)
    #dump(index, dump_path)
    train(root_path="corpus/identity", dump_path="dumps/identity_intent_pipeline.joblib")