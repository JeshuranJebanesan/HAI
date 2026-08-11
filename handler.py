import re

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from joblib import load

p_stemmer = PorterStemmer()
analyzer = CountVectorizer().build_analyzer()

def stemmed_words(doc):
    text = re.sub(r"[^\w\s]", "", doc.lower())
    return [p_stemmer.stem(w) for w in analyzer(text)]

top_level_intent_pipeline = None

def load_top_level_intent_pipeline():
    global top_level_intent_pipeline
    top_level_intent_pipeline = load("dumps/top_level_intent_pipeline.joblib")

def predict_top_level_intent(query):
    if top_level_intent_pipeline is None:
        load_top_level_intent_pipeline()
    return top_level_intent_pipeline.predict([query])[0]