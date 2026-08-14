import re
import math

import nltk
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from joblib import load
from collections import defaultdict

from db_manager import create_user, update_user_name

# nltk.download('stopwords', quiet=True)

conversation_context = {
    "user_id": None,
    "name": None,
    "awaiting_name" : True,
    "transaction" : {}
}

p_stemmer = PorterStemmer()
analyzer = CountVectorizer().build_analyzer()

def stemmed_words(doc):
    text = re.sub(r"[^\w\s]", "", doc.lower())
    return [p_stemmer.stem(w) for w in analyzer(text)]

def stemmed_stopped_words(doc):
    tokens = re.findall(r'\b\w+\b', doc.lower())
    return [p_stemmer.stem(w) for w in tokens if w not in stopwords.words('english') and not w.isdigit()]

### Load Models

top_level_intent_pipeline = None
qa_inverted_index = None
identity_intent_pipeline = None

def load_top_level_intent_pipeline():
    global top_level_intent_pipeline
    top_level_intent_pipeline = load("dumps/top_level_intent_pipeline.joblib")

def load_qa_inverted_index():
    global qa_inverted_index
    qa_inverted_index = load("dumps/qa_inverted_index.joblib")

def load_identity_intent_pipeline():
    global identity_intent_pipeline
    identity_intent_pipeline = load("dumps/identity_intent_pipeline.joblib")

### Top Level Intent

def predict_top_level_intent(query):
    if top_level_intent_pipeline is None:
        load_top_level_intent_pipeline()
    return top_level_intent_pipeline.predict([query])[0]

### Question Answer

def search_query(query, index_data, confidence_threshold=0.1):
    inverted_index = index_data["inverted_index"]
    doc_vectors = index_data["doc_vectors"]
    doc_norms = index_data["doc_norms"]
    corpus = index_data["corpus"]
    N = index_data["N"]

    q_tokens = stemmed_stopped_words(query)
    q_counts = {}
    q_vector = {}
    q_norm_sq = 0.0
    
    for w in q_tokens:
        q_counts[w] = q_counts.get(w, 0) + 1

    for term, tf in q_counts.items():
        if term in inverted_index:
            df = len(inverted_index[term])
            idf = math.log(N / df)
            weight = math.log(1 + tf) * idf
            q_vector[term] = weight
            q_norm_sq += weight ** 2

    q_norm = math.sqrt(q_norm_sq)

    if q_norm == 0.0:
        # this is fallback by warning message
        # if there's time can switch to fallback by query expansion
        return "Sorry, I don't understand. Could you rephrase your question?"

    scores = defaultdict(float)
    for term, q_weight in q_vector.items():
        for doc_id in inverted_index[term]:
            scores[doc_id] += q_weight * doc_vectors[doc_id][term]

    for doc_id in scores:
        if doc_norms[doc_id] > 0:
            scores[doc_id] /= (q_norm * doc_norms[doc_id])
        else:
            scores[doc_id] = 0.0

    best_doc_id, best_score = sorted(scores.items(), key=lambda x: x[1], reverse=True)[0]

    if best_score < confidence_threshold:
        return "Sorry, I don't understand. Could you rephrase your question?"

    return corpus[best_doc_id]['answers']

def answer_question(query):
    if qa_inverted_index is None:
        load_qa_inverted_index()
    return "Plotbot: search_query(query, qa_inverted_index)"

### Identity

def name_catch(query):
    tokens = word_tokenize(query.strip())
    tagged = pos_tag(tokens)
    chunks = ne_chunk(tagged)

    for chunk in chunks:
        if isinstance(chunk, Tree) and chunk.label() == "PERSON":
            return " ".join(c[0] for c in chunk.leaves()).title()

    patterns = [
            r"i'm ([a-zA-Z]+)",
            r"i am ([a-zA-z]+)",
            r"my name is ([a-zA-Z]+)",
            r"call me ([a-zA-Z]+)",
            r"it's ([a-zA-Z]+)",
            r"it is ([a-zA-Z]+)"
    ]   

    for p in patterns:
            m = re.search(p, query.lower())
            if m:
                return m.group(1).title()

    if re.fullmatch(r"[a-zA-Z]+", query.strip()):
        name = query.strip().title()

    if name:
        return name

    return None

def predict_identity_intent(query):
    if identity_intent_pipeline is None:
        load_identity_intent_pipeline()
    return identity_intent_pipeline.predict([query])[0]

def handle_identity_query(query):
    if conversation_context.get("awaiting_name"):
        name = name_catch(query)
        if name:
            conversation_context["awaiting_name"] = False
            if not conversation_context["name"]:
                conversation_context["name"] = name
                user_id = create_user(name)
                conversation_context["user_id"] = user_id
                return f"Plotbot: Nice to meet you {name}!"
            else:
                user_id = conversation_context["user_id"]
                conversation_context["name"] = name
                update_user_name(user_id, name)
                return f"Plotbot: Cool! Your name has been updated to {name}."
        return "Plotbot: Sorry I didn't get that. Could you tell me your name again (e.g 'Alice')"

    sub_intent = predict_identity_intent(query)

    if sub_intent == "get_name":
        name = conversation_context["name"]
        return f"Your name is {name}!"
    elif sub_intent == "change_name":
        name = name_catch(query)
        if name and name.lower() != conversation_context["name"].lower():
            user_id = conversation_context["user_id"]
            update_user_name(user_id, name)
            conversation_context["name"] = name
            return f"Plotbot: Updated! Your name is now {name}."
        else:
            conversation_context["awaiting_name"] = True
            return f"Plotbot: Sure thing! What would you like to change your name to?"

    return "Plotbot: Sorry, I didn't really understand that."