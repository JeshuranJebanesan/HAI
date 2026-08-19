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

from db_manager import create_user, update_user_name, get_available_crops_for_plot, get_user_plantings, create_planting, get_available_plots

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
transaction_intent_pipeline = None

def load_top_level_intent_pipeline():
    global top_level_intent_pipeline
    top_level_intent_pipeline = load("dumps/top_level_intent_pipeline.joblib")

def load_qa_inverted_index():
    global qa_inverted_index
    qa_inverted_index = load("dumps/qa_inverted_index.joblib")

def load_identity_intent_pipeline():
    global identity_intent_pipeline
    identity_intent_pipeline = load("dumps/identity_intent_pipeline.joblib")

def load_transaction_intent_pipeline():
    global transaction_intent_pipeline
    transaction_intent_pipeline = load("dumps/transaction_intent_pipeline.joblib")

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
        return query.strip().title()

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
        return "Plotbot: Sorry I didn't get that. Could you tell me your name again (e.g 'Alice')"/

    sub_intent = predict_identity_intent(query)

    if sub_intent == "request_name":
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

### Discoverability

def discoverability():
    return "Plotbot: I'm built to help you rent plots and plant crops. But you can also ask me general questions, make small talk and even change your name!\nPlotbot: Example phrases you can use are \"I want to rent a plot\", \"how are you\", \"help me\" and \"exit\"."

### Transactions

def predict_plot_intent(query):
    if transaction_intent_pipeline is None:
        load_transaction_intent_pipeline()
    return transaction_intent_pipeline.predict([query])[0]

def reset_transaction_context():
    conversation_context["transaction"] = {
        "step": None,
        "selected_plot": None,
        "selected_crop": None
    }

def format_plot_list(plots):
    if not plots:
        return "No available plots found matching your criteria."
    res = "Available Plots:\n"
    for p in plots[:5]:
        res += f" - Plot #{p[0]}: {p[1]} sqm, {p[2]} soil, {p[3]} sun (£{p[4]}/mo)\n"
    return res.strip()

def handle_plot_query(query):
    tx = conversation_context["transaction"]
    step = tx.get("step")
    user_id = conversation_context.get("user_id")

    # add another classifier here or can make general sentiment classifier if the user expresses negative confirmation
    if query.lower() in ["cancel", "stop", "abort"]:
        reset_transaction_context()
        return "Plotbot: Transaction cancelled."

    if step == "selecting_plot":
        plot_match = re.search(r'\b(\d+)\b', query)
        if plot_match:
            plot_id = int(plot_match.group(1))
            tx["selected_plot"] = plot_id
            tx["step"] = "selecting_crop"

            crops = get_available_crops_for_plot(plot_id)
            crop_names = ", ".join([c[1] for c in crops]) if crops else "Any standard crop"
            
            return (f"Plotbot: Selected Plot #{plot_id}.\n"
                    f"Which crop would you like to plant? Suitable crops for this plot: {crop_names}")
        
        sub_intent = predict_plot_intent(query)
        if sub_intent == "filter_plots":
            return handle_plot_filtering(query)

        return "Plotbot: Please select a valid Plot ID number (e.g., '10') or type 'cancel'."

    elif step == "selecting_crop":
        crops = get_available_crops_for_plot(tx["selected_plot"])
        selected_crop = None
        
        for c in crops:
            if c[1].lower() in query.lower():
                selected_crop = c
                break

        if selected_crop:
            tx["selected_crop"] = selected_crop
            tx["step"] = "confirming"
            return (f"Plotbot: Confirm rental of Plot #{tx['selected_plot']} "
                    f"planting '{selected_crop[1]}'? (Type 'yes' to confirm or 'no' to cancel)")
        
        return "Plotbot: Please choose a valid crop from the recommended list or type 'cancel'."

    elif step == "confirming":
        if "yes" in query.lower() or "confirm" in query.lower():
            plot_id = tx["selected_plot"]
            crop_id = tx["selected_crop"][0]
            
            create_planting(plot_id, crop_id, user_id)
            reset_transaction_context()
            return f"Plotbot: Success! Plot #{plot_id} has been rented and planted."
        else:
            reset_transaction_context()
            return "Plotbot: Transaction cancelled."

    sub_intent = predict_plot_intent(query)

    if sub_intent == "view_plots":
        plantings = get_user_plantings(user_id)
        if not plantings:
            return "Plotbot: You are not currently renting any plots."
        msg = "Plotbot: Your current plots:\n"
        for p in plantings:
            msg += f" - Plot #{p[0]}: Planted with {p[1]} ({p[2]} soil, ${p[3]}/mo)\n"
        return msg

    elif sub_intent == "rent_plot":
        tx["step"] = "selecting_plot"
        plots = get_available_plots()
        return f"Plotbot: Let's rent a plot!\n{format_plot_list(plots)}\nWhich plot ID would you like to rent?"

    elif sub_intent == "filter_plots":
        tx["step"] = "selecting_plot"
        return handle_plot_filtering(query)

    return "Plotbot: I didn't understand your plot request."

def handle_plot_filtering(query):
    query_lower = query.lower()
    soil_type = None
    crop_name = None
    sort_by = None

    if "cheapest" in query_lower or "cheap" in query_lower:
        sort_by = "cheapest"
    
    for soil in ["loam", "clay", "sandy"]:
        if soil in query_lower:
            soil_type = soil
            break

    for crop in ["tomato", "carrot", "lettuce", "potato", "spinach", "mint"]:
        if crop in query_lower:
            crop_name = crop
            break

    plots = get_available_plots(soil_type=soil_type, crop_name=crop_name, sort_by=sort_by)
    return f"Plotbot: Here are the matching plots:\n{format_plot_list(plots)}\nWhich plot ID would you like to choose?"
# train transaction classifier and test. can use similar keyword extraction and flow to identity
# need to form templates and ontology using database and can implement general discoverability from there
# can also ask at beginning while user specifies identity, if user is a beginner for specific adapted language