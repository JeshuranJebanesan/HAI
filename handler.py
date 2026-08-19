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
    "state": "awaiting_name",
    "user_id": None,
    "name": None,
    "transaction" : {
        "selected_plot": None,
        "selected_crop": None
    }
}

p_stemmer = PorterStemmer()
analyzer = CountVectorizer().build_analyzer()
sw = set(stopwords.words('english'))

def stemmed_words(doc):
    text = re.sub(r"[^\w\s]", "", doc.lower())
    return [p_stemmer.stem(w) for w in analyzer(text)]

def stemmed_stopped_words(doc):
    tokens = re.findall(r'\b\w+\b', doc.lower())
    return [p_stemmer.stem(w) for w in tokens if w not in sw and not w.isdigit()]

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

def handle_question_answer(query):
    if qa_inverted_index is None:
        load_qa_inverted_index()
    answer = search_query(query, qa_inverted_index)
    return f"Plotbot: {answer}"

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
    if conversation_context["state"] == "awaiting_name":
        name = name_catch(query)
        if name:
            conversation_context["state"] = "idle"
            if not conversation_context["name"]:
                conversation_context["name"] = name
                user_id = create_user(name)
                conversation_context["user_id"] = user_id
                return f"Plotbot: Nice to meet you {name}!\nPlotbot: I'm here to help you rent plots and plant crops. Ask me if you want any help or type 'exit' to quit."
            else:
                user_id = conversation_context["user_id"]
                conversation_context["name"] = name
                update_user_name(user_id, name)
                return f"Plotbot: Cool! Your name has been updated to {name}."
        return "Plotbot: Sorry I didn't get that. Could you tell me your name again (e.g 'Alice')."

    sub_intent = predict_identity_intent(query)

    if sub_intent == "request_name":
        name = conversation_context["name"]
        return f"Your name is {name}!"
    elif sub_intent == "change_name":
        name = name_catch(query)
        if name and name.lower() != conversation_context["name"].lower():
            user_id = conversation_context["user_id"]
            conversation_context["name"] = name
            update_user_name(user_id, name)
            return f"Plotbot: Updated! Your name is now {name}."
        else:
            conversation_context["state"] = "awaiting_name"
            return f"Plotbot: Sure thing! What would you like to change your name to?"

    return "Plotbot: Sorry, I didn't really understand that."

def handle_name_capture(query):
    name = name_catch(query)
    if not name:
        return "Plotbot: Sorry I didn't get that. Could you tell me your name again (e.g 'Alice')."

    conversation_context["state"] = "idle"
    if conversation_context["name"]:
        return handle_update_name(name)

    return handle_set_name(name)

def handle_request_name(query):
    name = conversation_context["name"]
    return f"Plotbot: Your name is {name}!"

def handle_set_name(name):
    conversation_context["name"] = name
    conversation_context["user_id"] = create_user(name)
    return f"Plotbot: Nice to meet you {name}!\nPlotbot: I'm here to help you rent plots and plant crops. Ask me if you want any help or type 'exit' to quit."

def handle_update_name(name):
    user_id = conversation_context["user_id"]
    conversation_context["name"] = name
    update_user_name(user_id, name)
    return f"Plotbot: Cool! Your name has been updated to {name}."

def route_identity_intent(query):
    state = conversation_context["state"]
    intent = predict_identity_intent(query) if state == "idle" else None

    match (state, intent):
        case ("awaiting_name", _):
            return handle_name_capture(query)

        case ("idle", _):
            return handlers[intent](query)

        case _:
            return "Plotbot: Sorry, I didn't really understand that."

### Discoverability

def discoverability(query):
    return "Plotbot: I'm built to help you rent plots and plant crops. But you can also ask me general questions, make small talk and even change your name!\nPlotbot: Example phrases you can use are \"I want to rent a plot\", \"how are you\", \"help me\" and \"exit\"."

### Transactions

def predict_transaction_intent(query):
    if transaction_intent_pipeline is None:
        load_transaction_intent_pipeline()
    return transaction_intent_pipeline.predict([query])[0]

def reset_transaction_context():
    conversation_context["transaction"] = {
        "selected_plot": None,
        "selected_crop": None
    }
    conversation_context["state"] = "idle"

def format_plot_list(plots):
    if not plots:
        return "No available plots found matching your criteria."
    res = "Plots:\n"
    for p in plots:
        res += f" - Plot #{p[0]}: {p[1]} sqm, {p[2]} soil, {p[3]} sun (£{p[4]}/mo)\n"
    return res.strip()

def plot_catch(query):
    match = re.search(r'\b(\d+)\b', query)
    return int(match.group(1)) if match else None

def crop_catch(query, available_crops):
    query_lower = query.lower()
    for crop in available_crops:
        if crop[1].lower() in query_lower:
            return crop
    return None

def handle_rent_plot(query):
    conversation_context["state"] = "selecting_plot"
    plots = get_available_plots()
    return f"Plotbot: Let's rent a plot!\n{format_plot_list(plots)}\nWhich plot ID would you like to rent?"

def handle_view_plots(query):
    return

def handle_filter_plots(query):
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

def handle_transaction_query(query):
    tx = conversation_context["transaction"]
    state = conversation_context["state"]
    user_id = conversation_context["user_id"]

    if state == "selecting_plot":
        plot_match = re.search(r'\b(\d+)\b', query)
        if plot_match:
            plot_id = int(plot_match.group(1))
            tx["selected_plot"] = plot_id
            tx["state"] = "selecting_crop"

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
        return handle_plot_filtering(query)

    return "Plotbot: I didn't understand your plot request."

def route_transaction_intent(query):
    state = conversation_context["state"]
    intent = predict_transaction_intent(query)
    user_id = conversation_context["user_id"]
    tx = conversation_context["transaction"]

    print(intent)

    match state:
        case ("selecting_plot", cancel):
        case ("selecting_plot", filter):
        case ("selecting_plot", _):

        case ("selecting_crop", cancel):
        case ("selecting_crop", _):

        case ("confirming_transaction", _):

        case("idle, _"):
                
                        
# train transaction classifier and test. can use similar keyword extraction and flow to identity
# need to form templates and ontology using database and can implement general discoverability from there
# can also ask at beginning while user specifies identity, if user is a beginner for specific adapted language

### Smalltalk

def handle_small_talk():
    return

### Top Level Intent

# can also refactor all intent handlers to implement sub intent handler functions and map the handlers here.
# for now identity is small enough without reused code to go without but can look at transaction handler.

def predict_top_level_intent(query):
    if top_level_intent_pipeline is None:
        load_top_level_intent_pipeline()
    return top_level_intent_pipeline.predict([query])[0]

# going to route given state then checking intent.
# should look into this as its interesting. theres a few global intents like terminate which always have the same result so no point writing for each state
# is there a cleanest coding for states and inputs e.g given n states, if there is an intent which has the same result for n-1 states, should i route that intent first instead of state

def route_top_level_intent(query):
    state = conversation_context["state"]
    intent = predict_top_level_intent(query) if state == "idle" else None
    print(state, intent)

    match (state, intent):
        case ("awaiting_name", _):
            return handle_identity_query(query)
        case (_, "terminate"):
            return intent
        case("selecting_plot" | "selecting_crop | confirming_transaction", _):
            return handle_transaction_query(query)
        case("idle", _):
            return handlers[intent](query)
        case _:
            return f"Unknown Error"

handlers = {
    "identity": handle_identity_query,
    "discoverability": discoverability,
    "questionanswer": handle_question_answer,
    "smalltalk": handle_small_talk,
    "transaction": handle_transaction_query,
    "rent_plot": handle_rent_plot,
    "view_plots": handle_view_plots,
    "filter_plots": handle_filter_plots
}