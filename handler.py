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

from db_manager import create_user, update_user_name, get_available_crops, get_user_plantings, create_planting, get_available_plots
from response import generate_response

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
sentiment_analysis_pipeline = None
small_talk_intent_pipeline = None

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

def load_small_talk_pipelines():
    global sentiment_analysis_pipeline, small_talk_intent_pipeline
    sentiment_analysis_pipeline = load("dumps/sentiment_analysis_pipeline.joblib")
    small_talk_intent_pipeline = load("dumps/small_talk_intent_pipeline.joblib")

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
        return generate_response("qa_fail")

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
        return generate_response("qa_low_confidence")

    reply = corpus[best_doc_id]['answers']
    return generate_response("qa_success", answer=reply)

def handle_question_answer(query):
    if qa_inverted_index is None:
        load_qa_inverted_index()
    answer = search_query(query, qa_inverted_index)
    return answer

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

def handle_name_capture(query):
    name = name_catch(query)
    if not name:
        return generate_response("identity_name_catch_failed")

    conversation_context["state"] = "idle"
    if conversation_context["name"]:
        return handle_update_name(name)

    return handle_set_name(name)

def handle_request_name(query):
    name = conversation_context["name"]
    return generate_response("request_name")

def handle_change_name(query):
    name = name_catch(query)
    if name and name.lower() != conversation_context["name"].lower():
        return handle_update_name(name)

    conversation_context["state"] = "awaiting_name"
    return generate_response("change_name")
    
def handle_set_name(name):
    conversation_context["name"] = name
    conversation_context["user_id"] = create_user(name)
    return generate_response("set_name")

def handle_update_name(name):
    user_id = conversation_context["user_id"]
    conversation_context["name"] = name
    update_user_name(user_id, name)
    return generate_response("update_name")

def route_identity_intent(query):
    state = conversation_context["state"]
    intent = predict_identity_intent(query) if state != "awaiting_name" else None

    match (state, intent):
        case ("awaiting_name", _):
            return handle_name_capture(query)
        case _:
            return handlers[intent](query)

### Discoverability

def discoverability(query):
    return "Plotbot: I'm built to help you rent plots and plant crops. But you can also ask me general questions, make small talk and even change your name!\nPlotbot: Example phrases you can use are \"I want to rent a plot\", \"how are you\", \"help me\" and \"exit\"."

### Transactions

# classify this if time
soil_types = ["clay", "loam", "sandy"]
sun_types = ["full sun", "partial shade", "shade"]
sort_types = [
    (("cheap", "lowest", "low", "cheapest", "price"), "price_asc"),
    (("expensive", "high", "highest"), "price_desc"),
    (("biggest", "big", "large", "largest", "size"), "size_desc"),
    (("smallest", "small", "tiny"), "size_asc")
]
crop_types = ["tomato", "carrot", "lettuce", "potato", "spinach", "mint", "cabbage", "onion", "radish", "green bean", "swiss chard", "peas", "kale", "zucchini", "cucumber", "pumpkin", "broccoli", "cauliflower", "brussels sprout", "asparagus"]

def predict_transaction_intent(query):
    if transaction_intent_pipeline is None:
        load_transaction_intent_pipeline()
    return transaction_intent_pipeline.predict([query])[0]

def plot_catch(query):
    q = query.lower()
    soil = next((s for s in soil_types if s in q), None)
    sun = next((s for s in sun_types if s in q), None)
    crop_name = next((s for s in crop_types if s in q), None)
    sort_by = next(
        (s for keywords, s in sort_types if any(kw in q for kw in keywords)), None
    )

    return soil, sun, crop_name, sort_by

def crop_catch(query):
    q = query.lower()
    soil = next((s for s in soil_types if s in q), None)
    sun = next((s for s in sun_types if s in q), None)
    plot_match = re.search(r'plot\s*(\d+)', q)
    plot_id = int(plot_match.group(1)) if plot_match else None

    return plot_id, sun, soil

def handle_cancel_transaction():
    conversation_context["transaction"] = {
        "selected_plot": None,
        "selected_crop": None
    }
    conversation_context["state"] = "idle"
    return generate_response("cancel_transaction")

def handle_view_transactions():
    plantings = get_user_plantings(conversation_context["user_id"])
    if not plantings:
        return generate_response("view_transactions_empty")

    reply = "\n".join([f"- {f'Plot #{p[0]}':<10} | {f'Crop: {p[1]}':<20} | Planted: {p[4]} | Harvest: {p[5]} | {f'Fee: £{p[3]}/mo':>7}" for p in plantings])
    return generate_response("view_transactions", lines = reply)

def handle_filter_options(query):
    q = query.lower()

    match = re.search(r'\b(plot|crop)s?\b', query.lower())
    first_keyword = match.group(1) if match else None

    if first_keyword == "crop":
        plot_id, sun, soil = crop_catch(q)
        table = get_available_crops(plot_id, sun, soil)
        if not table:
            return generate_response("filter_crops_empty")
        reply = "\n".join([f"- {f'{c[1]}':<20} | {f'Soil: {c[2]}':<20} | {f'Sun: {c[3]}':>20}" for c in table])
        return generate_response("filter_crops_success", crop_list = reply)
    elif first_keyword == "plot":
        soil, sun, crop_name, sort_by = plot_catch(q)
        table = get_available_plots(soil, sun, crop_name, sort_by)
        if not table:
            return generate_response("filter_plots_empty")
        reply = "\n".join([f"- {f'Plot #{p[0]}':<10} | {f'{p[1]}sqm':<10} | {f'{p[2]} soil':<10} | {f'{p[3]}':<10} | {f'${p[4]}/mo':>7}" for p in table])
        return generate_response("filter_plots_success", plot_list = reply)

    return generate_response("filter_invalid")

def handle_confirmation(query):
    q = query.lower()
    tx = conversation_context["transaction"]

    if any(kw in q for kw in ["yes", "confirm", "proceed"]):
        create_planting(tx["selected_plot"], tx["selected_crop"][0], conversation_context["user_id"])
        handle_cancel_transaction()
        return generate_response("confirm_success")
    elif any(kw in q for kw in ["cancel", "leave", "exit"]):
        return handle_cancel_transaction()
    elif "change plot" in q:
        tx["selected_plot"] = None
        conversation_context["state"] = "in_transaction"
        reply = handle_selection(tx['selected_crop'][1])
        return generate_response("clear_plot", nested_response = reply)
    elif "change crop" in q:
        tx["selected_crop"] = None
        conversation_context["state"] = "in_transaction"
        plot_str = f"Plot {tx['selected_plot']}"
        reply = handle_selection(plot_str)
        return generate_response("clear_crop", nested_response = reply)
    else:
        return generate_response("confirm_invalid")

def can_confirm():
    tx = conversation_context["transaction"]
    if tx["selected_plot"] and tx["selected_crop"]:
        conversation_context["state"] = "confirming_transaction"
        return generate_response("confirm_prompt", selected_plot = tx['selected_plot'], selected_crop = tx['selected_crop'][1])
    return None

def handle_selection(query):
    q = query.lower()
    tx = conversation_context["transaction"]
    conversation_context["state"] = "selecting_option"

    #valid plot num
    plot_match = re.search(r'plot\s*(\d+)', q)
    if plot_match:
        plot_id = int(plot_match.group(1))

        available_plots = get_available_plots()
        available_ids = [p[0] for p in available_plots]

        if plot_id not in available_ids:
            return generate_response("select_plot_invalid", id = plot_id)

        tx["selected_plot"] = plot_id
        prompt = can_confirm()
        if prompt: 
            return prompt
        matching_crops = get_available_crops(plot_id=plot_id)
        crops_str = ", ".join([c[1] for c in matching_crops])
        return generate_response("select_plot_need_crop", id = plot_id, crops = crops_str)
    
    all_crops = get_available_crops()
    for crop in all_crops:
        if crop[1].lower() in q:
            tx["selected_crop"] = (crop[0], crop[1])
            prompt = can_confirm()
            if prompt: 
                return prompt
            matching_plots = get_available_plots(crop_name=crop[1])
            reply = "\n".join([f"- {f'Plot #{p[0]}':<10} | {f'{p[1]}sqm':<10} | {f'£{p[4]}/mo':>10}" for p in matching_plots])
            return generate_response("select_crop_need_plot", crop_name=crop[1], plots_str = reply)

    return generate_response("select_invalid")

def handle_view_options():
    conversation_context["state"] = "in_transaction"
    plots = get_available_plots()
    crops = get_available_crops()

    # nlg this
    plot_str = "\n".join([f"- {f'Plot #{p[0]}':<10} | {f'{p[1]}sqm':<10} | {f'{p[2]} soil':<10} | {f'{p[3]}':<10} | {f'${p[4]}/mo':>7}" for p in plots])
    crop_str = ", ".join([c[1] for c in crops])

    return (
        f"Plotbot: Welcome to Plot Bookings!\n"
        f"Available Plots:\n{plot_str}\n"
        f"Available Crops:\n - {crop_str}\n"
        f"You can:\n"
        f"- Filter plots       e.g. 'Show cheap plots where I can plant tomatoes' or 'Show plots with loam soil'\n"
        f"- Filter crops       e.g. 'Show crops I can plant in plot 5' or 'Show crops that need partial shade'\n"
        f"- Select directly    e.g. 'Select Plot 2' or 'I want to plant apples'\n"
        f"- Cancel transaction e.g 'cancel'"
    )

def route_transaction_intent(query):
    state = conversation_context["state"]
    intent = predict_transaction_intent(query)

    # use text generation for training docs e.g i want to change {crop name}
    match (state, intent):
        case(_, "cancel"):
            return handle_cancel_transaction()
        case("confirming_transaction", _):
            return handle_confirmation(query)
        case("selecting_option", _):
            return handle_selection(query)
        case("in_transaction", "filter_options"):
            return handle_filter_options(query)
        case("in_transaction", "selection"):
            return handle_selection(query)
        case("idle" | "wellbeing_response", "view_options"):
            return handle_view_options()
        case("idle" | "wellbeing_response", "view_transactions"):
            return handle_view_transactions()
        case _:
            return f"Dont know {state} {intent}"
            
                        
# train transaction classifier and test. can use similar keyword extraction and flow to identity
# need to form templates and ontology using database and can implement general discoverability from there
# can also ask at beginning while user specifies identity, if user is a beginner for specific adapted language

### Smalltalk

def predict_small_talk_intent(query):
    if small_talk_intent_pipeline is None:
        load_small_talk_pipelines()
    return small_talk_intent_pipeline.predict([query])[0]

def predict_sentiment(query):
    if sentiment_analysis_pipeline is None:
        load_small_talk_pipelines()
    return sentiment_analysis_pipeline.predict([query])[0]

def handle_wellbeing_response(query):
    conversation_context["state"] = "idle"
    sentiment = predict_sentiment(query)
    return generate_response(sentiment)

def handle_wellbeing():
    conversation_context["state"] = "wellbeing_response"
    return generate_response("wellbeing")

def handle_weather():
    return generate_response("weather")

def handle_greetings():
    return generate_response("greeting")

def route_small_talk_intent(query):
    state = conversation_context["state"]
    intent = predict_small_talk_intent(query)

    match (state, intent):
        case("wellbeing_response", _):
            return handle_wellbeing_response(query)
        case(_, intent):
            handlers[intent](query)

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
    intent = predict_top_level_intent(query)

    match (state, intent):
        case ("awaiting_name", _):
            return route_identity_intent(query)
        case (_, "terminate"):
            return intent
        case("selecting_option" | "in_transaction" | "confirming_transaction", _):
            return route_transaction_intent(query)
        case("idle" | "wellbeing_response", _):
            return handlers[intent](query)
        case _:
            return f"Plotbot: Sorry, I didn't really understand that {state, intent}."

# inital traintest didnt keep underscores on qa and smalltalk. if time retrain with underscores for keys
handlers = {
    "identity": route_identity_intent,
    "discoverability": discoverability,
    "question_answer": handle_question_answer,
    "transaction": route_transaction_intent,
    "small_talk": route_small_talk_intent,
    "request_name": handle_request_name,
    "change_name": handle_change_name,
    "greetings": handle_greetings,
    "wellbeing": handle_wellbeing,
    "weather": handle_weather,
}