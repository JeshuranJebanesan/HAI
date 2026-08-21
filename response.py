import random
import nltk
import re
from nltk.corpus import wordnet

nltk.download('wordnet', quiet=True)

def synonym(word, pos = None):
    synonyms = []

    for syn in wordnet.synsets(word, pos) if pos else wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name().replace("_", " "))

    synonyms = list(set(synonyms))
    if word in synonyms:
        synonyms.remove(word)

    return random.choice(synonyms) if synonyms else word

def prefix_lines(text, prefix="Plotbot: "):
    lines = text.split("\n")
    formatted_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("-"):
            formatted_lines.append(f"         {line}")
        elif line.startswith(" "):
            formatted_lines.append(f"        {line}")
        else:
            formatted_lines.append(f"{prefix}{line}")

    return "\n".join(formatted_lines)

def apply_synonyms(template):
    # can use pos tagging here but might mess with qa or transactions so keep as regex
    def replace_pos(match):
        pos = match.group(1).lower()
        word = match.group(2)
        syn_word = synonym(word, pos)

        start_idx = match.start()
        prefix = template[:start_idx].rstrip()

        if not prefix or prefix[-1] in ".!?\n":
            return syn_word.capitalize()
        return syn_word

    template = re.sub(r"\{(a|n|v|r):([a-zA-Z]+)\}", replace_pos, template)

    template = re.sub(r"\{greet\}", lambda _: random.choice(["Hello", "Hey", "Hi", "Greetings", "What's up", "Good day"]), template)
    template = re.sub(r"\{sorry\}", lambda _: random.choice(["Sorry, ", "Apologies, ", "My bad but", "I apologise, ", "I am most remorseful but ", "Ah, ", "Forgive me, "]), template)

    # should implement a cleaner fix for getting name without non circular imports but format kwaargs method too inconsistent. could just implement oop
    from handler import conversation_context
    name = conversation_context.get("name") or "there"
    template = re.sub(r"\{name\}", name, template)

    return template

response_templates = {
    "general_fallback" : [
        "I don't really understand."
    ],

    # Question Answer
    "qa_success": [
        "{answer}",
        "Check out what I found: {answer}",
        "Hahahaha. I'm a genius. Here you go, {answer}"
    ],
    "qa_failure": [
        "{sorry} I don't know the answer to your question."
    ],
    "qa_low_confidence": [
        "{sorry} I am not confident I can answer your question. Could you rephrase it?",
        "{sorry} I couldn't find an answer for that. Could you ask in another way?"
    ],

    # Identity

    "identity_name_catch_failed": [
        "{sorry} I didnt catch your name.\n Please tell me one more time. For example:\n- 'My name is Alice'\n- 'Bob'\n- 'Call me Carl'",
        "{sorry} my ears don't work like they used to.\n You might have to repeat your name once more. For example:\n- 'I'm Dan'\n- 'It is Ethan'\n- 'Fyodor'"
    ],
    "set_name": [
        "Nice to meet you {name}!\n I'm here to help you rent plots and plant crops.\n Ask me if you want any help or type 'exit' to quit."
    ],
    "update_name": [
        "{a:wonderful}! Your name has been updated to {name}.",
        "{a:good}! You shall henceforth be known as {name}..."
    ],
    "request_name": [
        "Your name is {name}!",
        "It's somewhere here.\n ...\n ..\n Ah!\n It's {name}.\n Of course..."
    ],
    "change_name": [
        "Sure thing! What would you like to change your name to?"
    ],

    # Discoverability

    "discoverability": [
        "I'm built to help you rent plots and plant crops. But you can also ask me general questions, make small talk and even change your name!\nPlotbot: Example phrases you can use are \"Show me available plots\", \"how are you\", \"help me\" and \"exit\"."
    ],

    # Transaction

    "cancel_transaction": [
        "Understood. The booking has been cancelled.",
        "Affirmative. The transaction has been cancelled."
    ],
    "view_transactions_empty": [
        "You have no active plantings! Let's get to work.",
        "You haven't any bookings dear {n:friend}. Let's make a change!"
    ],
    "view_transactions": [
        "Your current active bookings:\n\n{lines}"
    ],
    "filter_crops_empty": [
        "No crops match those criteria!\n\n Keep in mind, if filtering for a plot, the plot will overwrite your sun and soil filters."
    ],
    "filter_crops_success": [
        "These crops match your criteria:\n\n{crop_list}"
    ],
    "filter_plots_empty": [
        "No plots match those criteria!\n\n Keep in mind, if filtering for a crop, the crop will overwrite your sun and soil filters."
    ],
    "filter_plots_success": [
        "These plots match your criteria:\n\n{plot_list}"
    ],
    "filter_invalid": [
        "{sorry} I didn't understand. As a tip,\n\n When filtering for crops, try 'Show crops where ...'\n\n When filtering for plots, try 'Show plots where ...'"
    ],
    "confirm_success": [
        "Transaction successfully confirmed!"
    ],
    "confirm_prompt": [
        "Selection Complete!\n\n- Selected Plot: #{selected_plot}\n- Selected Crop: {selected_crop}\n\nWould you like to confirm? (Type 'yes', 'cancel', 'change plot' or 'change crop')"
    ],
    "confirm_invalid": [
        "{sorry} I don't understand what you mean.\n\n Type 'yes', 'cancel', 'change plot' or 'change crop'."
    ],
    "clear_plot": [
        "Plot selection cleared.\n{nested_response}"
    ],
    "clear_crop": [
        "Crop selection cleared.\n{nested_response}"
    ],
    "select_plot_invalid": [
        "Plot #{id} is either invalid or currently occupied. Please choose an available plot."
    ],
    "select_plot_need_crop": [
        "Selected Plot #{id}.\n\n Suitable crops for this plot: {crops}.\n\n Which crop would you like?"
    ],
    "select_crop_need_plot": [
        "Selected Crop: {crop_name}.\nSuitable plots for this crop:\n{plots_str}\nWhich plot would you like?"
    ],
    "select_invalid": [
        "Please select a valid plot number (e.g. 'Plot 1') or crop name (e.g. 'Tomato')."
    ],
    "view_options": [
        "Welcome to Plot Bookings!\n\n Available Plots:\n{plot_str}\n\n Available Crops:\n- {crop_str}\n\nYou can:\n- Filter plots       e.g. 'Show cheap plots where I can plant tomatoes'\n- Filter crops       e.g. 'Show crops I can plant in plot 5'\n- Select directly    e.g. 'Select Plot 2'\n- Cancel transaction e.g 'cancel'"
    ],

    # Small Talk

    "positive": [
        "I'm {a:glad} to hear it {name}!",
        "Sounds {a:wonderful}. I hope it keeps up!",
        "Cool beans :) I'm happy to hear that from you!"
    ],
    "negative": [
        "That sounds tough. Keep your head up {name}.",
        "I believe in you {name}. You got this.",
        "I'm sorry to hear that. I just know you'll get over this {r:valiantly}."
    ],
    "neutral": [
        "Okay, understood.",
        "Fair enough.",
        "Chill."
    ],
    "weather": [
        "It's always sunny in Philadelphia!",
        "I heard there will be {a:unbearable} rain later today.\n Good thing I love jumping in muddy puddles!",
        "OH MY GOSH THERE'S A SOLAR ECLIPSE!!!"
    ],
    "greeting": [
        "{greet} {name}! What can I do for you?",
        "{greet}. Nice to see you today {name}!",
        "Well, well, well. I've been expecting you {name}...\n How can I help you today >:)"
    ],
    "wellbeing": [
        "I'm feeling {a:positive}, thanks for asking. How are you?",
        "I'm doing great {name}. What about you?",
        "It's a {a:simple} life being a transactional, conversational plot booking chatbot.\n Not a {n:worry} in the world!\n How is it on the other side of the screen?"
    ]
}

def generate_response(key, **kwargs):
    options = response_templates.get(key, response_templates["general_fallback"])
    template = random.choice(options)
    template = apply_synonyms(template)

    try:
        formatted = template.format(**kwargs)
    except KeyError:
        formatted = template

    return prefix_lines(formatted)