import random
import nltk
import re
from nltk.corpus import wordnet

nltk.download('wordnet')

def synonym(word, pos = None):
    synonyms = []

    for syn in wordnet.synsets(word, pos) if pos else wordnet.synset(word):
        for lemma in syn.lemmas:
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
        if stripped.startswith("-") or not stripped:
            formatted_lines.append(f"         {line}")
        else:
            formatted_lines.append(f"{prefix}{line}")

    return "\n".join(formatted_lines)

def apply_synonyms(template):
    # can use pos tagging here but might mess with qa or transactions so keep as regex
    def replace_pos(match):
        pos = match.group(1).lower()
        word = match.group(2)
        return synonym(word, pos)

    template = re.sub(r"\{(a|n|v|r):([a-zA-Z]+)\}", replace_pos, template)

    template = re.sub(r"\{greet\}", lambda _: random.choice(["Hello", "Hey", "Hi", "Greetings", "What's up", "Good day"]), template)

    from handler import conversation_context.get("name")


response_templates = {
    "general_fallback": [
        ""
    ]
}

def generate_response(key, **kwargs):
    options = response_templates["key"]
    template = random.choice(options)
    template = apply_synonyms(template)

    return prefix_lines(template.format(**kwargs))