from joblib import load

top_level_intent_pipeline = load("dumps/top_level_intent_pipeline.joblib")

def predict_top_level_intent(query):
    return top_level_intent_pipeline.predict([query])[0]