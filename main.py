from handler import predict_top_level_intent, load_top_level_intent_pipeline

def classify_intent(query):
    return predict_top_level_intent(query)

def main():
    print("Loading the top-level intent classification model...")
    load_top_level_intent_pipeline()
    print("Model loaded successfully!")
    print("Plotbot: Hi there! I'm Plotbot. I can help you rent public plots of land for gardening and decide what to plant. Type 'exit' to quit.")

    while True:
        query = input("You: ")
        if query.lower() == 'exit':
            print("Plotbot: Goodbye!")
            break
        intent = classify_intent(query)
        print(f"Plotbot: {intent}") 

if __name__ == "__main__":
    main()