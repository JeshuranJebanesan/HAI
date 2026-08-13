from handler import predict_top_level_intent, load_top_level_intent_pipeline

def main():
    print("Plotbot: Hi there! I'm Plotbot. I can help you rent public plots of land for gardening and decide what to plant.\nAsk me for help at any time.\nType 'exit' to quit.")
    print("Plotbot: Before we begin, may I know your name?")
    while True:
        query = input("You: ")
        if query.lower() == 'exit':
            print("Plotbot: Goodbye!")
            break
        intent = predict_top_level_intent(query)
        print(f"Plotbot: {intent}") 

if __name__ == "__main__":
    main()