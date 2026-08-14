from handler import predict_top_level_intent, answer_question

def main():
    print("Plotbot: Hi there! I'm Plotbot. I can help you rent public plots of land for gardening and decide what to plant.\nAsk me for help at any time.\nType 'exit' to quit.")
    while True:
        query = input("You: ")
        if query.lower() == 'exit':
            print("Plotbot: Goodbye!")
            break
        intent = predict_top_level_intent(query)
        if intent == "questionanswer":
            print(f"Plotbot: {answer_question(query)}")
        else:
            print(f"Plotbot: {intent}") 

if __name__ == "__main__":
    main()