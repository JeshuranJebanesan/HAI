from handler import predict_top_level_intent, answer_question, handle_identity_query, discoverability, conversation_context
from db_manager import init_database, seed_database, reset_database

def main():
    reset_database()

    print("Plotbot: Hi there! I'm Plotbot. Before we begin, what is your name?")

    while conversation_context["awaiting_name"]:
        potential_name_capture = input("You: ")
        print(handle_identity_query(potential_name_capture))

    print(f"Plotbot: I'm here to help you rent plots and plant crops. Ask me if you want any help or type 'exit to quit.")

    # need to make flow better, can check if user is in locked flow in conversation context and skip top level intent 
    # skip if else for domain routing

    while True:
        query = input("You: ")
        intent = predict_top_level_intent(query)
        if intent == "terminate":
            print("Plotbot: Goodbye!")
            break
        elif intent == "identity":
            print(handle_identity_query(query))
        elif intent == "questionanswer":
            print(answer_question(query))
        elif intent == "discoverability":
            print(discoverability())
        else:
            print(intent) 

if __name__ == "__main__":
    main()