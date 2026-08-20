from handler import route_top_level_intent
from db_manager import reset_database

def main():
    reset_database()

    print("Plotbot: Hi there! I'm Plotbot. Before we begin, what is your name?")

    while True:
        query = input("You: ")
        response = route_top_level_intent(query)
        # have a response code. can include nlg or templates depending. lookup the code and find random response unless generated
        # append plotbot: to each newline or if multiline sentence e.g plot viewings, pad front spaces

        if response == "terminate":
            break

        print(response)

if __name__ == "__main__":
    main()