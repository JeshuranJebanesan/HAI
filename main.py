from handler import route_intent
from db_manager import reset_database



def main():
    reset_database()

    print("Plotbot: Hi there! I'm Plotbot. Before we begin, what is your name?")

    while True:
        query = input("You: ")
        response = route_intent(query)

        if response == "terminate":
            break

        print(response)

if __name__ == "__main__":
    main()