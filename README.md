Plot Allotment Chatbot capable of

- identity management
  - stores, returns and changes name on request
- question answering
- small talk
- sentiment analysis
- transaction flow
- natural language generation

main.py
  entry for chatbot
  captures username on startup and begins conversation flow from user to chatbot
  routes user queries to handler.py for appropriate response
  capable of multi turn dialogue

handler.py
  loads models from dumps/
  routes user query from top level handler to sub domain handlers
  each handler acts as a state machine taking state and intent as input
  handlers use response.py to generate appropriate response

traintest.py
  used for training models
  loads input corpus, either self generated in corpus/, online from huggingface or a provided csv and returns X, y (data, label pair)
  creates inverted index for qa, otherwise pipeline for similarity based intent matching classifier models
  contains evaluation function to test capability of 5 models
  models stored in dumps/

db_manager.py
  inits/seeds/resets a 3nf relational database at database/transaction.db
  contains tables for users, plots, crops, plantings (transactions)
  can create filtered views of plots or crops

response.py
  stores response templates for handler.py to return to main.py
  capable of personalisation such as different responses based on sentiment
  can insert pos synonyms or general word categories such as greet, sorry...
