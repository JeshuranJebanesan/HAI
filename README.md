### Plotbot: An Allotment Plot Chatbot

#### Intro
Plotbot is designed to lease allotment plots with other conversational functionality

It is capable of:
- identity management
- question answering
- small talk
- sentiment analysis
- transaction flow
- natural language generation

main.py
  - entry for chatbot
  - captures username on startup and begins conversation flow from user to chatbot
  - routes user queries to handler.py for appropriate response
  - capable of multi turn dialogue

handler.py + dumps/
  - loads models from dumps/
  - assigns models confidence threshold
  - routes user query from top level handler to sub domain handlers
  - each handler acts as a state machine taking state and intent as input
  - handlers use response.py to generate appropriate response
  - handlers can manipulate transaction.db using calls to db_manager.py

traintest.py + corpus/ + evaluation/
  - used for training models
  - loads input corpus, either self generated in corpus/, online from huggingface or a provided csv and returns X, y (data, label pair)
  - creates inverted index for qa, otherwise pipeline for similarity based intent matching classifier models
  - contains evaluation function to test capability of 5 models with outputs stored in evaluation/
  - models stored in dumps/

db_manager.py + database/transaction.db
  - inits/seeds/resets a 3nf relational database at database/transaction.db
  - contains tables for users, plots, crops, plantings (transactions)
  - contains functions for manipulating database
  - can create filtered views of plots or crops

response.py
  - stores response templates for handler.py to return to main.py
  - capable of personalisation such as different responses based on sentiment
  - can insert pos synonyms or general word categories such as greet, sorry...

#### Chatbot Architecture
#### Conversational Design
#### Evaluation
#### Discussion
