#from bs4 import BeautifulSoup as bsoup
#from urllib import request
import nltk

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
#nltk.download('averaged_perceptron_tagger')
#nltk.download('universal_tagset')
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')
#import sqlite3

###nltk.download_gui()

#url = " http ://www.gutenberg.org/files/84/84-0.txt"
#content = request.urlopen(url).read().decode('utf-8', errors='ignore')

#connection = sqlite3.connect('lab0.db')
#cursor = connection.cursor()
#connection.close()

#text = " Artificial intelligence is cool but I am not too keen on Skynet."
#text_tokens = word_tokenize(text)
#tokens_without_sw = [word.lower() for word in text_tokens if not word in stopwords.words()]
#print(tokens_without_sw)
#filtered_sentence = (" ").join(tokens_without_sw)
#print(filtered_sentence)

#text_nltk = nltk.Text(tokens_without_sw)
#print(text_nltk.count('cool'))

#p_stemmer = PorterStemmer()
#sb_stemmer = SnowballStemmer("english")
#text = "This is a test sentence, and I am hoping it doesn’t get chopped up too much."
#for token in word_tokenize(text):
#    print(token, ":", p_stemmer.stem(token), ":", sb_stemmer.stem(token))

#lemmatizer = nltk.WordNetLemmatizer()
#text = "I am writing a few words, and I am hoping they don’t get chopped up too much."
#for token in word_tokenize(text):
#    print(token, ":", lemmatizer.lemmatize(token))

#lemmatizer = WordNetLemmatizer()
#text = "This is a test sentence, and I am hoping it doesn’t get chopped up too much."

#posmap = {
#    'ADJ': 'a',
#    'ADV': 'r',
#    'NOUN': 'n',
#    'VERB': 'v'
#}

#post = nltk.pos_tag(word_tokenize(text), tagset='universal')
#print(post)
#for token in post:
#    word = token[0]
#    tag = token[1]
#    if tag in posmap.keys():
#        print(word, ":", lemmatizer.lemmatize(word, posmap[tag]))
#    else:
#        print(word, ":", lemmatizer.lemmatize(word))

def process(user_input):
    tokens = nltk.word_tokenize(user_input)
    return tokens

def anglify(token):
    if token[-2:] == "or":
        print("1")
        return token[:-2] + "our"
    return token

def main():
#    while True:
#        user_input = input("You: ")
#        if user_input.lower() == "exit":
#            print("Exiting...")
#            break
#        processed_input = process(user_input)
#        print(f"Processed Input: {[anglify(token) for token in processed_input]}")
    print(stopwords.words('english'))


if __name__ == "__main__":
    main()