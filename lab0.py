#from bs4 import BeautifulSoup as bsoup
#from urllib import request
import nltk
#nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
#import sqlite3

###nltk.download_gui()

#url = " http ://www.gutenberg.org/files/84/84-0.txt"
#content = request.urlopen(url).read().decode('utf-8', errors='ignore')

#connection = sqlite3.connect('lab0.db')
#cursor = connection.cursor()
#connection.close()

text = " Artificial intelligence is cool but I am not too keen on Skynet."
text_tokens = word_tokenize(text)
tokens_without_sw = [word.lower() for word in text_tokens if not word in stopwords.words()]
print(tokens_without_sw)
filtered_sentence = (" ").join(tokens_without_sw)
print(filtered_sentence)

text_nltk = nltk.Text(tokens_without_sw)
print(text_nltk.count('cool'))

p_stemmer = PorterStemmer()
sb_stemmer = SnowballStemmer("english")
text = "This is a test sentence, and I am hoping it doesn’t get chopped up too much."
for token in word_tokenize(text):
    print(token, ":", p_stemmer.stem(token), ":", sb_stemmer.stem(token))

lemmatizer = nltk.WordNetLemmatizer()
text = "I am writing a few words, and I am hoping they don’t get chopped up too much."
for token in word_tokenize(text):
    print(token, ":", lemmatizer.lemmatize(token))
