import numpy
import scipy
import pickle
from joblib import dump, load
from urllib import request
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

some_object1 = {"a": 1, "b": 2, "c": 3}
some_object2 = [1, 2, 3, 4, 5]

with open("filename.pkl", "wb") as f:
    pickle.dump(some_object1, f )
with open("filename.pkl", "rb") as f:
    some_object1 = pickle.load(f)

dump(some_object2, "filename.joblib")
some_object2 = load("filename.joblib")

docs_urls = {
    "Russia": "http://www.gutenberg.org/cache/epub/13437/pg13437.txt",
    "France": "http://www.gutenberg.org/cache/epub/10577/pg10577.txt",
    "England": "http://www.gutenberg.org/cache/epub/10135/pg10135.txt",
    "USA": "http://www.gutenberg.org/cache/epub/10947/pg10947.txt",
    "Spain": "http://www.gutenberg.org/cache/epub/9987/pg9987.txt",
    "Scandinavia": "http://www.gutenberg.org/cache/epub/5336/pg5336.txt",
    "Iceland": "http://www.gutenberg.org/cache/epub/5603/pg5603.txt"
}

documents = {}

for country in docs_urls.keys():
    content = request.urlopen(docs_urls[country]).read().decode('utf-8', errors='ignore')
    documents[country] = content

all_text = documents.values()

count_vect = CountVectorizer(stop_words=stopwords.words('english'))
X_train_counts = count_vect.fit_transform(all_text)
tf_transformer = TfidfTransformer(use_idf=True, sublinear_tf=True).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)

print(X_train_tf.shape)