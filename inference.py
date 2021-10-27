import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

import pickle
import re
from string import punctuation

# Load the model
with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)
print('De-serializing the Model'+'...'*5)

# Load our word vectorizer
with open('transformed.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
print('Loading the Vectorized words'+'...'*5)

# Load the custom stopwords
with open('stopwords_json.txt', 'r') as f:
    stopwords_json = f.read()
    f.close()

# Load the Scrapped review of suicide_squad
reviews = pd.read_csv('Scrapped_review_suicide_squad.csv')
# Remove the Unnamed column from the dataFrame
reviews.drop('Unnamed: 0', axis=1, inplace=True)

# Initialize the Lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
stop_words_json = set(stopwords_json)
# Combine the punctuation with stopwords in nltk and stopwords in json
STOPWORDS_PUNCT = set.union(stop_words, stop_words_json, punctuation)

def preprocess_text(text):
    """Function to clean text from irrelevant words and symbols
    Args:
        text (string) : Data to clean
    
    Returns:
        string : Cleaned data
    """
    
    sentence = []
    # Tokenize and lowercase all alphabet
    tokens = [i.lower() for i in word_tokenize(text)]
    # Part of speech
    tags = pos_tag(tokens)
    
    for (token, tag) in tags:
        # Remove all irrelevant symbols from token
        token = re.sub(r"([0-9]+|[-_@./&+#]+|``)", '', token)
        token = re.sub(r"@[A-Za-z0-9_]+", '', token)

        # Grab the positions of the nouns(NN), verbs(VB), adverb(RB), and adjective(JJ)
        if tag.startswith('NN'):
            position = 'n'
        elif tag.startswith('VB'):
            position = 'v'
        elif tag.startswith('RB'):
            position = 'r'
        else:
            position = 'a'

        lemmatized_word = lemmatizer.lemmatize(token, position)
        if lemmatized_word not in STOPWORDS_PUNCT:
            sentence.append(lemmatized_word)
    final_sent = ' '.join(sentence)
    final_sent = final_sent.replace("n't", 'not').replace('br', '').replace('ii', '').replace('iii', '')
    final_sent = final_sent.replace("'s", " ").replace("''", " ")
    return final_sent

# Clean the reviews
reviews = reviews.Scrapped_review_suicide_squad.apply(preprocess_text)
for review in reviews:
    # keep movie review in a Numpy array
    movie_review_list = np.array([str(review)])
    # vectorize the movie reviews list
    data_count = vectorizer.transform(movie_review_list)
    # pass the data count to the model
    pred = model.predict(data_count)
    #print(pred)
    if pred == 1:
        print('\nThis is a Positive Sentiment!')
    else:
        print('\nThis is a Negative Sentiment!')

