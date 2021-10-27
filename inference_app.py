import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import numpy as np

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import re
from string import punctuation

# Load the Model
with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)
print('\nDe-serializing the Model'+'...'*5)

# Load the word vectorizer
with open('transformed.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
print('Loading the Vectorized words'+'...'*5)

# Load the custom stopwords
with open('stopwords_json.txt', 'r') as f:
    stopwords_json = f.read()
    f.close()
print('Loading the Custom Stopwords'+'...'*5)

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
    # Part of speech tagging
    tags = pos_tag(tokens)

    for (token, tag) in tags:
        # Remove all the irrelevant symbols from token
        token = re.sub(r'([0-9]+|[-_@./&+#]+|``)', '', token)
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

def get_sentiment(sent:str, loaded_model=model, vect=vectorizer):
    """
    Function to make predictions on the sentence.
    
    Args:
        sent (string) : The sentence entered by the user
        loaded_model : The sentiment Model
        vect : The word vectorizer

    Returns:
        String : The predicted Sentiment which can either be Positive or Negative.
    """
    # Clean the reviews
    review = preprocess_text(sent)
    movie_review_list = np.array([str(review)])
    data_count = vect.transform(movie_review_list)
    pred = loaded_model.predict(data_count)
    if pred == 1:
        return 'Positive ;)'
    return 'Negative :('