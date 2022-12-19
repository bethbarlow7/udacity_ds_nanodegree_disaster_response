import os
import json
import plotly
import numpy as np
import pandas as pd

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('wordnet')
import re

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.utils.multiclass import type_of_target


database_filepath = '/Users/bethbarlow/Documents/Nanodegree/udacity_ds_nanodegree_disaster_response/data/DisasterResponse.db'

model_filepath = '/Users/bethbarlow/Documents/Nanodegree/udacity_ds_nanodegree_disaster_response/models/classifier.pkl'


app = Flask(__name__)

def tokenize(text):
    
    """
    Word-tokenizes input text
    
    Inputs: text: text message requiring tokenization
    
    Outputs: clean_tokens: list of tokens from text message
    
    """
    
    # regular expression for urls
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # find urls in text message
    detected_urls = re.findall(url_regex, text)
    
    # loop over each detected url, replacing each with string
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # tokenize text message and save tokens in 'tokens' variable    
    tokens = word_tokenize(text)
    
    # initiate lemmatizer function
    lemmatizer = WordNetLemmatizer()

    # loop over tokens, lemmatizing, converting to lowercase and stripping whitespace
    # save cleaned tokens in clean_tokens variable
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    """
    Return single column dataframe of length equal to length of X, with booleans indicating whether first word in each entry of X is a verb or retweet
    
    Inputs: X: feature variable
    
    Outputs: Single column dataframe of booleans such that if row i = True, verb or retweet is first word in sentence of text entry i
    
    """
    
    
    def starting_verb(self, text):
        
    # tokenize by sentences
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            # tokenize each sentence into words and tag part of speech
                pos_tags = nltk.pos_tag(tokenize(sentence))

            # index pos_tags to get the first word and part of speech tag
                first_word, first_tag = pos_tags[0]
            
            # return true if the first word is an appropriate verb or RT for retweet
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return True
                return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        # apply starting_verb function to all values in X
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


# load data
engine = create_engine('sqlite:///' + database_filepath)
df = pd.read_sql_table(os.path.basename(database_filepath).replace(".db",""), engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()