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
 
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
                pos_tags = nltk.pos_tag(tokenize(sentence))
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return True
        return False

    def fit(self, X, Y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')

def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    category_names = df.iloc[:,4:].columns
    category_boolean = (df.iloc[:,4:] != 0).sum().values
    category_df = pd.DataFrame(list(zip(category_names.tolist(), category_boolean.tolist())), columns =['Category Name', 'Counts']).sort_values('Counts', ascending = False)
    
    def genre_count_df(genre):
        
        """
        Returns a dataframe containing counts of messages in each category where genre is equal to input genre
        
        Inputs: string containing name of input genre
        Output: dataframe with genre, category and message counts
        
        """
    
        category_names = df.iloc[:,4:].columns
        category_boolean = (df[df['genre'] == genre].iloc[:,4:] != 0).sum().values
        
        genre_list = [genre] * 36
        category_df = pd.DataFrame(list(zip(genre_list, category_names.tolist(), category_boolean.tolist())), columns =['Genre', 'Category Name', 'Counts'])
        
        return category_df.sort_values('Counts', ascending = False).head(5)
    
    # create visuals
    graphs = [
                # GRAPH 1 - genre graph
            {
                'data': [
                    Bar(
                        x=genre_names,
                        y=genre_counts,
                        marker={'color': 'firebrick'}
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
            },
        
                # GRAPH 2 - category graph    
            {
                'data': [
                    Bar(
                        x=category_df['Category Name'],
                        y=category_df['Counts'],
                        marker={'color': 'tomato'}
                    )
                ],

                'layout': {
                    'title': 'Distribution of Message Categories',
                    'yaxis': {
                        'title': "Count"
                    },
                    'xaxis': {
                        'title': "Category",
                        'tickangle': 35
                    }
                }
            },
        
                # GRAPH 3 - category graph for news messages
            {
                    'data': [
                        Bar(
                            x=genre_count_df('news')['Category Name'],
                            y=genre_count_df('news')['Counts'],
                            marker={'color': 'seagreen'}
                        )
                    ],

                    'layout': {
                        'title': 'Top 5 News Message Categories',
                        'yaxis': {
                            'title': "Count"
                        },
                        'xaxis': {
                            'title': "Category",
                            'tickangle': 35
                        }
                    }
              },
        
        
           # GRAPH 4 - category graph for direct messages
            {
                    'data': [
                        Bar(
                            x=genre_count_df('direct')['Category Name'],
                            y=genre_count_df('direct')['Counts'],
                            marker={'color': 'lightseagreen'}
                        )
                    ],

                    'layout': {
                        'title': 'Top 5 Direct Message Categories',
                        'yaxis': {
                            'title': "Count"
                        },
                        'xaxis': {
                            'title': "Category",
                            'tickangle': 35
                        }
                    }
              },
        
            # GRAPH 5 - category graph for social messages
             {
                    'data': [
                        Bar(
                            x=genre_count_df('social')['Category Name'],
                            y=genre_count_df('social')['Counts'],
                            marker={'color': 'darkturquoise'}
                        )
                    ],

                    'layout': {
                        'title': 'Top 5 Social Message Categories',
                        'yaxis': {
                            'title': "Count"
                        },
                        'xaxis': {
                            'title': "Category",
                            'tickangle': 35
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