#import libraries
import sys
from sqlalchemy import create_engine
import nltk

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.utils.multiclass import type_of_target


def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///' +  database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
    y = df.iloc[:, 4:]
    X = df.message

def tokenize(text):
    
    """
    Word-tokenizes input text
    
    Inputs: text - text message requiring tokenization
    
    Outputs: clean_tokens - list of tokens from text message
    
    """
    
    #regular expression for urls
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    #find urls in text message
    detected_urls = re.findall(url_regex, text)
    
    #loop over each detected url, replacing each with string
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    #tokenize text message and save tokens in 'tokens' variable    
    tokens = word_tokenize(text)
    
    #initiate lemmatizer function
    lemmatizer = WordNetLemmatizer()

    #loop over tokens, lemmatizing, converting to lowercase and stripping whitespace
    #save cleaned tokens in clean_tokens variable
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    
    model_pipeline = Pipeline([
       ('vect', CountVectorizer(tokenizer=tokenize)),
       ('tfidf', TfidfTransformer()),
       ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    model = model_pipeline.fit(X_train, y_train)
    
def evaluate_model(model, X_test, Y_test, category_names):
    
    y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred, columns = category_names)

    for column in y_test.columns:
        print('Category: {}\n'.format(column))
        print(classification_report(y_test[column],y_pred_df[column]))
        

def save_model(model, model_filepath):
    # save the model to disk
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))

    
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()