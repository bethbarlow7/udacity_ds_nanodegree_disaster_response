"""
Model Training Script

Arguments:
    -  Path to SQLite destination database (e.g. DisasterResponse.db)
    -  Path to trained model destination (e.g. classifier.pkl)
    
Sample Script Execution:
> python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl   
    
"""

#import libraries
import sys
import os
from sqlalchemy import create_engine
import nltk
import pickle

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

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
    
    """
    Loads data from database and splits into target and feature variables
    
    Inputs: df: dataframe created from DisasterResponse table in SQLite database
    
    Outputs: X: feature variable columns of df
             Y: target variable column of df
    """
    
    # load data from database
    engine = create_engine('sqlite:///' +  database_filepath)
    df = pd.read_sql_table(os.path.basename(database_filepath).replace(".db",""), engine)
    
    # split into target and feature variable columns
    Y = df.iloc[:, 4:]
    X = df['message']
    category_names = Y.columns
    
    return X, Y, category_names

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
    Returns single column dataframe of length equal to length of X, with booleans indicating whether first word in each entry of X is a verb or retweet
    
    Inputs: X: feature variable
    
    Outputs: Single column dataframe of booleans such that if row i = True, first word in sentence of text entry i is a verb or retweet
    
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

    def fit(self, X, Y=None):
        return self

    def transform(self, X):
        # apply starting_verb function to all values in X
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def build_model():
    
    """
    Returns optimal pipeline that comprises multiple feature engineering transforms and a classifier estimator
    
    Inputs: N/A
    
    Outputs: pipeline of transforms with a final classifier estimator
    
    """
    
    # original pipeline before model optimisation
    model_pipeline_1 = Pipeline([
       ('vect', CountVectorizer(tokenizer=tokenize)),
       ('tfidf', TfidfTransformer()),
       ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    # pipeline with inclusion of StartingVerbExtractor() result as additional feature
    model_pipeline_2 = Pipeline([
    
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            
            # add StartingVerbExtractor() to improves model performance
            ('starting_verb', StartingVerbExtractor())
        ])),
        
        # use Adaboost to improve model performance
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    return model_pipeline_2
    
def evaluate_model(model, X_test, Y_test, category_names):
    
    """
    Returns classification report for each category in test data
    
    Inputs: model - trained classification model
            X_test - feature variable test set
            Y_test - target variable test set
            category_names - list of column names representing message categories
            
    Outputs: classification report including precision, recall and f1 score for each category in test data
    
    """
    
    Y_pred = model.predict(X_test)
    Y_pred_df = pd.DataFrame(Y_pred, columns = category_names)

    # loop over categories and print classification report
    for column in Y_test.columns:
        print('Category: {}\n'.format(column))
        print(classification_report(Y_test[column],Y_pred_df[column]))
        

def save_model(model, model_filepath):
    
    
    
    # save the model to disk
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))

    
def main():
    
    """
    Main function that runs entire model training process. This comprises 6 main steps:
    
    1. load cleaned dataset from SQLite database and set target and feature variables
    2. build model object containing steps based on feature engineering and training with classifier
    3. optimise model by running pipeline with different hyperparameter sets using grid search
    4. train model with optimal set of hyperparameters
    5. evaluate optimal model using test set
    6. save optimal model as pickle file
    
    """
    
    if len(sys.argv) == 3:
        
        database_filepath, model_filepath = sys.argv[1:]
        
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        
        # split input data into train and test sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        # set hyperparameters to tune: maximum depth of decision tree and learning rate.
        parameters = {'clf__estimator__learning_rate': [x * 0.1 for x in range(1, 11)],
              'clf__estimator__n_estimators': list(range(1, 100, 5))}
        
        print('Building model...')
        model = build_model()
        
        print('Optimising model...')
        cv = GridSearchCV(model, param_grid = parameters)
        cv.fit(X_train, Y_train)
        
        print('Training model...')
        cv.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(cv, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(cv, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
