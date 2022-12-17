"""
Data Preprocessing Script

Arguments:
    -  Path to CSV file containing messages (e.g. messages.csv)
    -  Path to CSV file containing categories (e.g. categories.csv)
    -  Path to SQLite destination database (e.g. DisasterResponse.db)
    -  Name of Table in SQLite database (e.g. DisasterResponse)
    
"""

#import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


#import and merge datasets
def load_datasets(messages_uri, categories_uri):
    
    """
    Import messages and categories dataframes from csv and merge into a single dataframe
    
    Inputs: messages.csv, categories.csv - raw csv datasets
    Outputs: df - merged dataframe
    
    """
    try:
        messages = pd.read_csv('messages.csv')
        categories = pd.read_csv('categories.csv')
    except:
        print('Error: Files could not be loaded. Check file type.')
    
    if messages.shape[0] != categories.shape[0]:
        print('Error: Datasets have different number of rows and cannot be merged')
    else:
        df = pd.merge(messages, categories, on = ['id'])
    
    return df


def clean_categories(df):
    
    """
    Clean categories dataframe
    
    Inputs: df - combined messages and uncleaned categories data
    Outputs: df - combined messages and cleaned categories data
    
    """
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand = True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0, :].tolist()
    
    # extract a list of new column names for categories
    category_colnames = [item[:-2] for item in row]
    
    # rename the columns of categories dataframe
    categories.columns = category_colnames
    
    # convert category values to 0 or 1
    for column in categories:
        
    # set each value to be the last character of the string
    categories[column] = categories[column].astype(str).str[-1:]
    
    # convert column from string to numeric
    categories[column] = pd.to_numeric(categories[column])
    
    # drop the original categories column from df
    df.drop(columns = ['categories'], inplace = True)
    
    # concatenate the original dataframe with the new categories dataframe
    df = pd.concat([df, categories], axis = 1)
    
    # drop duplicates
    df = df.drop_duplicates()
    
    return df


def save_data(df, db_filename):
    
    """
    Save cleaned dataset to SQLite database
    
    Inputs: df - cleaned dataframe
            db_filename - name of database
            db_tablename - name of table in database
    """
    
    engine = create_engine('sqlite:///'+db_filename)
    df.to_sql(db_filename.replace('db', ''), engine, index=False, if_exists = 'replace')
    


def main():
    if len(sys.argv) == 4:

        messages_uri, categories_uri, db_filename = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_uri, categories_uri))
        df = load_datasets(messages_uri, categories_uri)

        print('Cleaning data...')
        df = clean_categories(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, db_filename)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')

> python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
    
if __name__ == '__main__':
    main()