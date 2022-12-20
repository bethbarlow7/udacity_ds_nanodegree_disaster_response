# Disaster Response Pipeline Project

## Table of Contents
1. [Description](#description)
2. [Getting Started](#getting_started)
	1. [Dependencies](#dependencies)
	2. [Installing](#installing)
	3. [Executing Program](#executing)
3. [Authors](#authors)

<a name="descripton"></a>
## Description

The aim of this project is to build a Natural Language Processing model for an API that categorises disaster messages. 

This project fulfills the requirements of the Data Science Nanodegree Program by Udacity in collaboration with Appen. 

The project is divided into the following sections:

1. Data Processing ETL Pipeline to extract data from csv format, clean, and save in a SQLite database
2. Machine Learning Pipeline to train an NLP multioutput classifier model that is able to classify text messages into categories
3. Web Application to allow a user to input a new text message and view the classification results

<a name="getting_started"></a>
## Getting Started

<a name="dependencies"></a>
### Dependencies
* Python 3.5+
* Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
* Natural Language Process Libraries: NLTK
* SQLlite Database Libraries: SQLalchemy
* Web App and Data Visualization: Flask, Plotly

<a name="installing"></a>
### Installing
Clone this GIT repository:
```
git clone https://github.com/bethbarlow7/udacity_ds_nanodegree_disaster_response
```
<a name="executing"></a>
### Executing Program:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that loads raw data, cleans and stores cleaned data in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves as pickle file
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run the web application.
    `python run.py`

3. Go to http://0.0.0.0:3001/

<a name="authors"></a>
## Authors

* [Beth Barlow](https://github.com/bethbarlow7)
