# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Note on data set:
The dataset was provided as part of an Udacity learning project. 
It contains 26,386 rows, of which, ~23% are not related to categories. We assumed that those rows were not yet classified, therefore they should be excluded from the model training. If they were classified, we can add another column 'Unclassified' for those. 

The data set is quite unbalanced in relation to the each of the categorization. Categories with small related samples could lead to low accuracy in its prediction, as not a lot of words could be identified that segregate that population vs the rest of the categories. 







