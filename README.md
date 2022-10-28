# Disaster Response Pipeline Project


### Project Purpose:
As part of my Udacity DataScience nanodegree, here is a great full data stack project to
1) Load 2 different Disaster response related messages csv files, blend them and save the cleansed data to a SQLite1 db
2) Load the db data and develop/ save as pickle a ML training model to forecast disaster response incident classifications based on the text of the message
3) Load the model into a Flask webserver, using plotly, visualize some of the data and have an interactive message form to use the ML model to predict the classification of the message. 

### Instructions how to run the app:
0. Ensure you pip installed the proper python modules as in the file app/requirements.txt
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves the model as a pickle file
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Note on data set:
The dataset was provided as part of an Udacity learning project. 
It contains 26,386 rows, of which, ~23% are not related to categories. We assumed that those rows were not yet classified, therefore they should be excluded from the model training. If they were classified, we can add another column 'Unclassified' for those. 

The data set is quite unbalanced in relation to the each of the categorization. Categories with small related samples could lead to low accuracy in its prediction, as not a lot of words could be identified that segregate that population vs the rest of the categories. 

### Note on files in this repository:
The file models/classifier.pikl was not uploaded in gitub given its size of 179MB. You can regenerate it by running the 2 steps in 1. above.
The gridseach parameters have been kept small to minimize the run time. A more precise model could be done through extensive parameter options tuning/ testing.





