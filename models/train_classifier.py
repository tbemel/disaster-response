"""
USER STORY: load the message database, clean the data, split it in X/y and train/test, build / train / test/ save the model
AUTHOR: Thierry Bemelmans
CREATED: 2022-OCT-15

EXAMPLE OF COMMAND + ARGUMENTS: python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

INPUT ARGUMENTS: 
(1) filepath to the sqlite db file  
(2) filepath of the ML model file to save as pickel

OUTPUT: saved pickle file on path provided in (2)

"""


import sys
from sqlalchemy import create_engine
import pandas as pd
import re
import pickle # used to save the model

# NLP
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
#from nltk.stem.wordnet import WordNetLemmatizer
from nltk import download

# ML modeling 
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import numpy as np


def load_data(database_filepath):
    """
    USER STORY: load a sqlite db file to a df, drop unnecessary columns, split data to X, y array
    INPUT: database path
    OUTPUT: X, y split data , y categories name (column names of the y array)
    """

    # 1) LOAD THE DB DATA
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('messages', engine)

    # getting X feature column as the text
    X = df['message']

    # getting the y columns as all columns from df, removing the undesired ones
    y_columns = df.columns.tolist()
    cols_2_remove = ['id', 'message', 'original', 'genre', 'related']
    for col in cols_2_remove:
        y_columns.remove(col)

    # copy the y values and set the upper limit to 1 max    
    y = df[y_columns].copy().clip(upper=1)

    return X, y, y_columns


def tokenize(text: str) -> str:
    """ remove punctuations, uppercase & stop words, stem & returns the joined string """
    text = text.lower()
    text = re.sub('[^0-9A-Za-z ]', '', text)
    
    # tokenize to words
    words = word_tokenize(text)
    
    # remove stopwords & keep only stem
    stopWordList = stopwords.words("english")
    stemmer = SnowballStemmer(language='english')
    words = [stemmer.stem(word) for word in words if word not in stopWordList]
    return ' '.join(words)


def build_model():
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    """
    USER STORY: build our model using a grid pipeline, optimizing the best modeling parameters to use
    INPUT: none
    OUTPUT: datamodel as a pipeline with gridsearch enabled to optimize the model when fitting 
    """

    pipeline = Pipeline(steps=[
        ('vect', TfidfVectorizer(sublinear_tf=True, min_df=5,ngram_range=(1, 2), stop_words='english')),
        ('classifier', MultiOutputClassifier(estimator=RandomForestClassifier(n_estimators =20,criterion="entropy",random_state =40)))
        ])

    """
    NOTE: I limited the grid pipeline feature as found best parameter set and did not want to wait for gridsearch
    """
    parameters = {
        'vect__ngram_range': [(1,1),(1,2)],
        'vect__min_df': [5]
    }
           
    grid_pipeline = GridSearchCV(pipeline,parameters, verbose=10)
    return grid_pipeline



def evaluate_model(model, X_test, y_test, category_names):
    """
    USER STORY: check the model accuracy by comparing y_test real values vs y_test predicted values by the model
    INPUT: model = model to use, X_test = features to test, y_test = real y outcomes, category_names = columns name for the y outcomes
    OUTPUT: none, just print the prediction scores as a matrix by outcome columns
    """

    # predict the y_test results using the model
    y_test_predict = model.predict(X_test)

    # report the scoring of the model predictions vs the real y_test values by outcome column
    target_names =category_names
    print(f'Predicted y outcomes shape: {y_test_predict.shape}')
    print(f'Categories column names count: {len(category_names)}')
    print(classification_report(np.hstack(y_test.values), np.hstack(y_test_predict)))


def save_model(model, model_filepath):  
    """
    USER STORY: save the model as a pickle file that can be loaded later on
    INPUT:  model=model to save, model_filepath=file path to save the model
    OUTPUT: none
    """

    with open(model_filepath, "wb") as f:
        pickle.dump(model, f)


def main():
    """
    Main function
    
    INPUT ARGUMENTS: 
    (1) filepath to the sqlite db file  
    (2) filepath of the ML model file to save as pickel

    """
    if len(sys.argv) == 3:
        # Download the stops words if not already in
        download('punkt')
        download('stopwords')
        
        # Loading DB data
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=32)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

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