import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
#from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)
app.debug = True

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# cleaned data
# filter only related messages
#df = df.loc[df['related'] == 1]
y_columns = df.columns.tolist().copy()
cols_2_remove = ['id', 'message', 'original', 'related']

for col in cols_2_remove:
    y_columns.remove(col)

# copy the y values and set the upper limit to 1 max    
y = df[y_columns].copy()

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals

    # data for chart on count by genre
    genre_counts = df.groupby('genre').count()['message']
    genre_names = genre_counts.index.tolist()

    # data for char on count per related category
    y_sum = y.sum()
    y_sum_sorted = y_sum.tolist()
    y_names = list(y_sum.index)
    
    # data for char on related or not
    related_name = ['Not related', 'Related']
    related_value = df.groupby('related').count()['message']

    # create visuals
    # see doc on https://plotly.com/chart-studio-help/json-chart-schema/
    # TODO: Below is an example - modify to create your own visuals

    graphs = [
        {
            'Info': '1. Count messages per related category',
            'data': [
                    {
                    'x': y_names,
                    'y': y_sum_sorted, 
                    'type': 'bar',
                    'color': '#aa22aa'

                    }
                ],
            'layout': {
                'title': 'Distribution of Related Message',
                'xaxis': {'title': "Message Categorization"},
                'yaxis': {'title': "Count" }
            }
        },

                {
            'Info': '2. Pie chart on related or not message count',
            'data': [
                    {
                    'labels': related_name,
                    'values': related_value, 
                    'type': 'pie',
                    'marker': {'colors': ['#aa2222', '#22aa22']},
                    'line': {'color':'#ffffff', 'width': 3 }
                    }
                ],
            'layout': {
                'title': 'Distribution of Related Message',
            }
        },

        {
            'Info': '3. Messages count per genre',
            'data': [
                    {
                    'x': genre_names,
                    'y': genre_counts, 
                    'type': 'bar',
                    }
                ],
            'layout': {
                'title': 'Distribution Messages per Genre',
                'xaxis': {'title': "Genre"},
                'yaxis': {'title': "Count" }
            }
        },


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