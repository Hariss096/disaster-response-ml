import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

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
df = pd.read_sql_table('transformed_data', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # fetching only categories
    only_categories_from_df = df.drop(['id','message','original','genre'], axis=1).sum().sort_values(ascending=False)
    categories_names = list(only_categories_from_df.index)
    # Plotting genre distribution
    genre_distribution = df.groupby("genre")["message"].count()
    genre_names = list(genre_distribution.index)
    
    # plotting Grouped bar chart for all 3 genres
    news_df = df[df['genre']=='news']
    news_df_distribution = news_df.drop(['id','message','original','genre'], axis=1).sum().sort_values(ascending=False)
    news_df_names = list(news_df_distribution.index)
    
    direct_df = df[df['genre']=='direct']
    direct_df_distribution = direct_df.drop(['id','message','original','genre'], axis=1).sum().sort_values(ascending=False)
    direct_df_names = list(direct_df_distribution.index)
    
    social_df = df[df['genre']=='social']
    social_df_distribution = social_df.drop(['id','message','original','genre'], axis=1).sum().sort_values(ascending=False)
    social_df_names = list(social_df_distribution.index)

    graphs = [
        {
            'data': [
                Bar(
                    name="News",
                    x=news_df_names,
                    y=news_df_distribution
                ),
                Bar(
                    name="Social",
                    x=social_df_names,
                    y=social_df_distribution
                ),
                Bar(
                    name="Direct",
                    x=direct_df_names,
                    y=direct_df_distribution
                )
            ],

            'layout': {
                'title': 'Count of Genres w.r.t. each Category',
            }
        },
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_distribution
                )
            ],

            'layout': {
                'title': 'Distribution of genres',
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