import datetime
import pandas as pd
import pickle
import json
import surprise
from surprise import Dataset, Reader, BaselineOnly

from flask import Flask, render_template

app = Flask(__name__)


@app.route("/<uid>")
def get_recommendation(uid):
    # For the sake of example, use static information to inflate the template.
    # This will be replaced with real information in later steps.
    user_id = int(uid)
    rating = pd.read_csv('data/ratings.csv')

    book = pd.read_csv('data/books.csv')
    def getBooksById(iids):
        return book[book['ISBN'].isin(iids)]
    user = pd.read_csv('data/users.csv')
    users = user[ (user['Age']>= 0) & (user['Age'] <= 150.) ]
    users['Country'] = users['Location'].str.split(',').str[-1]
    users['State'] = users['Location'].str.split(',').str[-2]
    users['City'] = users['Location'].str.split(',').str[-3]

    users['Country'] = users['Country'].str.replace('[^a-zA-Z\.\ ]', '')
    users['State'] = users['State'].str.replace('[^a-zA-Z\.\ ]', '')
    users['City'] = users['City'].str.replace('[^a-zA-Z\.\ ]', '')

    users['Country'] = users['Country'] .str.strip()
    users['State'] = users['State'] .str.strip()
    users['City'] = users['City'] .str.strip()
    usa_users = users[users['Country'].str.contains("usa")]
    usa_users_ids = usa_users['User-ID'].unique()
    rating = rating[rating['User-ID'].isin(usa_users_ids)]
    
    with open('model/svd_model.pkl', 'rb') as file:
        model = pickle.load(file)
    
    ISBN_to_predict = book['ISBN']
    ISBN_to_predict = ISBN_to_predict[~ISBN_to_predict.isin(rating[rating['User-ID']==user_id]['ISBN'])]
    
    user_predictions = [(item, model.predict(user_id, item)) for item in ISBN_to_predict.unique()]
    sorted_predictions = sorted(user_predictions, key=lambda x: x[1].est, reverse=True)
    top_10_reco_items = [item[1].iid for item in sorted_predictions[:10]]


    top_reco_books = json.loads(getBooksById(top_10_reco_items).to_json(orient='records'))

    sorted_ratings = rating[rating['User-ID']==user_id].sort_values(by='Book-Rating', ascending=False)
    top10_sorted_ratings = sorted_ratings[:10]['ISBN']
    last10_sorted_ratings = sorted_ratings[-10:]['ISBN']
    top_rated_books = json.loads(getBooksById(top10_sorted_ratings).to_json(orient='records'))
    lowest_rated_books = json.loads(getBooksById(last10_sorted_ratings).to_json(orient='records'))

    return render_template("results.html", top_reco_books=top_reco_books, top_rated_books=top_rated_books, lowest_rated_books=lowest_rated_books, uid=user_id)

@app.route("/")
def render_search():
    return render_template("search.html")
                           
if __name__ == "__main__":
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    # Flask's development server will automatically serve static files in
    # the "static" directory. See:
    # http://flask.pocoo.org/docs/1.0/quickstart/#static-files. Once deployed,
    # App Engine itself will serve those files as configured in app.yaml.
    app.run(host="127.0.0.1", port=8080, debug=True)