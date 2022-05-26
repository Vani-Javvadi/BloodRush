import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import  train_test_split
from sklearn.naive_bayes import MultinomialNB
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():
    df = pd.read_csv("C:\\Users\\javvadi lakshmi vani\\OneDrive\\Desktop\\Sentiment Analysis\\IMDB Dataset.csv")
    df.head()
    cv = CountVectorizer()
    text_counts = cv.fit_transform(df['review'])
    print(text_counts)
    X_train, X_test, Y_train, Y_test = train_test_split(text_counts, df['sentiment'], test_size=0.25, random_state=5)
    MNB = MultinomialNB()
    MNB.fit(X_train,Y_train)
    MNB.score(X_test,Y_test)

    print(MNB.score(X_test, Y_test)*100)

    if request.method == 'POST':
            message = request.form['message']
            data = [message]
            vect = cv.transform(data).toarray()
            my_prediction = MNB.predict(vect)
            print(my_prediction)
            return render_template('index.html', prediction=my_prediction)

if __name__ == '__main__':
    app.run()