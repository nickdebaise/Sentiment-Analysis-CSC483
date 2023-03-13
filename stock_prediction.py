"""
This file contains a class that is used to predict stock movements given news headlines

Honor Code: I affirm I have carried out the Union College Honor Code
"""

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import scripts
from constants import CSV_COLUMNS
from stock_price import StockPrices


def getClassFromPrice(open_price, close_price, plus_minus_range=0):
    """
    Return the class corresponding to the open and closing prices

    BUY : 2
    HOLD : 1
    SELL : 0

    :param open_price: the open price of the day
    :param close_price: the close price of the day
    :param plus_minus_range: the range for holding
    :return: 2, 1, or 0 depending on the values
    """
    # buy if close_price is plus_minus_range greater than open_price
    # sell if close_price is plus_minus_range less than open_price
    # hold else

    if plus_minus_range == 0:
        return 1 if close_price > open_price else 0

    if close_price > open_price + plus_minus_range:
        return 2
    elif close_price + plus_minus_range < open_price:
        return 0
    else:
        return 1


class Predictor:
    def __init__(self, stock):
        """
        A class to get X predictions (features of headlines) and Y values (stock buy/sell)
        It also:
            - Trains a given model given X and Y values
            - Predicts classifier outputs given X values
        :param stock: The stock that the predictor is working with
        """
        self.clf = None
        self.stock = stock
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.analyzer = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

    def get_sentiment_scores(self, headlines):
        """
        Given a list of headlines, return the sentiment score of each headline as tensors
        :param headlines: a list of headlines (i.e. ["Google rallied today", ...]
        :return: a list of sentiments of each headline in Tensor form
        """
        inputs = self.tokenizer(headlines, padding=True, truncation=True, return_tensors='pt')
        outputs = self.analyzer(**inputs)
        sentiment_scores = torch.nn.functional.softmax(outputs.logits, dim=-1)

        return sentiment_scores

    def get_and_clean_predictions(self, headlines):
        """
        Given a list of headlines, return the sentiment score as a Python List
        ^--- Will almost always want to use this method over get_predictions
        :param headlines: a list of headlines
        :return: a list of sentiments (positive, negative, neutral) corresponding to
        each headline in the given list i.e. [[0.5, 0.02, 0.7], ...]
        """
        scores = self.get_sentiment_scores(headlines)

        X = []
        print("Getting Predictions")

        for sentiment_score in scores:
            X.append(sentiment_score.cpu().detach().numpy())

        return X

    def get_stock_buy_sell(self, rows):
        """
        Given a list of rows containing headline, date, url, etc.
        return a list containing 1 if the stock rose on the given date in the list
                or 0 if the stock price fell on that date
        :param rows: the rows from a CSV file (given from scripts.get_rows_from_ticker())
        :return: list of values detailing stock price rising/falling i.e. [1, 1, 0, 0, 0, 1, 1, ...]
        """
        Y_actual = []

        for row in rows:
            date = row[CSV_COLUMNS['DATE']].split(" ")[0]
            price = self.stock.get_price_on_date(date)
            Y_actual.append(getClassFromPrice(price[0], price[1]))

        return Y_actual

    def train(self, X, Y, model):
        """
        Given a model and X and Y values, train the model
        :param X: The x values (features)
        :param Y: The y values (outputs)
        :param model: the model to be used i.e. Naive Bayes
        :return: None
        """
        self.clf = make_pipeline(StandardScaler(), model)
        self.clf.fit(X, Y)

    def predict(self, X):
        """
        Given a list of features, make a prediction using the trained model
        :param X: the list of features
        :return: the predictions from the model
        """
        return self.clf.predict(X)


if __name__ == "__main__":
    # PREDICTOR CONFIGURATION ###############################################################

    CSV_FILE = "raw_analyst_ratings.csv"

    # Change company here
    ticker = "GOOGL"

    rows = scripts.get_rows_from_ticker(ticker, CSV_FILE)
    min_date, max_date = scripts.get_date_range_from_rows(rows)

    stock = StockPrices(ticker, min_date, max_date)
    headlines_list = [row[CSV_COLUMNS["HEADLINE"]] for row in rows]

    # Number of training examples
    HEADLINE_AMOUNT = 800

    predictor = Predictor(stock)

    X = predictor.get_and_clean_predictions(headlines_list[:HEADLINE_AMOUNT])
    Y = predictor.get_stock_buy_sell(rows[:HEADLINE_AMOUNT])

    # Change Model Here
    predictor.train(X, Y, GaussianNB())

    #####
    # Predict
    #####

    # Number of testing examples
    NUM_TEST_EXAMPLES = 100

    prediction_rows = rows[HEADLINE_AMOUNT:HEADLINE_AMOUNT + NUM_TEST_EXAMPLES]
    prediction_headlines = headlines_list[HEADLINE_AMOUNT:HEADLINE_AMOUNT + NUM_TEST_EXAMPLES]

    X_predictions = predictor.get_and_clean_predictions(prediction_headlines)
    Y_predictions = predictor.get_stock_buy_sell(prediction_rows)

    predicted = predictor.predict(X_predictions)
    print("Accuracy: ", accuracy_score(predicted, Y_predictions))
    print("Precision: ", precision_score(predicted, Y_predictions))
    print("Recall: ", recall_score(predicted, Y_predictions))
    print("F Score: ", f1_score(predicted, Y_predictions))
