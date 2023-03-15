"""
This file contains a class that is used to predict stock movements given news headlines

Honor Code: I affirm I have carried out the Union College Honor Code
"""
import numpy
import numpy as np
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

from tqdm import tqdm


def chunk_data(data, chunk_size):
    """
    Split the given data in n chunks of chunk_size size
    :param data: the list to split
    :param chunk_size: how many items in each chunk
    :return: a new list of 2 dimensions with chunks of original data
    """
    chunks = []
    for i in range(0, len(data), chunk_size):
        chunks.append(data[i:i + chunk_size])
    return chunks


def getClassFromPrice(open_price, close_price, plus_minus_range=0):
    """
    Return the class corresponding to the open and closing prices

    if plus_minus_range = 0, return 1 or 0 based on the opening/closing price

    :param open_price: the open price of the day
    :param close_price: the close price of the day
    :param plus_minus_range: the range for holding
    :return: 2, 1, or 0 depending on the values
    """
    if plus_minus_range == 0:
        return 1 if close_price > open_price else 0

    # buy if close_price is plus_minus_range greater than open_price
    # sell if close_price is plus_minus_range less than open_price
    # hold else

    if close_price > open_price + plus_minus_range:
        return 2
    elif close_price + plus_minus_range < open_price:
        return 0
    else:
        return 1

def getClassFromSentimentProbabilities(sentiment_probs):
    """
    Return the class corresponding to the open and closing prices

    if plus_minus_range = 0, return 1 or 0 based on the opening/closing price

    :param open_price: the open price of the day
    :param close_price: the close price of the day
    :param plus_minus_range: the range for holding
    :return: 2, 1, or 0 depending on the values
    """

    max_value = max(sentiment_probs)
    index = np.where(sentiment_probs==max_value)[0][0]

    if index == 2:
        return 2
    elif index == 1:
        return 1
    else:
        return 0


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

        chunked_data = chunk_data(headlines, 5)
        X = []

        print("Extracting Sentiments from Articles")
        for chunk in tqdm(chunked_data, desc="Headline Chunk"):
            scores = self.get_sentiment_scores(chunk)

            for sentiment_score in scores:
                X.append(sentiment_score.cpu().detach().numpy())

        return X

    def get_and_clean_dated_predictions(self, rows):
        """
        Given a list of headlines, return the sentiment score as a Python List
        ^--- Will almost always want to use this method over get_predictions
        :param headlines: a list of headlines
        :return: a list of sentiments (positive, negative, neutral) corresponding to
        each headline in the given list i.e. [[0.5, 0.02, 0.7], ...]
        """

        chunked_data = chunk_data(rows, 5)
        X = {}

        print("Extracting Sentiments from Articles")
        for chunk in tqdm(chunked_data, desc="Headline Chunk"):
            headlines_list = [row[CSV_COLUMNS["HEADLINE"]] for row in chunk]

            scores = self.get_sentiment_scores(headlines_list)

            for i, sentiment_score in enumerate(scores):
                date_index = int(chunk[i][CSV_COLUMNS["DATE"]].split(" ")[0].replace("-", ""))
                if date_index not in X.keys():
                    X[date_index] = [sentiment_score.cpu().detach().numpy()]
                X[date_index].append(sentiment_score.cpu().detach().numpy())

        for date in X.keys():
            summated_sentiment = np.array([])
            for sentiment in X[date]:
                if len(summated_sentiment) == 0:
                    summated_sentiment = sentiment
                summated_sentiment += sentiment

            X[date] = getClassFromSentimentProbabilities(np.divide(summated_sentiment, 3))

        print(X.items())

        return list(X.items())

    def get_stock_buy_sell(self, rows):
        """
        Given a list of rows containing headline, date, url, etc.
        return a list containing 1 if the stock rose on the given date in the list
                or 0 if the stock price fell on that date
        :param rows: the rows from a CSV file (given from scripts.get_rows_from_ticker())
        :return: list of values detailing stock price rising/falling i.e. [1, 1, 0, 0, 0, 1, 1, ...]
        """
        Y_actual = []

        for row in tqdm(rows, desc="Parsing Historical Data"):
            date = row[CSV_COLUMNS['DATE']].split(" ")[0]
            price = self.stock.get_price_on_date(date)
            Y_actual.append(getClassFromPrice(price[0], price[1]))

        return Y_actual

    def get_day_buy_sell(self, rows):
        """
        Given a list of rows containing headline, date, url, etc.
        return a list containing 1 if the stock rose on the given date in the list
                or 0 if the stock price fell on that date
        :param rows: the rows from a CSV file (given from scripts.get_rows_from_ticker())
        :return: list of values detailing stock price rising/falling i.e. [1, 1, 0, 0, 0, 1, 1, ...]
        """
        Y_actual = {}

        for row in tqdm(rows, desc="Parsing Historical Data"):
            date = row[CSV_COLUMNS['DATE']].split(" ")[0]
            if date not in Y_actual.keys():
                Y_actual[ int(date.replace("-", ""))] = getClassFromPrice(*self.stock.get_price_on_date(date))

        return list(Y_actual.items())

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

    def evaluate(self, predicted, actual):
        print("\n############## Evaluation ##############\n")

        print("Accuracy: ", accuracy_score(predicted, actual) * 100)
        print("Precision: ", precision_score(predicted, actual) * 100)
        print("Recall: ", recall_score(predicted, actual) * 100)
        print("F Score: ", f1_score(predicted, actual) * 100)

if __name__ == "__main__":
    # PREDICTOR CONFIGURATION ###############################################################

    CSV_FILE = "raw_analyst_ratings.csv"

    # Change company here
    ticker = "NVDA"

    print("\n############## Preparing Data ##############\n")

    rows = scripts.get_rows_from_ticker(ticker, CSV_FILE)
    min_date, max_date = scripts.get_date_range_from_rows(rows)

    print("Getting Stock Historical Data")
    stock = StockPrices(ticker, min_date, max_date)

    print("Extracting Stock Headlines from Corpus")

    headlines_list = [row[CSV_COLUMNS["HEADLINE"]] for row in rows]

    # Number of training examples
    HEADLINE_AMOUNT = round(len(headlines_list) * 0.8)

    predictor = Predictor(stock)

    print("\n############## Training ##############\n")

    X = predictor.get_and_clean_predictions(headlines_list[:HEADLINE_AMOUNT])
    Y = predictor.get_stock_buy_sell(rows[:HEADLINE_AMOUNT])

    # Change Model Here
    predictor.train(X, Y, SVC())

    #####
    # Predict
    #####

    # Number of testing examples
    NUM_TEST_EXAMPLES = round(len(headlines_list) * 0.2)

    print("\n############## Testing ##############\n")

    prediction_rows = rows[HEADLINE_AMOUNT:HEADLINE_AMOUNT + NUM_TEST_EXAMPLES]
    prediction_headlines = headlines_list[HEADLINE_AMOUNT:HEADLINE_AMOUNT + NUM_TEST_EXAMPLES]

    X_predictions = predictor.get_and_clean_predictions(prediction_headlines)
    Y_actual = predictor.get_stock_buy_sell(prediction_rows)

    Y_predicted = predictor.predict(X_predictions)

    predictor.evaluate(Y_predicted, Y_actual)
