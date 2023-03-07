from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import recall_score, f1_score, precision_score

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import scripts
from constants import CSV_COLUMNS
from stock_price import StockPrices


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

    def get_predictions(self, headlines):
        """
        Given a list of headlines, return the sentiment score of each headline as tensors
        :param headlines: a list of headlines (i.e. ["Google rallied today", ...]
        :return: a list of sentiments of each headline in Tensor form
        """
        inputs = self.tokenizer(headlines, padding=True, truncation=True, return_tensors='pt')
        outputs = self.analyzer(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        return predictions

    def get_and_clean_predictions(self, headlines):
        """
        Given a list of headlines, return the sentiment score as a Python List
        ^--- Will almost always want to use this method over get_predictions
        :param headlines: a list of headlines
        :return: a list of sentiments (positive, negative, neutral) corresponding to
        each headline in the given list i.e. [[0.5, 0.02, 0.7], ...]
        """
        predictions = self.get_predictions(headlines)

        X_predictions = []

        for prediction in predictions:
            X_predictions.append(prediction.cpu().detach().numpy())

        return X_predictions

    def get_stock_buy_sell(self, rows):
        """
        Given a list of rows containing headline, date, url, etc.
        return a list containing 1 if the stock rose on the given date in the list
                or 0 if the stock price fell on that date
        :param rows: the rows from a CSV file (given from scripts.get_rows_from_ticker())
        :return: list of values detailing stock price rising/falling i.e. [1, 1, 0, 0, 0, 1, 1, ...]
        """
        Y_predictions = []

        for row in rows:
            date = row[CSV_COLUMNS['DATE']].split(" ")[0]
            price = self.stock.get_price_on_date(date)
            Y_predictions.append(1 if price[1] - price[0] > 0 else 0)

        return Y_predictions

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

ticker = "GOOGL"
CSV_FILE = "raw_analyst_ratings.csv"
rows = scripts.get_rows_from_ticker(ticker, CSV_FILE)
min_date, max_date = scripts.get_date_range_from_rows(rows)
stock = StockPrices(ticker, min_date, max_date)
headlines_list = [row[CSV_COLUMNS["HEADLINE"]] for row in rows]
HEADLINE_AMOUNT = 500

predictor = Predictor(stock)

X = predictor.get_and_clean_predictions(headlines_list[:HEADLINE_AMOUNT])
Y = predictor.get_stock_buy_sell(rows[:HEADLINE_AMOUNT])

predictor.train(X, Y, GaussianNB())

#####
# Predict
#####

NUM_TRAINING_EXAMPLES = 100
prediction_rows = rows[HEADLINE_AMOUNT + 1:HEADLINE_AMOUNT + NUM_TRAINING_EXAMPLES]
prediction_headlines = headlines_list[HEADLINE_AMOUNT + 1:HEADLINE_AMOUNT + NUM_TRAINING_EXAMPLES]

X_predictions = predictor.get_and_clean_predictions(prediction_headlines)
Y_predictions = predictor.get_stock_buy_sell(prediction_rows)

predicted = predictor.predict(X_predictions)
print(predicted)
print(Y_predictions)

print(precision_score(predicted, Y_predictions), recall_score(predicted, Y_predictions), f1_score(predicted, Y_predictions))
