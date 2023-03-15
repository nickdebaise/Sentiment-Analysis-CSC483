"""
This file contains a class that is used to predict stock movements given news headlines

Honor Code: I affirm I have carried out the Union College Honor Code
"""
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
from stock_prediction import Predictor
from stock_price import StockPrices

from tqdm import tqdm


# PREDICTOR CONFIGURATION ###############################################################

CSV_FILE = "raw_analyst_ratings.csv"

print("\n############## Preparing Data ##############\n")

# Change company here
tickers_dict = scripts.get_stocks_and_headlines(CSV_FILE)

dates_sentiments = {}

X, Y = [],[]


print("\n############## Individual ##############\n")

for ticker in tqdm(["AAPL"], desc="TICKER"):
    rows = scripts.get_rows_from_ticker(ticker, CSV_FILE)
    min_date, max_date = scripts.get_date_range_from_rows(rows)

    print("Getting Stock Historical Data")
    stock = StockPrices(ticker, min_date, max_date)

    headlines_list = tickers_dict[ticker]
    # Number of training examples
    HEADLINE_AMOUNT = round(len(headlines_list) * 0.9)

    predictor = Predictor(stock)
    X.append(predictor.get_and_clean_dated_predictions(rows[:HEADLINE_AMOUNT]))
    Y.append(predictor.get_day_buy_sell(rows[:HEADLINE_AMOUNT]))

print("\n############## Training ##############\n")

model = SVC()
clf = make_pipeline(StandardScaler(), model)

clf.fit(X, np.array(Y))


#####
# Predict
#####

ticker = "NVDA"

print("\n############## Testing ##############\n")

rows = scripts.get_rows_from_ticker(ticker, CSV_FILE)
min_date, max_date = scripts.get_date_range_from_rows(rows)

predictor = Predictor(ticker)

print("Getting Stock Historical Data")
stock = StockPrices(ticker, min_date, max_date)

# Number of training examples
ARTICLE_AMOUNT = round(len(tickers_dict[ticker]) * 0.9)

X_predict = [predictor.get_and_clean_dated_predictions(rows[:ARTICLE_AMOUNT])]
Y_actual  = [predictor.get_day_buy_sell(rows[:ARTICLE_AMOUNT])]

Y_predicted = clf.predict(X_predict)

predictor.evaluate(Y_predicted, Y_actual)


