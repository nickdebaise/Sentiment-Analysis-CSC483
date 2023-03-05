import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import scripts
from constants import CSV_COLUMNS
from stock_price import StockPrices

CSV_FILE = 'raw_analyst_ratings.csv'
ticker = "MRK"
rows = scripts.get_rows_from_ticker(ticker, CSV_FILE)
headlines_list = [row[CSV_COLUMNS["HEADLINE"]] for row in rows]

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

HEADLINE_AMOUNT = 500
inputs = tokenizer(headlines_list[:HEADLINE_AMOUNT], padding=True, truncation=True, return_tensors='pt')
outputs = model(**inputs)
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

X_predictions = []

for prediction in predictions:
    X_predictions.append(prediction.cpu().detach().numpy())


ticker = "MRK"
CSV_FILE = "raw_analyst_ratings.csv"
rows = scripts.get_rows_from_ticker(ticker, CSV_FILE)
min_date, max_date = scripts.get_date_range_from_rows(rows)
stock = StockPrices(ticker, min_date, max_date)

y_predictions = []

for row in rows[:HEADLINE_AMOUNT]:
    date = row[CSV_COLUMNS['DATE']].split(" ")[0]

    price = stock.get_price_on_date(date)

    y_predictions.append(1 if price[1] - price[0] > 0 else 0)

X = np.array(X_predictions)
y = np.array(y_predictions)

clf = make_pipeline(StandardScaler(), GaussianNB())
clf.fit(X, y)


#####
# Predict
#####

inputs = tokenizer(headlines_list[HEADLINE_AMOUNT + 1:HEADLINE_AMOUNT + 100], padding=True, truncation=True, return_tensors='pt')
outputs = model(**inputs)
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

X_predictions = []

for prediction in predictions:
    X_predictions.append(prediction.cpu().detach().numpy())


y_predictions = []

for row in rows[HEADLINE_AMOUNT + 1:HEADLINE_AMOUNT + 100]:
    date = row[CSV_COLUMNS['DATE']].split(" ")[0]

    price = stock.get_price_on_date(date)

    y_predictions.append(1 if price[1] - price[0] > 0 else 0)

X = np.array(X_predictions)
y = np.array(y_predictions)

from sklearn.metrics import recall_score, f1_score, precision_score

predicted = clf.predict(X)
print(predicted)
print(y)

print(precision_score(predicted, y), recall_score(predicted, y), f1_score(predicted, y))