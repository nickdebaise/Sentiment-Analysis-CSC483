import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

import torch
import torch.nn as nn
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

HEADLINE_AMOUNT = 400
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

X = torch.from_numpy(np.array(X_predictions).astype(np.float32))
y = torch.from_numpy(np.array(y_predictions).astype(np.float32))

n_input, n_hidden, n_out, batch_size, learning_rate = 3, 15, 1, 100, 0.01

model2 = nn.Sequential(nn.Linear(n_input, n_hidden),
                      nn.ReLU(),
                      nn.Linear(n_hidden, n_out),
                      nn.Sigmoid())

loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model2.parameters(), lr=learning_rate)

losses = []
for epoch in range(5000):
    pred_y = model2(X)
    loss = loss_function(pred_y, y)
    losses.append(loss.item())

    model2.zero_grad()
    loss.backward()

    optimizer.step()



# clf = make_pipeline(StandardScaler(), GaussianNB())
# clf.fit(X, y)


#####
# Predict
#####

inputs = tokenizer(headlines_list[HEADLINE_AMOUNT + 1:HEADLINE_AMOUNT + 300], padding=True, truncation=True, return_tensors='pt')
outputs = model(**inputs)
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

X_predictions = []

for prediction in predictions:
    X_predictions.append(prediction.cpu().detach().numpy())


y_predictions = []


for row in rows[HEADLINE_AMOUNT + 1:HEADLINE_AMOUNT + 300]:
    date = row[CSV_COLUMNS['DATE']].split(" ")[0]

    price = stock.get_price_on_date(date)

    y_predictions.append(1 if price[1] - price[0] > 0 else 0)

X = np.array(X_predictions)
y = np.array(y_predictions)

from sklearn.metrics import recall_score, f1_score, precision_score

p = model2(torch.from_numpy(X.astype(np.float32))).detach().numpy()
predicted = [1 if s[0] > 0 else 0 for s in [*p]]

print(precision_score(predicted, y), recall_score(predicted, y), f1_score(predicted, y))