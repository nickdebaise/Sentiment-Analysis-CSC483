'''
Test implementation using Hugging Face's FinBert model on Headline data
'''

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import scripts

CSV_COLUMNS = {"INDEX": 0, "HEADLINE": 1, "URL": 2, "Publisher": 3, "DATE": 4, "TICKER": 5}

if __name__ == "__main__":
    CSV_FILE = 'raw_analyst_ratings.csv'
    ticker, num_articles = scripts.find_stock_most_headlines(CSV_FILE)

    print(ticker + " had the most headlines with " + str(num_articles) + " headlines")

    rows = scripts.get_rows_from_ticker(ticker, CSV_FILE)
    headlines_list = [row[CSV_COLUMNS["HEADLINE"]] for row in rows]

    # Getting the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

    HEADLINE_AMOUNT = 10

    # Turning the headlines into tokens to be inputted into model
    inputs = tokenizer(headlines_list[:HEADLINE_AMOUNT], padding=True, truncation=True, return_tensors='pt')

    # Inference
    outputs = model(**inputs)
    print(outputs.logits.shape)

    # Postprocessing with softmax
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    print(predictions)
