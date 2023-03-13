"""
This file contains scripts that are commonly used to handle data from the data source (CSV file)
"""
import csv
import datetime

from stock_price import StockPrices
from constants import CSV_COLUMNS


def find_stock_most_headlines(file):
    """
    Find the stock with the most headlines in the given CSV
    :param file: the name of the CSV file to search
    :return: tuple containing the ticker and the number of articles i.e. (GOOGL, 1597)
    """
    stocks = {}

    with open(file, newline='', encoding="utf8") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            ticker = row[-1]
            if ticker in stocks:
                stocks[ticker] += 1
            else:
                stocks[ticker] = 1

    max_num_articles = float('-inf')
    max_ticker = None

    for key in stocks:
        num_articles = stocks[key]

        if num_articles > max_num_articles:
            max_num_articles = num_articles
            max_ticker = key

    return max_ticker, max_num_articles


def get_rows_from_ticker(ticker, file):
    """
    Given a ticker, return all of the rows as a list that contain that ticker
    :param ticker: The ticker (i.e. GOOGL) to sort by
    :param file: The CSV file to search on
    :return: the rows pertaining to the ticker as a list
    """

    rows = []

    with open(file, newline='', encoding="utf8") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            row_ticker = row[-1]
            if ticker == row_ticker:
                rows.append(row)

    return rows


def get_date_range_from_rows(rows):
    """
    Given rows from the analyst ratings, return the min and max date
    :param rows: the rows from the CSV
    :return: min and max date as a tuple
    """

    max_d = datetime.datetime(2008, 1, 1)  # Articles start at 2009
    min_d = datetime.datetime(2022, 1, 1)  # Articles are not older than 2020

    for row in rows:
        year, month, day = row[CSV_COLUMNS['DATE']].split(" ")[0].split("-")
        date = datetime.datetime(int(year), int(month), int(day))

        if date < min_d:
            min_d = date

        if date > max_d:
            max_d = date

    return min_d, max_d


if __name__ == "__main__":
    CSV_FILE = 'raw_analyst_ratings.csv'
    # ticker, num_articles = find_stock_most_headlines(CSV_FILE)
    # print(ticker + " had the most headlines with " + str(num_articles) + " headlines")

    ticker = "MRK"
    rows = get_rows_from_ticker(ticker, CSV_FILE)
    # print(rows)

    min_date, max_date = get_date_range_from_rows(rows)
    print(min_date, max_date)

    stock = StockPrices(ticker, min_date, max_date)
    print(stock.prices)
