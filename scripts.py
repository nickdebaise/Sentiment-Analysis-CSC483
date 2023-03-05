import csv


def find_stock_most_headlines(file):
    """
    Find the stock with the most headlines in the given CSV
    :param file: the name of the CSV file to search
    :return: tuple containing the ticker and the number of articles i.e. (GOOGL, 1597)
    """
    stocks = {}

    with open(file, newline='') as csvfile:
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

    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            row_ticker = row[-1]
            if ticker == row_ticker:
                rows.append(row)

    return rows


if __name__ == "__main__":
    CSV_FILE = 'raw_analyst_ratings.csv'
    ticker, num_articles = find_stock_most_headlines(CSV_FILE)

    print(ticker + " had the most headlines with " + str(num_articles) + " headlines")

    rows = get_rows_from_ticker(ticker, CSV_FILE)

    print(rows)