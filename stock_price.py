"""
This file contains two classes that help model a Stock and Stock Prices pertaining to a given stock
"""
import datetime
import yfinance as yf

import requests_cache

session = requests_cache.CachedSession('yfinance.cache')
session.headers['User-agent'] = 'my-program/1.0'

class Stock:
    def __init__(self, ticker):
        self.ticker = ticker
        self.api = yf.Ticker(ticker, session=session)

    def get_price_on_date(self, d):
        """
        Given a date, return the price on the given date
        :param d: the date to find, should be string like "2022-02-06"
        :return: the price on the given date in tuple form of (open price, close price)
        """
        year, month, day = d.split("-")
        start_date = datetime.datetime(int(year), int(month), int(day))
        end_date = start_date + datetime.timedelta(days=1)

        data = self.api.history(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"), debug=False)
        values = data.values

        # Check to make sure value exists (i.e. on a business day)
        if len(values) == 0:
            # Increase until the next business day
            while len(values) == 0:
                start_date = start_date + datetime.timedelta(days=1)
                end_date = start_date + datetime.timedelta(days=1)
                data = self.api.history(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"), debug=False)
                values = data.values

        return data['Open'].values[0], data['Close'].values[0]


class StockPrices(Stock):
    def __init__(self, ticker, start_date, end_date):
        """
        Get and store stock prices for a given date range
        :param ticker: The ticker of the company
        :param start_date: the start date of historical prices, can be a string like "2022-02-06" or a datetime
        :param end_date: the end date of historical prices, can be a string like "2022-02-07" or a datetime

        Example:
        merck = StockPrices("MRK", "2021-02-06", "2021-02-18")
        print(merck.get_price_on_date("2021-02-08"))
        """
        super().__init__(ticker)

        if isinstance(start_date, str):
            start_year, start_month, start_day = start_date.split("-")
            self.start_date = datetime.datetime(int(start_year), int(start_month), int(start_day))

            end_year, end_month, end_day = end_date.split("-")
            self.end_date = datetime.datetime(int(end_year), int(end_month), int(end_day))
        else:
            self.start_date = start_date
            self.end_date = end_date

        self.prices = {}
        self.__populate_prices()

    def __populate_prices(self):
        """
        Use the given date range to populate a dictionary of prices for each day in the range
        Do this to make lookup of prices quick
        """
        data = self.api.history(start=self.start_date.strftime("%Y-%m-%d"), end=self.end_date.strftime("%Y-%m-%d"))

        for index, row in data.iterrows():
            price = (row['Open'], row['Close'])
            date = index.to_pydatetime().strftime("%Y-%m-%d")

            self.prices[date] = price

    def get_price_on_date(self, d):
        """
        Given a date, return the price on the given date
        :param d: the date to find, should be string like "2022-02-06"
        :return: the price on the given date in tuple form of (open price, close price)
        """
        if d in self.prices:
            return self.prices[d]

        else:
            return super().get_price_on_date(d)


if __name__ == "__main__":
    merck = StockPrices("MRK", "2021-02-06", "2021-02-18")

    print(merck.get_price_on_date("2021-02-08"))
