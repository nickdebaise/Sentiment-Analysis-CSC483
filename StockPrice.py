import datetime
import yfinance as yf


class Stock:
    def __init__(self, ticker):
        self.ticker = ticker
        self.api = yf.Ticker(ticker)

    def get_price_on_date(self, d):
        year, month, day = d.split("-")
        start_date = datetime.datetime(int(year), int(month), int(day))
        end_date = start_date + datetime.timedelta(days=1)

        data = self.api.history(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
        values = data.values

        if len(values) == 0:
            # Increase until the next business day
            while len(values) == 0:
                start_date = start_date + datetime.timedelta(days=1)
                end_date = start_date + datetime.timedelta(days=1)
                data = self.api.history(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
                values = data.values

        return data['Close'].values[0]


class StockPrices(Stock):
    def __init__(self, ticker, start_date, end_date):
        super().__init__(ticker)
        self.ticker = ticker
        self.api = yf.Ticker(ticker)

        if type(start_date) == 'str':
            start_year, start_month, start_day = start_date.split("-")
            self.start_date = datetime.datetime(int(start_year), int(start_month), int(start_day))

            end_year, end_month, end_day = end_date.split("-")
            self.end_date = datetime.datetime(int(end_year), int(end_month), int(end_day))

        else:
            self.start_date = start_date
            self.end_date = end_date

        self.prices = {}

        self.populate_prices()

    def populate_prices(self):
        data = self.api.history(start=self.start_date.strftime("%Y-%m-%d"), end=self.end_date.strftime("%Y-%m-%d"))

        for index, row in data.iterrows():
            price = row['Close']
            date = index.to_pydatetime().strftime("%Y-%m-%d")

            self.prices[date] = price

        return self.prices

    def get_price_on_date(self, d):
        if d in self.prices:
            return self.prices[d]

        else:
            return super().get_price_on_date(d)


if __name__ == "__main__":
    merck = StockPrices("MRK", "2021-02-06", "2021-02-18")

    print(merck.get_price_on_date("2021-02-08"))
