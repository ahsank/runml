import numpy as np
import pandas as pd

from .findata import PreparedData, fetch_data, RNNModel, TradingResult

class NormalTrading:
  name = 'Normal'
  buy_profit  = lambda row : row.true_adjclose - row.adjclose if row.pred_adjclose > row.adjclose else 0
  sell_profit = lambda row: row.adjclose - row.true_adjclose if row.pred_adjclose < row.adjclose else 0

class LowTrading:
  name = 'Low'
  buy_profit  = lambda row :  row.true_adjclose - row.pred_low if row.true_low <= row.pred_low else 0
  sell_profit = lambda row: row.adjclose - row.true_adjclose if row.pred_low > row.true_low else row.adjclose - row.pred_low

class HighTrading:
  name = 'High'
  sell_profit  = lambda row:  row.pred_high-row.true_adjclose if row.true_high >= row.pred_high else 0
  buy_profit = lambda row: row.true_adjclose-row.adjclose  if row.pred_high < row.true_high else row.pred_high-row.adjclose


class NoModifier:
    name = 'Original'

    def change_prep(self, pdata):
        pass

    def change_model(self, mod):
        pass

    def change_data(self, data):
        return data

    def print(self, res):
      pass


class AddDay(NoModifier):
    def __init__(self):
        self.name = 'Day'

    def change_prep(self, pdata):
        pdata.FEATURE_COLUMNS = pdata.FEATURE_COLUMNS + ['day']
        pdata.ticker_data_filename += '-withday'
        pdata.data_prefix += '-withday'
        pass


    def change_data(self, data):
        # add date as a column
        if "date" not in data.columns:
            data["date"] = data.index

        df = data.apply(lambda row: row.date.timetuple().tm_yday, axis = 1)
        return df.to_frame('day').join(data)


class AddVWap(NoModifier):
    def change_prep(self, pdata):
        pdata.FEATURE_COLUMNS = pdata.FEATURE_COLUMNS + ['vwap']
        pdata.ticker_data_filename += '-wvwap'
        pdata.data_prefix += '-wvwap'
        pass


    def change_data(self, data):

        df = data.apply(lambda row: row.adjclose * row.volume, axis = 1)
        return df.to_frame('vwap').join(data)

class AddDayMonth(NoModifier):
    def __init__(self):
      self.name = 'DayMon'

    def change_prep(self, pdata):
        pdata.FEATURE_COLUMNS = pdata.FEATURE_COLUMNS + ['mday', 'month']
        pdata.ticker_data_filename += '-wdm'
        pdata.data_prefix += '-wdm'
        pass


    def change_data(self, data):
        # add date as a column
        if "date" not in data.columns:
            data["date"] = data.index

        dfd = data.apply(lambda row: row.date.timetuple().tm_mday, axis = 1)
        dfm = data.apply(lambda row: row.date.timetuple().tm_mon, axis = 1)
        df = dfd.to_frame('mday').join(data)
        return dfm.to_frame('month').join(df)


def getStatFrame():
    return pd.DataFrame(columns=['Ticker', 'Name', 'Buy', 'Sell', 'Total'])

def runTicker(ticker, models = [NoModifier], df=getStatFrame()):
    for model in models:
        prepare()
        result = runModel(ticker, model, NormalTrading, True)
        df = df.append(result, ignore_index=True)
    return df



def runTickers(tickers, models):
    df = getStatFrame()
    for ticker in tickers:
        df = runTicker(ticker, models, df)
    print(df)

class AddMA(NoModifier):
  def __init__(self, num, col='adjclose'):
    self.period = num
    self.colname = f"ma-{col}-{num}"
    self.col = col
    self.name = self.colname

  def change_prep(self, pdata):
        pdata.FEATURE_COLUMNS = pdata.FEATURE_COLUMNS + [self.colname]
        pdata.ticker_data_filename += f"-w{self.colname}"
        pdata.data_prefix += f"-w{self.colname}"

  def change_data(self, data):
    df = (data[self.col]
          .rolling(window=self.period, min_periods=1)
          .mean()
          .to_frame(self.colname))
#          .dropna()
    return df.join(data)

class Adj(NoModifier):
  def __init__(self, cols=['low', 'high']):
    self.colname = f"adj"
    self.cols = cols
    self.name = self.colname

  def change_prep(self, pdata):
    pdata.ticker_data_filename += f"-w{self.colname}"
    pdata.data_prefix += f"-w{self.colname}"

  def change_data(self, data):
    for col in self.cols:
      data[col] += (- data.close + data.adjclose)
    return data

class CropData(NoModifier):
    def __init__(self, num):
      self.num = num
      self.name = f"Crop{num}"

    def change_data(self, data):
      return data.tail(self.num)

class FeatureSeq(NoModifier):
  def __init__(self, classes):
    self.name = 'seq'
    self.modifiers = classes
    for cls in classes:
      self.name += cls.name

  def change_prep(self, pdata):
    for cls in self.modifiers:
      cls.change_prep(pdata)

  def change_model(self, mod):
    for obj in self.modifiers:
      obj.change_model(mod)

  def change_data(self, data):
    for cls in self.modifiers:
      data = cls.change_data(data)
    return data

  def print(self, res):
    for cls in self.modifiers:
      cls.print(res)


class RateReturnOnly(NoModifier):

  def __init__(self, next=None):
    self.name = 'RROnly'
    self.next = next
    self.lastdata = None
    # =['adjclose', 'volume', 'open', 'high', 'low']

  def change_data(self, data):
    self.lastdata = data
    # only use last 5 years data
    newdata = data.copy(deep=True).tail(1000)
    newdata.dropna(inplace=True)

    if self.next is None:
      return newdata
    else:
      return self.next.change_data(newdata)

  def change_prep(self, pdata):
    pdata.ticker_data_filename += f"-w{self.name}"
    pdata.data_prefix += f"-w{self.name}"
    if self.lastdata is not None:
      pdata.lastprice = self.lastdata.tail(1).adjclose.item()
      if IS_VERBOSE:
          print(f"{pdata.ticker} \t Price: {pdata.lastprice:.2f}")

    if self.next is not None:
      self.next.change_prep(pdata)


  def change_model(self, mod):
    if self.next is not None:
      self.next.change_model(mod)

  def predicted_price(self, pdata, res):
    return res.future_price

  def predicted_gain(self, pdata, res):
    return self.predicted_price(pdata, res)/pdata.lastprice-1


IS_VERBOSE = False

def runModelCombined(tickers, name, modifier, do_train=True, loss="huber_loss", output='adjclose', trading=NormalTrading):
  genpdata = PreparedData(name, output)
  genpdata.data = {}
  modifier.change_prep(genpdata)
  pdatas = []
  tickerset = set()
  for ticker in tickers:
    if ticker in tickerset:
      raise ValueError(f"Duplicate ticker {ticker}")
    tickerset.add(ticker)

  for ticker in tickers:
    data = fetch_data(ticker)
    data = modifier.change_data(data)
    pdata = PreparedData(ticker, output)
    pdatas.append(pdata)
    modifier.change_prep(pdata)
    pdata.prepare(data)
    mod = RNNModel(loss=loss)
    modifier.change_model(mod)
    mod.create(genpdata)

    if  do_train:
      if 'X_train' in genpdata.data:
        if IS_VERBOSE:
            print(f"Adding {pdata.ticker} {pdata.data['X_train'].shape}")
        genpdata.data['X_train'] = np.concatenate((genpdata.data['X_train'], pdata.data['X_train']))
        genpdata.data['y_train'] = np.concatenate((genpdata.data['y_train'], pdata.data['y_train']))
        genpdata.data['X_test'] = np.concatenate((genpdata.data['X_test'], pdata.data['X_test']))
        genpdata.data['y_test'] = np.concatenate((genpdata.data['y_test'], pdata.data['y_test']))

      else:
        genpdata.data['X_train'] = pdata.data['X_train']
        genpdata.data['y_train'] = pdata.data['y_train']
        genpdata.data['X_test'] = pdata.data['X_test']
        genpdata.data['y_test'] = pdata.data['y_test']

  if do_train:
    mod.train(genpdata.data)
  else:
    mod.load()


  df = getStatFrame()
  rows = []
  results = {}

  for pdata in pdatas:
    res = TradingResult(mod.model, pdata, mod.LOSS)
    results[pdata.ticker] = res
    res.eval()
    res.do_trade(trading)
    if IS_VERBOSE:
        res.print()
    modifier.print(res)
    rows.append(
        {'Ticker': pdata.ticker,
         'Name': modifier.name,
         'Error': res.mean_error,
         'Accuracy': res.accuracy_score,
         'Buy': round(res.total_buy_profit,2),
         'Sell': round(res.total_sell_profit,2),
         'Total': round(res.total_profit,2),
         'Last': round(pdata.lastprice,2),
         'Predicted': round(modifier.predicted_price(pdata, res),2),
         'Gain': round(modifier.predicted_gain(pdata, res),2)
         })

  return pd.DataFrame(rows), results

# Add adjhigh, adjlow
# Train for them

def runModelCombinedVola(tickers, name, modifier, do_train=True, loss="huber_loss"):
  dfs = []
  trading = {'adjclose' : NormalTrading, 'high' : HighTrading, 'low' : LowTrading }
  for col, cls in trading.items():
    df, results = runModelCombined(tickers, name, modifier, do_train, loss, col, cls)
    df["Col"] = col
    dfs.append(df)

  finaldf = pd.concat(dfs)
  return finaldf.set_index(['Ticker', 'Col']);
