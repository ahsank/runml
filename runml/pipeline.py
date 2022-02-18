import numpy as np
import pandas as pd

from .findata import PreparedData, fetch_data, RNNModel, TradingResult

class NormalTrading:
    buy_profit  = lambda current, pred_future, true_future: true_future - current if pred_future > current else 0
    sell_profit = lambda current, pred_future, true_future: current - true_future if pred_future < current else 0

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
            .rolling(window=self.period)
            .mean()
            .to_frame(self.colname)
            .dropna())
      return df.join(data)

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
    sdata = data.rolling(window=200, min_periods=1).mean()

    # only use last 5 years data
    newdata = data.copy(deep=True).tail(1000)
    # newdata['adjclose'] = (data['adjclose']-sdata['adjclose'])/sdata['adjclose']
    # newdata['volume'] = (data['volume']-sdata['volume'])/sdata['volume']
    # newdata['open'] = data['open']/data['adjclose']
    # newdata['high'] = data['high']/data['adjclose']-1
    # newdata['low'] = 1-data['low']/data['adjclose']
    # newdata['close'] =  data['close']/data['adjclose']
    # newdata['ticker'] = data['ticker']
    newdata.dropna(inplace=True)

    if self.next is None:
      return newdata
    else:
      return self.next.change_data(newdata)

  def change_prep(self, pdata):
    pdata.ticker_data_filename += f"-w{self.name}"
    pdata.data_prefix += f"-w{self.name}"
    if self.lastdata is not None:
      pdata.lastref = self.lastdata.rolling(window=200, min_periods=1).mean().tail(1).adjclose.item()
      pdata.lastprice = self.lastdata.tail(1).adjclose.item()
      print(f"{pdata.ticker} \t MA: {pdata.lastref:.2f}, \t Price: {pdata.lastprice:.2f}")

    if self.next is not None:
      self.next.change_prep(pdata)

    
  def change_model(self, mod):
    if self.next is not None:
      self.next.change_model(mod)
 
  def predicted_price(self, pdata, res):
    return pdata.lastref * (1 + res.future_price)
  
  def predicted_gain(self, pdata, res):
    return self.predicted_price(pdata, res)/pdata.lastprice-1

  def get_predict(self, data, future_price):
      print(f"Last 200 dma {self.lastref}")
      print(f"Future price {self.predicted_price(res)}")

def runModelCombined(tickers, name, modifier, do_train=True, trading=NormalTrading):
  genpdata = PreparedData(name)
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
    pdata = PreparedData(ticker)
    pdatas.append(pdata)
    modifier.change_prep(pdata)
    pdata.prepare(data)
    mod = RNNModel()
    modifier.change_model(mod)
    mod.create(genpdata)

    if  do_train:
      if 'X_train' in genpdata.data:
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

  for pdata in pdatas:
    res = TradingResult(mod.model, pdata, mod.LOSS)
    res.eval(trading)
    res.print()
    modifier.print(res)
    df = df.append(
        {'Ticker': pdata.ticker,
         'Name': modifier.name,
         'Buy': round(res.total_buy_profit,2),
         'Sell': round(res.total_sell_profit,2),
         'Total': round(res.total_profit,2),
         'Last': round(pdata.lastprice,2),
         'Predicted': round(modifier.predicted_price(pdata, res),2),
         'Gain': round(modifier.predicted_gain(pdata, res),2)
         }, ignore_index=True)

  return df

