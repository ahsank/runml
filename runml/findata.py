import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from yahoo_fin import stock_info as si
from collections import deque

import os
import numpy as np
import pandas as pd
import random
import time
from tensorflow.keras.layers import LSTM
import matplotlib.pyplot as plt

def prepare():
    # set seed, so we can get the same results after rerunning several times
    np.random.seed(314)
    tf.random.set_seed(314)
    random.seed(314)

def shuffle_in_unison(a, b):
    # shuffle two arrays in the same way
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)

# Following two functions code mainly from https://www.thepythoncode.com/article/stock-price-prediction-in-python-using-tensorflow-2-and-keras
def load_data(ticker, n_steps, scale, shuffle, lookup_step, split_by_date,
                test_size, feature_columns, output_column):
    """
    Loads data from Yahoo Finance source, as well as scaling, shuffling, normalizing and splitting.
    Params:
        ticker (str/pd.DataFrame): the ticker you want to load, examples include AAPL, TESL, etc.
        n_steps (int): the historical sequence length (i.e window size) used to predict, default is 50
        scale (bool): whether to scale prices from 0 to 1, default is True
        shuffle (bool): whether to shuffle the dataset (both training & testing), default is True
        lookup_step (int): the future lookup step to predict, default is 1 (e.g next day)
        split_by_date (bool): whether we split the dataset into training/testing by date, setting it 
            to False will split datasets in a random way
        test_size (float): ratio for test data, default is 0.2 (20% testing data)
        feature_columns (list): the list of features to use to feed into the model, default is everything grabbed from yahoo_fin
    """
    df = fetch_data(ticker)
    # this will contain all the elements we want to return from this function
    result = {}
    origdf = df.copy()
    origdf['unscaled_future_adjclose'] = origdf['adjclose'].shift(-lookup_step)
    result['df'] = origdf.copy()
    
    # make sure that the passed feature_columns exist in the dataframe
    for col in feature_columns:
        assert col in df.columns, f"'{col}' does not exist in the dataframe."
    # add date as a column
    if "date" not in df.columns:
        df["date"] = df.index
    if scale:
        column_scaler = {}
        # scale the data (prices) from 0 to 1
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler
        # add the MinMaxScaler instances to the result returned
        result["column_scaler"] = column_scaler
    # add the target column (label) by shifting by `lookup_step`
    df['future_adjclose'] = df['adjclose'].shift(-lookup_step)
    df['future_low'] = df['low'].rolling(lookup_step).min().shift(-lookup_step)
    df['future_high'] = df['high'].rolling(lookup_step).max().shift(-lookup_step)
    # we will also return the original dataframe itself
    future_column = f"future_{output_column}"

    # last `lookup_step` columns contains NaN in future column
    # get them before droping NaNs
    last_sequence = np.array(df[feature_columns].tail(lookup_step))
    # drop NaNs
    df.dropna(inplace=True)
    sequence_data = []
    sequences = deque(maxlen=n_steps)
    for entry, target in zip(df[feature_columns + ["date"]].values, df[future_column].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])
    # get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
    # for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 60 (that is 50+10) length
    # this last_sequence will be used to predict future stock prices that are not available in the dataset
    last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
    last_sequence = np.array(last_sequence).astype(np.float32)
    # add to result
    result['last_sequence'] = last_sequence
    # construct the X's and y's
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)
    # convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    if split_by_date:
        # split the dataset into training & testing sets by date (not randomly splitting)
        train_samples = int((1 - test_size) * len(X))
        result["X_train"] = X[:train_samples]
        result["y_train"] = y[:train_samples]
        result["X_test"]  = X[train_samples:]
        result["y_test"]  = y[train_samples:]
        if shuffle:
            # shuffle the datasets for training (if shuffle parameter is set)
            shuffle_in_unison(result["X_train"], result["y_train"])
            shuffle_in_unison(result["X_test"], result["y_test"])
    else:    
        # split the dataset randomly
        result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y, test_size=test_size, shuffle=shuffle)
    # get the list of test set dates
    dates = result["X_test"][:, -1, -1]
    # retrieve test features from the original dataframe
    result["test_df"] = result["df"].loc[dates]
    # remove duplicated dates in the testing dataframe
    result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(keep='first')]
    # remove dates from the training/testing sets & convert to float32
    result["X_train"] = result["X_train"][:, :, :len(feature_columns)].astype(np.float32)
    result["X_test"] = result["X_test"][:, :, :len(feature_columns)].astype(np.float32)
    return result

def get_final_df(model, data, scale, lookup_step, output_column):
    """
    This function takes the `model` and `data` dict to 
    construct a final dataframe that includes the features along 
    with true and predicted prices of the testing dataset
    """
    X_test = data["X_test"]
    y_test = data["y_test"]
    # perform prediction and get prices
    y_pred = model.predict(X_test)
    if scale:
        y_test = np.squeeze(data["column_scaler"][output_column].inverse_transform(np.expand_dims(y_test, axis=0)))
        y_pred = np.squeeze(data["column_scaler"][output_column].inverse_transform(y_pred))
    test_df = data["test_df"]
    # add predicted future prices to the dataframe
    test_df[f"{output_column}_{lookup_step}"] = y_pred
    # add true future prices to the dataframe
    test_df[f"true_{output_column}_{lookup_step}"] = y_test
    # sort the dataframe by date
    test_df.sort_index(inplace=True)
    final_df = test_df
    # add the buy profit column

    columns = [output_column, f"{output_column}_{lookup_step}", f"true_{output_column}_{lookup_step}"]
    if 'adjclose' not in columns:
        columns.append('adjclose')
    output = final_df[columns]
    return output.rename(columns={
        f"{output_column}_{lookup_step}": f"pred_{output_column}",
        f"true_{output_column}_{lookup_step}": f"true_{output_column}"})


def apply_trade(final_df, lookup_step, trade):
    # TODO: Use multi index
    final_df["buy_profit"] = list(final_df.apply(trade.buy_profit, axis=1)
                                    # since we don't have profit for last sequence, add 0's
                                    )
    # add the sell profit column
    final_df["sell_profit"] = list(final_df.apply(trade.sell_profit, axis=1)
                                    # since we don't have profit for last sequence, add 0's
                                    )
    # final_df["buy_profit"] = list(map(trade.buy_profit, 
    #                                 final_df[output_column], 
    #                                 final_df[f"{output_column}_{lookup_step}"], 
    #                                 final_df[f"true_{output_column}_{lookup_step}"])
    #                                 # since we don't have profit for last sequence, add 0's
    #                                 )
    # # add the sell profit column
    # final_df["sell_profit"] = list(map(trade.sell_profit, 
    #                                 final_df[output_column], 
    #                                 final_df[f"{output_column}_{lookup_step}"], 
    #                                 final_df[f"true_{output_column}_{lookup_step}"])
    #                                 # since we don't have profit for last sequence, add 0's
    #                                 )

    return final_df

class TradingResult:
    def __init__(self, model, prepdata, lossn):
        self.model = model
        self.pdata = prepdata
        self.data = prepdata.data
        self.output_column = prepdata.OUTPUT_COLUMN
        self.LOSSN = lossn
        
    def predict(self):
        # retrieve the last sequence from data
        last_sequence = self.data["last_sequence"][-self.pdata.N_STEPS:]
        # expand dimension
        last_sequence = np.expand_dims(last_sequence, axis=0)
        # get the prediction (scaled from 0 to 1)
        prediction = self.model.predict(last_sequence)
        # get the price (by inverting the scaling)
        if self.pdata.SCALE:
            predicted_price = self.data["column_scaler"][self.output_column].inverse_transform(prediction)[0][0]
        else:
            predicted_price = prediction[0][0]
        return predicted_price

    def eval(self):
        # evaluate the model
        loss, merr = self.model.evaluate(self.data["X_test"], self.data["y_test"], verbose=0)
        # calculate the mean absolute error (inverse scaling)
        if self.pdata.SCALE:
            self.mean_error = self.data["column_scaler"][self.output_column].inverse_transform([[merr]])[0][0]
        else:
            self.mean_error = merr
            
        # get the final dataframe for the testing set
        self.final_df = get_final_df(self.model, self.data, self.pdata.SCALE, self.pdata.LOOKUP_STEP, self.output_column)
        self.loss = loss
        # predict the future price
        self.future_price = self.predict()

    def do_trade(self, trade):
        final_df = self.final_df
        if 'true_adjclose' not in final_df:
            final_df['true_adjclose'] = self.data['test_df']['unscaled_future_adjclose']
        apply_trade(final_df,  self.pdata.LOOKUP_STEP, trade)
        

        # we calculate the accuracy by counting the number of positive profits
        self.accuracy_score = (len(final_df[final_df['sell_profit'] >= 0]) + len(final_df[final_df['buy_profit'] >= 0])) / len(final_df)
        # calculating total buy & sell profit
        self.total_buy_profit  = final_df["buy_profit"].sum()
        self.total_sell_profit = final_df["sell_profit"].sum()
        # total profit by adding sell & buy together
        self.total_profit = self.total_buy_profit + self.total_sell_profit
        # dividing total profit by number of testing samples (number of trades)
        self.profit_per_trade = self.total_profit / len(final_df)
        self.final_df = final_df

    def print(self):
        # printing metrics
        print(f"Ticker {self.pdata.ticker}")
        print(f"Future price after {self.pdata.LOOKUP_STEP} days is {self.future_price:.2f}$")
        print(f"{self.LOSSN}_loss:", self.loss)
        print("Mean Error:", self.mean_error)
        print("Accuracy score:", self.accuracy_score)
        print("Total buy profit:", self.total_buy_profit)
        print("Total sell profit:", self.total_sell_profit)
        print("Total profit:", self.total_profit)
        print("Profit per trade:", self.profit_per_trade)

G_LOOKUP_STEP = 15

class PreparedData:
    def __init__(self, ticker, output):
        # Window size or the sequence length
        self.N_STEPS = 50
	# Lookup step, 1 is the next day
        self.LOOKUP_STEP = G_LOOKUP_STEP
        # whether to scale feature columns & output price as well
        self.SCALE = True
        self.scale_str = f"sc-{int(self.SCALE)}"
        # whether to shuffle the dataset
        self.SHUFFLE = True
        self.shuffle_str = f"sh-{int(self.SHUFFLE)}"
        # whether to split the training/testing set by date
        self.SPLIT_BY_DATE = False
        self.split_by_date_str = f"sbd-{int(self.SPLIT_BY_DATE)}"
        # test ratio size, 0.2 is 20%
        self.TEST_SIZE = 0.2
        # features to use
        self.FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]
        self.OUTPUT_COLUMN = output
        self.ticker = ticker
        # date now
        self.date_now = time.strftime("%Y-%m-%d")        
        self.ticker_data_filename = os.path.join("data", f"{self.ticker}_{self.date_now}.csv")
        self.data_prefix = f"{self.ticker}-{self.OUTPUT_COLUMN}-{self.shuffle_str}-{self.scale_str}-{self.split_by_date_str}-seq-{self.N_STEPS}-step-{self.LOOKUP_STEP}"
        
    def prepare(self,  df):
        # load the data
        self.data = load_data(df, self.N_STEPS, scale=self.SCALE, split_by_date=self.SPLIT_BY_DATE, 
                shuffle=self.SHUFFLE, lookup_step=self.LOOKUP_STEP, test_size=self.TEST_SIZE, 
                              feature_columns=self.FEATURE_COLUMNS, output_column=self.OUTPUT_COLUMN)

        # save the dataframe
        self.data["df"].to_csv(self.ticker_data_filename)


def fetch_data(ticker):
# see if ticker is already a loaded stock from yahoo finance
    if isinstance(ticker, str):
        # load it from yahoo_fin library
        df = si.get_data(ticker)
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
    elif isinstance(ticker, pd.DataFrame):
        # already loaded, use it directly
        df = ticker
    else:
        raise TypeError("ticker can be either a str or a `pd.DataFrame` instances")
    return df

def create_model(sequence_length, n_features, units=256, cell=LSTM, n_layers=2, dropout=0.3,
                loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False):
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            # first layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True), batch_input_shape=(None, sequence_length, n_features)))
            else:
                model.add(cell(units, return_sequences=True, batch_input_shape=(None, sequence_length, n_features)))
        elif i == n_layers - 1:
            # last layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=False)))
            else:
                model.add(cell(units, return_sequences=False))
        else:
            # hidden layers
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True)))
            else:
                model.add(cell(units, return_sequences=True))
        # add dropout after each layer
        model.add(Dropout(dropout))
    model.add(Dense(1, activation="linear"))
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
    return model


import os
import time
from tensorflow.keras.layers import LSTM

EPOCHS = 20

class RNNModel:
    def __init__(self, loss="huber_loss"):
        self.date_now = time.strftime("%Y-%m-%d")
        ### model parameters
        self.N_LAYERS = 2
        # LSTM cell
        self.CELL = LSTM
        # 256 LSTM neurons
        self.UNITS = 256
        # 40% dropout
        self.DROPOUT = 0.4
        # whether to use bidirectional RNNs
        self.BIDIRECTIONAL = False
        ### training parameters
        # mean absolute error loss
        # LOSS = "mae"
        # huber loss
        self.LOSS = loss
        self.OPTIMIZER = "adam"
        self.BATCH_SIZE = 64
        self.EPOCHS = EPOCHS # 500

    def create(self, prepdata):
        # model name to save, making it as unique as possible based on parameters
        self.model_name = f"{prepdata.data_prefix}-model-{self.LOSS}-{self.OPTIMIZER}-{self.CELL.__name__}-layers-{self.N_LAYERS}-units-{self.UNITS}"
        self.model_path = os.path.join("results", self.model_name + ".h5")
        if self.BIDIRECTIONAL:
            self.model_name += "-b"

        # create these folders if they does not exist
        if not os.path.isdir("results"):
            os.mkdir("results")
        if not os.path.isdir("logs"):
            os.mkdir("logs")
        if not os.path.isdir("data"):
            os.mkdir("data")

        self.model = create_model(prepdata.N_STEPS, len(prepdata.FEATURE_COLUMNS), loss=self.LOSS, units=self.UNITS, cell=self.CELL, n_layers=self.N_LAYERS,
                    dropout=self.DROPOUT, optimizer=self.OPTIMIZER, bidirectional=self.BIDIRECTIONAL)
        # some tensorflow callbacks
        self.checkpointer = ModelCheckpoint(self.model_path, save_weights_only=True, save_best_only=True, verbose=1)
        self.tensorboard = TensorBoard(log_dir=os.path.join("logs", self.model_name))
        self.earlystopping = EarlyStopping(monitor='loss', patience=5)

    def train(self, data):
        # train the model and save the weights whenever we see 
        # a new optimal model using ModelCheckpoint
        history = self.model.fit(data["X_train"], data["y_train"],
                                 batch_size=self.BATCH_SIZE,
                                 epochs=self.EPOCHS,
                                 validation_data=(data["X_test"], data["y_test"]),
                                 callbacks=[self.checkpointer, self.tensorboard, self.earlystopping],
                                 verbose=1)
        self.load()

    def load(self):
        # load optimal model weights from results folder
        print(f"loading model from {self.model_path}")
        self.model.load_weights(self.model_path)



def runModel(ticker, modifier, trading, do_train=True):
    data = fetch_data(ticker)

    data = modifier.change_data(data)
    
    pdata = PreparedData(ticker)
    modifier.change_prep(pdata)
        
    pdata.prepare(data)
    
    mod = RNNModel()

    modifier.change_model(mod)
    mod.create(pdata)
    if not do_train:
        # TODO train only when not already trained
        mod.load()
    else:
        mod.train(pdata.data)
    res = TradingResult(mod.model, pdata, mod.LOSS)
    res.eval(trading)
    res.print()
    modifier.print(res)
    #df.set_index(['Ticker', 'Name'])
    return {'Ticker': ticker, 'Name': modifier.name, 'Buy': res.total_buy_profit,
               'Sell': res.total_sell_profit, 'Total': res.total_profit}
              



