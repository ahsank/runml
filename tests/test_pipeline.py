import unittest
import runml.pipeline as pipeline
import pandas as pd
import runml.findata as findata


class TestNormalTradingRR(unittest.TestCase):
    def test_profit_pred(self):
        columns = ['pred_adjclose_period_change', 'true_adjclose_period_change',
            'expected_buy_profit', 'expected_sell_profit']
        testcases = [
            # Predicted change, True change, buy_profit, sell_profit
            [10, 5, 5, 0], # Buy only
            [10, -5, -5, 0],
            [0, 1, 0, 0], # didn't buy, or sell
            [0, -1, 0, 0],
            [-1, 5, 0, -5], # Shorted only
            [-1, -2, 0, 2]
        ]
        df = pd.DataFrame(testcases, columns=columns)
        findata.apply_trade(df, pipeline.NormalTradingRR)
        pd.testing.assert_series_equal(df.expected_buy_profit, df.buy_profit, check_names=False)
        pd.testing.assert_series_equal(df.expected_sell_profit, df.sell_profit, check_names=False)

class TestHighTradingRR(unittest.TestCase):
    def test_profit_pred(self):
        columns=['pred_high_period_change', 'true_high_period_change',
            'true_adjclose_period_change', 'expected_buy_profit',
            'expected_sell_profit']
        testcases = [
            # Columns: Predicted high change, True high change, true adjclose change,
            # buy_profit, sell_profit
            [10,  10, 5,  10,  5], # Buy at start, sold at predicted high and shorted
            [10,  11, 5,  10,  5],
            [10,  11, 15, 10, -5],
            [10,  5,  2,  2,   0], # Buy at start, closed at period end by close  price
            [10,  5,  -2, -2,  0], # Buy at start, closed at period end by close  price
            [0,   5,  2,  0,  -2], # shorted
            [0,   -5,  2,  0, -2], # Didn't buy or shorted
            [-10, 5,  6,  0,  -6], # Shorted as  predicted high is negative and closed at end
            [-10, -15,  6,  0,  -6], # Shorted as  predicted high is negative and closed at end
        ]
        df = pd.DataFrame(testcases, columns=columns)
        findata.apply_trade(df, pipeline.HighTradingRR)
        pd.testing.assert_series_equal(df.expected_buy_profit, df.buy_profit, check_names=False)
        pd.testing.assert_series_equal(df.expected_sell_profit, df.sell_profit, check_names=False)

class TestLowTradingRR(unittest.TestCase):
    def test_profit_pred(self):
        columns=['pred_low_period_change', 'true_low_period_change',
            'true_adjclose_period_change', 'expected_buy_profit',
            'expected_sell_profit']
        testcases = [
            # Columns: Predicted high change, True high change, true adjclose change,
            # buy_profit, sell_profit
            [-10,  -10, 5,  15,  10], # Short at start, cover at predicted low and buy
            [-10,  -11, 5,  15,  10],
            [-10,  -11, -15, -5, 10],
            [-10,  -5,  -2,  0,   2], # Short at start, closed at period end by close  price
            [-10,  5,  2, 0, -2], # short at start, closed at period end by close  price
            [0,   -5,  2,  2,   0], # Didn't short but bought and closed at end
            [0,   5,  2,  0,   0], # Didn't get chance to buy
            [10, -5,  6,  6,  0], # Bought  and and closed at end
            [10, 5,  6,  6,  0], # Bought  and and closed at end
        ]
        df = pd.DataFrame(testcases, columns=columns)
        findata.apply_trade(df, pipeline.LowTradingRR)
        pd.testing.assert_series_equal(df.expected_buy_profit, df.buy_profit, check_names=False)
        pd.testing.assert_series_equal(df.expected_sell_profit, df.sell_profit, check_names=False)

# if __name__ == '__main__':
#     unittest.main()
