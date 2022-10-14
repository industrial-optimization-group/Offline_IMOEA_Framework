import numpy as np
from pymoo.core.problem import Problem


class PortfolioSelection(Problem):
    """
    Portfolio selection problem
    choose weight w_i for each asset n to
    maximize expected return 
    and minimize risk
    and maximize environmental, social and governance (ESG) score
    subject to
        w_min <= wi <= w_max
        sum(wi) = 1
        Cardinality constraint
            there need to be between c_min and c_max stocks
            in the portfolio
    note: expected return is negated here so
          both objective can be minimized
    """
    def __init__(self, df, prices, esg, w_min: float, w_max: float, c_min: int, c_max: int, 
                n_stocks: int, predictionPeriod: int):
        """
        Params:
            df: pandas.DataFrame of stock prices
            prices: predicted prices for stocks
            esg: pandas.Series of esg scores for each stock
            w_min: minimum weight for stock
            w_max: maximum weight for stock
            c_min: smallest amount of stocks allowed in a portfolio
            c_max: largest amount of stocks allowed in a portfolio
            n_stocks: how many stocks there are in total
            predictionPeriod: how many time periods to predict into future
        """
        xl = np.concatenate((np.full(n_stocks, w_min), np.zeros(n_stocks, dtype=int)))
        xu = np.concatenate((np.full(n_stocks, w_max), np.ones(n_stocks, dtype=int)))

        super().__init__(n_var=2 * n_stocks, n_obj=3, n_constr =2, xl=xl, xu=xu)

        self.w_min = w_min
        self.w_max = w_max
        self.c_min = c_min
        self.c_max = c_max
        self.n_stocks = n_stocks
        
        self.prices = prices
        self.esg = esg

        change = df.pct_change()
        self.cov = change.cov()

        self.predictionPeriod = predictionPeriod


    def _evaluate(self, X, out, *args, **kwargs):
        """
        Calculate objective and constraint function values
        """
        W = X[:, :self.n_stocks]
        Y = X[:, self.n_stocks:]
        WW = W * Y
        expected_return = WW @ self.prices
        
        risk = np.array([(w.T @ self.cov @ w) for w in WW]) * self.predictionPeriod
        esg_score = WW @ self.esg

        n_stocks_in_portfolio = Y.sum(axis=1)
        #n_stocks in solution needs to be between c_min and c_max
        g1 = n_stocks_in_portfolio - self.c_max
        g2 = self.c_min - n_stocks_in_portfolio

        out["F"] = np.array([-expected_return, risk, -esg_score]).T
        out["G"] = np.array([g1, g2]).T
