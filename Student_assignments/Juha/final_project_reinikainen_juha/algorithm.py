import numpy as np
from pymoo.core.repair import Repair
from pymoo.core.population import Population

class OneSumRepair(Repair):
    """
    Repair solutions in population
    sum of weights equals to one
    weights are between w_min and w_max
    for each w_i where y_i == 1

    Kaucic, M., Moradi, M., & Mirzazadeh, M. (2019). 
    Portfolio optimization by improved NSGA-II and SPEA 2 based on different risk measures. 
    Financial Innovation, 5(1), 1-28.
    """
    def __init__(self, w_min, w_max, n_stocks) -> None:
        super().__init__()
        self.w_min = w_min
        self.w_max = w_max
        self.n_stocks = n_stocks

    def _do(self, problem, pop: Population, **kwargs):
        X = pop.get("X")
        W = X[:, :self.n_stocks] #weight vectors
        Y = X[:, self.n_stocks:] #boolean flags
        #TODO: probably not neccessary if mutation and crossover dont do unfeasible
        Wminmax = W.clip(self.w_min, self.w_max)
        #set weights of stocks that are not selected to zero
        Wminmax = W * Y
        Wsum = Wminmax.sum(axis=1)
        try:
            Wnorm = Wminmax.T / Wsum
        except ZeroDivisionError:
            Wsum = Wsum.astype(float)
            Wminmax = Wminmax.astype(float)
            Wnorm = Wminmax.T / Wsum
        #Wsum can be zero if none stocks are chosen
        nanValue = 1/self.n_stocks
        Wnorm = np.nan_to_num(Wnorm, nan=nanValue)

        WnormY = np.hstack((Wnorm.T, Y))
        pop.set("X", WnormY)
        return pop

