import numpy as np
from pmdarima.arima import auto_arima
from pmdarima import model_selection

def predictWithArima(df, predictionPeriod: int):
    """
    Predicts the change in prices of stocks predictionPeriod amounts of 
    steps into future
    also returns crossvalidation scores for models
    """
    changes = []
    cv_scores = []
    for stockName in df:
        x = df[stockName]
        # arima doesn't allow Nans
        x = x.fillna(method="backfill").fillna(method="ffill")
        
        arima = auto_arima(x, start_p=1, max_p=5, start_d=1,
                            max_d=5, start_q=1, max_q=5, maxiter=100)
        pricesForecast = arima.predict(predictionPeriod)
        
        next = pricesForecast[-1]
        prev = x[-1]
        change = (next - prev) / prev
        changes.append(change)

        cv_score = crossValidateModel(arima, x, predictionPeriod)
        cv_scores.append(cv_score)

    return np.array(changes), np.array(cv_scores)

def crossValidateModel(model, x, predictionPeriod):
    """
    Compute mean cross validation score for model
    """
    cv = model_selection.SlidingWindowForecastCV(h=predictionPeriod)
    model_cv_scores = model_selection.cross_val_score(
        model, x, scoring='smape', cv=cv)
    mean_cv = model_cv_scores.mean()
    
    return mean_cv