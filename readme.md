# Motivation/problem statement: 

Crypto markets continue to gain adoption but the lack of equity like fundamentals makes forecasting prices a challenge.  In response, traders have come to heavily rely on technical indicators which try and capture short term opportunities.  Traders monitor these indicators for signals suggesting a price movement is imminent.  The goal of this project was to identify a few common indicators, strip them of their signals, and explore whether a machine learning algorithm could leverage the same underlying metric and produce actionable investment decisions.  

# Methods/procedure/approach: 

Several momentum indicators (50/200 rolling avg ratio, RSI, MACD, OBV) were evaluated against historical daily ETH-USD prices and volumes.  The indicators are all common technical indicators that try and capture shifts in buy/sell activity, short-term investor sentiment and short-term price momentum.  These indicators were calculated using the historical data, fed into a panda dataframe and incorporated into machine learning classification models.  A target variable was created identifying whether Close price (date+1) was higher or lower than (date) to see if the models could predict price (date+1) and outperform the baseline given current observations and indicators. Baseline was set at 1 (date+1 Close price is 'up' vs current observation) as it was the data set mode.

# Results/findings: 

My Random Forest and Logistic Regression models outperformed baseline accuracy of 51% at 55% & 54%, repectively.  More importantly, the Random Forest model was also able to achieve 55-56% precision on both prediction scenarios.  My Decision Tree model slightly underperformed the baseline.  All of the models significantly outperformed the benchmark accuracy by 15% points or more in the training datasets but failed to reproduce the results in the validate data sets suggesting some overfit.  Nevertheless, the validate results supported the train findings in my Random Forest & Logistic Regression models.  I tested my Random Forest model and the results were in line with my validate results.  

# Conclusion: 

The Random Forest machine learning model was able to utilize the momentum indicators included as features and provide value in its predictions. Specifically, it was 3% more accurate than the Baseline but, more importantly, the precision metrics were equally as robust across both 'up' (56%) and 'down' (55%) delta predictions. These recommendations are actionable considering we outperform the Baseline with either a 'long' or 'short' prediction. The goal was to identify reliable 'long' and 'short' predictions with a level of confidence that exceeds the Baseline so the precision metrics were significant in determining confidence in either of the model's predictions. Recall metrics weren't considered because the goal was not to capture as many of either the 'long' or 'short' opportunities but to maximize predictive performance in either direction. Considering the Baseline has a 100% 'long' recall with 52% overall accuracy, I believe the Random Forest model achieved the goal.

* Considering the fact that a lot of the momentum indicators included in the model are usually deployed in conjunction with a signal, I'd say the model did a good job of interpreting the indicators and extrapolating predictive trends.

*  In the Random Forest model, the features that were weighted more heavily in the model were the 'observed' delta features and the simple momentum indicators that weight the most recent data more heavily.

    * The indicators that are normally coupled with a signal event (RSI & MACD) were significantly weighted but carried the lowest importance within the model. The assigned importance ('weightings') suggests the model was able to extrapolate useful information from the indicators but left room for improvement by incorporating a 'signal' event. In fact, the MACD indicator is a relationship between the 12-period EMA & 26-period EMA features. The 12-period EMA indicator was the most heavily weighted feature in the model.

# Next Steps:

* In order to improve on the predictive value of the model, we could add signals to those indicators where normally deployed. The signals would try and capture the most likely scenarios and would decrease the noise by focusing on these high probability selections. We could then evaluate the model performance incorporating this selectivity.

* Further evaluation: We can see from the model results that we can outperform the Baseline and do it reliably in both directions. A 3-4% outperformance doesn't appear to be significant but, in terms of an investment, suggests the model is profitable. Performance here only encompasses correctly predicting the opportunities, the model did not measure what kind of returns could be expected with a 55-56% success rate. Additionally, we did not incorporate the costs of execution into the evaluation. In order to do so, we could run the predictions versus a mock scenario to further evaluate how what a 55-56% success rate translates into return wise. 

# Data Dictionary

   Feature      |  Data Type   | Description    |
| :------------- | :----------: | -----------: |
| Close| float64 | daily closing price |
| Volume | float64   | total daily trading volume  |
| close_dod   | float64 | Day over Day % change in Close  |
| volume_dod  | float64 | Day over Day % change in Volume |
| rolling_50C | float64 | rolling 50 day Close average |
| rolling_100C | float64 | rolling 100 day Close average |
| rolling_50V  | float64 | rolling 50 day Volume average |
| rolling_100V | float64 | rolling 100 day Volume average |
| RSI* | float64 | Relative Strenth Index |
| RSI7 | float64 | RSI 7 day period |
| RSI12 | float64 | RSI 12 day period  |
| RSI26 | float64 | RSI 26 day period |
| EMA* | float64 | Exponential Moving Average |
| EMA7 | float64 | EMA 7 day period |
| EMA12 | float64 | EMA 12 day period |
| EMA26 | float64 | EMA 26 day period |
| momentum-cross* | float64 | ratio of 50 day Close avg / 200 day Close avg |
| OBV* | float64 | On Balance Volume |
| MACD* | float64 | Moving Average Convergence Divergence |
| * calculation formulas included in notebook |
