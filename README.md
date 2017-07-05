# Electric Vehicle Detection
Gridcure predictive modeling challenge

Work in progress (7/4/2017)...

1. Which residences have electric vehicles?

I threw together a quick Random Forest on just 3 engineered features (`Pwr_Rng`, `Pwr_Std`, `Avg_Norm_Rng`) and performed a grid search to establish a baseline. I chose to optimize the model parameters based on F1 score, as accuracy is not appropriate for imbalanced classes.

My best Random Forest had an `F1 score` of `0.686` and an `accuracy` using 3-fold cross-validation. This is what I will use as my baseline comparative model for predicting which residences have EVs.

I next wanted to try gradient boosting. An out of the box gradient boosted classifier (using xgboost) yielded an `F1 score` of `0.716`. Very promising, considering boosted models typically yield good improvements with detailed tuning.

Next step is to perform a quick grid search on the xgboost model, but I ultimately expect going back for some additional feature engineering to be most critical.

To be continued...
