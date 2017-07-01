# Electric Vehicle Detection
Gridcure predictive modeling challenge

Work in progress (7/1/2017)...

1. Which residences have electric vehicles?

I threw together a quick Random Forest on just 2 engineered features (average power and power range (max - min)) and performed a grid search to establish a baseline. I chose to optimize the model parameter based on F1 score, as accuracy is not appropriate for severely unbalanced classes (AUC score would have also been appropriate).

My best Random Forest had an `F1 score` of `0.684` and an `accuracy` of `81%` (a 17% improvement over guessing in proportion to the known number of EVs) using 3-fold cross-validation. This is what I will use as my baseline comparative model for predicting which residences have EVs. I suspect a bit of feature engineering and applying Gradient Boosting will go a long way.

To be continued...
