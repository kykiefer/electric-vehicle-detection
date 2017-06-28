# Electric Vehicle Detection
Gridcure predictive modeling challenge

Work in progress (6/28/2017)...

1. Which residences have electric vehicles?
I threw together a quick Random Forest on 4 engineered features (min power, max power, (max - min power), avg power) and performed a grid search to establish a baseline. I chose to optimize the model parameter based on f1 score as accuracy is not appropriate for severely unbalanced classes (AUC score would have also been appropriate).

With a bit of tuning, my best Random Forest had an `F1 score` of `0.715` and an `accuracy` of `84%` (a 21% improvement over guessing in proportion to the known number of EVs) using 3-fold cross-validation. This is what I will use as my baseline comparative model for predicting which residences have EVs. I suspect a bit of feature engineering and perhaps applying Gradient Boosting will go a long way.

To be continued...
