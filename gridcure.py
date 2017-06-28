import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
np.random.seed(15)  # for reproducibility

# load developement data
df = pd.read_csv('EV_files/EV_train.csv')
labels = pd.read_csv('EV_files/EV_train_labels.csv')

# set index as House ID
df.set_index('House ID', inplace=True)
labels.set_index('House ID', inplace=True)

# get has_EV label for household
df['has_EV'] = labels.max(axis=1)

# create avg, stdev, min,  max, range and other values per house
df['Pwr_Min'] = df.max(axis=1)
df['Pwr_Max'] = df.max(axis=1)
df['Pwr_Avg'] = df.mean(axis=1)
df['Pwr_Rng'] = df.max(axis=1) - df.min(axis=1)
df['Pwr_Std'] = df.std(axis=1)
df['Pwr_IQR'] = df.quantile(q=0.75, axis=1) - df.quantile(q=0.25, axis=1)

# select features to fit model on
y = df.pop('has_EV')
features = ['Pwr_Min', 'Pwr_Max', 'Pwr_Avg', 'Pwr_Rng']
X = df[features]

# # grid parameters
# param_grid = {'n_estimators': [15, 20, 25],
#               'max_depth': [12, 15, 18, None],
#               'max_features': ['auto', 'log2'],
#               'min_samples_split': [2, 3, 4],
#               'min_samples_leaf': [1, 2],
#               'bootstrap': [True, False],
#               'criterion': ['gini', 'entropy']}
#
# # tune for f1 score to establish baseline
# scores = ['f1']
#
# for score in scores:
#     print('Tuning hyper-parameters for {}'.format(score))
#
#     rf = GridSearchCV(RandomForestClassifier(n_jobs=-1), param_grid=param_grid, cv=3, scoring=score)
#     rf.fit(X, y)
#
#     print('Best parameters set found on development set: {}'.format(rf.best_params_))
#     print()
#     print('Grid scores on development set:')
#     print()
#     means = rf.cv_results_['mean_test_score']
#     stds = rf.cv_results_['std_test_score']
#     for mean, std, params in zip(means, stds, rf.cv_results_['params']):
#         print('%0.3f (+/-%0.03f) for %r'
#               % (mean, std * 2, params))

# from the grid search these are our best model hyper-parameters
best_rf = RandomForestClassifier(n_estimators=25, criterion='entropy',
                                 max_features='log2', max_depth=None,
                                 min_samples_leaf=1, min_samples_split=2,
                                 bootstrap=True, n_jobs=-1)
best_rf.fit(X, y)
print('features:  ', features)
print('importance:', best_rf.feature_importances_)
scores = cross_val_score(best_rf, X, y, cv=3)
print('Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))



# # fill missing intervals with house mean
# # don't love this approach but good for first pass
# house_mean = pd.DataFrame({col: X.mean(axis=1) for col in X.columns})
# X.fillna(house_mean, inplace=True)
