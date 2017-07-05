import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
np.random.seed(15)  # for reproducibility


def load_data():
    # load developement data
    df = pd.read_csv('EV_files/EV_train.csv')
    labels = pd.read_csv('EV_files/EV_train_labels.csv')

    # set index as House ID
    df.set_index('House ID', inplace=True)
    labels.set_index('House ID', inplace=True)

    # create series with index as House ID and Has_EV label (0/1)
    ev_label = labels.max(axis=1).rename('Has_EV')

    return df, ev_label


def feature_engineer(df, ev_label):
    # create avg, stdev, min,  max, range and other values per house
    X = df.mean(axis=1).rename('Pwr_Avg').to_frame()
    X['Pwr_Med'] = df.median(axis=1)
    X['Pwr_Rng'] = df.max(axis=1) - df.min(axis=1)
    X['Pwr_Std'] = df.std(axis=1)
    X['Pwr_IQR'] = df.quantile(q=0.75, axis=1) - df.quantile(q=0.25, axis=1)
    X['Avg_Norm_Rng'] = X['Pwr_Rng'] / X['Pwr_Avg']
    X['IQR_Norm'] = X['Pwr_IQR'] / X['Pwr_Avg']

    # inner join on House ID index
    X = pd.concat((X, ev_label), axis=1, join='inner')

    # select features to fit model on
    y = X.pop('Has_EV')
    return X, y


def rf_grid(X, y):
    # grid parameters
    param_grid = {'n_estimators': [15, 20, 25],
                  'max_depth': [12, 15, 18, None],
                  'max_features': ['auto', 'log2'],
                  'min_samples_split': [2, 3, 4],
                  'min_samples_leaf': [1, 2],
                  'bootstrap': [True, False],
                  'criterion': ['gini', 'entropy']}

    # tune for f1 score to establish baseline
    scores = ['f1']

    for score in scores:
        print('Tuning hyper-parameters for {}'.format(score))

        rf = GridSearchCV(RandomForestClassifier(n_jobs=-1), param_grid=param_grid, cv=3, scoring=score)
        rf.fit(X, y)

        print('Best parameters set found on development set: {}'.format(rf.best_params_))
        print()
        print('Grid scores on development set:')
        print()
        means = rf.cv_results_['mean_test_score']
        stds = rf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, rf.cv_results_['params']):
            print('%0.3f (+/-%0.03f) accuracy for %r' % (mean, std * 2, params))
        return rf

def boosted_trees(X, y):
    pass


if __name__ == '__main__':
    df, ev_label = load_data()
    X, y = feature_engineer(df, ev_label)

    # # gradient boosted features on selected features
    # features = ['Pwr_Avg', 'Pwr_Rng', 'Normed_Pwr_Rng']
    # X = df[features]
    # boosted_tree = XGBClassifier(max_depth=2)
    # scores = cross_val_score(boosted_tree, X, y, cv=3, scoring='f1')
    # print('F1', scores.mean())

    # random forest features
    features = ['Pwr_Rng', 'Pwr_Std', 'Avg_Norm_Rng']
    X = X[features]

    print('Performing grid search...')
    rf = rf_grid(X, y)

    # # from the f1 sscore grid search these are our best model hyper-parameters
    # best_rf = RandomForestClassifier(n_estimators=15, criterion='entropy',
    #                                  max_features='log2', max_depth=12,
    #                                  min_samples_leaf=2, min_samples_split=2,
    #                                  bootstrap=True, n_jobs=-1)
    #
    # scores = cross_val_score(best_rf, X, y, cv=3)
    # print('Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))
    # best_rf.fit(X, y)
    # print('feature importance:', best_rf.feature_importances_)
