import itertools
from os import replace
import pandas as pd
import numpy as np
from pandas.core.series import Series
from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
import argparse

def balance(dataframe, label, n=500):
    
    """
    Create a balanced sample from imbalanced datasets.
    
    dataframe: 
        Pandas dataframe with a column called 'ls
        text' and one called 'label'
    n:         
        Number of samples from each label, defaults to 500
    """
    # Use pandas select a random bunch of examples from each label
    import numpy as np
    np.random.seed(2022)
    if n>1500:
        out = (dataframe.groupby(label, as_index=False)
            .apply(lambda x: x.sample(n=n, replace=True))
            .reset_index(drop=True))
    else:    
        out = (dataframe.groupby(label, as_index=False)
                    .apply(lambda x: x.sample(n=n))
                    .reset_index(drop=True))
    
    return out

def prep_data(data, dependent_var, predictor_list):
    '''
    data = entire dataframe, pandas DataFrame
    dependent_var = name of dependent variable, str
    predictor_list = list of predictor columns, list of str variables
    returns =  X_train, X_test, y_train, y_test
    '''
    # Split X and y
    X = np.array(data[predictor_list])
    y = data[dependent_var]
    # train and test
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    random_state=22,
                                                    train_size=.7, 
                                                    test_size=.3)
    return X_train, X_test, y_train, y_test

def rf_search_params(X_train, y_train):
    """
    Create a grid search of values in random forest
    
    X_train & y_train: 
        train sets
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10, 20, 40, 80]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4, 5]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune

    rf = RandomForestClassifier(n_estimators=100, random_state=2022)
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, 
    n_iter = 200, 
    cv = 3, 
    verbose=2, 
    random_state=2022, 
    n_jobs = -1)
    # Fit the random search model
    rf_random.fit(X_train, y_train)

    return rf_random.best_params_, rf_random.best_estimator_, random_grid

def model_evaluation(data,dependent_var, clf, predictor_list, X_train, y_train, X_test, y_test, performance_sheet):
    from sklearn import metrics
    # Fit model
    clf.fit(X_train, y_train)
    # Predictions
    y_pred=clf.predict(X_test)
    # Accuracy
    acc = metrics.accuracy_score(y_test, y_pred)
    # Random classification
    import random
    random.seed(2022) 
    y_pred_random = data[dependent_var].sample(n=len(y_test))
    random_accuracy = metrics.accuracy_score(y_test, y_pred_random)
    print(f"Empirical classification returns an accuracy of {round(acc, ndigits = 3)} and random classification returns an accuracy of {round(random_accuracy, ndigits = 3)}")
    # Important features
    import pandas as pd
    feature_imp = pd.Series(clf.feature_importances_, 
                            index=predictor_list).sort_values(ascending=False)
    
    performance_sheet.append([dependent_var, predictor_list, round(acc, ndigits = 3), round(random_accuracy, ndigits = 3)])

    
    return acc, random_accuracy, feature_imp, performance_sheet


def main():
    ap = argparse.ArgumentParser(description = "[INFO] Random Forest model with grid search") # Defining an argument parse

    ap.add_argument("-d", 
                    "--data_folder",  # Argument 1
                    required=True, # Not required
                    type = str, # The input type should be a string
                    help = "str of data_folder") # Help function

    # Adding them together
    args = vars(ap.parse_args()) 
    # Data
    data = pd.read_csv(args['data_folder'])
    # Data
    performance_sheet = []

    #### GENDER ####
    # Balancing data
    data_balanced = balance(data, 'dominance', n=1500)
    data_balanced['privacy'] = data_balanced['privacy'].apply(lambda x : 0 if x == 'OPEN' else (1 if x == 'CLOSED' else 2))

    # Defining necessary variables
    dependent_var = 'dominance'
    predictor_list = ['dominant_topic', 'new_days', 'total_unique_P_C', 'privacy']
    # Prep data
    np.random.seed(2030)
    X_train, X_test, y_train, y_test = prep_data(data_balanced, dependent_var, predictor_list)
    # Grid search
    best_params, clf, __ = rf_search_params(X_train, y_train)
    # Performance evaluation 
    acc, random_accuracy, feature_imp, performance_sheet = model_evaluation(data_balanced, dependent_var, clf, predictor_list, X_train, y_train, X_test, y_test, performance_sheet)
    all_feature_importances = feature_imp.copy()
    all_feature_importances = Series.append(all_feature_importances, feature_imp)

    predictor_list = ['dominant_topic']
    # Prep data
    np.random.seed(2030)
    X_train, X_test, y_train, y_test = prep_data(data_balanced, dependent_var, predictor_list)
    # Grid search
    best_params, clf, __ = rf_search_params(X_train, y_train)
    # Performance evaluation 
    acc, random_accuracy, feature_imp, performance_sheet = model_evaluation(data_balanced,dependent_var, clf, predictor_list, X_train, y_train, X_test, y_test, performance_sheet)
    
    all_feature_importances = Series.append(all_feature_importances, feature_imp)


    fifty = pd.DataFrame({'dominant_topic':range(1, 51)})
    # pred_list = pd.DataFrame()
    # for model in range(1, 1001):
        # make the model
    #    rf = RandomForestClassifier()
    #    rf.set_params(**best_params) # Passing the optimal parameters
    #    rf.fit(X_train, y_train)
    #    probs = pd.DataFrame(rf.predict_proba(fifty))
    #    probs['topic_n'] = range(1, 51)
    #    pred_list = pred_list.append(probs)

    new_topic_numbers = pd.DataFrame({'old': range(1, 51),'new':[6,49,14,40,19,36,4,42,7,28,29,26,34,17,27,8,21,9,38,23,10,44,25,22,24,48,46,16,47,31,18,11,35,2,37,12,30,13,43,45,32,1,39,15,20,50,5,41,33,3]}) 

   # mean = pred_list.groupby('topic_n', as_index=False)[[0,1,2]].mean()
   # sds = pred_list.groupby('topic_n',as_index=False)[[0,1,2]].std()
   # mean.sort_values(1, ascending=False)
   # mean.to_csv('mean_probs_gender.csv')

    fifty = pd.DataFrame({'dominant_topic':range(1, 51)})
    probs = pd.DataFrame(clf.predict_proba(fifty))
    probs['topic_n'] = range(1, 51)
    # probs['topic_n']=(probs['topic_n']-1).map(new_topic_numbers['new'].astype(int))

    probs.sort_values(by=0, ascending=False)
    pd.DataFrame(probs).to_csv('prob_table_gender_topics.csv')

    #### GENDER 2 LEVELS ####
    data_balanced = balance(data, 'dominance', n=2000)
    data_balanced = data_balanced[data_balanced['dominance']>1]
    # Changing privacy to int type
    data_balanced['privacy'] = data_balanced['privacy'].apply(lambda x : 0 if x == 'OPEN' else (1 if x == 'CLOSED' else 2))

    # Defining necessary variables
    dependent_var = 'dominance'
    predictor_list = ['dominant_topic']
    np.random.seed(2030)
    # Prep data
    X_train, X_test, y_train, y_test = prep_data(data_balanced, dependent_var, predictor_list)
    # Grid search
    best_params, clf, __ = rf_search_params(X_train, y_train)
    # Performance evaluation 
    acc, random_accuracy, feature_imp, performance_sheet = model_evaluation(data_balanced, dependent_var, clf, predictor_list, X_train, y_train, X_test, y_test, performance_sheet)

    all_feature_importances = Series.append(all_feature_importances, feature_imp)



    #### PRIVACY ####
    # Balancing data
    data_balanced = balance(data, 'privacy', n=1500)
    # Changing privacy to int type
    data_balanced['privacy'] = data_balanced['privacy'].apply(lambda x : 0 if x == 'OPEN' else (1 if x == 'CLOSED' else 2))
    # Defining necessary variables
    dependent_var = 'privacy'
    predictor_list = ['dominant_topic', 'new_days', 'post_total', 'dominance']

    # Prep data
    X_train, X_test, y_train, y_test = prep_data(data_balanced, dependent_var, predictor_list)
    # Grid search
    best_params, clf, __ = rf_search_params(X_train, y_train)
    # Performance evaluation 
    acc, random_accuracy, feature_imp, performance_sheet = model_evaluation(data_balanced, dependent_var, clf, predictor_list, X_train, y_train, X_test, y_test, performance_sheet)
    all_feature_importances = Series.append(all_feature_importances, feature_imp)


    # Defining necessary variables
    dependent_var = 'privacy'
    predictor_list = ['dominant_topic']
    data_balanced = balance(data, 'privacy', n=1500)

    # Changing privacy to int type
    data_balanced['privacy'] = data_balanced['privacy'].apply(lambda x : 0 if x == 'OPEN' else (1 if x == 'CLOSED' else 2))
    np.random.seed(2030)
    X_train, X_test, y_train, y_test = prep_data(data_balanced, dependent_var, predictor_list)
    best_params, clf, __ = rf_search_params(X_train, y_train)
    acc, random_accuracy, feature_imp, performance_sheet = model_evaluation(data_balanced, dependent_var, clf, predictor_list, X_train, y_train, X_test, y_test, performance_sheet)
    all_feature_importances = Series.append(all_feature_importances, feature_imp)


    fifty = pd.DataFrame({'dominant_topic':range(1, 51)})
    #pred_list = pd.DataFrame()
    #for model in range(1, 1001):
    #    # make the model
    #    rf = RandomForestClassifier()
    #    rf.set_params(**best_params) # Passing the optimal parameters
    #    rf.fit(X_train, y_train)
    #    probs = pd.DataFrame(rf.predict_proba(fifty))
    #    probs['topic_n'] = range(1, 51)
    #    pred_list = pred_list.append(probs)

    #mean = pred_list.groupby('topic_n', as_index=False)[[0,1,2]].mean()
    #sds = pred_list.groupby('topic_n',as_index=False)[[0,1,2]].std()
    #mean.sort_values(1, ascending=False)
    #mean.to_csv('mean_probs_privacy.csv')


    probs = pd.DataFrame(clf.predict_proba(fifty))
    probs['topic_n'] = range(1, 51)
    new_topic_numbers = pd.DataFrame({'new':[6,49,14,40,19,36,4,42,7,28,29,26,34,17,27,8,21,9,38,23,10,44,25,22,24,48,46,16,47,31,18,11,35,2,37,12,30,13,43,45,32,1,39,15,20,50,5,41,33,3]}) 
    probs['topic_n']=(probs['topic_n']-1).map(new_topic_numbers['new'].astype(int))
    probs.to_csv('prob_table_privacy_topics.csv')

    performance_sheet =  pd.DataFrame(performance_sheet, columns=['dependent_var', 'independent_var', 'empirical_accuracy', 'random_accuracy'])
    performance_sheet.to_csv("performance_finalfeatures.csv")

    all_feature_importances.to_csv("feature_importances_finalfeatures.csv")


if __name__ == "__main__":
    main()