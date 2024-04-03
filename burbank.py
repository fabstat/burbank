"""Burbank - Classification of Agronomic Trials Data

This file contains methods to analyze and classify 
preprocessed data of Russet potato trials.

"""

# basic data packages
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# data processing
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# models
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

# model selection
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SequentialFeatureSelector

# metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

# inspection
from sklearn.inspection import permutation_importance

# stats
from scipy import stats
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform


def feature_correlation_plot(X, output_path):
    """Plots a dendrogram and a correlation figure of a training dataframe
    
    Parameters
    ----------
    X: pandas.Dataframe
        The training dataset
    output_path: str
        The path to store the plots
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    corr = spearmanr(X).correlation

    # Ensure the correlation matrix is symmetric
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)

    # We convert the correlation matrix to a distance matrix before performing
    # hierarchical clustering using Ward's linkage.
    distance_matrix = 1 - np.abs(corr)
    dist_linkage = hierarchy.ward(squareform(distance_matrix))
    dendro = hierarchy.dendrogram(
        dist_linkage, labels=X.columns, ax=ax1, leaf_rotation=90
    )
    dendro_idx = np.arange(0, len(dendro["ivl"]))

    ax2.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]])
    ax2.set_xticks(dendro_idx)
    ax2.set_yticks(dendro_idx)
    ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
    ax2.set_yticklabels(dendro["ivl"])
    fig.tight_layout()
    plt.savefig(os.path.join(output_path, "correlation_plot.png"))
    plt.close()
    
    
def print_classification_reports(model, X_test, y_test, output_path):
    """Prints to screen and saves plots reporting on the classification results
    
    Parameters
    ----------
    X_test: pandas.Dataframe
        The test dataset
    y_test: pandas.Series
        The test results, aka the truth
    output_path: str
        The path to store the plots
    """
    
    print("Accuracy score: " + str(model.score(X_test, y_test)))
    print()
    y_true, y_pred = y_test, model.predict(X_test)

    # Show the ROC score
    print(f"Area Under the Receiver Operating Characteristic Curve: {roc_auc_score(y_true, y_pred):.3f}")
    print()
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % auc(fpr, tpr),
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_path, "roc_curve.png"))
    plt.close()

    # Show the classification report
    print(classification_report(y_true, y_pred))
    print()
    m = matthews_corrcoef(y_true, y_pred)
    p_val, l, h = cc_ci(m, y_pred.size)
    print(f"MCC: {m:.3f}, CI: ({l:.3f}, {h:.3f}), P-val: {p_val:.16f}")
    print()
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test,
                                          display_labels=['Drop', 'Keep'],
                                          cmap=plt.cm.Blues)
    plt.savefig(os.path.join(output_path, "confusion_matrix_display.png"))
    plt.close()
    print()

    # Get feature importance for best classifier (takes a long time to run!)
    # r = permutation_importance(best_pipe, X_test, y_test, n_repeats=10, random_state=0)
    # important = list()
    # for i in r.importances_mean.argsort()[::-1]:
    #     if r.importances_mean[i] - 1.96 * r.importances_std[i] > 0:
    #         important.append(i)
    #         print(f"{X.columns[i]:<8} "
    #           f"{r.importances_mean[i]:.3f}"
    #           f" +/- {r.importances_std[i]:.3f}")
    # print()
    
    
    
    
def cc_ci(r, n, alpha=0.5):
    r_z = np.arctanh(r)
    se = 1/np.sqrt(n-3)
    z = stats.norm.ppf(1-alpha/2)
    lo_z, hi_z = r_z-z*se, r_z+z*se
    lo, hi = np.tanh((lo_z, hi_z))
    t = r*np.sqrt(n-2)/np.sqrt(1-r**2)
    p_val = (1 - stats.t.cdf(x=t, df=n-1)) * 2
    return p_val, lo, hi



def learner(df, output_path, new_data=None, region='all', max_na_col=1129, impute='n', model='svc'):
    df = df.set_index(["Clone", "Year"])
    # define categorical variables
    df[["Trial Region"]] = df[["Trial Region"]].astype("category")
    # dropping colums with over max_na_col NaNs
    sub_df = df.dropna(axis="columns", thresh=df.shape[0]-max_na_col)
    
    X = sub_df.drop('Keep', axis=1)
    X = X[(X['true_keeps'] == 0) | (X['true_keeps'] == 1)]
    if region == 'ONT':
        X = X[X["Trial Region"] == 'ONT'].drop(["Trial Region"], axis=1)
    elif region == 'KF':
        X = X[X["Trial Region"] == 'KF'].drop(["Trial Region"], axis=1)
    elif region == 'HER_Early':
        X = X[X["Trial Region"] == 'HER_Early'].drop(["Trial Region"], axis=1)
    elif region == 'HER_Late':
        X = X[X["Trial Region"] == 'HER_Late'].drop(["Trial Region"], axis=1)
    elif region == 'HER':
        X = X[X["Trial Region"] == 'HER'].drop(["Trial Region"], axis=1)
    elif region == 'COR':
        X = X[X["Trial Region"] == 'COR'].drop(["Trial Region"], axis=1)
    #else:
        #X = X.drop(["Trial Region"], axis=1)
        
    # Feature correlation
    if region == 'all' and spearmanr(X).correlation is not np.nan:
        DM = X.drop(["true_keeps"], axis=1).dropna()
        feature_correlation_plot(DM, output_path)
            
    # scoring function
    mcc = make_scorer(matthews_corrcoef) 
    
    #imputer
    imp = IterativeImputer(n_nearest_features=3, sample_posterior=True, random_state=0)   
   
    # Classification models parameters' grids
    if model == 'hgbc':
         # HBGB model and param search grid
        clf = HistGradientBoostingClassifier()
        param = {'max_iter':[200], 
                 'max_depth': [2,3,4,5], 
                 'l2_regularization': np.linspace(21, 30, num=10),
                 'class_weight': [{0:1, 1:2}, {0:1, 1:3}, {0:1, 1:3.5}, {0:1, 1:4},
                                  {0:1, 1:4.5}, {0:1, 1:10}, {0:1, 1:12}],
                 'random_state':[0]}
    elif model=='mlpc':
        # MLP model
        clf = MLPClassifier()
        param = [{'solver': ['lbfgs'], 'alpha': 10.0 ** -np.arange(1, 5), 
                  'hidden_layer_sizes': np.arange(10, 20),
                  'max_iter':[10000], 'random_state':[0]}]
    else:
        # SVC model
        clf = SVC(probability=True, random_state=42)
        param = [{'kernel': ['rbf'], 'gamma': np.linspace(0.001, 0.01, num=10),
                  'C': np.linspace(5, 15, num=5), 'class_weight': [{0:1, 1:2}, {0:1, 1:3},
                                                                   {0:1, 1:3.5}, {0:1, 1:4},
                                                                   {0:1, 1:4.5}, {0:1, 1:10},
                                                                   {0:1, 1:12}]}]
     

    # To impute or not to impute?    
    if impute == 'n':
        X = X.dropna() # must drop NA's if no impute
        y = X["true_keeps"] # response
        X = X.drop(["true_keeps"], axis=1)
        
        # if new data is not provided, make a train/test dataset
        if new_data == None:
            # uncontaminated split 70/30
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=0)
        else:
            X_train = X
            y_train = y
        
        # Normalize numeric data and encode categorical data
        cat_feats = X_train.select_dtypes(include="category").columns
        num_feats = X_train.select_dtypes(include="number").columns
        preprocessor = ColumnTransformer(
            [
                ("num", StandardScaler(), num_feats),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse=False),
                    cat_feats,
                ),
            ],
        )
         
        # Train the classifier
        num_ex = min(len([y for y in y_train if y == 0]), len([y for y in y_train if y == 1]))
        n_splits = 5 if num_ex > 5 else num_ex
        clf_search = GridSearchCV(estimator=clf, param_grid=param, scoring=mcc, refit=True, cv=n_splits)
        pipe = make_pipeline(preprocessor, clf_search)
        pipe.fit(X_train, y_train)
        best_parameters = clf_search.best_estimator_.get_params()
        best_score = clf_search.best_score_
        
        # Uncomment to print the top 5 parameter sets:
        #print(pd.DataFrame.from_dict(clf_search.cv_results_).sort_values(by=['rank_test_score']).head())
        clf_best = clf.set_params(**best_parameters)
        sfs = SequentialFeatureSelector(clf_best, 
                                    n_features_to_select=0.88,
                                    scoring=mcc, cv=n_splits)
        best_pipe = make_pipeline(preprocessor, sfs, clf_best)
        best_pipe.fit(X_train, y_train)
        #print(best_pipe[:-1].get_feature_names_out())
       
        
    else: # impute
        y = X["true_keeps"] # response
        X = X.drop(["true_keeps"], axis=1)
        
        # if new data is not provided, make a train/test dataset
        if new_data == None:
            # uncontaminated split 70/30
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=0)
        else:
            X_train = X
            y_train = y
        
        # Normalize numeric data and encode categorical data
        cat_feats = X_train.select_dtypes(include="category").columns
        num_feats = X_train.select_dtypes(include="number").columns
        preprocessor = ColumnTransformer(
            [
                ("num", StandardScaler(), num_feats),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse=False),
                    cat_feats,
                ),
            ],
        )
        
        
        # Train the classifier with imputing
        num_ex = min(len([y for y in y_train if y == 0]), len([y for y in y_train if y == 1]))
        n_splits = 5 if num_ex > 5 else num_ex
        clf_search = GridSearchCV(estimator=clf, param_grid=param, scoring=mcc, refit=True, cv=n_splits)
        pipe = make_pipeline(preprocessor, imp, clf_search)
        pipe.fit(X_train, y_train)
        best_parameters = clf_search.best_estimator_.get_params()
        best_score = clf_search.best_score_
        
        # Uncomment to print the top 5 parameter sets:
        #print(pd.DataFrame.from_dict(clf_search.cv_results_).sort_values(by=['rank_test_score']).head())
        clf_best = clf.set_params(**best_parameters)
        sfs = SequentialFeatureSelector(clf_best, 
                                    n_features_to_select=0.72,
                                    scoring=mcc, cv=n_splits)
        best_pipe = make_pipeline(preprocessor, imp, sfs, clf_best) 
        best_pipe.fit(X_train, y_train)
        #print(best_pipe[:-1].get_feature_names_out())


    # Test the classifier and get the prediction
    if new_data != None:
        # dropping colums with over max_na_col NaNs
        new_data = new_data.set_index(["Clone", "Year"])
        # define categorical variables
        new_data[["Trial Region"]] = new_data[["Trial Region"]].astype("category")
        # Here X_test is data to predict and y_test is the prediction
        X_test = new_data[list(X_train.columns)]
        y_test = best_pipe.predict(X_test)
    
    
    # Print classification reports and scores
    print_classification_reports(best_pipe, X_test, y_test, output_path)
    
    # Predicted y values from test data or new data
    y_pred = best_pipe.predict(X_test)
    
    # If SVC, get probabilities
    # Warning: likely to disagree with predictions on small datasets
    if model == 'svc':
        y_probs = best_pipe.predict_proba(X_test) 
    

    # Make results dataframe, save it and return it
    clones = list([i[0] for i in X_test.index])
    res = {}
    if model == 'svc':
        res = pd.DataFrame({"Clone": clones, "Trial Region": X_test['Trial Region'].to_list(), "pred": y_pred, "prob of 0": y_probs.T[0], "prob of 1": y_probs.T[1]})
    else: 
        res = pd.DataFrame({"Clone": clones, "Trial Region": X_test['Trial Region'].to_list(), "pred": y_pred})
    model_name = model + impute
    res.to_csv(os.path.join(output_path, f"{model_name}.csv"))
    print(f"Results dataframe saved to {output_path} as {model_name}.csv")
    
    return res