<<<<<<< HEAD
import time
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt


from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import xgboost as xgb

from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import normalize
from sklearn.utils.multiclass import unique_labels
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold


def roc_plot(Y_test, Y_pred, title=''):
    fpr, tpr, thresholds = metrics.roc_curve(Y_test, Y_pred)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % metrics.auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


def results(clf, clf_name, X_test, Y_test):
    Y_pred = clf.predict(X_test)
    conf_matrix = confusion_matrix(Y_test, Y_pred)
    print(' - ', conf_matrix)
    print(' - ', clf_name, ' : TP : ', conf_matrix[1][1], ' || FP : ', conf_matrix[0][1])


def classifier_train(clf, X_train, Y_train, X_test, Y_test):
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    conf_matrix = confusion_matrix(Y_test, Y_pred)
    Y_pred_probab = clf.predict_proba(X_test)
    print(' - Conf Matrix : ')
    print(conf_matrix)
    print(' - F1 score    : ', round(metrics.f1_score(Y_test, Y_pred, pos_label=1), 3))
    print(' - Precision   : ', round(metrics.precision_score(Y_test, Y_pred, pos_label=1), 3))
    print(' - Recall      : ', round(metrics.recall_score(Y_test, Y_pred, pos_label=1), 3))
    return clf, conf_matrix, Y_pred_probab, Y_pred


def experiments(X, Y, cv, smote, args):
    classifiers = []
    conf_matrixes = []
    Y_tests = []
    Y_tests_preds = []
    Y_tests_preds_probabs = []

    roc_title = ''
    if smote == 0:
        roc_title = 'ROC - unSMOTEd - '
    elif smote == 1:
        roc_title = 'ROC - SMOTEd (%.1f)' % (args['sampling_ratio'])

    n_splits = cv

    if cv == 1:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=args['random_state'])
        n_splits = 2
        print(' - Solving Q2')
    else:
        print(' - Solving Q3')

    for i, (train, test) in enumerate(
            StratifiedKFold(n_splits=n_splits, random_state=args['random_state']).split(X, Y)):
        print('')
        print(' -------------------- Fold : ', i, ' --------------------')
        if (cv != 1):
            X_train = X[train]
            Y_train = Y[train]
            X_test = X[test]
            Y_test = Y[test]

        print(' - Y_train : ', Counter(Y_train))
        print(' - Y_test  : ', Counter(Y_test))

        if (smote == 1):
            sm = SMOTE(sampling_strategy=args['sampling_ratio'], k_neighbors=5
                       , random_state=args['random_state'])  # 'sampling_strategy': 'auto',
            X_train, Y_train = sm.fit_resample(X_train, Y_train)
            print(' - SMOTEd Y : ', Counter(Y_train))
            print ('')

        if (args['LogisticRegression']['bool']):
            print(' - LogisticRegression')
            clf_args = args['LogisticRegression']
            clf = LogisticRegression(C=clf_args['C'], max_iter=clf_args['max_iter'], solver='lbfgs', random_state=42)
            if i == 0:
                roc_title += ' Logistic Regression'

        if (args['RandomForestClassifier']['bool']):
            print(' - RandomForestClassifier')
            clf_args = args['RandomForestClassifier']
            clf = RandomForestClassifier(n_estimators=clf_args['n_estimators'], n_jobs=3, random_state=42)
            if i == 0:
                roc_title += ' Random Forest'

        if (args['SVC']['bool']):
            clf = classifier_SVC(X_orig_smote, Y_orig_smote, X_orig_test, Y_orig_test, 'SVC')
            classifiers.append(('svc', clf))

        if args['AdaBoostClassifier']['bool']:
            print(' - AdaBoostClassifier')
            clf_args = args['AdaBoostClassifier']
            print(' - Clf Args : ', clf_args)
            clf = AdaBoostClassifier(n_estimators=clf_args['n_estimators'], learning_rate=clf_args['learning_rate'],
                                     random_state=42)
            if i == 0:
                roc_title += ' AdaBoost'

        if args['XGBClassifier']['bool']:
            print(' - XGBClassifier')
            clf = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
            classifiers.append(('xgb', clf))
            if i == 0:
                roc_title += ' XGB'

        if (args['ensemble']['bool']):
            print('')
            print(' - Ensembling')
            if (1):
                clf_xg = xgb.XGBClassifier(objective="binary:logistic", subsample=0.5, random_state=42)
                clf_log = LogisticRegression(C=100, max_iter=500, random_state=42, solver='lbfgs')
                clf_ada = AdaBoostClassifier(n_estimators=250, learning_rate=1, random_state=42)

                classifiers = [('clg_xg', clf_xg), ('clf_log', clf_log)('clf_ada', clf_ada)]

            clf = VotingClassifier(classifiers, voting='soft', n_jobs=3)  # voting='hard'
            if i == 0:
                roc_title += ' Ensemble'

        # TRAIN
        clf, conf_matrix, Y_pred_probab, Y_pred = classifier_train(clf, X_train, Y_train, X_test, Y_test)
        classifiers.append(clf)
        conf_matrixes.append(conf_matrix)
        Y_tests.append(Y_test)
        Y_tests_preds.append(Y_pred)
        Y_tests_preds_probabs.append(Y_pred_probab)

        if (cv == 1):
            break

    print ('')
    print (' -------------------------------------------------------------- ')
    # Confusion Matrices
    conf_matrix_final = []
    for i, each in enumerate(conf_matrixes):
        if i == 0:
            conf_matrix_final = each.copy()
        else:
            conf_matrix_final += each.copy()
    print (' - Final Conf Matrix : ')
    print (conf_matrix_final)

    # ROC-CURVEs
    Y_tests_final = []
    for i, each in enumerate(Y_tests):
        Y_tests_final.extend(each.tolist())
    Y_tests_preds_final = []
    for i, each in enumerate(Y_tests_preds):
        Y_tests_preds_final.extend(each.tolist())
    Y_tests_preds_probabs_final = []
    for i, each in enumerate(Y_tests_preds_probabs):
        Y_tests_preds_probabs_final.extend(each[:, 1].tolist())

    # Final Metrics
    print(' - F1 score    : ', round(metrics.f1_score(Y_tests_final, Y_tests_preds_final, pos_label=1), 3))
    print(' - Precision   : ', round(metrics.precision_score(Y_tests_final, Y_tests_preds_final, pos_label=1), 3))
    print(' - Recall      : ', round(metrics.recall_score(Y_tests_final, Y_tests_preds_final, pos_label=1), 3))
    roc_plot(Y_tests_final, Y_tests_preds_probabs_final, roc_title)



    return classifiers, conf_matrixes, Y_tests_final, Y_tests_preds_final, Y_tests_preds_probabs_final




=======
import time
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt


from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import xgboost as xgb

from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import normalize
from sklearn.utils.multiclass import unique_labels
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold


def roc_plot(Y_test, Y_pred, title=''):
    fpr, tpr, thresholds = metrics.roc_curve(Y_test, Y_pred)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % metrics.auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


def results(clf, clf_name, X_test, Y_test):
    Y_pred = clf.predict(X_test)
    conf_matrix = confusion_matrix(Y_test, Y_pred)
    print(' - ', conf_matrix)
    print(' - ', clf_name, ' : TP : ', conf_matrix[1][1], ' || FP : ', conf_matrix[0][1])


def classifier_train(clf, X_train, Y_train, X_test, Y_test):
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    conf_matrix = confusion_matrix(Y_test, Y_pred)
    Y_pred_probab = clf.predict_proba(X_test)
    print(' - Conf Matrix : ')
    print(conf_matrix)
    print(' - F1 score    : ', round(metrics.f1_score(Y_test, Y_pred, pos_label=1), 3))
    print(' - Precision   : ', round(metrics.precision_score(Y_test, Y_pred, pos_label=1), 3))
    print(' - Recall      : ', round(metrics.recall_score(Y_test, Y_pred, pos_label=1), 3))
    return clf, conf_matrix, Y_pred_probab, Y_pred


def experiments(X, Y, cv, smote, args):
    classifiers = []
    conf_matrixes = []
    Y_tests = []
    Y_tests_preds = []
    Y_tests_preds_probabs = []

    roc_title = ''
    if smote == 0:
        roc_title = 'ROC - unSMOTEd - '
    elif smote == 1:
        roc_title = 'ROC - SMOTEd (%.1f)' % (args['sampling_ratio'])

    n_splits = cv

    if cv == 1:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=args['random_state'])
        n_splits = 2
        print(' - Solving Q2')
    else:
        print(' - Solving Q3')

    for i, (train, test) in enumerate(
            StratifiedKFold(n_splits=n_splits, random_state=args['random_state']).split(X, Y)):
        print('')
        print(' -------------------- Fold : ', i, ' --------------------')
        if (cv != 1):
            X_train = X[train]
            Y_train = Y[train]
            X_test = X[test]
            Y_test = Y[test]

        print(' - Y_train : ', Counter(Y_train))
        print(' - Y_test  : ', Counter(Y_test))

        if (smote == 1):
            sm = SMOTE(sampling_strategy=args['sampling_ratio'], k_neighbors=5
                       , random_state=args['random_state'])  # 'sampling_strategy': 'auto',
            X_train, Y_train = sm.fit_resample(X_train, Y_train)
            print(' - SMOTEd Y : ', Counter(Y_train))
            print ('')

        if (args['LogisticRegression']['bool']):
            print(' - LogisticRegression')
            clf_args = args['LogisticRegression']
            clf = LogisticRegression(C=clf_args['C'], max_iter=clf_args['max_iter'], solver='lbfgs', random_state=42)
            if i == 0:
                roc_title += ' Logistic Regression'

        if (args['RandomForestClassifier']['bool']):
            print(' - RandomForestClassifier')
            clf_args = args['RandomForestClassifier']
            clf = RandomForestClassifier(n_estimators=clf_args['n_estimators'], n_jobs=3, random_state=42)
            if i == 0:
                roc_title += ' Random Forest'

        if (args['SVC']['bool']):
            clf = classifier_SVC(X_orig_smote, Y_orig_smote, X_orig_test, Y_orig_test, 'SVC')
            classifiers.append(('svc', clf))

        if args['AdaBoostClassifier']['bool']:
            print(' - AdaBoostClassifier')
            clf_args = args['AdaBoostClassifier']
            print(' - Clf Args : ', clf_args)
            clf = AdaBoostClassifier(n_estimators=clf_args['n_estimators'], learning_rate=clf_args['learning_rate'],
                                     random_state=42)
            if i == 0:
                roc_title += ' AdaBoost'

        if args['XGBClassifier']['bool']:
            print(' - XGBClassifier')
            clf = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
            classifiers.append(('xgb', clf))
            if i == 0:
                roc_title += ' XGB'

        if (args['ensemble']['bool']):
            print('')
            print(' - Ensembling')
            if (1):
                clf_xg = xgb.XGBClassifier(objective="binary:logistic", subsample=0.5, random_state=42)
                clf_log = LogisticRegression(C=100, max_iter=500, random_state=42, solver='lbfgs')
                clf_randfor = RandomForestClassifier(n_estimators=250, random_state=42)
                clf_ada = AdaBoostClassifier(n_estimators=250, learning_rate=1, random_state=42)

                classifiers = [('clg_xg', clf_xg), ('clf_log', clf_log)
                    , ('clf_randfor', clf_randfor), ('clf_ada', clf_ada),
                               ]

            clf = VotingClassifier(classifiers, voting='soft', n_jobs=3)  # voting='hard'
            if i == 0:
                roc_title += ' Ensemble'

        # TRAIN
        clf, conf_matrix, Y_pred_probab, Y_pred = classifier_train(clf, X_train, Y_train, X_test, Y_test)
        classifiers.append(clf)
        conf_matrixes.append(conf_matrix)
        Y_tests.append(Y_test)
        Y_tests_preds.append(Y_pred)
        Y_tests_preds_probabs.append(Y_pred_probab)

        if (cv == 1):
            break

    print ('')
    print (' -------------------------------------------------------------- ')
    # Confusion Matrices
    conf_matrix_final = []
    for i, each in enumerate(conf_matrixes):
        if i == 0:
            conf_matrix_final = each.copy()
        else:
            conf_matrix_final += each.copy()
    print (' - Final Conf Matrix : ')
    print (conf_matrix_final)

    # ROC-CURVEs
    Y_tests_final = []
    for i, each in enumerate(Y_tests):
        Y_tests_final.extend(each.tolist())
    Y_tests_preds_final = []
    for i, each in enumerate(Y_tests_preds):
        Y_tests_preds_final.extend(each.tolist())
    Y_tests_preds_probabs_final = []
    for i, each in enumerate(Y_tests_preds_probabs):
        Y_tests_preds_probabs_final.extend(each[:, 1].tolist())

    # Final Metrics
    print(' - F1 score    : ', round(metrics.f1_score(Y_tests_final, Y_tests_preds_final, pos_label=1), 3))
    print(' - Precision   : ', round(metrics.precision_score(Y_tests_final, Y_tests_preds_final, pos_label=1), 3))
    print(' - Recall      : ', round(metrics.recall_score(Y_tests_final, Y_tests_preds_final, pos_label=1), 3))
    roc_plot(Y_tests_final, Y_tests_preds_probabs_final, roc_title)



    return classifiers, conf_matrixes, Y_tests_final, Y_tests_preds_final, Y_tests_preds_probabs_final




>>>>>>> c245a7c9ce67639dee047a7c31592bf1700026af
