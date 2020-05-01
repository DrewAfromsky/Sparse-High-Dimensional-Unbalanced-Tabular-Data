# -*- coding: utf-8 -*-
#!/usr/bin/env python
 
#################################
# author = Drew Afromsky        #
# email = dafromsky@gmail.com   #
#################################

import matplotlib.pylab as pyl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import make_scorer
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, r2_score, roc_auc_score, average_precision_score
from tensorflow import feature_column
from imblearn.pipeline import make_pipeline as make_imb_pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import matplotlib as mpl
import seaborn as sns
from tqdm import tqdm
import itertools
import glob
import re
import os


def stat_printer(y_test, y_pred):
    f1 = f1_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    precision = precision_score(y_test, y_pred, average='binary')
    accuracy = accuracy_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred)
    metrics = ['F1 Score', 'Recall', 'Precision', 'Accuracy', 'AUC ROC']
    scores = [f1, recall, precision, accuracy, auc_roc]
    stat_df = pd.DataFrame(list(zip(metrics, scores)), columns =['Metric', 'Score']) 
    display(stat_df)
    return None

def roc_maker(arch_str, y_pred, y_test):
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
    auc = metrics.roc_auc_score(y_test, y_pred)
    plt.plot(fpr,tpr,label="ROC, auc="+str(auc))
    plt.legend(loc=4)
    plt.title(arch_str + ' ROC')
    plt.xlabel('False Positive Rate')
    plt.ylabel('Tue Positive Rate')
    plt.show()
    return None

def plot_cm(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    print('True Negatives: ', cm[0][0])
    print('False Positives: ', cm[0][1])
    print('False Negatives: ', cm[1][0])
    print('True Positives: ', cm[1][1])
    print('Total: ', np.sum(cm[1]))

def preliminary_testing(X, y, i):    
    # Initialize lists to hold models and metrics
    premodels = []
    model_names = []
    f1_scores = []
    recalls = []
    precisions = []
    average_precisions = []
    accuracies = []
    auc = []
    
    # Initialize and add models to premodels list
#     premodels.append(('K-NearestNeighbors', KNeighborsClassifier()))
    premodels.append(('SVC', SVC()))
#     premodels.append(('LogisticRegression', LogisticRegression()))
    premodels.append(('DecisionTree', DecisionTreeClassifier()))
#     premodels.append(('GaussianNaiveBayes', GaussianNB()))
    premodels.append(('RandomForest', RandomForestClassifier()))
    premodels.append(('XGBoost', XGBClassifier()))
    premodels.append(('GradientBoostingClassifier', GradientBoostingClassifier()))
    premodels.append(('MLPClassifier', MLPClassifier()))
    
    # Iteratively call each model and append metrics to corresponding lists
    for name, premodel in premodels:
        premodel.fit(X_train, y_train)
        y_pred = premodel.predict(X_test)
        
        f1_scores.append(f1_score(y_test, y_pred, average='binary'))
        recalls.append(recall_score(y_test, y_pred, average='binary'))
        average_precisions.append(average_precision_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, average='binary'))
        auc.append(roc_auc_score(y_test, y_pred))
        accuracies.append(accuracy_score(y_test, y_pred))
        model_names.append(name)
        
    # Make dataframe of each model's performance and return it    
    preliminary_test = pd.DataFrame({'Model Name': model_names, 'F1-Score': f1_scores, 'Recall':recalls, 'Average Precision': average_precisions, 
                                     'Precision':precisions, 'Accuracy':accuracies, 'AUC-ROC':auc})
    return preliminary_test