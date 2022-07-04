from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import scikitplot as skplt
import pandas as pd

#Make our customer score

def classification_report_with_accuracy_score(y_true, y_pred, originalclass, predictedclass):
    originalclass.extend(y_true)
    predictedclass.extend(y_pred)
    return accuracy_score(y_true, y_pred)  # return accuracy score


def print_best_classifier(best_model):
    print('----------------------------------------------------------')
    print(best_model['classifier'])
    print('best validation macro f1 score: ', best_model['best score'])
    print('Best Parameter:', best_model['best params'])
    print('Models best Validation Report: \n', best_model['Average report'])



def save_roc_curve(pipe, gt_lables, y_probas):
    clf = str(pipe['classifier']['clf'])
    clf_name = clf.split('(')[0]
    clf_formated = " ".join([ele + ' \n' for ele in clf.split(',')])
    
    ax = skplt.metrics.plot_roc_curve(gt_lables, y_probas, figsize=(16, 12), 
                                 title=f"ROC Curves: {clf_name}")
    plt.legend(fontsize=10)
    ax.text(1.01, 0.85, clf_formated + format_params(pipe['best params']), fontsize=11)
    plt.tight_layout()
    plt.savefig(f"./images/genre_pediction/roc_{clf_name}.png", bbox_inches='tight')
    plt.close()

def save_confusion_matrix(pipe, gt_labels, predictions):
    # sns.set(font_scale=1.2)
    clf = str(pipe['classifier']['clf'])
    clf_name = clf.split('(')[0]
    clf_formated = " ".join([ele + ' \n' for ele in clf.split(',')])
    cm1 = confusion_matrix(gt_labels, predictions)
    plt.figure(figsize=(16, 12))
    ax = sns.heatmap(cm1, annot=True,  xticklabels=pipe['classifier'].classes_, 
                     yticklabels=pipe['classifier'].classes_ ,fmt='g')
    ax.set_title(f'Confusion Matrix {clf_name}')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    plt.text(24, 2, clf_formated + format_params(pipe['best params']), fontsize=11)
    plt.tight_layout()
    plt.savefig(f"./images/genre_pediction/cm_{clf_name}.png", bbox_inches='tight')
    plt.close()
    

def save_report(pipe):
    pd.options.display.float_format = "{:,.2f}".format
    clf = str(pipe['classifier']['clf'])
    clf_name = clf.split('(')[0]
    clf_formated = " ".join([ele + ' \n' for ele in clf.split(',')])
    df = pd.DataFrame(pipe['Report dict']).T
    df['support'] = df['support'].astype(int)
    ax = sns.heatmap(df, annot=True, fmt='.4g')
    ax.set_title(f'Scores: {clf_name}')
    plt.text(5, 2, clf_formated + format_params(pipe['best params']), fontsize=8)
    plt.savefig(f"./images/genre_pediction/report_{clf_name}.png", bbox_inches='tight')
    plt.close()

class OutlierCleaner(BaseEstimator, TransformerMixin):
    '''
    This Class performs all the transformation jobs on the numerical features.
    Outliners are getting cleaned.
    '''
    def __init__(self):
        return None

    def fit(self, X, y =None):
        return self

    def remove_outliers(self, X, range_cut_off=(0.2, 0.8)):
        for col in X.columns:
            Q1 = X[col].quantile(range_cut_off[0]) # Ideally it is (.25 - .75 but some outliers are wanted)
            Q3 = X[col].quantile(range_cut_off[1])
            IQR = Q3 - Q1
            # Adjusting outliers with their percentile values

            low = Q1 - 1.5 * IQR
            high = Q3 + 1.5 * IQR

            X[col] = np.where(X[col] < low, low, X[col])
            X[col] = np.where(X[col] > high, high, X[col])

        return X


    def transform(self, X, y = None):
       
        # Removing Outliers
        X = self.remove_outliers(X)
        
        return X
    
    
def format_params(parameters):
    text = []
    for param, value in parameters.items():
        text.append(f"{param}: {value} \n")
        
    return " ".join(text)
