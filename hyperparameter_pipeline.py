from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from operator import itemgetter
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, make_scorer
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from prince import PCA, MCA

from sklearn.decomposition import NMF

import pandas as pd
import numpy as np
import joblib

from pipeline_helper import (OutlierCleaner, print_best_classifier, 
                             save_confusion_matrix, save_roc_curve,
                             save_report)

# Parameters:
drop_columns = ['title',
                # 'artist',
                #  'album',
                'release_date',
                'sub_genre',  # First try without
                'genre']

# First do a random search for time reasons and then grid search with tuned parameters
grid_search = False  # If False, performs random search
grid_search_scoring = 'f1_macro' # Best for imbalanced datasets
cv_splits = 3  # 5 would be better
n_jobs = -1

train_ratio = 0.70
test_ratio = 0.15
validation_ratio = 0.15

save_models = True  # Save the models and test data
save_test_data = False

categorical_features = [# 'title',
                        'artist',
                        'release_type',
                        # 'sub_genre', 
                        'album',
                        ]

numeric_features = ['popularity', 'artist_followers', 'danceability', 'energy',
                    'loudness', 'speechiness', 'acoustics', 'instrumentalness',
                    'liveness', 'valence', 'tempo', 'duration_min']

#############################################################################
# Modeling and Tuning 
set_config(display="diagram")

# Load Data
df = pd.read_pickle('./dataset_/dataset_genre.pkl')

y = df['genre']
# Manipulate features
df.drop(columns=drop_columns, inplace=True, errors='ignore')
df.reset_index(drop=True, inplace=True)

# train is now 75% of the entire data set
# the _junk suffix means that we drop that variable completely
x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=1 - train_ratio, 
                                                    stratify=y, random_state=42, shuffle=True)

x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio),
                                                stratify=y_test, random_state=42, shuffle=True)


# Setup Preprocessing
numeric_transformer = 'passthrough'
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
num_red_transformer = 'passthrough'
cat_red_transformer = 'passthrough'
cleaner = 'passthrough'


preprocessor = ColumnTransformer(
    transformers=[
        ('scaler', numeric_transformer, numeric_features),
        ('num_outliers', cleaner, numeric_features),
        ('num_red', num_red_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
        # ('cat_red', cat_red_transformer, categorical_features)
        ], remainder='drop'
)

inner_cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

#pipeline parameters
parameters = \
    [ \
        # {
        #     'clf': [MultinomialNB()],
        #     'clf__alpha': [0.001, 0.1, 1, 10, 100],
        #     'preproc__scaler': [MinMaxScaler()]
        # },

        # { # Takes to much time on my side
        #     'clf': [SVC()],
        #     'clf__C': [0.001, 0.1, 1, 10, 100, 10e5],
        #     'clf__kernel': ['linear', 'rbf'],
        #     'clf__class_weight': ['balanced'],
        #     'clf__probability': [True]
        # },

        # {
        #     'clf': [DecisionTreeClassifier()],
        #     'clf__criterion': ['gini','entropy'],
        #     'clf__max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
        #     'clf__splitter': ['best','random'],
        #     'clf__class_weight':['balanced', None],
        #     'preproc__scaler': [MinMaxScaler(), StandardScaler(), RobustScaler(), 'passthrough'],
        #     'preproc__num_red': [PCA(n_components=4), PCA(n_components=8), PCA(n_components=12), 'passthrough'],
        #     'preproc__num_outliers': [OutlierCleaner(), 'passthrough']
        #     # 'preproc__cat_red': [MCA(n_components=4), MCA(n_components=8), MCA(n_components=12), 'passthrough']
        #     # 'preproc__num_red__n_components': list(range(4, len(numeric_features) + 2, 2))
        # }, 
        
        # {
        #     'clf': [RandomForestClassifier()],
        #     'clf__criterion': ['gini','entropy'],
        #     'clf__bootstrap': [True, False],
        #     'clf__max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
        #     'clf__max_features': ['auto', 'sqrt'],
        #     'clf__min_samples_leaf': [1, 2, 4],
        #     'clf__min_samples_split': [2, 5, 10],
        #     'clf__n_estimators': [100, 400, 600, 800, 1000, 1200],
        #     'preproc__scaler': [MinMaxScaler(), StandardScaler(), RobustScaler(), 'passthrough'], 
        #     'preproc__num_red': [PCA(n_components=4), PCA(n_components=8), PCA(n_components=12), 'passthrough'],
        #     'preproc__num_outliers': [OutlierCleaner(), 'passthrough']

        # }, 
        
        # {
        #     'clf': [KNeighborsClassifier()],
        #     'clf__weights': ['uniform', 'distance'],
        #     'clf__n_neighbors': range(1, 31), 
        #     'clf__leaf_size': range(10, 60), 
        #     'clf__p': [1, 2, 3], 
        #     'clf__metric': ['manhattan', 'mahalanobis', 'euclidean', 'cosine'],
        #     'preproc__scaler': [MinMaxScaler(), StandardScaler(), RobustScaler(), 'passthrough'],
        #     'preproc__num_red': [PCA(n_components=4), PCA(n_components=8), PCA(n_components=12), 'passthrough'],
        #     'preproc__num_outliers': [OutlierCleaner(), 'passthrough']

        # }, 
        
        {
            'clf': [LogisticRegression()],
            'clf__class_weight': ['balanced', None],
            'clf__max_iter': [500],
            'clf__multi_class': ['multinomial']
            # # 'clf__C': np.logspace(-4, 4, 20),
            # 'preproc__scaler': [MinMaxScaler(), StandardScaler(), RobustScaler(), 'passthrough'],
            # 'preproc__num_red': [PCA(n_components=4), PCA(n_components=8), PCA(n_components=12), 'passthrough'],
            # 'preproc__num_outliers': [OutlierCleaner(), 'passthrough']

        }
    ]

# evaluating multiple classifiers
# based on pipeline parameters
#-------------------------------
result = []

# Variables for average classification report
originalclass = []
predictedclass = []
i = 42

def classification_report_with_accuracy_score(y_true, y_pred):
    originalclass.extend(y_true)
    predictedclass.extend(y_pred)
    return accuracy_score(y_true, y_pred)   # return accuracy score


for params in parameters:

    # classifier
    clf = params['clf'][0]

    # getting arguments by
    # popping out classifier
    params.pop('clf')

    #pipeline
    steps = [('preproc', preprocessor),
             ('clf', clf)]

    # cross validation using
    # Grid Search or Random Search
    if grid_search:
        grid = GridSearchCV(Pipeline(steps), param_grid=params, cv=inner_cv, verbose=2, 
                            n_jobs=n_jobs, scoring=grid_search_scoring)
    else:
        grid = RandomizedSearchCV(Pipeline(steps), param_distributions=params, cv=inner_cv, verbose=2, 
                    n_jobs=n_jobs, scoring=grid_search_scoring)
    grid.fit(x_train, y_train)
    
    report = classification_report(y_val, grid.best_estimator_.predict(x_val), digits=3)
    report_dict = classification_report(y_val, grid.best_estimator_.predict(x_val), digits=3, output_dict=True)

    print(f'Current best Estimator: {grid.best_estimator_._final_estimator} \n', report)
    # storing result
    result.append \
    (
        {
            'grid': grid,
            'classifier': grid.best_estimator_,
            'best score': grid.best_score_,
            'best params': grid.best_params_,
            'cv': grid.cv,
            'Average report': report,
            'Report dict': report_dict
        }
    )


# Sorting result by best score
result = sorted(result, key=itemgetter('best score'), reverse=True)
best_model = result[0]
print_best_classifier(best_model)
print('Finished')

# Visualizations
for pipe in result:
    save_confusion_matrix(pipe, y_val, pipe['classifier'].predict(x_val))
    y_probas = pipe['classifier'].predict_proba(x_val)
    save_roc_curve(pipe, y_val, y_probas)
    save_report(pipe)

# # Saving best classifier
# if save_models: joblib.dump((best_model, result), './models/best_classifiers.pkl')
# if save_test_data: joblib.dump((x_test, y_test), './dataset_/test_data_genre.pkl')
