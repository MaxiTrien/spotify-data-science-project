from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from operator import itemgetter
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, make_scorer, f1_score
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from prince import PCA, MCA


import pandas as pd
import numpy as np
import joblib

from pipeline_helper import (OutlierCleaner, print_best_classifier, 
                             save_confusion_matrix, save_pr_curve,
                             save_report)

# Parameters:
drop_columns = ['title',
                'track_number',
                'release_date',
                'tracks_in_album',
                # 'days_since_release',
                # 'release_year',
                # 'release_month',
                # 'release_day',
                'explicit_false', 
                'explicit_true', 
                'sub_genre', # First try without
                'genre'
                ]

# First do a random search for time reasons and then grid search with tuned parameters
grid_search = False  # If False, performs random search
random_picks = 10  # Random iterations of combinations from random gridsearch
grid_search_scoring = 'accuracy' # Best for imbalanced datasets
cv_splits = 5  # 5 would be better
n_jobs = -1
random_state = 42

train_ratio = 0.60
test_ratio = 0.20
validation_ratio = 0.20

save_models = True  # Save the models and test data
save_test_data = True

categorical_features = [
                        'artist', 
                        'album',
                        'release_type', 
                        'explicit', # Already binary
                        'key',
                        'mode',  # Already binary
                        'time_signature',
                        'days_since_release', 
                        'release_year',
                        'release_month',
                        'release_day',
                        'released_after_2017',
                        # 'sub_genre'
                        ]

numeric_features = ['popularity', 
                    'artist_followers', 
                    'danceability', 
                    'energy',
                    'loudness', 
                    'speechiness', 
                    'acoustics', 
                    'instrumentalness',
                    'liveness', 
                    'valence', 
                    'tempo', 
                    'duration_min'
                    ]

# All other columns will be dropped automatically

#############################################################################
# Modeling and Tuning 
set_config(display="diagram")

# Load Data
df = pd.read_pickle('./dataset_/three_genres.pkl')

y = df['genre']
# Manipulate features
df.drop(columns=drop_columns, inplace=True, errors='ignore')
df.reset_index(drop=True, inplace=True)

# train is now 75% of the entire data set
# the _junk suffix means that we drop that variable completely
x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=1 - train_ratio, 
                                                    stratify=y, random_state=random_state, shuffle=True)

x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio),
                                                stratify=y_test, random_state=random_state, shuffle=True)


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
        ], remainder='drop'
)

cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

parameters = \
    [ \

        {
        
            'clf': [DecisionTreeClassifier()],
            'clf__criterion': ['gini'],
            'clf__class_weight': ['balanced'],
            'clf__max_depth': [17500, 18000, 18500],
            'clf__max_features': ['auto', 'sqrt'],
            'clf__min_samples_leaf': [1],
            'clf__min_samples_split': [2, 3, 4],
            'preproc__scaler':  ['passthrough'],
            'preproc__num_red': ['passthrough'],
            'preproc__num_outliers': ['passthrough']

        },
        {
            'clf': [RandomForestClassifier()],
            'clf__criterion': ['gini'],
            'clf__bootstrap': [True],
            'clf__max_depth': [ 55, 60, 65],
            'clf__max_features': ['auto', 'sqrt'],
            'clf__min_samples_leaf': [1],
            'clf__min_samples_split': [3],
            'clf__n_estimators': [550],
            'preproc__scaler': ['passthrough'], 
            'preproc__num_red': [ 'passthrough'],
            'preproc__num_outliers': ['passthrough']

        }, 
        
        {
            'clf': [KNeighborsClassifier()],
            'clf__weights': ['distance'],
            'clf__n_neighbors': range(2, 20, 2), 
            'clf__leaf_size': range(45, 60, 2), 
            'clf__p': [2], 
            'clf__metric': ['manhattan'],
            'preproc__scaler': [StandardScaler()],
            'preproc__num_red': [PCA(n_components=4), PCA(n_components=6)],
            'preproc__num_outliers': ['passthrough']

        }, 
        
        {
            'clf': [LogisticRegression()],
            'clf__class_weight': ['balanced'],
            'clf__max_iter': [20000],
            'clf__multi_class': ['multinomial'],
            'preproc__scaler': [MinMaxScaler()],
            'preproc__num_red': ['passthrough'],
            'preproc__num_outliers': ['passthrough']

        }
    ]

# evaluating multiple classifiers
# based on pipeline parameters
#-------------------------------
result = []


for params in parameters:

    # classifier
    clf = params['clf'][0]
    
    #pipeline
    pipe = Pipeline(steps=[('preproc', preprocessor),
                            ('clf', clf)])

    # cross validation using
    # Grid Search or Random Search
    if grid_search:
        grid = GridSearchCV(pipe, params, cv=cv, verbose=1,
                            n_jobs=n_jobs, scoring=grid_search_scoring)   
    else:
        grid = RandomizedSearchCV(pipe, params, cv=cv, verbose=1,
                    n_jobs=n_jobs, scoring=grid_search_scoring, n_iter=random_picks)
    
    grid.fit(x_train, y_train)
    report = classification_report(y_val, grid.best_estimator_.predict(x_val), digits=3)
    report_dict = classification_report(y_val, grid.best_estimator_.predict(x_val), digits=3, output_dict=True)

    print()
    print(f'Current best Estimator: {grid.best_params_} \n', report)
    print()
    # print("Grid scores on development set:")
    # print()
    # means = grid.cv_results_["mean_test_score"]
    # stds = grid.cv_results_["std_test_score"]
    # for mean, std, params in zip(means, stds, grid.cv_results_["params"]):
    #     print("f1 macro: %0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    # print()
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


# # Sorting result by best score
result = sorted(result, key=itemgetter('best score'), reverse=True)
best_model = result[0]
# print_best_classifier(best_model)


# Visualizations
for pipe in result:
    save_confusion_matrix(pipe, y_test, pipe['classifier'].predict(x_test))
    y_probas = pipe['classifier'].predict_proba(x_test)
    save_pr_curve(pipe, y_test, y_probas)
    save_report(pipe)
    print('Test data \n')
    print(f"Type: {pipe['best params']}")
    print(classification_report(y_test, pipe['classifier'].predict(x_test), digits=3))

# # Saving best classifier
if save_models: joblib.dump(best_model['classifier'], './models/best_classifiers_knn.pkl')
if save_test_data: joblib.dump([x_test, y_test], './dataset_/test_data_genre.pkl')
