# spotify-data-science-project
<<<<<<< HEAD
A Data Science Project for a University Lab Course.

<img width="777" alt="Spotify" src="https://user-images.githubusercontent.com/47475044/170836844-b69d2309-17c8-4f71-bb8f-7cd276cd0c70.png">

=======

![](images/readme/header.jpg)


A Data Science Projekt for a University Lab Course.
>>>>>>> dev

__Welcome to the Spotify Group page for the Data Mining Lab SoSe 2022__

Here we have taken a Spotify dataset and applying data mining techniques on it. Since Spotify is one of the largest platform for music, it also interesting from a business case perspective.

## Topics Explored

### Popularity Analysis and Predictions

Final Results:

<img src="images/readme/pop_prediction_results.jpg" width="500"/>

### Genre Clustering Visualization and Predictions

Final Results:

- Predictions
    ![](images/readme/results_genre.png)

- Clustering Example PCA
    ![](images/readme/pca_rock_hiphop.gif)

### Recommender System of Songs

- ... Flowchart with input song in model and output recommendations (if possible use albums)

## Structure of the Code

- Notebooks for EDA and first Modeling
- Sklearn Pipeline Setup for automated hyperparameter Tuning
  - Choose between gird or random search
  - Possibility for tuning multiple models at ones and choose best preforming one
  - PCA, Scaling & Outlier Cleaning as hyperparameters

### How to handle Conflicts for Notebooks when merging

```python
pip install nbdime
nbdime config-git --enable --global
```

Then if you have a merge conflicts, do:

```python
nbdime mergetool
```

This should allow you to use a gui to fix the conflics.
