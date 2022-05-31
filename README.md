# spotify-data-science-project
A Data Science Project for a University Lab Course.

<img width="777" alt="Spotify" src="https://user-images.githubusercontent.com/47475044/170836844-b69d2309-17c8-4f71-bb8f-7cd276cd0c70.png">


#### Welcome to the Spotify Group page for the Data Mining Lab SoSe 2022

Here we have taken a Spotify dataset and applying data mining techniques on it. Since Spotify is one of the largest platform for music, it also interesting from a business case perspective.

**At first we want to fokus on the following aspects:**

1. Popular artist and genres
2. Trends in terms of song characteristics
3. Ideal song length
4. Correlation between song characteristics and popularity

## Structure

- Master Branch: Production Code / Finished work (e.g. upload to kaggle)
- Dev Branch: Weekly Exploration and Modeling

### How to handle conflicts

```python
pip install nbdime
nbdime config-git --enable --global
```

Then if you have a merge conflicts, do:

```python
nbdime mergetool
```

This should allow you to use a gui to fix the conflics.
