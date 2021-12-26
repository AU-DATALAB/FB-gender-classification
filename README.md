<h1 align="left"> Random Forest model of Facebook groups </h1>
<h2 align="left">  DATALAB Aarhus University</h1>
<h3 align="left"> November 2021</h1>
    

## Content 

Using demographics and topic distributions of Facebook groups this projects attempts to see which parts can be classified using Random Forest models. 

| File | Description|
|--------|:-----------|
| random_forest_all_predictors_v3.py | Script performing Random Forest model with varying dependent and independent predictors without grid search |
| random_forest_finalfeatures_v2.py | Models with most important features |

## Data description

The file used as input can be found at grundtvig and is called . It contains 13672 rows with data from Facebook groups. The columns used in the models in these scripts are described below

| Column | Description|
|--------|:-----------|
| total_unique_p_c | amount of unique users that either comment or post in a facebook group |
|total_post | total amount of posts|
|total_comment | total amount of comments|
|dominance | gender dominance calculated with a 20/80 threshold (if males > 80% = 1, if females > 80 % = 2, else neutral = 3)|
|new_days|group age / longevity of group in days |


#### SETUP

```
cd FB-gender-classification # Change directory to the repository folder

bash create_venv.sh 

source FB_RANDOMFOREST/bin/activate

```
Run script by typing
```
python3 random_forest_all_predictors_v3.py # Run script
```

To deactivate and remove the environment, the following commands need to be executed:
```
deactivate 

bash kill_vision_venv.sh

```

    
