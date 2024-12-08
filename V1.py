!git clone https://github.com/dirty-cat/dirty_cat.git

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import holidays

data = pd.read_parquet("../input/msdb-2024/train.parquet")
test = pd.read_parquet("../input/msdb-2024/final_test.parquet") # import test 

# 1. Handle Missing Data
# Check for missing values
missing_data = data.isnull().sum()
print("Missing data:\n", missing_data)

# Drop rows with missing values in crucial columns
data = data.dropna(subset=['bike_count', 'date', 'latitude', 'longitude'])

# 3. Remove Duplicates
print("Number of duplicates before removal:", data.duplicated().sum())
data = data.drop_duplicates(subset=['counter_id', 'date'])
print("Number of duplicates after removal:", data.duplicated().sum())

# 4. Handle Outliers (check negative values in bike_count)
if (data['bike_count'] < 0).any():
    print("Negative bike counts detected.")
    data['bike_count'] = data['bike_count'].clip(lower=0)  # Replace negative values with 0

# 5. Validate Geographic Coordinates
# Check for invalid latitude and longitude values
invalid_coords = data[(data['latitude'] < -90) | (data['latitude'] > 90) |
                      (data['longitude'] < -180) | (data['longitude'] > 180)]
if not invalid_coords.empty:
    print(f"Invalid coordinates detected:\n{invalid_coords}")
    data = data.dropna(subset=['latitude', 'longitude'])  # Optionally drop invalid coordinates

# 6. Clean Text Columns (remove extra spaces, convert to lowercase)
text_columns = ['counter_name', 'site_name', 'counter_technical_id']
for col in text_columns:
    data[col] = data[col].str.strip().str.lower()

    # 1. Convert `date` column to datetime type
data['date'] = pd.to_datetime(data['date'])

# 2. Extract temporal components
data['date_only'] = data['date'].dt.date
data['hour_only'] = data['date'].dt.hour
data['day_of_week'] = data['date'].dt.dayofweek
data['month'] = data['date'].dt.month
data['year'] = data['date'].dt.year

# 3. Add temporal indicators
data['is_weekend'] = data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
data['is_rush_hour'] = data['hour_only'].apply(lambda x: 1 if (7 <= x <= 9 or 17 <= x <= 19) else 0)
data['season'] = data['month'].apply(lambda x: 'winter' if x in [12, 1, 2] \
                                      else 'spring' if x in [3, 4, 5] \
                                      else 'summer' if x in [6, 7, 8] \
                                      else 'autumn')

# 4. Add holiday indicators
def vacation(date):
    summer_vacation = (date.month == 7 or date.month == 8)
    christmas_vacation = (date.month == 12 and date.day >= 20)
    return 1 if (summer_vacation or christmas_vacation) else 0

data['vacation'] = data['date'].apply(vacation)

# 5. Add public holidays using the holidays library
fr_holidays = holidays.France(years=data['year'].unique().tolist())
def is_public_holiday(date):
    return 1 if date in fr_holidays else 0

data['is_public_holiday'] = data['date'].apply(is_public_holiday)

# Create features (X) and response variable (y)
X = data.drop(columns=['bike_count', 'log_bike_count', 'counter_id', 'site_id', 'coordinates', 'counter_technical_id', 'date_only']) 
y = data['log_bike_count']

# train-test split
from sklearn.model_selection import train_test_split

#data_train, data_test, target_train, target_test = train_test_split(X, y, random_state=2408)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from dirty_cat import TableVectorizer
from sklearn.pipeline import make_pipeline

# Select only numerical features
X_num = X.select_dtypes(include="number")
#data_train_cat = data_train.select_dtypes(include="object")

# store parameters to test in a dictionnary
param_grid = {'max_depth': range(10, 15), 
    'min_samples_split': range(1,3),  
    'min_samples_leaf': range(8,16)} 

# initialize the decision tree regressor
dtr_model = DecisionTreeRegressor(random_state=2408)

# initialize GridSearchCV
grid_search = GridSearchCV(dtr_model, param_grid=param_grid, n_jobs=4, cv=5, return_train_score=True)

# fit GridSearchCV
grid_search.fit(X_num, y) # REGLER CE PB, TU PRENDS QUE LES NUMERIQUES

# get the best max_depth
best_max_depth = grid_search.best_params_['max_depth']
best_split = grid_search.best_params_['min_samples_split']
best_leaf = grid_search.best_params_['min_samples_leaf']

print(best_max_depth, best_split, best_leaf)


# DEFINE PIPELINES and test RMSE - WITHOUT SCALER

# Pipeline
pipeline_no_scaler_dtr = make_pipeline(TableVectorizer(),
    DecisionTreeRegressor(max_depth=best_max_depth,min_samples_split=best_split,min_samples_leaf=best_leaf)
)

# RMSE Without scaler
pipeline_no_scaler_dtr.fit(X, y)
#target_pred_no_scaler = pipeline_no_scaler_dtr.predict(data_test)
#rmse_no_scaler = mean_squared_error(target_test, target_pred_no_scaler, squared=False)
#print(f"RMSE without scaler: {rmse_no_scaler}")



# DEFINE PIPELINES and test RMSE - WITH SCALER

# split numerical and categorical
dfcat= X.select_dtypes(include="object").columns
dfnum= X.select_dtypes(include="number").columns


# define ColumnTransformer
preprocessor = ColumnTransformer([
    ('table_vectorizer', TableVectorizer(),
     dfcat),
    ('standard-scaler', StandardScaler(), dfnum)
])

# with scaler
pipeline_scaler_dtr = make_pipeline(
    preprocessor,
    DecisionTreeRegressor(max_depth=best_max_depth, min_samples_split=best_split, min_samples_leaf=best_leaf)
)

# RMSE with scaler
pipeline_scaler_dtr.fit(X, y)

# predict
#target_pred_scaler = pipeline_scaler_dtr.predict(data_test)
#rmse_scaler = mean_squared_error(target_test, target_pred_scaler, squared=False)
#print(f"RMSE with scaler: {rmse_scaler}")







# 1. Convert `date` column to datetime type
test['date'] = pd.to_datetime(test['date'])

# 2. Extract temporal components
test['date_only'] = test['date'].dt.date
test['hour_only'] = test['date'].dt.hour
test['day_of_week'] = test['date'].dt.dayofweek
test['month'] = test['date'].dt.month
test['year'] = test['date'].dt.year

# 3. Add temporal indicators
test['is_weekend'] = test['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
test['is_rush_hour'] = test['hour_only'].apply(lambda x: 1 if (7 <= x <= 9 or 17 <= x <= 19) else 0)
test['season'] = test['month'].apply(lambda x: 'winter' if x in [12, 1, 2] \
                                      else 'spring' if x in [3, 4, 5] \
                                      else 'summer' if x in [6, 7, 8] \
                                      else 'autumn')

# 4. Add vacation indicators
def vacation(date):
    summer_vacation = (date.month == 7 or date.month == 8)
    christmas_vacation = (date.month == 12 and date.day >= 20)
    return 1 if (summer_vacation or christmas_vacation) else 0

test['vacation'] = test['date'].apply(vacation)

# 5. Add public holidays using the holidays library
fr_holidays = holidays.France(years=test['year'].unique().tolist())
def is_public_holiday(date):
    return 1 if date in fr_holidays else 0

test['is_public_holiday'] = test['date'].apply(is_public_holiday)

# 6. Drop

test = test.drop(columns=['counter_id', 'site_id', 'coordinates', 'counter_technical_id', 'date_only']) 

ypred_scaler = pipeline_scaler_dtr.predict(test)
ypred_no_scaler = pipeline_no_scaler_dtr.predict(test)

results = pd.DataFrame(
    dict(
        Id=np.arange(ypred_scaler.shape[0]),
        log_bike_count=ypred_scaler,
    )
)
results.to_csv("submission.csv", index=False)
