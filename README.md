# Car Price Prediction Project

This project aims to predict car prices using data obtained from car sale advertisements. The project includes data preprocessing, feature analysis, and the application of two machine learning models: Decision Tree Regressor and Random Forest Regressor.

## Project Goal

The main goal of this project is to develop a model that can accurately predict car prices based on various characteristics such as model, year of manufacture, engine type, mileage, color, body type, condition, and engine volume.

## Data Loading

You can find the dataset used for this project on Kaggle here: https://www.kaggle.com/competitions/car-price-prediction-x/data

## Data

The data consists of two files: `train.csv` (training set) and `test.csv` (test set).

**Features:**
- `model`: Car brand and model (categorical).
- `year`: Year of car manufacture (numerical).
- `motor_type`: Engine type (categorical: petrol, diesel, hybrid, gas, petrol+gas).
- `running`: Car mileage (numerical, initially string with units).
- `wheel`: Steering wheel location (categorical).
- `color`: Car color (categorical).
- `type`: Car body type (categorical).
- `status`: Car condition (categorical).
- `motor_volume`: Engine volume (numerical).
- `price`: Car price (target variable, only in the training set).

## Methodology

### 1. Data Preprocessing

- **`wheel` feature processing**: It was found that most cars have left-hand drive. The `wheel` feature was removed as it did not carry significant information or had very little variation.
- **`Id` feature processing**: The `Id` feature from the test set was removed as it is an identifier and not needed for model training.
- **`motor_type` feature processing**: The `diesel` and `hybrid` categories were combined with the `petrol` category due to their small number and possible absence in the test set, to avoid encoding issues.
- **`running` feature processing**: Mileage was extracted as a numerical value, and units (`km`, `miles`) were discarded. The data type was converted to integer.
- **`type` feature processing**: The `pickup` and `minivan / minibus` categories were combined with the `Universal` category due to their small number.
- **Categorical feature encoding**: All categorical features were converted to a numerical format using One-Hot Encoding (`pd.get_dummies`).
- **Feature alignment**: Features in the training and test sets were aligned to ensure an identical set of columns with the same order.

### 2. Modeling

Two machine learning models were used to predict car prices:

#### a) Decision Tree Regressor
- **Hyperparameters used**: `criterion` (splitting criterion), `max_depth` (maximum tree depth), `min_samples_split` (minimum number of samples required to split an internal node), `min_samples_leaf` (minimum number of samples required to be at a leaf node).
- **Hyperparameter tuning**: `GridSearchCV` with 5-fold cross-validation was used to find optimal hyperparameters.

#### b) Random Forest Regressor
- **Hyperparameters used**: `n_estimators` (number of trees in the forest), `max_depth` (maximum tree depth), `min_samples_split`, `min_samples_leaf`, `max_features` (number of features to consider when looking for the best split).
- **Hyperparameter tuning**: `GridSearchCV` with 5-fold cross-validation was also used to find optimal hyperparameters.

### 3. Model Evaluation and Results

The models were evaluated on the test set, and predictions were saved to CSV files for submission. Evaluation was performed using a metric related to prediction error.

- **Decision Tree Regressor Score**: `2016.0097`
- **Random Forest Regressor Score**: `1891.7341`

*Note: The lower the error value (score), the better the model. Random Forest showed better results in this case.*

## Output Files

- `sub.csv`: File with price predictions obtained using the Decision Tree Regressor.
- `rf_sub.csv`: File with price predictions obtained using the Random Forest Regressor.

## Libraries Used

- `pandas`
- `numpy`
- `seaborn`
- `matplotlib.pyplot`
- `sklearn` (for models, preprocessing, and hyperparameter tuning)
