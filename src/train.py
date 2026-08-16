import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

DATA_PATH = "data/insurance.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# --------------------------------------------------
# 2. BASIC CLEANING
# --------------------------------------------------

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

df = df.drop_duplicates()

print("Shape after removing duplicates:", df.shape)


# --------------------------------------------------
# 3. TARGET AND FEATURES
# --------------------------------------------------

TARGET = "charges"

X = df.drop(columns=[TARGET])
y = df[TARGET]


# --------------------------------------------------
# 4. IDENTIFY COLUMN TYPES
# --------------------------------------------------

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X.select_dtypes(
    exclude=["object"]
).columns.tolist()

print("\nNumerical features:")
print(numerical_features)

print("\nCategorical features:")
print(categorical_features)


# --------------------------------------------------
# 5. NUMERICAL PIPELINE
# --------------------------------------------------

numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


# --------------------------------------------------
# 6. CATEGORICAL PIPELINE
# --------------------------------------------------

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


# --------------------------------------------------
# 7. COLUMN TRANSFORMER
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_pipeline,
            numerical_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# --------------------------------------------------
# 8. COMPLETE ML PIPELINE
# --------------------------------------------------

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            DecisionTreeRegressor(
                random_state=42
            )
        )
    ]
)


# --------------------------------------------------
# 9. TRAIN TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining rows:", X_train.shape[0])
print("Testing rows:", X_test.shape[0])


# --------------------------------------------------
# 10. HYPERPARAMETER TUNING
# --------------------------------------------------

param_grid = {
    "model__max_depth": [
        3,
        5,
        7,
        10,
        None
    ],

    "model__min_samples_split": [
        2,
        5,
        10
    ],

    "model__min_samples_leaf": [
        1,
        2,
        4
    ]
}


grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    verbose=1
)


# --------------------------------------------------
# 11. TRAIN
# --------------------------------------------------

print("\nTraining model...")

grid_search.fit(
    X_train,
    y_train
)


# --------------------------------------------------
# 12. BEST MODEL
# --------------------------------------------------

best_model = grid_search.best_estimator_

print("\nBest parameters:")
print(grid_search.best_params__)


# --------------------------------------------------
# 13. PREDICTION
# --------------------------------------------------

y_pred = best_model.predict(X_test)


# --------------------------------------------------
# 14. EVALUATION
# --------------------------------------------------

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = mse ** 0.5

r2 = r2_score(
    y_test,
    y_pred
)


print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2  :", r2)


# --------------------------------------------------
# 15. SAVE MODEL
# --------------------------------------------------

MODEL_PATH = "model/model.pkl"

joblib.dump(
    best_model,
    MODEL_PATH
)

print("\nModel saved successfully:")
print(MODEL_PATH)
