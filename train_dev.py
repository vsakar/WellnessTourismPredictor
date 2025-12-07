import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.metrics import classification_report
import joblib
import mlflow

DEV_TRACKING_URI = "https://nonexcitatory-zayn-unhoned.ngrok-free.dev/"

# Point MLflow to Dev server
mlflow.set_tracking_uri(DEV_TRACKING_URI)
mlflow.set_experiment("wellness-tourism-training-dev")

# --- Load dataset ---
DATASET_PATH = "hf://datasets/vsakar/wellness-tourism-prediction/tourism.csv"
df = pd.read_csv(DATASET_PATH)

# --- Define target and features ---
target = "ProdTaken"
numeric_features = [
    "Age",
    "NumberOfPersonVisiting",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "Passport",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "MonthlyIncome",
    "PitchSatisfactionScore",
    "NumberOfFollowups",
    "DurationOfPitch"
]
categorical_features = [
    "TypeofContact",
    "CityTier",
    "Occupation",
    "Gender",
    "MaritalStatus",
    "Designation",
    "ProductPitched"
]

X = df[numeric_features + categorical_features]
y = df[target]

# --- Train-test split ---
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Handle class imbalance ---
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# --- Preprocessing pipeline ---
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

# --- Base XGBoost model ---
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)

pipeline = make_pipeline(preprocessor, xgb_model)

# --- Parameter grid for XGB ---
'''
param_grid = {
    "xgbclassifier__n_estimators": [100, 200],
    "xgbclassifier__max_depth": [3, 5, 7],
    "xgbclassifier__learning_rate": [0.01, 0.1, 0.2],
    "xgbclassifier__subsample": [0.8, 1.0],
    "xgbclassifier__colsample_bytree": [0.8, 1.0]
}
'''
param_grid = {
    "xgbclassifier__n_estimators": [100, 200],
    "xgbclassifier__max_depth": [3, 5],
    "xgbclassifier__learning_rate": [0.05, 0.1],
}


grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring="f1",
    n_jobs=-1,
    verbose=2
)

# --- MLflow run ---
with mlflow.start_run():
    grid_search.fit(Xtrain, ytrain)

    best_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Predictions
    y_pred_train = best_pipeline.predict(Xtrain)
    y_pred_test = best_pipeline.predict(Xtest)

    # Reports
    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log best params
    mlflow.log_params(best_params)

    # Log metrics
    mlflow.log_metrics({
        "train_accuracy": train_report["accuracy"],
        "train_precision": train_report["1"]["precision"],
        "train_recall": train_report["1"]["recall"],
        "train_f1": train_report["1"]["f1-score"],
        "test_accuracy": test_report["accuracy"],
        "test_precision": test_report["1"]["precision"],
        "test_recall": test_report["1"]["recall"],
        "test_f1": test_report["1"]["f1-score"]
    })

    # Save best model
    model_path = "dev_wellness_model.joblib"
    joblib.dump(best_pipeline, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"✅ Best model saved and logged to MLflow: {model_path}")
    print(f"Best hyperparameters: {best_params}")
