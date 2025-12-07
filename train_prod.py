import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.metrics import classification_report
import joblib
import mlflow
from mlflow import MlflowClient
from huggingface_hub import HfApi, upload_file
import os

# --- MLflow URIs ---
DEV_TRACKING_URI = "https://nonexcitatory-zayn-unhoned.ngrok-free.dev/"
PROD_TRACKING_URI = "https://nonexcitatory-zayn-unhoned.ngrok-free.dev/"

# --- Load dataset ---
DATASET_PATH = "hf://datasets/vsakar/wellness-tourism-prediction/tourism.csv"
df = pd.read_csv(DATASET_PATH)

# --- Define target and features ---
target = "ProdTaken"
numeric_features = [
    "Age","NumberOfPersonVisiting","PreferredPropertyStar","NumberOfTrips","Passport",
    "OwnCar","NumberOfChildrenVisiting","MonthlyIncome","PitchSatisfactionScore",
    "NumberOfFollowups","DurationOfPitch"
]
categorical_features = [
    "TypeofContact","CityTier","Occupation","Gender","MaritalStatus","Designation","ProductPitched"
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

# --- Step 1: Fetch best hyperparameters from Dev MLflow ---
dev_client = MlflowClient(tracking_uri=DEV_TRACKING_URI)
experiment = dev_client.get_experiment_by_name("wellness-tourism-training-dev")
runs = dev_client.search_runs(experiment.experiment_id, order_by=["metrics.test_f1 DESC"], max_results=1)
best_run = runs[0]
best_params = best_run.data.params
print("✅ Best params from Dev:", best_params)

# --- Step 2: Build XGB model with best params ---
'''
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    eval_metric="logloss",
    n_estimators=int(best_params["xgbclassifier__n_estimators"]),
    max_depth=int(best_params["xgbclassifier__max_depth"]),
    learning_rate=float(best_params["xgbclassifier__learning_rate"]),
    subsample=float(best_params["xgbclassifier__subsample"]),
    colsample_bytree=float(best_params["xgbclassifier__colsample_bytree"])
)
'''
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    eval_metric="logloss",
    n_estimators=int(best_params.get("xgbclassifier__n_estimators", 100)),
    max_depth=int(best_params.get("xgbclassifier__max_depth", 3)),
    learning_rate=float(best_params.get("xgbclassifier__learning_rate", 0.1)),
    subsample=float(best_params.get("xgbclassifier__subsample", 1.0)),
    colsample_bytree=float(best_params.get("xgbclassifier__colsample_bytree", 1.0))
)


pipeline = make_pipeline(preprocessor, xgb_model)

# --- Step 3: Train and log to Prod MLflow ---
mlflow.set_tracking_uri(PROD_TRACKING_URI)
mlflow.set_experiment("wellness-tourism-training-prod")

with mlflow.start_run():
    pipeline.fit(Xtrain, ytrain)
    y_pred_test = pipeline.predict(Xtest)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log params + metrics
    mlflow.log_params(best_params)
    mlflow.log_metrics({
        "test_accuracy": test_report["accuracy"],
        "test_precision": test_report["1"]["precision"],
        "test_recall": test_report["1"]["recall"],
        "test_f1": test_report["1"]["f1-score"]
    })

    # Save model
    model_path = "tourism_project/prod_wellness_model.joblib"
    joblib.dump(pipeline, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"✅ Model saved and logged to MLflow Prod: {model_path}")

    # --- Step 4: Push to Hugging Face Hub ---
    api = HfApi(token=os.getenv("HF_TOKEN"))
    api.create_repo(repo_id="vsakar/wellness_tourism_model", repo_type="model", exist_ok=True)
    upload_file(
        path_or_fileobj=model_path,
        path_in_repo="prod_wellness_model.joblib",
        repo_id="vsakar/wellness_tourism_model",
        repo_type="model"
    )
    print("🚀 Production model pushed to Hugging Face Hub!")
