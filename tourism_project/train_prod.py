import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.metrics import classification_report
import joblib
import mlflow

PROD_TRACKING_URI = "https://nonexcitatory-zayn-unhoned.ngrok-free.dev/"

# Point MLflow to Prod server
mlflow.set_tracking_uri(PROD_TRACKING_URI)
mlflow.set_experiment("wellness-tourism-training-prod")

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

# --- Base XGBoost model ---
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    eval_metric="logloss"
)

pipeline = make_pipeline(preprocessor, xgb_model)

# --- MLflow run ---
with mlflow.start_run():
    pipeline.fit(Xtrain, ytrain)

    # Predictions
    y_pred_test = pipeline.predict(Xtest)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log metrics
    mlflow.log_metrics({
        "test_accuracy": test_report["accuracy"],
        "test_precision": test_report["1"]["precision"],
        "test_recall": test_report["1"]["recall"],
        "test_f1": test_report["1"]["f1-score"]
    })

    # Save model
    model_path = "prod_wellness_model.joblib"
    joblib.dump(pipeline, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"✅ Model saved and logged to MLflow Prod: {model_path}")
