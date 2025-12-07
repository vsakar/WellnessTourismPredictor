import pandas as pd
from sklearn.model_selection import train_test_split
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
target = "ProdTaken"   # whether customer purchased the package
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

# --- MLflow run ---
with mlflow.start_run():
    pipeline.fit(Xtrain, ytrain)

    # Predictions
    y_pred_train = pipeline.predict(Xtrain)
    y_pred_test = pipeline.predict(Xtest)

    # Reports
    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

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

    # Save model
    model_path = "dev_wellness_model.joblib"
    joblib.dump(pipeline, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved and logged to MLflow: {model_path}")
