
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import sklearn
import joblib
import argparse
import os
import pandas as pd

# Required by SageMaker for inference
def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model


if __name__ == "__main__":
    print("[INFO] Extracting arguments")

    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=0)

    # SageMaker paths
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))

    # File names
    parser.add_argument("--train-file", type=str, default="train-V-1.csv")
    parser.add_argument("--test-file", type=str, default="test-V-1.csv")

    args, _ = parser.parse_known_args()

    print("SKLearn Version:", sklearn.__version__)
    print("Joblib Version:", joblib.__version__)

    print("\n[INFO] Reading data")

    train_df = pd.read_csv(os.path.join(args.train, args.train_file))
    test_df = pd.read_csv(os.path.join(args.test, args.test_file))

    features = list(train_df.columns)
    label = features.pop(-1)

    X_train = train_df[features]
    y_train = train_df[label]

    X_test = test_df[features]
    y_test = test_df[label]

    print("\nColumn order:")
    print(features)

    print("\nLabel column:", label)

    print("\nTraining data shape:", X_train.shape)
    print("Testing data shape:", X_test.shape)

    print("\nTraining RandomForest model...")

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.random_state
    )

    model.fit(X_train, y_train)

    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)

    print("Model persisted at:", model_path)

    y_pred_test = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)

    print("\n--- TEST METRICS ---")
    print("Accuracy:", test_acc)
    print("Classification Report:")
    print(classification_report(y_test, y_pred_test))

