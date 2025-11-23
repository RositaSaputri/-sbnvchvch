import pandas as pd
import joblib
import mlflow
import os
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# =======================
# Fungsi Helper
# =======================

def load_data(path):
    print(f"Memuat data dari {path}...")
    return pd.read_csv(path)

def split_data(df, target_col='Exited', test_size=0.2, random_state=42):
    print("Membagi data menjadi train dan test set...")
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

# =======================
# Main Function
# =======================

def main():
    # Ambil parameter dari 'mlflow run'
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=150)
    parser.add_argument("--max_depth", type=int, default=20)
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--data_path", type=str, default="preprocessing/bank_churn_preprocessed.csv")
    args = parser.parse_args()

    # Set experiment MLflow
    mlflow.set_experiment("Bank_Churn_CI_Workflow")

    with mlflow.start_run():
        print("Memulai MLflow Run...")

        # --- Log parameter secara manual ---
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("min_samples_split", args.min_samples_split)
        mlflow.log_param("data_path", args.data_path)

        # --- Load dan split data ---
        df = load_data(args.data_path)
        X_train, X_test, y_train, y_test = split_data(df)

        # --- Latih RandomForest ---
        print("Melatih model RandomForestClassifier...")
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            random_state=42
        )
        model.fit(X_train, y_train)

        # --- Evaluasi model ---
        print("Mengevaluasi performa model...")
        y_pred = model.predict(X_test)
        metrics = {
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_precision": precision_score(y_test, y_pred),
            "test_recall": recall_score(y_test, y_pred),
            "test_f1": f1_score(y_test, y_pred)
        }

        # --- Log metrics ---
        mlflow.log_metrics(metrics)

        # --- Log model artifact ---
        mlflow.sklearn.log_model(model, "model")

        print(f"\nModel berhasil dilatih. Metrics: {metrics}")
        print("Run ID:", mlflow.active_run().info.run_id)
        print("Workflow CI selesai.")

if __name__ == "__main__":
    main()
