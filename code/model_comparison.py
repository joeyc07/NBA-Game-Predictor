import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from config import PROCESSED_GAMES_FILE
from model_utils import get_feature_columns, load_processed_games, split_features_and_target


def load_data():
    if not PROCESSED_GAMES_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {PROCESSED_GAMES_FILE}. Run feature_engineering.py first."
        )

    df = load_processed_games(PROCESSED_GAMES_FILE)
    feature_columns = get_feature_columns(df)
    X_train, X_test, y_train, y_test = split_features_and_target(df, feature_columns)
    return X_train, X_test, y_train, y_test, feature_columns


def evaluate_model(model_name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print(f"\n{'=' * 60}")
    print(model_name)
    print(f"{'=' * 60}")
    print("Training Accuracy:", train_acc)
    print("Test Accuracy:", test_acc)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, test_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, test_pred))

    return {
        "Model": model_name,
        "Training Accuracy": train_acc,
        "Test Accuracy": test_acc
    }


def main():
    X_train, X_test, y_train, y_test, feature_columns = load_data()

    print("Using features:")
    for col in feature_columns:
        print("-", col)

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=3000))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42
        ),
        "MLP Classifier": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=0.001,
                batch_size=32,
                learning_rate_init=0.001,
                max_iter=500,
                random_state=42
            ))
        ])
    }

    results = []

    for model_name, model in models.items():
        result = evaluate_model(model_name, model, X_train, X_test, y_train, y_test)
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("Test Accuracy", ascending=False).reset_index(drop=True)

    print(f"\n{'=' * 60}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'=' * 60}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
