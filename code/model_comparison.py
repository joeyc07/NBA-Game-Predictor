import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


DATA_FILE = "../data/processed/games_with_features.csv"

FEATURE_COLUMNS = [
    "home_last10_win_pct",
    "away_last10_win_pct",
    "last10_win_pct_diff",
    "home_last10_home_win_pct",
    "away_last10_away_win_pct",
    "home_away_form_diff",
    "home_rest_days",
    "away_rest_days",
    "rest_diff",
    "home_off_vs_away_def",
    "away_off_vs_home_def",
    "home_net_rating",
    "away_net_rating",
    "net_rating_diff",
    "home_turnover_rate",
    "away_turnover_rate",
    "turnover_rate_diff",
]

MATCHUP_FEATURE_ALIASES = [
    "offensive_defensive_matchup_diff",
    "off_def_matchup_diff",
]

OPTIONAL_FEATURE_COLUMNS = [
    "home_last10_efg",
    "away_last10_efg",
    "home_star_available",
    "away_star_available"
]

OPTIONAL_EFG_DIFF_ALIASES = [
    "effective_fg_pct_diff",
    "efg_diff",
]


def load_data():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"Could not find {DATA_FILE}. Run feature_engineering.py first."
        )

    df = pd.read_csv(DATA_FILE)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")

    feature_columns = FEATURE_COLUMNS.copy()

    for col in MATCHUP_FEATURE_ALIASES:
        if col in df.columns:
            feature_columns.append(col)
            break

    for col in OPTIONAL_FEATURE_COLUMNS:
        if col in df.columns:
            feature_columns.append(col)

    for col in OPTIONAL_EFG_DIFF_ALIASES:
        if col in df.columns:
            feature_columns.append(col)
            break

    X = df[feature_columns]
    y = df["HOME_TEAM_WINS"]

    split_index = int(len(df) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

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
                hidden_layer_sizes=(48, 24),
                activation="relu",
                solver="adam",
                alpha=0.01,
                early_stopping=True,
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
