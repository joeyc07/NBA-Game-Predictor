import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from config import (
    BASE_FEATURE_COLUMNS,
    MATCHUP_FEATURE_ALIASES,
    OPTIONAL_EFG_DIFF_ALIASES,
    OPTIONAL_FEATURE_COLUMNS,
    PROCESSED_GAMES_FILE,
)


def get_feature_columns(df):
    feature_columns = BASE_FEATURE_COLUMNS.copy()

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

    return feature_columns


def load_processed_games(path=PROCESSED_GAMES_FILE):
    df = pd.read_csv(path)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    return df


def split_features_and_target(df, feature_columns):
    X = df[feature_columns]
    y = df["HOME_TEAM_WINS"]

    split_index = int(len(df) * 0.8)
    return (
        X.iloc[:split_index],
        X.iloc[split_index:],
        y.iloc[:split_index],
        y.iloc[split_index:],
    )


def train_home_win_model(df, feature_columns, max_iter=2000, print_report=True):
    X_train, X_test, y_train, y_test = split_features_and_target(df, feature_columns)

    model = LogisticRegression(class_weight="balanced", max_iter=max_iter)
    model.fit(X_train, y_train)

    if print_report:
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        print("Training Accuracy:", accuracy_score(y_train, train_pred))
        print("Baseline Model Accuracy:", accuracy_score(y_test, test_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, test_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, test_pred))

    return model
