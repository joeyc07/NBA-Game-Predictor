import pandas as pd

from config import NBA_TEAMS, NUMERIC_BOXSCORE_COLUMNS


REQUIRED_TEAM_GAME_COLUMNS = [
    "GAME_ID", "GAME_DATE", "TEAM_ID", "TEAM_NAME",
    "MATCHUP", "WL", "PTS", "SEASON"
]


def clean_team_name(name):
    name = str(name).strip()
    replacements = {
        "Los Angeles Clippers": "Los Angeles Clippers",
        "LA Clippers": "Los Angeles Clippers",
        "L.A. Clippers": "Los Angeles Clippers"
    }
    return replacements.get(name, name)


def clean_team_games(df, validate_columns=False):
    df = df.copy()

    if validate_columns:
        missing = [col for col in REQUIRED_TEAM_GAME_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required raw columns: {missing}")

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    df["TEAM_NAME"] = df["TEAM_NAME"].apply(clean_team_name)
    df["MATCHUP"] = df["MATCHUP"].astype(str).str.strip()
    df["WL"] = df["WL"].astype(str).str.strip()

    for col in NUMERIC_BOXSCORE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["TEAM_NAME"].isin(NBA_TEAMS)].copy()
    df = df.dropna(subset=REQUIRED_TEAM_GAME_COLUMNS)
    return df.sort_values(["GAME_DATE", "GAME_ID", "TEAM_NAME"]).reset_index(drop=True)
