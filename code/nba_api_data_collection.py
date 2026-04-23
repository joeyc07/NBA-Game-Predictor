import os
import time
import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder

from config import RAW_CACHE_DIR, RAW_GAMES_FILE, SEASONS
from data_utils import clean_team_games


def ensure_directories():
    os.makedirs(RAW_CACHE_DIR, exist_ok=True)
    os.makedirs(RAW_GAMES_FILE.parent, exist_ok=True)


def get_season_games(season):
    cache_file = os.path.join(RAW_CACHE_DIR, f"{season}_team_games.csv")

    try:
        if os.path.exists(cache_file):
            print(f"Using cached games for {season}")
            return pd.read_csv(cache_file)

        print(f"Downloading games for {season}...")
        games = leaguegamefinder.LeagueGameFinder(
            player_or_team_abbreviation="T",
            season_nullable=season
        )

        df = games.get_data_frames()[0]

        if df.empty:
            print(f"No games found for {season}")
            return pd.DataFrame()

        df["SEASON"] = season
        df.to_csv(cache_file, index=False)
        print(f"Saved cached games for {season} to {cache_file}")

        time.sleep(1.5)
        return df

    except Exception as e:
        print(f"Error fetching games for {season}: {e}")
        return pd.DataFrame()


def load_all_seasons():
    season_frames = [get_season_games(season) for season in SEASONS]
    season_frames = [df for df in season_frames if not df.empty]

    if not season_frames:
        return pd.DataFrame()

    return pd.concat(season_frames, ignore_index=True)


def clean_raw_data(df):
    return clean_team_games(df, validate_columns=True)


def main():
    ensure_directories()

    print("Loading season data...")
    raw_df = load_all_seasons()

    if raw_df.empty:
        print("No season data was collected. Exiting.")
        return

    print("Cleaning raw data...")
    raw_df = clean_raw_data(raw_df)

    raw_df.to_csv(RAW_GAMES_FILE, index=False)
    print(f"Saved combined raw data to {RAW_GAMES_FILE}")

    print("\nRaw dataset preview:")
    print(raw_df.head())

    print("\nRaw dataset shape:")
    print(raw_df.shape)

    print("\nRaw dataset columns:")
    print(raw_df.columns.tolist())


if __name__ == "__main__":
    main()
