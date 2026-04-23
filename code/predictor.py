from datetime import datetime

import pandas as pd

try:
    from nba_api.stats.endpoints import leaguegamefinder
except ImportError:
    leaguegamefinder = None

from config import OPTIONAL_FEATURE_COLUMNS
from data_utils import clean_team_games
from feature_engineering import (
    add_efg_features,
    add_last10_features,
    add_matchup_features,
    add_net_rating_and_turnover_features,
    add_rest_features,
    add_scoring_features,
    build_game_level_dataset,
)
from model_utils import get_feature_columns, load_processed_games, train_home_win_model


class NBAPredictorService:
    def __init__(self, data_file):
        self.df = load_processed_games(data_file)
        self.live_feature_df = None
        self.live_data_status = "Local processed dataset"
        self.live_season = self.get_default_live_season()
        self.feature_columns = get_feature_columns(self.df)
        self.model = train_home_win_model(self.df, self.feature_columns)

    def get_default_live_season(self):
        today = datetime.now()
        start_year = today.year if today.month >= 10 else today.year - 1
        end_year = str((start_year + 1) % 100).zfill(2)
        return f"{start_year}-{end_year}"

    def prepare_live_team_games(self, raw_df):
        return clean_team_games(raw_df)

    def build_live_feature_dataset(self):
        if leaguegamefinder is None:
            raise RuntimeError("nba_api is not installed in this environment.")

        season = self.live_season
        raw_df = leaguegamefinder.LeagueGameFinder(
            player_or_team_abbreviation="T",
            season_nullable=season
        ).get_data_frames()[0]

        if raw_df.empty:
            raise RuntimeError(f"NBA API returned no games for season {season}.")

        if "SEASON" not in raw_df.columns:
            raw_df["SEASON"] = season

        team_games_df = self.prepare_live_team_games(raw_df)
        games_df = build_game_level_dataset(team_games_df)
        games_df = add_last10_features(games_df)
        games_df = add_rest_features(games_df)
        games_df = add_scoring_features(games_df)
        games_df = add_matchup_features(games_df)
        games_df = add_efg_features(games_df)
        games_df = add_net_rating_and_turnover_features(games_df)
        return games_df

    def get_prediction_dataset(self):
        # Prefer current NBA API data; fall back to the processed local dataset if live data is unavailable.
        if self.live_feature_df is not None:
            return self.live_feature_df

        try:
            self.live_feature_df = self.build_live_feature_dataset()
            self.live_data_status = f"NBA API live data ({self.live_season})"
            return self.live_feature_df
        except Exception as exc:
            self.live_data_status = f"Local processed dataset fallback ({exc})"
            return self.df

    def get_latest_home_row(self, source_df, team_name):
        rows = source_df[source_df["HOME_TEAM_NAME"] == team_name].sort_values(["GAME_DATE", "GAME_ID"])
        if rows.empty:
            return None
        return rows.iloc[-1]

    def get_latest_away_row(self, source_df, team_name):
        rows = source_df[source_df["AWAY_TEAM_NAME"] == team_name].sort_values(["GAME_DATE", "GAME_ID"])
        if rows.empty:
            return None
        return rows.iloc[-1]

    def get_latest_team_row(self, source_df, team_name):
        home_rows = source_df[source_df["HOME_TEAM_NAME"] == team_name].copy()
        if not home_rows.empty:
            home_rows["TEAM_ROLE"] = "home"

        away_rows = source_df[source_df["AWAY_TEAM_NAME"] == team_name].copy()
        if not away_rows.empty:
            away_rows["TEAM_ROLE"] = "away"

        team_rows = pd.concat([home_rows, away_rows], ignore_index=True)
        if team_rows.empty:
            return None

        team_rows = team_rows.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
        return team_rows.iloc[-1]

    def get_team_snapshot(self, latest_team_row, team_name):
        if latest_team_row is None:
            raise ValueError(f"No team data found for {team_name}.")

        team_role = latest_team_row["TEAM_ROLE"]

        if team_role == "home":
            snapshot = {
                "last10_win_pct": latest_team_row["home_last10_win_pct"],
                "rest_days": latest_team_row["home_rest_days"],
                "last10_pts_scored": latest_team_row["home_last10_pts_scored"],
                "last10_pts_allowed": latest_team_row["home_last10_pts_allowed"],
                "net_rating": latest_team_row["home_net_rating"],
                "turnover_rate": latest_team_row["home_turnover_rate"],
            }
            if "home_last10_efg" in latest_team_row.index:
                snapshot["last10_efg"] = latest_team_row["home_last10_efg"]
        else:
            snapshot = {
                "last10_win_pct": latest_team_row["away_last10_win_pct"],
                "rest_days": latest_team_row["away_rest_days"],
                "last10_pts_scored": latest_team_row["away_last10_pts_scored"],
                "last10_pts_allowed": latest_team_row["away_last10_pts_allowed"],
                "net_rating": latest_team_row["away_net_rating"],
                "turnover_rate": latest_team_row["away_turnover_rate"],
            }
            if "away_last10_efg" in latest_team_row.index:
                snapshot["last10_efg"] = latest_team_row["away_last10_efg"]

        return snapshot

    def build_prediction_input(self, home_team, away_team):
        # Build an upcoming matchup row from each team's latest available form and location-specific stats.
        source_df = self.get_prediction_dataset()

        latest_home_team_row = self.get_latest_team_row(source_df, home_team)
        latest_away_team_row = self.get_latest_team_row(source_df, away_team)
        latest_home_home_row = self.get_latest_home_row(source_df, home_team)
        latest_away_away_row = self.get_latest_away_row(source_df, away_team)

        if latest_home_team_row is None:
            raise ValueError(f"No team data found for {home_team}.")
        if latest_away_team_row is None:
            raise ValueError(f"No team data found for {away_team}.")
        if latest_home_home_row is None:
            raise ValueError(f"No home-game data found for {home_team}.")
        if latest_away_away_row is None:
            raise ValueError(f"No away-game data found for {away_team}.")

        home_snapshot = self.get_team_snapshot(latest_home_team_row, home_team)
        away_snapshot = self.get_team_snapshot(latest_away_team_row, away_team)

        home_last10_win_pct = home_snapshot["last10_win_pct"]
        away_last10_win_pct = away_snapshot["last10_win_pct"]

        home_last10_home_win_pct = latest_home_home_row["home_last10_home_win_pct"]
        away_last10_away_win_pct = latest_away_away_row["away_last10_away_win_pct"]

        home_rest_days = home_snapshot["rest_days"]
        away_rest_days = away_snapshot["rest_days"]

        home_off_vs_away_def = (
            home_snapshot["last10_pts_scored"] - away_snapshot["last10_pts_allowed"]
        )
        away_off_vs_home_def = (
            away_snapshot["last10_pts_scored"] - home_snapshot["last10_pts_allowed"]
        )

        home_net_rating = home_snapshot["net_rating"]
        away_net_rating = away_snapshot["net_rating"]

        home_turnover_rate = home_snapshot["turnover_rate"]
        away_turnover_rate = away_snapshot["turnover_rate"]

        feature_dict = {
            "home_last10_win_pct": home_last10_win_pct,
            "away_last10_win_pct": away_last10_win_pct,
            "last10_win_pct_diff": home_last10_win_pct - away_last10_win_pct,

            "home_last10_home_win_pct": home_last10_home_win_pct,
            "away_last10_away_win_pct": away_last10_away_win_pct,
            "home_away_form_diff": home_last10_home_win_pct - away_last10_away_win_pct,

            "home_rest_days": home_rest_days,
            "away_rest_days": away_rest_days,
            "rest_diff": home_rest_days - away_rest_days,

            "home_off_vs_away_def": home_off_vs_away_def,
            "away_off_vs_home_def": away_off_vs_home_def,
            "offensive_defensive_matchup_diff": home_off_vs_away_def - away_off_vs_home_def,
            "off_def_matchup_diff": home_off_vs_away_def - away_off_vs_home_def,

            "home_net_rating": home_net_rating,
            "away_net_rating": away_net_rating,
            "net_rating_diff": home_net_rating - away_net_rating,

            "home_turnover_rate": home_turnover_rate,
            "away_turnover_rate": away_turnover_rate,
            "turnover_rate_diff": home_turnover_rate - away_turnover_rate,
        }

        if all(col in source_df.columns for col in OPTIONAL_FEATURE_COLUMNS):
            home_last10_efg = home_snapshot["last10_efg"]
            away_last10_efg = away_snapshot["last10_efg"]

            feature_dict["home_last10_efg"] = home_last10_efg
            feature_dict["away_last10_efg"] = away_last10_efg
            feature_dict["effective_fg_pct_diff"] = home_last10_efg - away_last10_efg
            feature_dict["efg_diff"] = home_last10_efg - away_last10_efg

        return pd.DataFrame([feature_dict])[self.feature_columns]

    def predict_game(self, home_team, away_team):
        input_df = self.build_prediction_input(home_team, away_team)
        prediction = self.model.predict(input_df)[0]
        probabilities = self.model.predict_proba(input_df)[0]

        return {
            "prediction": prediction,
            "away_prob": probabilities[0],
            "home_prob": probabilities[1],
            "data_source": self.live_data_status,
        }
