from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_CACHE_DIR = RAW_DATA_DIR / "nba_api_cache"
RAW_GAMES_FILE = RAW_DATA_DIR / "nba_team_games_combined.csv"
PLAYER_GAMES_CACHE_DIR = RAW_DATA_DIR / "nba_api_games_cache"
PLAYER_GAMES_FILE = RAW_DATA_DIR / "nba_player_games_combined.csv"
PROCESSED_GAMES_FILE = PROCESSED_DATA_DIR / "games_with_features.csv"

SEASONS = ["2022-23", "2023-24", "2024-25"]

NBA_TEAMS = [
    "Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets",
    "Chicago Bulls", "Cleveland Cavaliers", "Dallas Mavericks", "Denver Nuggets",
    "Detroit Pistons", "Golden State Warriors", "Houston Rockets", "Indiana Pacers",
    "Los Angeles Clippers", "Los Angeles Lakers", "Memphis Grizzlies", "Miami Heat",
    "Milwaukee Bucks", "Minnesota Timberwolves", "New Orleans Pelicans", "New York Knicks",
    "Oklahoma City Thunder", "Orlando Magic", "Philadelphia 76ers", "Phoenix Suns",
    "Portland Trail Blazers", "Sacramento Kings", "San Antonio Spurs", "Toronto Raptors",
    "Utah Jazz", "Washington Wizards"
]

NUMERIC_BOXSCORE_COLUMNS = [
    "PTS", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
    "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF", "PLUS_MINUS"
]

BASE_FEATURE_COLUMNS = [
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
    "home_star_available",
    "away_star_available",
]

MATCHUP_FEATURE_ALIASES = [
    "offensive_defensive_matchup_diff",
    "off_def_matchup_diff",
]

OPTIONAL_FEATURE_COLUMNS = [
    "home_last10_efg",
    "away_last10_efg",
]

OPTIONAL_EFG_DIFF_ALIASES = [
    "effective_fg_pct_diff",
    "efg_diff",
]
