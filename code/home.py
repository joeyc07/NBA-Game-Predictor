import tkinter as tk
from tkinter import ttk, messagebox

from config import NBA_TEAMS, PROCESSED_GAMES_FILE
from predictor import NBAPredictorService


class NBAPredictorApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("NBA Game Predictor")
        self.geometry("650x450")

        if not PROCESSED_GAMES_FILE.exists():
            messagebox.showerror(
                "Missing Data",
                f"Could not find {PROCESSED_GAMES_FILE}.\nRun nba_api_data_collection.py and feature_engineering.py first."
            )
            self.destroy()
            return

        self.predictor = NBAPredictorService(PROCESSED_GAMES_FILE)
        self.create_widgets()

    def create_widgets(self):
        title_label = tk.Label(
            self,
            text="NBA Home Team Win Predictor",
            font=("Arial", 20, "bold")
        )
        title_label.pack(pady=20)

        home_frame = tk.Frame(self)
        home_frame.pack(pady=10)

        home_label = tk.Label(home_frame, text="Home Team:", font=("Arial", 12))
        home_label.pack(side="left", padx=10)

        self.home_team_var = tk.StringVar()
        self.home_team_dropdown = ttk.Combobox(
            home_frame,
            textvariable=self.home_team_var,
            values=NBA_TEAMS,
            state="readonly",
            width=25
        )
        self.home_team_dropdown.pack(side="left")

        away_frame = tk.Frame(self)
        away_frame.pack(pady=10)

        away_label = tk.Label(away_frame, text="Away Team:", font=("Arial", 12))
        away_label.pack(side="left", padx=10)

        self.away_team_var = tk.StringVar()
        self.away_team_dropdown = ttk.Combobox(
            away_frame,
            textvariable=self.away_team_var,
            values=NBA_TEAMS,
            state="readonly",
            width=25
        )
        self.away_team_dropdown.pack(side="left")

        predict_button = tk.Button(
            self,
            text="Predict Winner",
            font=("Arial", 12),
            command=self.predict_game
        )
        predict_button.pack(pady=20)

        self.result_label = tk.Label(
            self,
            text="Select two teams to predict the game result.",
            font=("Arial", 12),
            wraplength=550,
            justify="center"
        )
        self.result_label.pack(pady=20)

    def predict_game(self):
        home_team = self.home_team_var.get()
        away_team = self.away_team_var.get()

        if home_team == "" or away_team == "":
            messagebox.showerror("Input Error", "Please select both teams.")
            return

        if home_team == away_team:
            messagebox.showerror("Input Error", "Home team and away team cannot be the same.")
            return

        try:
            result = self.predictor.predict_game(home_team, away_team)
        except ValueError as e:
            messagebox.showerror("Data Error", str(e))
            return

        if result["prediction"] == 1:
            winner_text = f"Predicted Winner: {home_team}"
        else:
            winner_text = f"Predicted Winner: {away_team}"

        result_text = (
            f"{winner_text}\n\n"
            f"{home_team} win probability: {result['home_prob']:.2%}\n"
            f"{away_team} win probability: {result['away_prob']:.2%}\n\n"
            f"Data source: {result['data_source']}"
        )

        self.result_label.config(text=result_text)


def main():
    app = NBAPredictorApp()
    app.mainloop()


if __name__ == "__main__":
    main()
