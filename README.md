# NBA Game Predictor

## Abstract

NBA Game Predictor is a Python machine learning project that predicts whether the home team will win an NBA matchup. The project collects NBA team game data, creates rolling team performance features, compares classification models, and provides a Tkinter desktop app where users can select a home team and away team to generate a win prediction.

## Developers

- Eric Noga
- Joey Carney
- Joseph
- Lexi Sung
- Justin
##Link for Web Version

https://nba-outcome-predictor.vercel.app/

## How to Run

1. Create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Collect NBA team game data:

```powershell
py code\nba_api_data_collection.py
```

4. Build the feature-engineered dataset:

```powershell
py code\feature_engineering.py
```

5. Optional: compare model performance:

```powershell
py code\model_comparison.py
```

6. Run the prediction app:

```powershell
py code\home.py
```
