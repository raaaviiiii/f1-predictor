<div align="center">

<img src="https://upload.wikimedia.org/wikipedia/commons/3/33/F1.svg" width="80px" alt="F1 Logo"/>

# 🏎 F1 Prediction System

### Machine learning system that predicts Formula 1 race winners and championship standings

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1-orange?style=flat-square)](https://xgboost.readthedocs.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.6-green?style=flat-square)](https://lightgbm.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=flat-square&logo=streamlit)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

</div>

---

## 🎯 What it does

This system predicts:
- **Race winners** — win probability for every driver at every race
- **WDC standings** — projected final drivers championship after each round
- **WCC standings** — projected final constructors championship after each round
- **Future races** — predictions for upcoming rounds using current form + circuit history
- **2026 season** — live predictions updated after every race weekend

---

## 📊 Model Performance

| Metric | Score | Benchmark |
|---|---|---|
| Top-1 Accuracy | **73.9%** | Bookmakers: ~65-70% |
| Top-3 Accuracy | **87.0%** | Random guess: ~14% |
| AUC Score | **0.964** | Perfect: 1.0 |
| Log-loss | **0.107** | Lower is better |

> **73.9% top-1 accuracy** means the model correctly identifies the race winner almost 3 out of 4 times on completely unseen 2023–2024 data — beating bookmaker implied accuracy.

---

## 🔮 2026 Season Live Predictions

### Round 5 — Canadian Grand Prix (Upcoming)

| Pos | Driver | Team | Win Probability |
|---|---|---|---|
| 🥇 | ANT | Mercedes | 41.90% |
| 🥈 | LEC | Ferrari | 26.94% |
| 🥉 | VER | Red Bull | 15.98% |
| 4 | NOR | McLaren | 5.84% |
| 5 | HAM | Ferrari | 4.18% |

> Predictions based on Round 4 form + historical circuit performance at Montreal

### Championship after Round 4

| Pos | Driver | Team | Current Pts | Projected Final |
|---|---|---|---|---|
| 🥇 | ANT | Mercedes | 74 | 256 |
| 🥈 | VER | Red Bull | 13 | 245 |
| 🥉 | RUS | Mercedes | 63 | 238 |
| 4 | LEC | Ferrari | 49 | 225 |
| 5 | NOR | McLaren | 25 | 184 |

| Pos | Constructor | Current Pts | Projected Final |
|---|---|---|---|
| 🥇 | Mercedes | 137 | 552 |
| 🥈 | Ferrari | 90 | 370 |
| 🥉 | McLaren | 46 | 332 |
| 4 | Red Bull | 17 | 201 |

---

## 🏗 Architecture
f1-predictor/
├── config/
│   └── config.py              # Seasons, points system, model params
├── src/
│   ├── pipeline/
│   │   ├── loader.py          # CSV data loader (TracingInsights dataset)
│   │   ├── assembler.py       # Master dataset builder + standings
│   │   └── elo.py             # Driver & constructor ELO engine
│   ├── features/
│   │   └── engineer.py        # 67 engineered features
│   ├── models/
│   │   ├── race_winner.py     # XGBoost race winner classifier
│   │   ├── championship.py    # LightGBM championship projector
│   │   └── tuner.py           # Optuna hyperparameter tuner
│   └── evaluation/
│       └── shap_analysis.py   # SHAP feature importance
├── app.py                     # Streamlit dashboard
├── run_pipeline.py            # Full pipeline runner
├── predict.py                 # CLI prediction tool
└── update.py                  # Post-race data updater

---

## 🔬 How it works

### 1. Data Pipeline
Historical F1 data from 2010–2026 sourced from [TracingInsights](https://tracinginsights.com/data/) — race results, qualifying times, pit stops, sprint races, and championship standings for every round. Updates automatically after each race weekend.

### 2. ELO Rating Engine
Every driver and constructor gets a continuous ELO rating that updates after each race using pairwise comparisons. Key design decisions:
- **Off-season decay** — 10% regression toward mean each year
- **Regulation era decay** — 35% constructor reset at major rule changes (2014, 2017, 2019, 2022, 2026)
- **Rookie seeding** — new drivers seeded from F2 championship performance
- **Driver portability** — ELO travels with the driver across team changes

### 3. Feature Engineering (67 features)

| Category | Features |
|---|---|
| Qualifying | Grid position, gap to pole %, teammate delta |
| Driver form | Rolling avg finish (3/5/10 races), points, DNF rate |
| Pace | Lap time vs field median, rolling pace delta |
| Circuit history | Historical avg finish, win rate, podium rate per circuit |
| Constructor | Rolling points, ELO, regulation era age |
| Championship | Points before race, gap to leader, races remaining |
| ELO | Driver rating, constructor rating, delta from last race |

### 4. Models
- **Race winner:** XGBoost classifier with Optuna-tuned hyperparameters, temporal cross-validation (train 2010–2022, test 2023–2024)
- **Championship:** LightGBM regressor predicting final season points from mid-season state (WDC MAE: 32 pts, WCC MAE: 57 pts)

### 5. Circuit History Multiplier
For future race predictions, a circuit-specific multiplier adjusts base probabilities based on each driver's historical performance at that circuit. Drivers with 2+ appearances get up to a 1.5x boost (or 0.7x penalty) based on their historical win rate. New drivers receive a neutral multiplier.

### 6. 2026 Regulation Awareness
The 2026 season introduces active aerodynamics and a new 50/50 hybrid power unit formula. The model handles this by:
- Applying 35% constructor ELO decay at the 2026 boundary
- Using driver-level ELO (portable across teams) separately from constructor context
- Updating predictions live as 2026 race data becomes available

---

## 🚀 Quick Start

### 1. Clone and install
```bash
git clone https://github.com/YOUR_USERNAME/f1-predictor.git
cd f1-predictor
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the pipeline
```bash
python run_pipeline.py
```

### 3. Launch the dashboard
```bash
streamlit run app.py
```

### 4. Or use the CLI
```bash
# Race prediction
python predict.py --race "Monaco" --season 2026 --round 6

# Championship projection
python predict.py --championship --season 2026 --after-round 4
```

---

## 🔄 Updating after each race

The TracingInsights dataset updates automatically after every race weekend. To refresh your predictions:

```bash
python update.py
```

This rebuilds the full feature matrix and your dashboard will reflect the updated standings and form immediately.

---

## 🧠 Feature Importance (SHAP)

The top predictive features ranked by mean absolute SHAP value:

| Rank | Feature | Description |
|---|---|---|
| 1 | pace_vs_field | Lap time advantage vs field median |
| 2 | grid_position | Starting position |
| 3 | quali_position | Qualifying result |
| 4 | quali_gap_to_pole_pct | Gap to pole as percentage |
| 5 | roll_avg_finish_5 | Rolling average finish (last 5 races) |
| 6 | elo_rating | Driver ELO skill rating |
| 7 | roll_avg_points_3 | Rolling points (last 3 races) |
| 8 | overtake_index | Positions gained from grid (rolling) |
| 9 | pts_gap_to_leader | Championship pressure |
| 10 | constructor_elo | Team performance rating |

---

## 📈 Industry Comparison

| Model | Top-1 Accuracy | Notes |
|---|---|---|
| **Our Model** | **73.9%** | XGBoost + ELO + circuit history |
| Bookmaker implied | 65-70% | Market consensus |
| Top Kaggle notebooks | 65-75% | Public F1 ML notebooks |
| Academic papers (2019-2022) | 60-72% | Published research |
| Pick championship leader | ~45% | Simple heuristic |
| Pick pole position | ~35% | Simple heuristic |
| Random guess | ~5% | Baseline |

---

## 🛠 Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| ML Models | XGBoost, LightGBM |
| Hyperparameter Tuning | Optuna (100 trials) |
| Feature Importance | SHAP |
| Dashboard | Streamlit + Plotly |
| Data | TracingInsights CSV dataset (2010–2026) |
| Ratings | Custom ELO engine |

---

## 📝 License

MIT License — free to use, modify, and distribute.

---

<div align="center">
Built with ❤️ and way too much coffee ☕
</div>