# app.py — F1 Prediction Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.models.race_winner import RaceWinnerModel
from src.models.championship import ChampionshipModel
from src.utils.logger import get_logger

logger = get_logger("app")

st.set_page_config(
    page_title="F1 Prediction System",
    page_icon="🏎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .block-container { padding-top: 2rem; }
    .metric-card {
        background: #1a1d27;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #e8393a;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #a0aec0;
        margin-top: 0.3rem;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #ffffff;
        border-left: 4px solid #e8393a;
        padding-left: 0.8rem;
        margin: 1.5rem 0 1rem 0;
    }
    div[data-testid="stSidebar"] {
        background-color: #1a1d27;
    }
</style>
""", unsafe_allow_html=True)

TEAM_COLORS = {
    "Red Bull Racing": "#3671C6",
    "Red Bull":        "#3671C6",
    "Ferrari":         "#E8002D",
    "Mercedes":        "#27F4D2",
    "McLaren":         "#FF8000",
    "Aston Martin":    "#229971",
    "Alpine":          "#FF87BC",
    "Alpine F1 Team":  "#FF87BC",
    "Williams":        "#64C4FF",
    "RB":              "#6692FF",
    "RB F1 Team":      "#6692FF",
    "Haas":            "#B6BABD",
    "Haas F1 Team":    "#B6BABD",
    "Kick Sauber":     "#52E252",
    "Sauber":          "#52E252",
    "Audi":            "#C0C0C0",
    "Cadillac":        "#FFB81C",
}


@st.cache_resource
def load_models():
    winner_model = RaceWinnerModel()
    winner_model.load()
    champ_model  = ChampionshipModel()
    champ_model.load()
    return winner_model, champ_model


@st.cache_data
def load_features():
    path = Path("data/processed/features.parquet")
    if not path.exists():
        return None
    return pd.read_parquet(path)


@st.cache_data
def load_schedule():
    path = Path("data/kaggle/races.csv")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def get_race_features(features_df, season, round_num):
    race_df = features_df[
        (features_df["season"] == season) &
        (features_df["round"] == round_num)
    ].copy()

    if race_df.empty:
        # Future race — use latest available round as baseline
        season_df = features_df[features_df["season"] == season]
        if season_df.empty:
            last_season = features_df["season"].max()
            season_df   = features_df[features_df["season"] == last_season]
        latest_round = season_df["round"].max()
        race_df      = season_df[season_df["round"] == latest_round].copy()

        # Fold the latest results into rolling features
        for idx, row in race_df.iterrows():
            drv = row["driver"]
            driver_history = features_df[
                (features_df["driver"] == drv) &
                ((features_df["season"] < season) |
                 ((features_df["season"] == season) &
                  (features_df["round"] <= latest_round)))
            ]

            last5  = driver_history.tail(5)
            last3  = driver_history.tail(3)
            last10 = driver_history.tail(10)

            if not last5.empty:
                race_df.at[idx, "roll_avg_finish_5"] = last5["finish_position"].mean()
                race_df.at[idx, "roll_avg_points_5"] = last5["points_scored"].mean()
            if not last3.empty:
                race_df.at[idx, "roll_avg_finish_3"] = last3["finish_position"].mean()
                race_df.at[idx, "roll_avg_points_3"] = last3["points_scored"].mean()
            if not last10.empty:
                race_df.at[idx, "roll_avg_finish_10"] = last10["finish_position"].mean()
                race_df.at[idx, "roll_avg_points_10"] = last10["points_scored"].mean()

            # Season-specific updates
            s_curr = features_df[
                (features_df["driver"] == drv) &
                (features_df["season"] == season) &
                (features_df["round"] <= latest_round)
            ]
            if not s_curr.empty:
                race_df.at[idx, "season_avg_finish"]   = s_curr["finish_position"].mean()
                race_df.at[idx, "season_avg_points"]   = s_curr["points_scored"].mean()
                race_df.at[idx, "season_win_count"]    = (s_curr["finish_position"] == 1).sum()
                race_df.at[idx, "season_podium_count"] = (s_curr["finish_position"] <= 3).sum()

        race_df["round"] = round_num

    return race_df


def get_circuit_id_for_round(schedule_df, season, round_num, features_df):
    past = features_df[
        (features_df["season"] == season) &
        (features_df["round"] == round_num)
    ]
    if not past.empty:
        return past["circuit_id"].iloc[0]

    season_schedule = schedule_df[schedule_df["year"] == season]
    race_row        = season_schedule[season_schedule["round"] == round_num]
    if race_row.empty:
        return None

    race_name = race_row["name"].iloc[0].lower()
    race_keywords = [
        w for w in race_name.replace(" grand prix", "").split()
        if len(w) > 3
    ]

    all_circuits = features_df["circuit_id"].unique()
    for circuit in all_circuits:
        circuit_clean = circuit.replace("_", " ").lower()
        if any(kw in circuit_clean for kw in race_keywords):
            return circuit

    return None


def build_round_labels(schedule_df, season, available_rounds, max_round):
    labels          = {}
    season_schedule = schedule_df[schedule_df["year"] == season]

    for r in available_rounds:
        row = season_schedule[season_schedule["round"] == r]
        if not row.empty:
            labels[r] = f"R{r} — {row['name'].iloc[0]}"
        else:
            labels[r] = f"Round {r}"

    for r in range(max_round + 1, 25):
        row = season_schedule[season_schedule["round"] == r]
        if not row.empty:
            labels[r] = f"R{r} 🔮 — {row['name'].iloc[0]}"
        else:
            break

    return labels


def main():

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("# 🏎 F1 Prediction System")
        st.markdown(
            "*Machine learning predictions for race winners "
            "and championship standings*"
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "![F1](https://img.shields.io/badge/Model_Accuracy-73.9%25"
            "-red?style=for-the-badge)"
        )

    st.divider()

    features = load_features()
    if features is None:
        st.error("No feature data found. Run `python run_pipeline.py` first.")
        return

    schedule = load_schedule()

    try:
        winner_model, champ_model = load_models()
    except Exception as e:
        st.error(f"Could not load models: {e}")
        return

    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        st.divider()

        available_seasons = sorted(features["season"].unique(), reverse=True)
        season            = st.selectbox("Season", available_seasons, index=0)

        season_df        = features[features["season"] == season]
        available_rounds = sorted(season_df["round"].unique())
        max_round        = max(available_rounds)

        round_labels = build_round_labels(
            schedule, season, available_rounds, max_round
        )
        all_rounds = list(round_labels.keys())

        round_num = st.selectbox(
            "Round",
            all_rounds,
            index=len(available_rounds) - 1,
            format_func=lambda x: round_labels.get(x, f"Round {x}"),
        )

        is_future = round_num > max_round

        if is_future:
            st.info(
                f"🔮 Future race — predictions based on "
                f"Round {max_round} form + circuit history"
            )
        else:
            circuit = (
                season_df[season_df["round"] == round_num]["circuit_id"].iloc[0]
                if not season_df[season_df["round"] == round_num].empty
                else "Unknown"
            )
            st.markdown(
                f"**Circuit:** `{circuit.replace('_', ' ').title()}`"
            )

        st.divider()
        st.markdown("### 📊 Model Info")
        st.markdown("- **Algorithm:** XGBoost")
        st.markdown("- **Top-1 Accuracy:** 73.9%")
        st.markdown("- **Top-3 Accuracy:** 87.0%")
        st.markdown("- **AUC:** 0.964")
        st.markdown("- **Training data:** 2010–2025")
        st.markdown("- **Test data:** 2024–2025")

        st.divider()
        st.markdown("### 🔄 Update Data")
        if st.button("Refresh Data", type="primary"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

    tab1, tab2, tab3 = st.tabs([
        "🏁 Race Prediction",
        "🏆 Championship",
        "📊 Model Insights",
    ])

    # ── Tab 1: Race Prediction ─────────────────────────────────────────────
    with tab1:
        race_label = round_labels.get(round_num, f"Round {round_num}")
        st.markdown(
            f'<div class="section-header">Win Probabilities — '
            f'{race_label} · {season}'
            f'{"  🔮 Prediction" if is_future else ""}</div>',
            unsafe_allow_html=True,
        )

        if is_future:
            st.caption(
                f"Based on driver form through Round {max_round} "
                f"+ historical circuit performance"
            )

        race_df    = get_race_features(features, season, round_num)
        circuit_id = get_circuit_id_for_round(
            schedule, season, round_num, features
        )

        if race_df.empty:
            st.warning("No data available for this race.")
        else:
            predictions = winner_model.predict(
                race_df, circuit_id=circuit_id
            )
            predictions["win_pct"] = (
                predictions["win_probability"] * 100
            ).round(2)

            top3   = predictions.head(3)
            medals = ["🥇", "🥈", "🥉"]
            cols   = st.columns(3)

            for i, (col, (_, row)) in enumerate(zip(cols, top3.iterrows())):
                color = TEAM_COLORS.get(row["team"], "#666666")
                with col:
                    st.markdown(f"""
                    <div class="metric-card" style="border-top:4px solid {color}">
                        <div style="font-size:2rem">{medals[i]}</div>
                        <div class="metric-value" style="color:{color}">
                            {row['driver']}
                        </div>
                        <div style="color:#a0aec0;font-size:0.9rem;margin:0.3rem 0">
                            {row['team']}
                        </div>
                        <div style="font-size:1.8rem;font-weight:700;color:white">
                            {float(row['win_pct']):.2f}%
                        </div>
                        <div class="metric-label">Win Probability</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            if circuit_id and is_future:
                st.caption(
                    f"🏁 Circuit history applied for: "
                    f"`{circuit_id.replace('_', ' ').title()}`"
                )

            st.markdown(
                '<div class="section-header">All Drivers</div>',
                unsafe_allow_html=True,
            )

            fig = go.Figure()
            for _, row in predictions.head(20).iterrows():
                color   = TEAM_COLORS.get(row["team"], "#666666")
                win_pct = round(float(row["win_pct"]), 2)
                fig.add_trace(go.Bar(
                    x=[win_pct],
                    y=[f"{row['driver']} ({row['team'][:12]})"],
                    orientation="h",
                    marker_color=color,
                    text=f"{win_pct:.2f}%",
                    textposition="outside",
                    showlegend=False,
                ))

            fig.update_layout(
                height=600,
                plot_bgcolor="#0e1117",
                paper_bgcolor="#0e1117",
                font=dict(color="white", size=12),
                xaxis=dict(
                    title="Win Probability (%)",
                    gridcolor="#2d3748",
                    color="white",
                ),
                yaxis=dict(
                    autorange="reversed",
                    gridcolor="#2d3748",
                    color="white",
                ),
                margin=dict(l=200, r=80, t=20, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("📋 Full predictions table"):
                display_df = predictions[["driver", "team", "win_pct"]].copy()
                display_df.columns = ["Driver", "Team", "Win Probability (%)"]
                display_df.index   = range(1, len(display_df) + 1)
                st.dataframe(display_df, use_container_width=True)

    # ── Tab 2: Championship ────────────────────────────────────────────────
    with tab2:
        st.markdown(
            f'<div class="section-header">Championship Projections — '
            f'{season} after Round {round_num}</div>',
            unsafe_allow_html=True,
        )

        season_rounds = sorted(
            features[features["season"] == season]["round"].unique()
        )
        closest_round = max(
            (r for r in season_rounds if r <= round_num),
            default=season_rounds[-1],
        )

        try:
            projections = champ_model.predict_season(
                features, season, closest_round
            )

            # If user selected a future round, project additional rounds forward
            if is_future:
                rounds_ahead = round_num - closest_round
                race_df_future = get_race_features(features, season, round_num)
                future_preds = winner_model.predict(race_df_future, circuit_id=circuit_id)

                # Expected points from upcoming race(s) using win probabilities
                F1_POINTS = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
                future_preds = future_preds.reset_index(drop=True)

                # Add expected points to current standings for each round ahead
                wdc = projections["wdc"].copy()
                wcc = projections["wcc"].copy()

                for _ in range(rounds_ahead):
                    # Expected points per driver based on win prob ranking
                    for i, row in future_preds.iterrows():
                        if i < len(F1_POINTS):
                            exp_pts = F1_POINTS[i] * float(row["win_probability"]) * 4
                            mask = wdc["driver"] == row["driver"]
                            wdc.loc[mask, "current_pts"] += exp_pts
                            wdc.loc[mask, "projected_final_pts"] += exp_pts

                            team_mask = wcc["team"] == row["team"]
                            wcc.loc[team_mask, "current_pts"] += exp_pts
                            wcc.loc[team_mask, "projected_final_pts"] += exp_pts

                wdc = wdc.sort_values("current_pts", ascending=False).reset_index(drop=True)
                wdc["position"] = wdc.index + 1
                wcc = wcc.sort_values("current_pts", ascending=False).reset_index(drop=True)
                wcc["position"] = wcc.index + 1
            else:
                wdc = projections["wdc"]
                wcc = projections["wcc"]


            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 🏆 Drivers Championship (WDC)")

                fig_wdc = go.Figure()
                for _, row in wdc.head(22).iterrows():
                    color = TEAM_COLORS.get(str(row["team"]), "#666666")
                    fig_wdc.add_trace(go.Bar(
                        x=[int(row["projected_final_pts"])],
                        y=[row["driver"]],
                        orientation="h",
                        marker_color=color,
                        text=f"{int(row['projected_final_pts'])} pts",
                        textposition="outside",
                        showlegend=False,
                        customdata=[[int(row["current_pts"])]],
                        hovertemplate=(
                            "<b>%{y}</b><br>"
                            "Current: %{customdata[0]} pts<br>"
                            "Projected: %{x} pts<extra></extra>"
                        ),
                    ))

                fig_wdc.update_layout(
                    height=600,
                    plot_bgcolor="#0e1117",
                    paper_bgcolor="#0e1117",
                    font=dict(color="white", size=11),
                    xaxis=dict(
                        title="Projected Final Points",
                        gridcolor="#2d3748",
                        color="white",
                    ),
                    yaxis=dict(
                        autorange="reversed",
                        gridcolor="#2d3748",
                        color="white",
                    ),
                    margin=dict(l=60, r=80, t=10, b=40),
                )
                st.plotly_chart(fig_wdc, use_container_width=True)

                wdc_display = wdc[[
                    "position", "driver", "team",
                    "current_pts", "projected_final_pts"
                ]].copy()
                wdc_display.columns = [
                    "Pos", "Driver", "Team",
                    "Current Pts", "Projected Pts"
                ]
                wdc_display["Projected Pts"] = (
                    wdc_display["Projected Pts"].astype(int)
                )
                wdc_display["Current Pts"] = (
                    wdc_display["Current Pts"].astype(int)
                )
                wdc_display = wdc_display.set_index("Pos")
                st.dataframe(wdc_display, use_container_width=True)

            with col2:
                st.markdown("#### 🏗 Constructors Championship (WCC)")

                fig_wcc = go.Figure()
                for _, row in wcc.head(12).iterrows():
                    color = TEAM_COLORS.get(str(row["team"]), "#666666")
                    fig_wcc.add_trace(go.Bar(
                        x=[int(row["projected_final_pts"])],
                        y=[str(row["team"])[:20]],
                        orientation="h",
                        marker_color=color,
                        text=f"{int(row['projected_final_pts'])} pts",
                        textposition="outside",
                        showlegend=False,
                        customdata=[[int(row["current_pts"])]],
                        hovertemplate=(
                            "<b>%{y}</b><br>"
                            "Current: %{customdata[0]} pts<br>"
                            "Projected: %{x} pts<extra></extra>"
                        ),
                    ))

                fig_wcc.update_layout(
                    height=600,
                    plot_bgcolor="#0e1117",
                    paper_bgcolor="#0e1117",
                    font=dict(color="white", size=11),
                    xaxis=dict(
                        title="Projected Final Points",
                        gridcolor="#2d3748",
                        color="white",
                    ),
                    yaxis=dict(
                        autorange="reversed",
                        gridcolor="#2d3748",
                        color="white",
                    ),
                    margin=dict(l=140, r=80, t=10, b=40),
                )
                st.plotly_chart(fig_wcc, use_container_width=True)

                wcc_display = wcc[[
                    "position", "team",
                    "current_pts", "projected_final_pts"
                ]].copy()
                wcc_display.columns = [
                    "Pos", "Team", "Current Pts", "Projected Pts"
                ]
                wcc_display["Projected Pts"] = (
                    wcc_display["Projected Pts"].astype(int)
                )
                wcc_display["Current Pts"] = (
                    wcc_display["Current Pts"].astype(int)
                )
                wcc_display = wcc_display.set_index("Pos")
                st.dataframe(wcc_display, use_container_width=True)

        except Exception as e:
            st.error(f"Championship projection error: {e}")

    # ── Tab 3: Model Insights ──────────────────────────────────────────────
    with tab3:
        st.markdown(
            '<div class="section-header">Model Performance</div>',
            unsafe_allow_html=True,
        )

        col1, col2, col3, col4 = st.columns(4)
        perf_metrics = [
            ("73.9%", "Top-1 Accuracy",  "Correctly predicts winner"),
            ("87.0%", "Top-3 Accuracy",  "Winner in top 3 predicted"),
            ("0.964", "AUC Score",        "Discrimination ability"),
            ("0.107", "Log-loss",         "Probability calibration"),
        ]
        for col, (val, label, desc) in zip(
            [col1, col2, col3, col4], perf_metrics
        ):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{val}</div>
                    <div style="color:white;font-weight:600;margin-top:0.3rem">
                        {label}
                    </div>
                    <div class="metric-label">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            '<div class="section-header">Feature Importance (SHAP)</div>',
            unsafe_allow_html=True,
        )

        plot_path = Path("outputs/plots/feature_importance.png")
        bee_path  = Path("outputs/plots/shap_beeswarm.png")

        col1, col2 = st.columns(2)
        with col1:
            if plot_path.exists():
                st.image(
                    str(plot_path),
                    caption="Mean |SHAP| per feature",
                    use_container_width=True,
                )
            else:
                st.info("Run SHAP analysis to generate this plot.")
        with col2:
            if bee_path.exists():
                st.image(
                    str(bee_path),
                    caption="SHAP beeswarm",
                    use_container_width=True,
                )
            else:
                st.info("Run SHAP analysis to generate this plot.")

        st.markdown(
            '<div class="section-header">Race Explanation</div>',
            unsafe_allow_html=True,
        )
        shap_plots = list(Path("outputs/plots").glob("shap_race_*.png"))
        if shap_plots:
            selected = st.selectbox(
                "Select race explanation",
                [p.name for p in shap_plots],
            )
            st.image(
                str(Path("outputs/plots") / selected),
                use_container_width=True,
            )
        else:
            st.info("No race explanations generated yet.")

        st.markdown(
            '<div class="section-header">Comparison with Industry</div>',
            unsafe_allow_html=True,
        )
        comparison = pd.DataFrame({
            "Model": [
                "Our Model",
                "Bookmaker implied",
                "Top Kaggle notebooks",
                "Academic papers (2019-2022)",
                "Pick championship leader",
                "Pick pole position",
                "Random guess",
            ],
            "Top-1 Accuracy": [
                "73.9%", "65-70%", "65-75%",
                "60-72%", "~45%", "~35%", "~5%",
            ],
            "Notes": [
                "XGBoost + ELO + circuit history",
                "Market consensus",
                "Public F1 ML notebooks",
                "Published research",
                "Simple heuristic",
                "Simple heuristic",
                "Baseline",
            ],
        })
        st.dataframe(comparison, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()