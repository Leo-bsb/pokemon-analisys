"""
Pokémon Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import pickle

# ─────────────────────────────────────────────
# CONFIGURAÇÃO
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Pokémon Analysis",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# DESIGN
# ─────────────────────────────────────────────

COLORS = {
    "red":       "#E63946",
    "blue":      "#457B9D",
    "dark_blue": "#1D3557",
    "green":     "#2D9E6B",
    "amber":     "#F4A261",
    "bg":        "#F8F9FA",
    "card":      "#FFFFFF",
    "border":    "#E9ECEF",
    "text":      "#212529",
    "muted":     "#6C757D",
    "light":     "#F1F3F5",
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif;
    background-color: {COLORS['bg']};
    color: {COLORS['text']};
}}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background-color: {COLORS['dark_blue']};
    border-right: none;
}}
section[data-testid="stSidebar"] * {{
    color: #FFFFFF !important;
}}
section[data-testid="stSidebar"] .stRadio > label {{
    color: rgba(255,255,255,0.7) !important;
    font-size: 0.85rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}}

/* ── Main content ── */
.main .block-container {{
    padding: 2rem 3rem;
    max-width: 1200px;
}}

/* ── Headings ── */
h1, h2, h3 {{
    font-family: 'Syne', sans-serif !important;
}}

/* ── Chapter header ── */
.chapter-header {{
    padding: 2.5rem 3rem;
    border-radius: 16px;
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
}}

/* ── Insight card ── */
.insight-card {{
    background: {COLORS['card']};
    border: 1px solid {COLORS['border']};
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}}

.insight-card.highlight {{
    border-left: 4px solid {COLORS['red']};
}}

.insight-card.info {{
    border-left: 4px solid {COLORS['blue']};
}}

.insight-card.success {{
    border-left: 4px solid {COLORS['green']};
}}

/* ── Stat metric ── */
.stat-block {{
    background: {COLORS['card']};
    border: 1px solid {COLORS['border']};
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    text-align: center;
    box-shadow: 0 2px 6px rgba(0,0,0,0.04);
}}
.stat-block .stat-value {{
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: {COLORS['dark_blue']};
    line-height: 1;
    margin-bottom: 0.25rem;
}}
.stat-block .stat-label {{
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: {COLORS['muted']};
}}
.stat-block .stat-delta {{
    font-size: 0.85rem;
    margin-top: 0.4rem;
    font-weight: 500;
}}

/* ── Pull quote ── */
.pull-quote {{
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: {COLORS['dark_blue']};
    border-left: 5px solid {COLORS['red']};
    padding: 1rem 1.5rem;
    margin: 2rem 0;
    background: {COLORS['light']};
    border-radius: 0 8px 8px 0;
}}

/* ── Tags ── */
.tag {{
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 99px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}}

/* ── Battle result ── */
.battle-winner {{
    background: linear-gradient(135deg, {COLORS['dark_blue']} 0%, {COLORS['blue']} 100%);
    border-radius: 20px;
    padding: 3rem;
    text-align: center;
    margin: 2rem 0;
    color: white;
    box-shadow: 0 12px 40px rgba(29,53,87,0.25);
}}
.battle-winner h2 {{
    color: rgba(255,255,255,0.7) !important;
    font-size: 1rem !important;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.5rem !important;
}}
.battle-winner h1 {{
    color: white !important;
    font-size: 3.5rem !important;
    margin: 0.5rem 0 !important;
}}
.battle-winner .prob {{
    font-size: 1.3rem;
    color: rgba(255,255,255,0.85);
}}

/* ── Nav pill in sidebar ── */
div[data-testid="stRadio"] label {{
    background: rgba(255,255,255,0.07);
    border-radius: 8px;
    padding: 0.6rem 1rem;
    margin: 0.2rem 0;
    display: block;
    transition: background 0.2s;
    cursor: pointer;
}}
div[data-testid="stRadio"] label:hover {{
    background: rgba(255,255,255,0.15);
}}

/* ── Dataframe ── */
.dataframe-container {{
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid {COLORS['border']};
}}

/* ── Progress bar ── */
.stProgress > div > div > div {{
    background: linear-gradient(90deg, {COLORS['blue']}, {COLORS['red']});
    border-radius: 99px;
}}

/* ── Buttons ── */
.stButton > button {{
    background-color: {COLORS['red']};
    color: white;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.05rem;
    border: none;
    border-radius: 10px;
    padding: 0.75rem 2.5rem;
    letter-spacing: 0.03em;
    transition: all 0.2s ease;
    box-shadow: 0 4px 12px rgba(230,57,70,0.3);
}}
.stButton > button:hover {{
    background-color: #C1121F;
    box-shadow: 0 6px 18px rgba(230,57,70,0.4);
    transform: translateY(-2px);
}}

/* ── Selectbox ── */
.stSelectbox > div > div {{
    border-radius: 10px;
    border-color: {COLORS['border']};
    background: white;
}}

/* ── Divider ── */
hr {{
    border: none;
    border-top: 1px solid {COLORS['border']};
    margin: 2.5rem 0;
}}

/* ── Tab overrides ── */
.stTabs [data-baseweb="tab-list"] {{
    background: {COLORS['light']};
    border-radius: 10px;
    padding: 0.4rem;
    gap: 0.5rem;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 8px;
    color: {COLORS['muted']};
    font-weight: 500;
}}
.stTabs [aria-selected="true"] {{
    background: white;
    color: {COLORS['dark_blue']};
    box-shadow: 0 2px 6px rgba(0,0,0,0.08);
}}

.stAlert {{
    border-radius: 10px;
}}
html {{
    zoom: 95%;
}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CAMINHOS
# ─────────────────────────────────────────────

PROCESSED_DATA_PATH = Path("data/processed")
MODELS_PATH = Path("data/models")

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data():
    df_pokemon = pd.read_parquet(PROCESSED_DATA_PATH / "pokemon_processed.parquet")
    df_combats = pd.read_parquet(PROCESSED_DATA_PATH / "combats_processed.parquet")
    if "winner_1" not in df_combats.columns:
        df_combats["winner_1"] = (df_combats["winner_name"] == df_combats["pokemon_1"]).astype(int)
    return df_pokemon, df_combats


@st.cache_resource(show_spinner=False)
def load_models():
    try:
        with open(MODELS_PATH / "logistic_regression_model.pkl", "rb") as f:
            logistic_data = pickle.load(f)
        with open(MODELS_PATH / "catboost_model.pkl", "rb") as f:
            catboost_data = pickle.load(f)
        return logistic_data, catboost_data
    except FileNotFoundError:
        return None, None


# ─────────────────────────────────────────────
# FUNÇÕES DE ANALYTICS
# ─────────────────────────────────────────────

def compute_speed_stats(df):
    df = df.copy()
    df["p1_is_faster"] = (df["speed_diff"] > 0).astype(int)
    win_if_faster = df[df["p1_is_faster"] == 1]["winner_1"].mean()
    win_if_slower = df[df["p1_is_faster"] == 0]["winner_1"].mean()
    return win_if_faster, win_if_slower


def compute_type_advantage_stats(df):
    df = df.copy()
    battles_with_adv = df[df["type_advantage"] != 0]
    winner_had_advantage = (
        ((df["type_advantage"] == 1) & (df["winner_1"] == 1)) |
        ((df["type_advantage"] == 2) & (df["winner_1"] == 0))
    ).astype(int)
    df["winner_had_advantage"] = winner_had_advantage
    pct = battles_with_adv["winner_1"].apply(
        lambda x: x
    )
    adv_win_rate = df[df["type_advantage"] == 1]["winner_1"].mean()
    neutral_win_rate = df[df["type_advantage"] == 0]["winner_1"].mean()
    disadv_win_rate = df[df["type_advantage"] == 2]["winner_1"].mean()
    overall_adv_pct = df[df["type_advantage"] != 0]["winner_had_advantage"].mean()
    return adv_win_rate, neutral_win_rate, disadv_win_rate, overall_adv_pct


def compute_correlations(df):
    features = ["stats_diff", "speed_diff", "attack_diff", "defense_diff"]
    corr = df[features + ["winner_1"]].corr()["winner_1"].drop("winner_1")
    return corr.sort_values(ascending=False)


def build_pokemon_stats(df_combats):
    cols_base = ["pokemon", "generation", "legendary", "type1", "type2",
                 "hp", "attack", "defense", "sp_attack", "sp_defense", "speed", "total_stats"]
    df1 = df_combats[["pokemon_1", "generation_1", "legendary_1", "type1_1", "type2_1",
                       "hp_1", "attack_1", "defense_1", "sp_attack_1", "sp_defense_1",
                       "speed_1", "total_stats_1"]].copy()
    df1.columns = cols_base
    df2 = df_combats[["pokemon_2", "generation_2", "legendary_2", "type1_2", "type2_2",
                       "hp_2", "attack_2", "defense_2", "sp_attack_2", "sp_defense_2",
                       "speed_2", "total_stats_2"]].copy()
    df2.columns = cols_base
    ps = pd.concat([df1, df2], ignore_index=True).drop_duplicates(subset="pokemon")
    ps["type1"] = ps["type1"].fillna("None").astype(str)
    ps["type2"] = ps["type2"].fillna("None").astype(str)
    ps.set_index("pokemon", inplace=True)
    return ps


# ─────────────────────────────────────────────
# FUNÇÕES DE PREDIÇÃO
# ─────────────────────────────────────────────

TYPE_CHART = {
    "Fire": ["Grass", "Ice", "Bug", "Steel"],
    "Water": ["Fire", "Ground", "Rock"],
    "Electric": ["Water", "Flying"],
    "Grass": ["Water", "Ground", "Rock"],
    "Ice": ["Grass", "Ground", "Flying", "Dragon"],
    "Fighting": ["Normal", "Ice", "Rock", "Dark", "Steel"],
    "Poison": ["Grass", "Fairy"],
    "Ground": ["Fire", "Electric", "Poison", "Rock", "Steel"],
    "Flying": ["Grass", "Fighting", "Bug"],
    "Psychic": ["Fighting", "Poison"],
    "Bug": ["Grass", "Psychic", "Dark"],
    "Rock": ["Fire", "Ice", "Flying", "Bug"],
    "Ghost": ["Psychic", "Ghost"],
    "Dragon": ["Dragon"],
    "Dark": ["Psychic", "Ghost"],
    "Steel": ["Ice", "Rock", "Fairy"],
    "Fairy": ["Fighting", "Dragon", "Dark"],
}


def get_type_advantage(types1, types2):
    for t1 in types1:
        if t1 in TYPE_CHART:
            for t2 in types2:
                if t2 in TYPE_CHART[t1]:
                    return 1
    return 0


def predict_battle(pokemon_stats, name1, name2, model_data, is_catboost=False):
    if name1 not in pokemon_stats.index or name2 not in pokemon_stats.index:
        return None
    p1, p2 = pokemon_stats.loc[name1], pokemon_stats.loc[name2]
    types1 = [t for t in [p1["type1"], p1.get("type2", "None")] if t not in ["None", "", None]]
    types2 = [t for t in [p2["type1"], p2.get("type2", "None")] if t not in ["None", "", None]]
    adv = get_type_advantage(types1, types2)

    base_features = {
        "stats_diff":   p1["total_stats"] - p2["total_stats"],
        "speed_diff":   p1["speed"] - p2["speed"],
        "attack_diff":  p1["attack"] - p2["attack"],
        "defense_diff": p1["defense"] - p2["defense"],
        "vantagem_p1":  adv,
    }

    if is_catboost:
        features_dict = {
            **base_features,
            "generation_1": str(p1["generation"]),
            "generation_2": str(p2["generation"]),
            "legendary_1":  str(p1["legendary"]),
            "legendary_2":  str(p2["legendary"]),
            "type1_1":      p1["type1"],
            "type1_2":      p2["type1"],
            "type2_1":      p1.get("type2", "None"),
            "type2_2":      p2.get("type2", "None"),
        }
        X = pd.DataFrame([features_dict])[model_data["all_features"]]
    else:
        X = pd.DataFrame([base_features])[model_data["features"]]

    prob = model_data["model"].predict_proba(X)[0, 1]
    winner = name1 if prob >= 0.5 else name2
    confidence = max(prob, 1 - prob)
    return {
        "pokemon1": name1, "pokemon2": name2,
        "prob_p1": prob, "prob_p2": 1 - prob,
        "winner": winner, "confidence": confidence,
    }


# ─────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────

PLOT_LAYOUT = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="DM Sans, sans-serif", color=COLORS["text"]),
    margin=dict(t=50, b=40, l=40, r=30),
)


# ─────────────────────────────────────────────
# FUNÇÕES DE VISUALIZAÇÃO
# ─────────────────────────────────────────────

def fig_correlations(df_combats):
    corr = compute_correlations(df_combats)
    labels = {
        "speed_diff": "Speed",
        "stats_diff": "Total Stats",
        "attack_diff": "Attack",
        "defense_diff": "Defense",
    }
    bar_colors = [COLORS["red"] if v == corr.max() else COLORS["blue"] for v in corr.values]
    fig = go.Figure(go.Bar(
        x=[labels.get(k, k) for k in corr.index],
        y=corr.values,
        marker_color=bar_colors,
        text=[f"{v:.3f}" for v in corr.values],
        textposition="outside",
        width=0.55,
    ))
    # Remove eixos do layout base para evitar duplicidade
    base_yaxis = PLOT_LAYOUT.get('yaxis', {})
    base_xaxis = PLOT_LAYOUT.get('xaxis', {})
    layout_sem_eixos = {k: v for k, v in PLOT_LAYOUT.items() if k not in ('yaxis', 'xaxis')}
    fig.update_layout(
        **layout_sem_eixos,
        title=dict(text="<b>Correlação com Vitória</b>", font=dict(family="Syne, sans-serif", size=16)),
        height=380,
        yaxis=dict(range=[0, corr.max() * 1.3], title="Correlação de Pearson", **base_yaxis),
        xaxis=dict(title="", **base_xaxis),
        showlegend=False,
    )
    return fig


def fig_speed_win_rate(win_faster, win_slower):
    categories = ["Pokémon mais rápido", "Pokémon mais lento"]
    values = [win_faster, win_slower]
    colors = [COLORS["red"], COLORS["blue"]]
    fig = go.Figure(go.Bar(
        x=categories, y=values,
        marker_color=colors,
        text=[f"{v:.1%}" for v in values],
        textposition="outside",
        width=0.4,
    ))
    fig.add_hline(y=0.5, line_dash="dot", line_color=COLORS["muted"],
                  annotation_text="50% — Chance aleatória", annotation_position="right")
    base_yaxis = PLOT_LAYOUT.get('yaxis', {})
    base_xaxis = PLOT_LAYOUT.get('xaxis', {})
    layout_sem_eixos = {k: v for k, v in PLOT_LAYOUT.items() if k not in ('yaxis', 'xaxis')}
    fig.update_layout(
        **layout_sem_eixos,
        title=dict(text="<b>Taxa de Vitória por Velocidade</b>", font=dict(family="Syne, sans-serif", size=16)),
        height=380,
        yaxis=dict(tickformat=".0%", range=[0, 1], title="Probabilidade de Vitória", **base_yaxis),
        xaxis=dict(title="", **base_xaxis),
    )
    return fig


def fig_speed_scatter(df_combats):
    df = df_combats.copy()
    df["outcome"] = df["winner_1"].map({1: "P1 venceu", 0: "P2 venceu"})
    sample = df.sample(min(3000, len(df)), random_state=42)

    # Gráfico de dispersão original
    fig = px.scatter(
        sample, x="speed_diff", y="stats_diff",
        color="outcome",
        color_discrete_map={"P1 venceu": COLORS["red"], "P2 venceu": COLORS["blue"]},
        opacity=0.35,
        labels={"speed_diff": "Diferença de Velocidade (P1 - P2)", "stats_diff": "Diferença de Stats Totais"},
    )
    fig.update_traces(marker=dict(size=4))

    # --- Linha de regressão (geral) ---
    # Ajuste linear com numpy
    coef = np.polyfit(sample["speed_diff"], sample["stats_diff"], 1)
    linha_x = np.array([sample["speed_diff"].min(), sample["speed_diff"].max()])
    linha_y = np.polyval(coef, linha_x)

    fig.add_trace(go.Scatter(
        x=linha_x,
        y=linha_y,
        mode="lines",
        name="Regressão (geral)",
        line=dict(color="black", width=2, dash="dash"),
        showlegend=True
    ))

    # Layout (mantendo as configurações originais)
    layout_sem_eixos = {k: v for k, v in PLOT_LAYOUT.items() if k not in ('yaxis', 'xaxis')}
    fig.update_layout(
        **layout_sem_eixos,
        title=dict(text="<b>Speed × Stats Totais — Quem vence?</b>", font=dict(family="Syne, sans-serif", size=16)),
        height=420,
        legend=dict(
            title="",
            orientation="h",
            yanchor="bottom",
            y=0.95,
            font=dict(size=12),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="lightgray",
            borderwidth=1
        ),
    )
    return fig


def fig_type_bars(adv, neutral, disadv):
    labels = ["Com vantagem de tipo", "Neutro", "Com desvantagem de tipo"]
    values = [adv, neutral, disadv]
    colors = [COLORS["green"], COLORS["blue"], COLORS["red"]]
    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=[f"{v:.1%}" for v in values],
        textposition="outside",
        width=0.45,
    ))
    fig.add_hline(y=0.5, line_dash="dot", line_color=COLORS["muted"],
                  annotation_text="50% baseline", annotation_position="right")
    base_yaxis = PLOT_LAYOUT.get('yaxis', {})
    base_xaxis = PLOT_LAYOUT.get('xaxis', {})
    layout_sem_eixos = {k: v for k, v in PLOT_LAYOUT.items() if k not in ('yaxis', 'xaxis')}
    fig.update_layout(
        **layout_sem_eixos,
        title=dict(text="<b>Vitórias do P1 por Cenário de Tipo</b>", font=dict(family="Syne, sans-serif", size=16)),
        height=380,
        yaxis=dict(tickformat=".0%", range=[0, 1], title="Taxa de Vitória", **base_yaxis),
        xaxis=dict(title="", **base_xaxis),
    )
    return fig


def fig_stats_distribution(df_pokemon):
    vars_ = ["hp", "attack", "defense", "sp_attack", "sp_defense", "speed"]
    titles = ["HP", "Attack", "Defense", "Sp. Attack", "Sp. Defense", "Speed"]
    palette = [COLORS["red"], COLORS["blue"], COLORS["green"],
               COLORS["amber"], COLORS["dark_blue"], COLORS["muted"]]

    fig = make_subplots(rows=2, cols=3, subplot_titles=titles,
                        horizontal_spacing=0.1, vertical_spacing=0.18)
    for i, (var, color) in enumerate(zip(vars_, palette)):
        row, col = i // 3 + 1, i % 3 + 1
        fig.add_trace(go.Histogram(
            x=df_pokemon[var], nbinsx=28,
            marker_color=color, marker_line_color="white", marker_line_width=0.8,
            showlegend=False,
        ), row=row, col=col)

    layout_sem_eixos = {k: v for k, v in PLOT_LAYOUT.items() if k not in ('yaxis', 'xaxis')}
    fig.update_layout(
        **layout_sem_eixos,
        title=dict(text="<b>Distribuição das Estatísticas Base</b>", font=dict(family="Syne, sans-serif", size=16)),
        height=500,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#F0F0F0")
    fig.update_yaxes(showgrid=True, gridcolor="#F0F0F0")
    return fig


def fig_battle_radar(p1_stats, p2_stats, name1, name2):
    attrs = ["hp", "attack", "defense", "sp_attack", "sp_defense", "speed"]
    labels = ["HP", "ATK", "DEF", "SP.ATK", "SP.DEF", "SPD"]

    fig = go.Figure()
    for name, stats, color in [(name1, p1_stats, COLORS["red"]), (name2, p2_stats, COLORS["blue"])]:
        vals = [stats[a] for a in attrs] + [stats[attrs[0]]]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=labels + [labels[0]],
            fill="toself", fillcolor=color,
            line_color=color, opacity=0.35,
            name=name,
        ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 260], gridcolor="#EEE"),
            angularaxis=dict(gridcolor="#EEE"),
            bgcolor="white",
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
        height=380,
        margin=dict(t=30, b=60, l=40, r=40),
        paper_bgcolor="white",
        font=dict(family="DM Sans, sans-serif"),
    )
    return fig


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────

def chapter_header(icon, eyebrow, title, subtitle, accent=COLORS["red"]):
    st.markdown(f"""
    <div class="chapter-header" style="background: linear-gradient(135deg, {COLORS['dark_blue']} 0%, {COLORS['blue']} 100%);">
        <div style="font-size:0.75rem; font-weight:700; letter-spacing:0.15em; text-transform:uppercase;
                    color:rgba(255,255,255,0.6); margin-bottom:0.5rem;">{eyebrow}</div>
        <h1 style="color:white; font-size:2.2rem; margin:0 0 0.5rem 0; font-family:'Syne',sans-serif;">
            {icon} {title}
        </h1>
        <p style="color:rgba(255,255,255,0.75); font-size:1.05rem; margin:0; max-width:600px;">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def stat_block(value, label, delta=None, delta_positive=True):
    delta_html = ""
    if delta:
        color = COLORS["green"] if delta_positive else COLORS["red"]
        delta_html = f'<div class="stat-delta" style="color:{color};">{delta}</div>'
    st.markdown(f"""
    <div class="stat-block">
        <div class="stat-value">{value}</div>
        <div class="stat-label">{label}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def pull_quote(text):
    st.markdown(f'<div class="pull-quote">{text}</div>', unsafe_allow_html=True)


def insight_card(text, variant="highlight"):
    st.markdown(f'<div class="insight-card {variant}">{text}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE
# ─────────────────────────────────────────────

def page_overview(df_pokemon, df_combats):
    chapter_header(
        "🔍", "Capítulo 1 — Os Dados",
        "Conhecendo o Dataset",
        "Antes de responder qualquer pergunta, precisamos entender o que temos. Escala, cobertura e qualidade dos dados."
    )

    n_battles  = len(df_combats)
    n_pokemon  = len(df_pokemon)
    n_types    = df_pokemon["type1"].nunique()
    n_legends  = df_pokemon["legendary"].sum() if "legendary" in df_pokemon.columns else "—"

    cols = st.columns(4)
    with cols[0]: stat_block(f"{n_battles:,}", "Batalhas registradas")
    with cols[1]: stat_block(f"{n_pokemon}", "Pokémon únicos")
    with cols[2]: stat_block(f"{n_types}", "Tipos primários")
    with cols[3]: stat_block(f"{int(n_legends)}", "Pokémon Lendários")

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns([1.1, 1])

    with col_a:
        st.subheader("Distribuição das Estatísticas Base")
        fig_dist = fig_stats_distribution(df_pokemon)
        st.plotly_chart(fig_dist, width='stretch')

    with col_b:
        st.subheader("O que observamos?")
        st.markdown("<br>", unsafe_allow_html=True)

        insight_card("""
        <b>HP e Defesa</b> apresentam distribuições concentradas em valores baixos com longas caudas à direita —
        indicando que a maioria dos Pokémon é frágil, com poucos outliers extremamente resistentes.
        """, "info")

        insight_card("""
        <b>Velocidade (Speed)</b> tem uma distribuição mais uniforme, cobrindo uma faixa maior de valores.
        Essa heterogeneidade vai ser crucial para a análise adiante.
        """, "highlight")

        insight_card("""
        <b>Pokémon Lendários</b> concentram-se no topo de todas as estatísticas — mas são raros.
        """, "success")

    st.markdown("---")
    pull_quote("Com {n:,} batalhas e dados estruturais completos, temos material suficiente para testar hipóteses com rigor estatístico.".format(n=n_battles))


def page_hypothesis_speed(df_combats):
    chapter_header(
        "⚡", "Capítulo 2 — Hipótese 1",
        "O que faz um Pokémon vencer uma Batalha?",
        "Nossa primeira hipótese: Existe algum atributo que promova vantagem estrutural?"
    )

    win_faster, win_slower = compute_speed_stats(df_combats)
    corr = compute_correlations(df_combats)
    speed_corr = corr.get("speed_diff", 0)

    cols = st.columns(3)
    with cols[0]:
        stat_block(f"{win_faster:.1%}", "Vitórias sendo mais rápido",
                   f"+{win_faster - 0.5:.1%} acima do baseline", True)
    with cols[1]:
        stat_block(f"{win_slower:.1%}", "Vitórias sendo mais lento",
                   f"{win_slower - 0.5:.1%} abaixo do baseline", False)
    with cols[2]:
        stat_block(f"{speed_corr:.3f}", "Correlação com vitória",
                   "Maior entre todos os atributos", True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(fig_speed_win_rate(win_faster, win_slower), width='stretch')
    with col_b:
        st.plotly_chart(fig_correlations(df_combats), width='stretch')

    st.markdown("<br>", unsafe_allow_html=True)
    st.plotly_chart(fig_speed_scatter(df_combats), width='stretch')

    st.markdown("---")
    pull_quote(f"Pokémon mais rápidos vencem {win_faster:.0%} das batalhas. A velocidade é o atributo com maior correlação individual com o resultado.")

    col_a, col_b = st.columns(2)
    with col_a:
        insight_card("""
        <b>Hipótese confirmada.</b> A diferença de velocidade é o preditor mais forte de vitória,
        superando diferenças de ataque, defesa e stats totais individualmente.
        """, "highlight")
    with col_b:
        insight_card("""
        <b>Mas não é absoluto.</b> Mesmo o Pokémon mais lento ainda vence 4,3% das batalhas,
        o que indica que velocidade é o atributo mais determinante para a vitória, mas não exclusivo.
        """, "info")


def page_hypothesis_type(df_combats):
    chapter_header(
        "🔥", "Capítulo 3 — Hipótese 2",
        "Tipo do Pokémon Importa?",
        "Vantagem de tipo é um dos pilares do design Pokémon. Mas quanto ela influencia o resultado real das batalhas?"
    )

    adv_win, neutral_win, disadv_win, overall_pct = compute_type_advantage_stats(df_combats)

    cols = st.columns(3)
    with cols[0]:
        stat_block(f"{adv_win:.1%}", "Vitória com vantagem de tipo",
                   f"+{adv_win - 0.5:.1%} vs baseline", True)
    with cols[1]:
        stat_block(f"{neutral_win:.1%}", "Vitória em cenário neutro", None)
    with cols[2]:
        stat_block(f"{disadv_win:.1%}", "Vitória em desvantagem",
                   f"{disadv_win - 0.5:.1%} vs baseline", False)

    st.markdown("<br>", unsafe_allow_html=True)
    st.plotly_chart(fig_type_bars(adv_win, neutral_win, disadv_win), width='stretch')
    st.markdown("---")

    pull_quote(f"Ter vantagem de tipo eleva a chance de vitória para {adv_win:.0%} — mas não elimina o risco. Uma desvantagem ainda permite ~{disadv_win:.0%} de chance.")

    col_a, col_b = st.columns(2)
    with col_a:
        insight_card("""
        <b>Tipo importa — mas menos do que speed.</b> A vantagem de tipo eleva a probabilidade de vitória
        em ~{delta:.0f} pontos percentuais acima do baseline. É relevante, mas não dominante.
        """.format(delta=(adv_win - 0.5) * 100), "highlight")
    with col_b:
        insight_card("""
        <b>Stats superam tipo.</b> Um Pokémon com Stats Totais muito superiores pode vencer mesmo com
        desvantagem de tipo. Isso sugere que atributos estruturais têm precedência sobre vantagens táticas.
        """, "info")

    st.markdown("<br>", unsafe_allow_html=True)
    insight_card("""
    <b>Síntese das duas hipóteses:</b> Velocidade é o fator estrutural mais determinante.
    Tipo é um modificador relevante, mas secundário. Juntos, eles explicam grande parte do resultado —
    e é exatamente isso que os modelos de ML vão formalizar no próximo capítulo.
    """, "success")


def page_modeling(logistic_data, catboost_data):
    chapter_header(
        "🤖", "Capítulo 4 — Modelagem",
        "Machine Learning Confirma a Análise",
        "Dois modelos preditivos foram treinados. As métricas formalizam o que os dados já sinalizavam."
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(f"""
        <div class="insight-card info">
            <div style="font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:0.1em;
                        color:{COLORS['blue']}; margin-bottom:0.75rem;">Modelo 1</div>
            <h3 style="margin:0 0 0.5rem 0; font-family:'Syne',sans-serif; color:{COLORS['dark_blue']};">
                Regressão Logística
            </h3>
            <p style="color:{COLORS['muted']}; font-size:0.9rem; margin-bottom:1rem;">
                Modelo linear probabilístico. Interpretável, coeficientes diretamente mapeáveis a impacto.
            </p>
            <div style="display:flex; gap:1.5rem;">
                <div><div style="font-size:1.6rem; font-weight:800; font-family:'Syne',sans-serif; color:{COLORS['dark_blue']};">~84%</div>
                    <div style="font-size:0.75rem; color:{COLORS['muted']};">Acurácia</div></div>
                <div><div style="font-size:1.6rem; font-weight:800; font-family:'Syne',sans-serif; color:{COLORS['dark_blue']};">~0.88</div>
                    <div style="font-size:0.75rem; color:{COLORS['muted']};">AUC-ROC</div></div>
            </div>
            <div style="margin-top:1rem; font-size:0.85rem; color:{COLORS['muted']};">
                <b>5 features:</b> stats_diff, speed_diff, attack_diff, defense_diff, type_advantage
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown(f"""
        <div class="insight-card highlight">
            <div style="font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:0.1em;
                        color:{COLORS['red']}; margin-bottom:0.75rem;">Modelo 2 — Melhor Performance</div>
            <h3 style="margin:0 0 0.5rem 0; font-family:'Syne',sans-serif; color:{COLORS['dark_blue']};">
                CatBoost
            </h3>
            <p style="color:{COLORS['muted']}; font-size:0.9rem; margin-bottom:1rem;">
                Gradient boosting com suporte nativo a variáveis categóricas. Captura interações não-lineares.
            </p>
            <div style="display:flex; gap:1.5rem;">
                <div><div style="font-size:1.6rem; font-weight:800; font-family:'Syne',sans-serif; color:{COLORS['dark_blue']};">~88%</div>
                    <div style="font-size:0.75rem; color:{COLORS['muted']};">Acurácia</div></div>
                <div><div style="font-size:1.6rem; font-weight:800; font-family:'Syne',sans-serif; color:{COLORS['dark_blue']};">~0.94</div>
                    <div style="font-size:0.75rem; color:{COLORS['muted']};">AUC-ROC</div></div>
            </div>
            <div style="margin-top:1rem; font-size:0.85rem; color:{COLORS['muted']};">
                <b>13 features:</b> numéricas + geração, lendário, tipos (p1 e p2)
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("O que os modelos aprenderam?")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        insight_card("""
        <b>Speed domina.</b> Em ambos os modelos, <code>speed_diff</code> é consistentemente
        a feature mais importante — confirmando o que a análise exploratória já indicava.
        """, "highlight")
    with col_b:
        insight_card("""
        <b>Stats totais complementam.</b> A diferença de stats totais é o segundo preditor mais forte,
        capturando o efeito geral de superioridade estatística.
        """, "info")
    with col_c:
        insight_card("""
        <b>Tipo e geração adicionam margem.</b> O CatBoost melhora ~4pp de acurácia ao incluir
        tipo e geração, provando que esses fatores contribuem — mas não dominam.
        """, "success")

    st.markdown("---")
    pull_quote("Um AUC-ROC de 0.94 significa que o modelo ordena corretamente o vencedor em 94% dos pares. Isso é poder preditivo real.")


def page_simulator(df_combats, logistic_data, catboost_data):
    chapter_header(
        "⚔️", "Capítulo 5 — Simulador",
        "Teste Você Mesmo",
        "Coloque dois Pokémon frente a frente. Os modelos fazem a predição em tempo real."
    )

    if not logistic_data or not catboost_data:
        st.error("⚠️ Modelos não carregados. Execute o pipeline de treinamento primeiro.")
        return

    pokemon_stats = build_pokemon_stats(df_combats)
    pokemon_list = sorted(pokemon_stats.index.tolist())

    # Seleção dos competidores
    col_l, col_vs, col_r = st.columns([5, 1, 5])

    with col_l:
        st.markdown(f"""
        <div style="text-align:center; padding:0.5rem; font-size:0.75rem; font-weight:700;
                    text-transform:uppercase; letter-spacing:0.1em; color:{COLORS['red']};">
            🔴 Primeiro Competidor
        </div>
        """, unsafe_allow_html=True)
        default1 = pokemon_list.index("Pikachu") if "Pikachu" in pokemon_list else 0
        name1 = st.selectbox("Pokémon 1", pokemon_list, index=default1, label_visibility="collapsed")
        if name1:
            p1 = pokemon_stats.loc[name1]
            st.caption(f"⚡ {p1['speed']} Speed  ·  ⚔️ {p1['attack']} Atk  ·  🛡️ {p1['defense']} Def  ·  ❤️ {p1['hp']} HP")
            st.caption(f"Tipo: **{p1['type1']}** {('/ ' + p1['type2']) if p1['type2'] not in ['None',''] else ''}  ·  Gen {int(p1['generation'])}  ·  Total: **{int(p1['total_stats'])}**")

    with col_vs:
        st.markdown(f"""
        <div style="text-align:center; padding:3rem 0; font-family:'Syne',sans-serif;
                    font-weight:800; font-size:1.4rem; color:{COLORS['muted']};">VS</div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown(f"""
        <div style="text-align:center; padding:0.5rem; font-size:0.75rem; font-weight:700;
                    text-transform:uppercase; letter-spacing:0.1em; color:{COLORS['blue']};">
            🔵 Segundo Competidor
        </div>
        """, unsafe_allow_html=True)
        default2 = pokemon_list.index("Charizard") if "Charizard" in pokemon_list else 1
        name2 = st.selectbox("Pokémon 2", pokemon_list, index=default2, label_visibility="collapsed")
        if name2:
            p2 = pokemon_stats.loc[name2]
            st.caption(f"⚡ {p2['speed']} Speed  ·  ⚔️ {p2['attack']} Atk  ·  🛡️ {p2['defense']} Def  ·  ❤️ {p2['hp']} HP")
            st.caption(f"Tipo: **{p2['type1']}** {('/ ' + p2['type2']) if p2['type2'] not in ['None',''] else ''}  ·  Gen {int(p2['generation'])}  ·  Total: **{int(p2['total_stats'])}**")

    st.markdown("<br>", unsafe_allow_html=True)

    col_btn = st.columns([1, 2, 1])
    with col_btn[1]:
        battle_clicked = st.button("⚔️ Simular Batalha", width='stretch')

    if battle_clicked:
        if name1 == name2:
            st.warning("⚠️ Selecione dois Pokémon diferentes para simular.")
            return

        with st.spinner("Analisando..."):
            res_lr = predict_battle(pokemon_stats, name1, name2, logistic_data, False)
            res_cb = predict_battle(pokemon_stats, name1, name2, catboost_data, True)

        if not res_lr or not res_cb:
            st.error("Erro ao calcular predição.")
            return

        winner = res_cb["winner"]
        conf   = res_cb["confidence"]
        prob1  = res_cb["prob_p1"]
        prob2  = res_cb["prob_p2"]

        # Winner banner
        st.markdown(f"""
        <div class="battle-winner">
            <h2>🏆 Vencedor Previsto pelo CatBoost</h2>
            <h1>{winner}</h1>
            <div class="prob">{conf:.1%} de probabilidade de vitória</div>
        </div>
        """, unsafe_allow_html=True)

        # Confidence bar
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:1rem; margin:0.5rem 0 1.5rem;">
            <div style="font-size:0.85rem; color:{COLORS['red']}; font-weight:600; min-width:80px; text-align:right;">{name1}<br>{prob1:.1%}</div>
            <div style="flex:1;">
        """, unsafe_allow_html=True)
        st.progress(prob1)
        st.markdown(f"""
            </div>
            <div style="font-size:0.85rem; color:{COLORS['blue']}; font-weight:600; min-width:80px;">{name2}<br>{prob2:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

        # Confidence interpretation
        if conf > 0.90:
            st.success("💪 **Alta confiança.** Os stats estruturais apontam claramente para um vencedor.")
        elif conf > 0.75:
            st.info("✅ **Vitória provável.** Boa vantagem, mas o combate ainda tem imprevisibilidade.")
        elif conf > 0.60:
            st.warning("⚖️ **Resultado incerto.** Pequena margem — uma estratégia diferente pode mudar tudo.")
        else:
            st.warning("🎲 **Batalha equilibrada.** Os modelos divergem. Qualquer resultado é plausível.")

        st.markdown("---")

        # Model comparison + Radar
        col_a, col_b = st.columns([1, 1.3])

        with col_a:
            st.subheader("Comparação entre Modelos")
            st.markdown("<br>", unsafe_allow_html=True)
            for label, res, color in [
                ("📈 Regressão Logística", res_lr, COLORS["blue"]),
                ("🌲 CatBoost", res_cb, COLORS["red"]),
            ]:
                st.markdown(f"""
                <div class="stat-block" style="margin-bottom:1rem; border-left:4px solid {color}; text-align:left;">
                    <div style="font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em;
                                color:{color}; margin-bottom:0.3rem;">{label}</div>
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div style="font-size:1.1rem; font-weight:700; font-family:'Syne',sans-serif;">{res['winner']}</div>
                        <div style="font-size:1.4rem; font-weight:800; font-family:'Syne',sans-serif; color:{COLORS['dark_blue']};">
                            {res['confidence']:.1%}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Stats comparison table
            st.markdown("<br>", unsafe_allow_html=True)
            p1s = pokemon_stats.loc[name1]
            p2s = pokemon_stats.loc[name2]
            attrs = ["total_stats", "speed", "attack", "defense", "sp_attack", "hp"]
            icons = ["⭐", "⚡", "⚔️", "🛡️", "✨", "❤️"]
            rows = []
            for attr, icon in zip(attrs, icons):
                v1, v2 = int(p1s[attr]), int(p2s[attr])
                diff = v1 - v2
                rows.append({
                    "Stat": f"{icon} {attr.replace('_',' ').title()}",
                    name1: v1,
                    name2: v2,
                    "Δ": f"+{diff}" if diff > 0 else str(diff),
                })
            df_cmp = pd.DataFrame(rows).set_index("Stat")
            st.dataframe(df_cmp, width='stretch')

        with col_b:
            st.subheader("Radar de Atributos")
            p1s = pokemon_stats.loc[name1]
            p2s = pokemon_stats.loc[name2]
            st.plotly_chart(fig_battle_radar(p1s, p2s, name1, name2), width='stretch')


def page_conclusion():
    chapter_header(
        "🎯", "Conclusão",
        "O que Realmente Determina a Vitória?",
        "Uma síntese executiva da investigação analítica."
    )

    pull_quote("A resposta não está em um único fator — mas a hierarquia é clara.")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown(f"""
        <div class="insight-card highlight">
            <div style="font-size:2rem; margin-bottom:0.5rem;">⚡</div>
            <h3 style="font-family:'Syne',sans-serif; color:{COLORS['dark_blue']}; margin:0 0 0.5rem;">Velocidade</h3>
            <p style="color:{COLORS['muted']}; font-size:0.9rem; margin:0;">
                Principal preditor. Alta correlação com vitória. Quem ataca primeiro tem vantagem estrutural.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown(f"""
        <div class="insight-card info">
            <div style="font-size:2rem; margin-bottom:0.5rem;">📊</div>
            <h3 style="font-family:'Syne',sans-serif; color:{COLORS['dark_blue']}; margin:0 0 0.5rem;">Stats Totais</h3>
            <p style="color:{COLORS['muted']}; font-size:0.9rem; margin:0;">
                Segundo fator mais relevante. Pokémon superiores em stats gerais têm vantagem consistente.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_c:
        st.markdown(f"""
        <div class="insight-card success">
            <div style="font-size:2rem; margin-bottom:0.5rem;">🔥</div>
            <h3 style="font-family:'Syne',sans-serif; color:{COLORS['dark_blue']}; margin:0 0 0.5rem;">Tipo</h3>
            <p style="color:{COLORS['muted']}; font-size:0.9rem; margin:0;">
                Fator relevante, mas secundário. Eleva a chance de vitória, mas não garante e pode ser superado por stats.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="insight-card" style="border-left:4px solid {COLORS['dark_blue']}; background:{COLORS['light']};">
        <h3 style="font-family:'Syne',sans-serif; color:{COLORS['dark_blue']}; margin-top:0;">
            Resultado dos Modelos
        </h3>
        <p style="color:{COLORS['text']}; margin:0;">
            Regressão Logística alcançou <b>~84% de acurácia</b> e AUC-ROC de <b>~0.88</b> apenas com features numéricas.
            CatBoost atingiu <b>~88% de acurácia</b> e AUC-ROC de <b>~0.94</b> ao incorporar tipo e geração.
            A diferença de ~4pp confirma que tipo e geração adicionam valor, mas não são os protagonistas.
        </p>
        <br>
        <p style="color:{COLORS['muted']}; margin:0; font-size:0.9rem;">
            <b>Implicação:</b> É possível prever o vencedor de uma batalha Pokémon com alta confiança
            usando apenas estatísticas estruturais — sem qualquer conhecimento de movimentos, estratégia ou contexto de batalha.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"""
    <div style="text-align:center; padding:2rem 0; color:{COLORS['muted']}; font-size:0.85rem;">
        ⚔️ Pokémon Analysis · Leonardo Braga<br>
        <span style="color:{COLORS['border']};">Regressão Logística + CatBoost · Plotly · Streamlit</span>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div style="padding:1.5rem 0 2rem;">
            <div style="font-family:'Syne',sans-serif; font-size:1.4rem; font-weight:800;
                        color:white; line-height:1.1;">
                ⚔️ Pokémon<br>Analysis
            </div>
            <div style="font-size:0.75rem; color:rgba(255,255,255,0.5); margin-top:0.5rem;
                        letter-spacing:0.05em;">ML · Data Analysis</div>
        </div>
        <hr style="border-color:rgba(255,255,255,0.1); margin-bottom:1.5rem;">
        """, unsafe_allow_html=True)

        nav_options = [
            "🔍  Os Dados",
            "⚡  Hipótese: Velocidade",
            "🔥  Hipótese: Tipo",
            "🤖  Modelagem",
            "⚔️  Simulador",
            "🎯  Conclusão",
        ]

        page = st.radio(
            "Navegação",
            nav_options,
            label_visibility="collapsed",
        )


    return page


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    page = render_sidebar()

    with st.spinner("Carregando dados..."):
        df_pokemon, df_combats = load_data()
        logistic_data, catboost_data = load_models()

    routes = {
        "🔍  Os Dados":           lambda: page_overview(df_pokemon, df_combats),
        "⚡  Hipótese: Velocidade": lambda: page_hypothesis_speed(df_combats),
        "🔥  Hipótese: Tipo":      lambda: page_hypothesis_type(df_combats),
        "🤖  Modelagem":           lambda: page_modeling(logistic_data, catboost_data),
        "⚔️  Simulador":           lambda: page_simulator(df_combats, logistic_data, catboost_data),
        "🎯  Conclusão":           page_conclusion,
    }

    routes[page]()


if __name__ == "__main__":
    main()
