import pandas as pd
import numpy as np
import plotly.express as px
from dash import Dash, dcc, html, Input, Output

# =========================================================
# 1. CARGAR DATOS
# =========================================================
df = pd.read_csv("movies_final.csv", low_memory=False)

# =========================================================
# 2. LIMPIEZA BÁSICA
# =========================================================
numeric_cols = [
    "budget",
    "revenue",
    "runtime",
    "popularity",
    "vote_average",
    "vote_count"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

if "release_date" in df.columns:
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = df["release_date"].dt.year

text_cols = [
    "genre",
    "spoken_languages",
    "countries_name",
    "countries_iso",
    "original_language",
    "title",
    "title_x",
    "title_y"
]

for col in text_cols:
    if col in df.columns:
        df[col] = df[col].astype("string")

if "title" in df.columns:
    title_col = "title"
elif "title_x" in df.columns:
    title_col = "title_x"
elif "title_y" in df.columns:
    title_col = "title_y"
else:
    title_col = None

df = df.dropna(subset=["release_year", "genre"]).copy()
df["release_year"] = df["release_year"].astype(int)
df = df[(df["release_year"] >= 1950) & (df["release_year"] <= 2020)].copy()

if "countries_name" in df.columns:
    country_replacements = {
        "United States of America": "United States",
        "Russian Federation": "Russia",
        "Republic of Korea": "South Korea",
        "Korea": "South Korea",
        "Iran, Islamic Republic of": "Iran",
        "Viet Nam": "Vietnam",
        "Syrian Arab Republic": "Syria",
        "Czech Republic": "Czechia",
        "United Kingdom of Great Britain and Northern Ireland": "United Kingdom"
    }
    df["countries_name"] = df["countries_name"].replace(country_replacements)

genre_options = sorted(df["genre"].dropna().unique().tolist())
language_options = sorted(df["spoken_languages"].dropna().unique().tolist()) if "spoken_languages" in df.columns else []

min_year = int(df["release_year"].min())
max_year = int(df["release_year"].max())

# =========================================================
# 3. ESTILO AUXILIAR
# =========================================================
def make_light(fig):
    fig.update_layout(
        plot_bgcolor="rgba(255,255,255,0.55)",
        paper_bgcolor="rgba(255,255,255,0)",
        font=dict(color="#1f2a44"),
        title_font=dict(color="#1f2a44", size=18),
        legend=dict(
            bgcolor="rgba(255,255,255,0.35)",
            font=dict(color="#1f2a44")
        ),
        xaxis=dict(
            gridcolor="rgba(120,120,120,0.18)",
            zerolinecolor="rgba(120,120,120,0.18)",
            color="#1f2a44"
        ),
        yaxis=dict(
            gridcolor="rgba(120,120,120,0.18)",
            zerolinecolor="rgba(120,120,120,0.18)",
            color="#1f2a44"
        ),
        margin=dict(l=40, r=20, t=60, b=40)
    )
    return fig


def kpi_card(title, value):
    return html.Div(
        style={
            "backgroundColor": "rgba(255,255,255,0.72)",
            "padding": "12px",
            "borderRadius": "12px",
            "border": "1px solid rgba(0,0,0,0.08)",
            "textAlign": "center",
            "color": "#1f2a44",
            "boxShadow": "0 2px 6px rgba(0,0,0,0.08)"
        },
        children=[
            html.Div(title, style={"fontSize": "14px", "opacity": 0.85}),
            html.H3(value, style={"margin": "8px 0 0 0"})
        ]
    )

PANEL_STYLE = {
    "backgroundColor": "rgba(255,255,255,0.58)",
    "padding": "10px",
    "borderRadius": "12px",
    "border": "1px solid rgba(0,0,0,0.08)",
    "boxShadow": "0 2px 6px rgba(0,0,0,0.06)"
}

# =========================================================
# 4. APP
# =========================================================
app = Dash(__name__)
app.title = "Dashboard de Películas"

app.layout = html.Div(
    style={
        "backgroundColor": "#eef2f7",
        "minHeight": "100vh",
        "padding": "20px 24px 40px 24px",
        "fontFamily": "Arial, sans-serif"
    },
    children=[
        html.Div(
            style={
                "maxWidth": "1200px",
                "margin": "0 auto 16px auto",
                "textAlign": "center",
                "color": "#1f2a44"
            },
            children=[
                html.H1(
                    "Dashboard de películas",
                    style={"marginBottom": "6px", "fontSize": "40px"}
                ),
                html.P(
                    "Exploración de presupuesto, revenue, rating, géneros, países e idiomas.",
                    style={"marginTop": "0", "fontSize": "17px"}
                )
            ]
        ),

        html.Div(
            style={
                "position": "relative",
                "maxWidth": "1200px",
                "margin": "0 auto",
                "minHeight": "1500px",
                "backgroundImage": "url('/assets/cinta.png')",
                "backgroundSize": "100% 100%",
                "backgroundRepeat": "no-repeat",
                "backgroundPosition": "center",
                # AJUSTE CLAVE: pantalla útil más angosta y más centrada
                "padding": "115px 135px 95px 135px",
                "boxSizing": "border-box"
            },
            children=[
                html.Div(
                    style={
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "14px"
                    },
                    children=[
                        html.Div(
                            style={
                                "display": "grid",
                                "gridTemplateColumns": "1fr 1fr 1fr",
                                "gap": "10px"
                            },
                            children=[
                                html.Div(
                                    style=PANEL_STYLE,
                                    children=[
                                        html.Label("Géneros", style={"color": "#1f2a44"}),
                                        dcc.Dropdown(
                                            id="genre_filter",
                                            options=[{"label": g, "value": g} for g in genre_options],
                                            value=genre_options,
                                            multi=True,
                                            style={"color": "black"}
                                        )
                                    ]
                                ),
                                html.Div(
                                    style=PANEL_STYLE,
                                    children=[
                                        html.Label("Idiomas", style={"color": "#1f2a44"}),
                                        dcc.Dropdown(
                                            id="language_filter",
                                            options=[{"label": l, "value": l} for l in language_options],
                                            value=language_options,
                                            multi=True,
                                            style={"color": "black"}
                                        )
                                    ]
                                ),
                                html.Div(
                                    style=PANEL_STYLE,
                                    children=[
                                        html.Label("Rango de años", style={"color": "#1f2a44"}),
                                        dcc.RangeSlider(
                                            id="year_filter",
                                            min=min_year,
                                            max=max_year,
                                            step=1,
                                            value=[min_year, max_year],
                                            marks={
                                                y: str(y)
                                                for y in range(
                                                    min_year,
                                                    max_year + 1,
                                                    max(1, (max_year - min_year) // 8)
                                                )
                                            }
                                        )
                                    ]
                                )
                            ]
                        ),

                        html.Div(
                            id="kpi_row",
                            style={
                                "display": "grid",
                                "gridTemplateColumns": "repeat(4, 1fr)",
                                "gap": "10px"
                            }
                        ),

                        html.Div(
                            style={
                                "display": "grid",
                                "gridTemplateColumns": "1fr 1fr",
                                "gap": "10px"
                            },
                            children=[
                                dcc.Graph(id="fig_movies_year", style={"height": "300px"}),
                                dcc.Graph(id="fig_budget_revenue_time", style={"height": "300px"})
                            ]
                        ),

                        html.Div(
                            style={
                                "display": "grid",
                                "gridTemplateColumns": "1fr 1fr",
                                "gap": "10px"
                            },
                            children=[
                                dcc.Graph(id="fig_genres_bar", style={"height": "300px"}),
                                dcc.Graph(id="fig_vote_hist", style={"height": "300px"})
                            ]
                        ),

                        html.Div(
                            style={
                                "display": "grid",
                                "gridTemplateColumns": "1fr 1fr",
                                "gap": "10px"
                            },
                            children=[
                                dcc.Graph(id="fig_budget_revenue_scatter", style={"height": "340px"}),
                                dcc.Graph(id="fig_country_map", style={"height": "340px"})
                            ]
                        ),

                        dcc.Graph(id="fig_dynamic_bubble", style={"height": "430px"})
                    ]
                )
            ]
        )
    ]
)

# =========================================================
# 5. CALLBACK
# =========================================================
@app.callback(
    Output("kpi_row", "children"),
    Output("fig_movies_year", "figure"),
    Output("fig_budget_revenue_time", "figure"),
    Output("fig_genres_bar", "figure"),
    Output("fig_vote_hist", "figure"),
    Output("fig_budget_revenue_scatter", "figure"),
    Output("fig_country_map", "figure"),
    Output("fig_dynamic_bubble", "figure"),
    Input("genre_filter", "value"),
    Input("language_filter", "value"),
    Input("year_filter", "value"),
)
def update_dashboard(selected_genres, selected_languages, selected_years):
    dff = df.copy()

    if selected_genres:
        dff = dff[dff["genre"].isin(selected_genres)]

    if selected_languages and "spoken_languages" in dff.columns:
        dff = dff[dff["spoken_languages"].isin(selected_languages)]

    dff = dff[
        (dff["release_year"] >= selected_years[0]) &
        (dff["release_year"] <= selected_years[1])
    ].copy()

    total_movies = dff["movieId"].nunique() if "movieId" in dff.columns else len(dff)
    avg_vote = round(dff["vote_average"].mean(), 2) if "vote_average" in dff.columns else np.nan
    avg_budget = round(dff.loc[dff["budget"] > 0, "budget"].mean(), 0) if "budget" in dff.columns else np.nan
    avg_revenue = round(dff.loc[dff["revenue"] > 0, "revenue"].mean(), 0) if "revenue" in dff.columns else np.nan

    kpis = [
        kpi_card("Películas", f"{total_movies:,}"),
        kpi_card("Rating promedio", f"{avg_vote}" if pd.notnull(avg_vote) else "N/A"),
        kpi_card("Budget promedio", f"${avg_budget:,.0f}" if pd.notnull(avg_budget) else "N/A"),
        kpi_card("Revenue promedio", f"${avg_revenue:,.0f}" if pd.notnull(avg_revenue) else "N/A"),
    ]

    yearly_movies = (
        dff.groupby("release_year", as_index=False)
        .agg(n_movies=("movieId", "nunique"))
    )

    fig_movies_year = px.line(
        yearly_movies,
        x="release_year",
        y="n_movies",
        markers=True,
        title="Cantidad de películas por año"
    )
    fig_movies_year = make_light(fig_movies_year)
    fig_movies_year.update_layout(xaxis_title="Año", yaxis_title="Películas")

    time_df = (
        dff.groupby("release_year", as_index=False)
        .agg(
            avg_budget=("budget", "mean"),
            avg_revenue=("revenue", "mean")
        )
    )

    time_long = time_df.melt(
        id_vars="release_year",
        value_vars=["avg_budget", "avg_revenue"],
        var_name="metric",
        value_name="value"
    )

    time_long["metric"] = time_long["metric"].replace({
        "avg_budget": "Budget promedio",
        "avg_revenue": "Revenue promedio"
    })

    fig_budget_revenue_time = px.line(
        time_long,
        x="release_year",
        y="value",
        color="metric",
        markers=True,
        title="Budget promedio vs revenue promedio por año"
    )
    fig_budget_revenue_time = make_light(fig_budget_revenue_time)
    fig_budget_revenue_time.update_layout(xaxis_title="Año", yaxis_title="Valor promedio")

    genre_counts = (
        dff.groupby("genre", as_index=False)
        .agg(n_movies=("movieId", "nunique"))
        .sort_values("n_movies", ascending=False)
        .head(10)
    )

    fig_genres_bar = px.bar(
        genre_counts,
        x="genre",
        y="n_movies",
        title="Top 10 géneros por cantidad de películas"
    )
    fig_genres_bar = make_light(fig_genres_bar)
    fig_genres_bar.update_layout(xaxis_title="Género", yaxis_title="Cantidad")

    fig_vote_hist = px.histogram(
        dff.dropna(subset=["vote_average"]),
        x="vote_average",
        nbins=20,
        title="Distribución del rating promedio"
    )
    fig_vote_hist = make_light(fig_vote_hist)
    fig_vote_hist.update_layout(xaxis_title="Vote average", yaxis_title="Frecuencia")

    scatter_df = dff[
        (dff["budget"] > 0) &
        (dff["revenue"] > 0)
    ].copy()

    fig_budget_revenue_scatter = px.scatter(
        scatter_df,
        x="budget",
        y="revenue",
        color="genre",
        size="vote_count" if "vote_count" in scatter_df.columns else None,
        hover_name=title_col if title_col else None,
        title="Budget vs revenue por película"
    )
    fig_budget_revenue_scatter = make_light(fig_budget_revenue_scatter)
    fig_budget_revenue_scatter.update_layout(xaxis_title="Budget", yaxis_title="Revenue")

    map_df = (
        dff.dropna(subset=["countries_name"])
        .groupby("countries_name", as_index=False)
        .agg(n_movies=("movieId", "nunique"))
    )

    fig_country_map = px.choropleth(
        map_df,
        locations="countries_name",
        locationmode="country names",
        color="n_movies",
        hover_name="countries_name",
        color_continuous_scale="Plasma",
        title="Cantidad de películas por país"
    )
    fig_country_map.update_layout(
        paper_bgcolor="rgba(255,255,255,0)",
        font=dict(color="#1f2a44"),
        title_font=dict(color="#1f2a44", size=18),
        coloraxis_colorbar=dict(title="Películas"),
        geo=dict(
            showframe=False,
            showcoastlines=True,
            bgcolor="rgba(255,255,255,0.35)",
            landcolor="rgba(255,255,255,0.35)",
            lakecolor="rgba(255,255,255,0)",
            oceancolor="rgba(255,255,255,0)"
        ),
        margin=dict(l=40, r=20, t=60, b=20)
    )

    bubble_df = df.copy()

    if selected_languages and "spoken_languages" in bubble_df.columns:
        bubble_df = bubble_df[bubble_df["spoken_languages"].isin(selected_languages)]

    bubble_df = bubble_df[
        (bubble_df["release_year"] >= selected_years[0]) &
        (bubble_df["release_year"] <= selected_years[1])
    ].copy()

    bubble_df = bubble_df[
        (bubble_df["budget"] > 0) &
        (bubble_df["revenue"] > 0)
    ].copy()

    year_genre = (
        bubble_df.groupby(["release_year", "genre"], as_index=False)
        .agg(
            avg_revenue=("revenue", "mean"),
            avg_budget=("budget", "mean"),
            n_movies=("movieId", "nunique")
        )
        .sort_values(["release_year", "genre"])
    )

    year_genre["point_id"] = (
        year_genre["genre"].astype(str) + "_" +
        year_genre["release_year"].astype(str)
    )

    years = sorted(year_genre["release_year"].unique().tolist())

    frames = []
    for year in years:
        temp = year_genre[year_genre["release_year"] <= year].copy()
        temp["frame_year"] = year
        frames.append(temp)

    bubble_anim = pd.concat(frames, ignore_index=True) if len(frames) > 0 else year_genre.copy()

    if "frame_year" not in bubble_anim.columns:
        bubble_anim["frame_year"] = bubble_anim["release_year"]

    fig_dynamic_bubble = px.scatter(
        bubble_anim,
        x="release_year",
        y="avg_revenue",
        size="avg_budget",
        color="genre",
        animation_frame="frame_year",
        animation_group="point_id",
        hover_name="genre",
        hover_data={
            "release_year": True,
            "avg_revenue": ":,.0f",
            "avg_budget": ":,.0f",
            "n_movies": True,
            "frame_year": False,
            "point_id": False
        },
        size_max=42,
        title="Revenue promedio anual por género acumulado (tamaño = budget promedio anual)",
        range_x=[selected_years[0], selected_years[1]]
    )
    fig_dynamic_bubble = make_light(fig_dynamic_bubble)
    fig_dynamic_bubble.update_layout(
        xaxis_title="Año",
        yaxis_title="Revenue promedio anual",
        yaxis=dict(range=[0, 550_000_000])
    )

    return (
        kpis,
        fig_movies_year,
        fig_budget_revenue_time,
        fig_genres_bar,
        fig_vote_hist,
        fig_budget_revenue_scatter,
        fig_country_map,
        fig_dynamic_bubble
    )

# =========================================================
# 6. RUN
# =========================================================
if __name__ == "__main__":
    app.run(debug=True)