import base64
from io import BytesIO

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from dash import Dash, dcc, html, Input, Output

# =========================================================
# 1. CARGA Y LIMPIEZA
# =========================================================
df = pd.read_csv("movies_final.csv", low_memory=False)

numeric_cols = ["budget", "revenue", "runtime", "popularity", "vote_average", "vote_count"]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

if "release_date" in df.columns:
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")

if "release_year" not in df.columns and "release_date" in df.columns:
    df["release_year"] = df["release_date"].dt.year

df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
df = df.dropna(subset=["release_year", "genre"]).copy()
df["release_year"] = df["release_year"].astype(int)

# rango razonable
df = df[(df["release_year"] >= 1950) & (df["release_year"] <= 2020)].copy()

# strings
for col in ["genre", "spoken_languages", "countries_name", "countries_iso", "title", "title_x", "title_y"]:
    if col in df.columns:
        df[col] = df[col].astype("string")

# elegir columna título
if "title" in df.columns:
    title_col = "title"
elif "title_x" in df.columns:
    title_col = "title_x"
elif "title_y" in df.columns:
    title_col = "title_y"
else:
    title_col = None

# normalización simple para mapa
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
# 2. ESTILO
# =========================================================
BG = "#eef2f7"
TEXT = "#1f2a44"
CARD_BG = "rgba(255,255,255,0.88)"
PLOT_BG = "rgba(255,255,255,0.82)"
BORDER = "1px solid rgba(31,42,68,0.08)"
SHADOW = "0 4px 12px rgba(31,42,68,0.08)"

def empty_figure(title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text="No hay datos para los filtros seleccionados",
        x=0.5, y=0.5, xref="paper", yref="paper",
        showarrow=False, font=dict(size=16, color=TEXT)
    )
    fig.update_layout(
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=60, b=20),
        height=330
    )
    return fig

def style_fig(fig, title, height=330):
    fig.update_layout(
        title=title,
        title_font=dict(size=18, color=TEXT),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT),
        margin=dict(l=20, r=20, t=60, b=25),
        height=height,
        legend=dict(
            bgcolor="rgba(255,255,255,0.65)",
            bordercolor="rgba(31,42,68,0.08)",
            borderwidth=1
        )
    )
    fig.update_xaxes(gridcolor="rgba(31,42,68,0.08)")
    fig.update_yaxes(gridcolor="rgba(31,42,68,0.08)")
    return fig

def card(title, value, icon=""):
    return html.Div(
        style={
            "background": CARD_BG,
            "border": BORDER,
            "borderRadius": "14px",
            "boxShadow": SHADOW,
            "padding": "14px 10px",
            "textAlign": "center"
        },
        children=[
            html.Div(title, style={"fontSize": "14px", "color": "#55627a"}),

            html.Div(
                children=[
                    html.Span(icon, style={"marginRight": "8px", "fontSize": "26px"}),
                    html.Span(value)
                ],
                style={
                    "fontSize": "30px",
                    "fontWeight": "700",
                    "color": TEXT,
                    "marginTop": "6px"
                }
            )
        ]
    )

panel_style = {
    "background": CARD_BG,
    "border": BORDER,
    "borderRadius": "14px",
    "boxShadow": SHADOW,
    "padding": "12px"
}

def make_wordcloud_src(dff: pd.DataFrame) -> str:
    wc_df = dff.dropna(subset=["genre"]).copy()
    if wc_df.empty:
        return ""

    genre_counts = wc_df["genre"].value_counts().to_dict()
    if not genre_counts:
        return ""

    wordcloud = WordCloud(
        width=1200,
        height=550,
        background_color=None,
        mode="RGBA",
        colormap="viridis",
        prefer_horizontal=0.95
    ).generate_from_frequencies(genre_counts)

    img = wordcloud.to_image()
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

# =========================================================
# 3. APP
# =========================================================
app = Dash(__name__)
server = app.server

app.layout = html.Div(
    style={
        "background": BG,
        "minHeight": "100vh",
        "padding": "20px 24px 40px 24px",
        "fontFamily": "Arial, sans-serif"
    },
    children=[
        # Encabezado con palomitas
        html.Div(
            style={
                "maxWidth": "1280px",
                "margin": "0 auto 18px auto",
                "position": "relative",
                "textAlign": "center"
            },
            children=[
                html.Img(
                    src="/assets/maiz.png",
                    style={
                        "position": "absolute",
                        "left": "0px",
                        "top": "0px",
                        "height": "80px",
                        "opacity": "0.9"
                    }
                ),
                html.Img(
                    src="/assets/maiz.png",
                    style={
                        "position": "absolute",
                        "right": "0px",
                        "top": "0px",
                        "height": "80px",
                        "opacity": "0.9"
                    }
                ),
                html.H1(
                    "Inside the Movie Industry",
                    style={
                        "marginBottom": "6px",
                        "fontSize": "42px",
                        "color": TEXT
                    }
                ),
                html.P(
                    "Cómo se relacionan presupuesto, ingresos, rating y género en la industria del cine.",
                    style={
                        "margin": "0",
                        "fontSize": "18px",
                        "color": "#55627a"
                    }
                )
            ]
        ),

        # Marco cinta
        html.Div(
            style={
                "position": "relative",
                "maxWidth": "1280px",
                "margin": "0 auto",
                "backgroundImage": "url('/assets/cinta.png')",
                "backgroundSize": "100% 100%",
                "backgroundRepeat": "no-repeat",
                "backgroundPosition": "center",
                "padding": "92px 110px 88px 110px",
                "boxSizing": "border-box"
            },
            children=[
                # Filtros
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "1.2fr 1fr 1fr",
                        "gap": "12px",
                        "marginBottom": "14px"
                    },
                    children=[
                        html.Div(
                            style=panel_style,
                            children=[
                                html.Label("Géneros", style={"color": TEXT, "fontWeight": "600"}),
                                dcc.Dropdown(
                                    id="genre_filter",
                                    options=[{"label": g, "value": g} for g in genre_options],
                                    value=genre_options,
                                    multi=True,
                                    style={"marginTop": "6px"}
                                )
                            ]
                        ),
                        html.Div(
                            style=panel_style,
                            children=[
                                html.Label("Idiomas", style={"color": TEXT, "fontWeight": "600"}),
                                dcc.Dropdown(
                                    id="language_filter",
                                    options=[{"label": l, "value": l} for l in language_options],
                                    value=language_options,
                                    multi=True,
                                    style={"marginTop": "6px"}
                                )
                            ]
                        ),
                        html.Div(
                            style=panel_style,
                            children=[
                                html.Label("Rango de años", style={"color": TEXT, "fontWeight": "600"}),
                                dcc.RangeSlider(
                                    id="year_filter",
                                    min=min_year,
                                    max=max_year,
                                    step=1,
                                    value=[min_year, max_year],
                                    marks={
                                        y: str(y)
                                        for y in range(min_year, max_year + 1, max(1, (max_year - min_year) // 8))
                                    }
                                )
                            ]
                        ),
                    ]
                ),

                # KPIs
                html.Div(
                    id="kpi_row",
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(4, 1fr)",
                        "gap": "12px",
                        "marginBottom": "18px"
                    }
                ),

                # Sección 1
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "1.25fr 1fr",
                        "gap": "12px",
                        "marginBottom": "12px"
                    },
                    children=[
                        dcc.Graph(id="fig_budget_vs_rating", style={"height": "360px"}),
                        dcc.Graph(id="fig_movies_year", style={"height": "360px"})
                    ]
                ),

                # Sección 2
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "1fr 1fr",
                        "gap": "12px",
                        "marginBottom": "12px"
                    },
                    children=[
                        dcc.Graph(id="fig_budget_revenue_time", style={"height": "330px"}),

                        html.Div(
                            children=[
                                # título fuera del panel blanco
                                html.H3(
                                    "Generos más populares del cine",
                                    style={
                                        "margin": "0 0 10px 0",
                                        "textAlign": "center",
                                        "color": TEXT,
                                        "fontSize": "18px",
                                        "fontWeight": "400"
                                    }
                                ),

                                # panel solo para el wordcloud
                                html.Div(
                                    style={
                                        **panel_style,
                                        "height": "290px",
                                        "display": "flex",
                                        "flexDirection": "column",
                                        "justifyContent": "center"
                                    },
                                    children=[
                                        html.Img(
                                            id="fig_genres_wordcloud",
                                            style={
                                                "width": "100%",
                                                "height": "240px",
                                                "objectFit": "contain"
                                            }
                                        ),
                                        html.Div(
                                            id="fig_genres_wordcloud_empty",
                                            style={
                                                "display": "none",
                                                "textAlign": "center",
                                                "color": "#55627a",
                                                "fontSize": "15px",
                                                "marginTop": "8px"
                                            }
                                        )
                                    ]
                                )
                            ]
                        )
                    ]
                ),

                # Sección 3
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "1fr 1fr",
                        "gap": "12px",
                        "marginBottom": "12px"
                    },
                    children=[
                        dcc.Graph(id="fig_vote_genre", style={"height": "330px"}),
                        dcc.Graph(id="fig_country_map", style={"height": "330px"})
                    ]
                ),

                # Sección 4
                dcc.Graph(id="fig_dynamic_bubble", style={"height": "500px"})
            ]
        )
    ]
)

# =========================================================
# 4. CALLBACK
# =========================================================
@app.callback(
    Output("kpi_row", "children"),
    Output("fig_budget_vs_rating", "figure"),
    Output("fig_movies_year", "figure"),
    Output("fig_budget_revenue_time", "figure"),
    Output("fig_genres_wordcloud", "src"),
    Output("fig_genres_wordcloud_empty", "children"),
    Output("fig_genres_wordcloud_empty", "style"),
    Output("fig_vote_genre", "figure"),
    Output("fig_country_map", "figure"),
    Output("fig_dynamic_bubble", "figure"),
    Input("genre_filter", "value"),
    Input("language_filter", "value"),
    Input("year_filter", "value"),
)
def update_dashboard(selected_genres, selected_languages, year_range):
    dff = df.copy()

    if selected_genres:
        dff = dff[dff["genre"].isin(selected_genres)]

    if selected_languages and "spoken_languages" in dff.columns:
        dff = dff[dff["spoken_languages"].isin(selected_languages)]

    dff = dff[
        (dff["release_year"] >= year_range[0]) &
        (dff["release_year"] <= year_range[1])
    ].copy()

    # KPIs
    total_movies = dff["movieId"].nunique() if "movieId" in dff.columns else len(dff)
    avg_vote = dff["vote_average"].mean() if "vote_average" in dff.columns else np.nan
    avg_budget = dff.loc[dff["budget"] > 0, "budget"].mean() if "budget" in dff.columns else np.nan
    avg_revenue = dff.loc[dff["revenue"] > 0, "revenue"].mean() if "revenue" in dff.columns else np.nan

    kpis = [
        card("Total Películas", f"{total_movies:,}", "🎬"),
        card("Rating Promedio", f"{avg_vote:.2f}" if pd.notnull(avg_vote) else "N/A", "⭐"),
        card("Presupuesto Promedio", f"${avg_budget:,.0f}" if pd.notnull(avg_budget) else "N/A", "💰"),
        card("Ganancias Promedio", f"${avg_revenue:,.0f}" if pd.notnull(avg_revenue) else "N/A", "📈"),
    ]

    # 1. Budget vs rating
    scatter_df = dff.dropna(subset=["budget", "vote_average", "genre"]).copy()
    scatter_df = scatter_df[scatter_df["budget"] > 0]
    if not scatter_df.empty:
        scatter_df = scatter_df[scatter_df["budget"] < scatter_df["budget"].quantile(0.98)]

    if scatter_df.empty:
        fig_budget_vs_rating = empty_figure("Relación entre presupuesto y rating de películas")
    else:
        fig_budget_vs_rating = px.scatter(
            scatter_df,
            x="budget",
            y="vote_average",
            color="genre",
            hover_name=title_col if title_col else None,
            opacity=0.55
        )
        if len(scatter_df) >= 2:
            x = scatter_df["budget"].values
            y = scatter_df["vote_average"].values
            coef = np.polyfit(x, y, 1)
            trend = np.poly1d(coef)
            line_df = scatter_df.sort_values("budget")
            fig_budget_vs_rating.add_trace(
                go.Scatter(
                    x=line_df["budget"],
                    y=trend(line_df["budget"]),
                    mode="lines",
                    name="Tendencia global",
                    line=dict(color="black", width=3)
                )
            )
        fig_budget_vs_rating = style_fig(fig_budget_vs_rating, "Relación entre presupuesto y rating de películas", 360)
        fig_budget_vs_rating.update_layout(xaxis_title="Presupuesto", yaxis_title="Rating")

    # 2. Películas por año
    movies_year = (
        dff.groupby("release_year", as_index=False)
        .agg(n_movies=("movieId", "nunique"))
    )
    if movies_year.empty:
        fig_movies_year = empty_figure("Producción cinematográfica a lo largo del tiempo")
    else:
        fig_movies_year = px.line(movies_year, x="release_year", y="n_movies", markers=True)
        fig_movies_year = style_fig(fig_movies_year, "Producción cinematográfica a lo largo del tiempo", 360)
        fig_movies_year.update_layout(xaxis_title="Año", yaxis_title="Películas")

    # 3. Budget vs revenue por año
    time_df = (
        dff.groupby("release_year", as_index=False)
        .agg(avg_budget=("budget", "mean"), avg_revenue=("revenue", "mean"))
    )
    if time_df.empty:
        fig_budget_revenue_time = empty_figure("Optimización del presupuesto en el tiempo")
    else:
        time_long = time_df.melt(
            id_vars="release_year",
            value_vars=["avg_budget", "avg_revenue"],
            var_name="metric",
            value_name="value"
        )
        time_long["metric"] = time_long["metric"].replace({
            "avg_budget": "Presupuesto promedio",
            "avg_revenue": "Ganancias promedio"
        })
        fig_budget_revenue_time = px.line(
            time_long,
            x="release_year",
            y="value",
            color="metric",
            markers=True
        )
        fig_budget_revenue_time = style_fig(fig_budget_revenue_time, "Optimización del presupuesto en el tiempo", 330)
        fig_budget_revenue_time.update_layout(
            xaxis_title="Año",
            yaxis_title="Valor promedio",
            legend_title="",
            legend=dict(
                x=0.02,
                y=0.98,
                xanchor="left",
                yanchor="top",
                bgcolor="rgba(255,255,255,0.75)",
                bordercolor="rgba(31,42,68,0.08)",
                borderwidth=1
            ),
            margin=dict(l=20, r=20, t=60, b=25)
        )

    # 4. Word cloud de géneros
    wordcloud_src = make_wordcloud_src(dff)
    if wordcloud_src:
        wc_empty_text = ""
        wc_empty_style = {"display": "none"}
    else:
        wc_empty_text = "No hay datos para los filtros seleccionados"
        wc_empty_style = {
            "display": "block",
            "textAlign": "center",
            "color": "#55627a",
            "fontSize": "15px",
            "marginTop": "8px"
        }

    # 5. Top 10 géneros por promedio de votos
    vote_df = dff.dropna(subset=["vote_average", "genre"]).copy()
    if vote_df.empty:
        fig_vote_genre = empty_figure("Top 10 Géneros por Promedio de Votos")
    else:
        vote_by_genre = vote_df.groupby("genre", as_index=False)["vote_average"].mean()
        vote_by_genre = vote_by_genre.sort_values("vote_average", ascending=False).head(10)
        vote_by_genre = vote_by_genre.sort_values("vote_average", ascending=True)

        fig_vote_genre = px.bar(
            vote_by_genre,
            x="vote_average",
            y="genre",
            orientation="h",
            color="vote_average",
            color_continuous_scale="Viridis",
            text_auto=".2f"
        )
        fig_vote_genre = style_fig(fig_vote_genre, "Top 10 Géneros por Promedio de Votos", 330)
        fig_vote_genre.update_layout(
            xaxis_title="Promedio de votos",
            yaxis_title="Género",
            showlegend=False,
            margin=dict(l=150, r=20, t=60, b=25)
        )

    # 6. Mapa
    map_df = (
        dff.dropna(subset=["countries_name"])
        .groupby("countries_name", as_index=False)
        .agg(n_movies=("movieId", "nunique"))
    )
    if map_df.empty:
        fig_country_map = empty_figure("¿Quién domina la producción cinematográfica?")
    else:
        fig_country_map = px.choropleth(
            map_df,
            locations="countries_name",
            locationmode="country names",
            color="n_movies",
            hover_name="countries_name",
            color_continuous_scale="Plasma"
        )
        fig_country_map.update_layout(
            title="¿Quién domina la producción cinematográfica?",
            title_font=dict(size=18, color=TEXT),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT),
            geo=dict(
                showframe=False,
                showcoastlines=True,
                bgcolor="rgba(0,0,0,0)"
            ),
            margin=dict(l=20, r=20, t=60, b=20),
            height=330
        )

    # 7. Gráfico dinámico
    bubble_df = dff.dropna(subset=["release_year", "genre", "budget", "revenue"]).copy()
    bubble_df = bubble_df[(bubble_df["budget"] > 0) & (bubble_df["revenue"] > 0)]

    if bubble_df.empty:
        fig_dynamic_bubble = empty_figure("Evolución de las ganancias al aumentar el presupuesto")
    else:
        year_genre = (
            bubble_df.groupby(["release_year", "genre"], as_index=False)
            .agg(
                avg_revenue=("revenue", "mean"),
                avg_budget=("budget", "mean"),
                n_movies=("movieId", "nunique")
            )
            .sort_values(["release_year", "genre"])
        )
        year_genre["point_id"] = year_genre["genre"].astype(str) + "_" + year_genre["release_year"].astype(str)

        years = sorted(year_genre["release_year"].unique().tolist())
        frames = []
        for year in years:
            temp = year_genre[year_genre["release_year"] <= year].copy()
            temp["frame_year"] = year
            frames.append(temp)

        bubble_anim = pd.concat(frames, ignore_index=True)

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
            range_x=[year_range[0], 2020]
        )
        fig_dynamic_bubble = style_fig(fig_dynamic_bubble, "Evolución de las ganancias al aumentar el presupuesto", 500)
        fig_dynamic_bubble.update_layout(xaxis_title="Año", yaxis_title="Promedio de ganancias anuales")
        fig_dynamic_bubble.update_yaxes(range=[0, 550_000_000])

    return (
        kpis,
        fig_budget_vs_rating,
        fig_movies_year,
        fig_budget_revenue_time,
        wordcloud_src,
        wc_empty_text,
        wc_empty_style,
        fig_vote_genre,
        fig_country_map,
        fig_dynamic_bubble
    )

# =========================================================
# 5. RUN
# =========================================================
if __name__ == "__main__":
    app.run(debug=True)