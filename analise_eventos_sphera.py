from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Analise Sphera - Ontologia de Eventos",
    layout="wide",
    initial_sidebar_state="expanded",
)


APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data" / "Sphera.xlsx"
DEFAULT_PASSWORD = "cdshell"

SHEET_EVENTS = "Ontologia_Eventos"
SHEET_VARIABLES = "Variaveis_Utilidade"
SHEET_ANALYSES = "Analises"

EVENT_ID_COL = "Event ID"
DATE_COL = "Date Occurred"
YEAR_COL = "Year"
EVENT_TYPE_COL = "Event Type"
LOCATION_COL = "Location"
RAM_POTENTIAL_COL = "RAM Potential"
ONTOLOGY_COL = "Tipo ontológico"
SCENARIO_COL = "Cenário acidental"
BARRIER_COL = "Barreira crítica"
BARRIER_STATE_COL = "Estado da barreira"
ESCALATION_COL = "Mecanismo de escalada"
TASK_COL = "Task / Activity (consolidado)"
RISK_AREA_COL = "Risk Area (consolidado)"
HUMAN_FACTOR_COL = "Human Factors"
FPI_COL = "Potencial FPI/SIF - Pessoas"
FPI_DAMAGE_COL = "Tipo de dano FPI/SIF potencial"
CURATION_PRIORITY_COL = "Prioridade de curadoria"
CONFIDENCE_COL = "Confiança"
QUALITY_EVIDENCE_COL = "Qualidade da evidência"
QUALITY_DESCRIPTION_COL = "Qualidade da descrição Sphera"
DIRECT_EVIDENCE_COL = "Evidência textual direta?"
INFERENCE_COL = "Nível de inferência"
GAP_COL = "Observação sobre lacuna de dados"
CURATION_FLAG_COL = "Flag de curadoria FPI/SIF"
DOMAIN_COL = "Domínio de manifestação / precursor"
SCOPE_COL = "Escopo do precursor"
TITLE_COL = "Title"
DESCRIPTION_COL = "Description"
OBSERVED_EVENT_COL = "Evento observado"
EVIDENCE_COL = "Evidência"
POTENTIAL_SEVERITY_PEOPLE_COL = "Potential Severity - People"

CORE_COMPLETENESS_COLUMNS = [
    EVENT_ID_COL,
    DATE_COL,
    EVENT_TYPE_COL,
    LOCATION_COL,
    ONTOLOGY_COL,
    SCENARIO_COL,
    BARRIER_COL,
    BARRIER_STATE_COL,
    ESCALATION_COL,
    FPI_COL,
    DOMAIN_COL,
    TASK_COL,
    RISK_AREA_COL,
]

PAGE_OPTIONS = [
    "1. Visão Executiva da Ontologia",
    "2. FPI/SIF e Severidade Potencial",
    "3. Sinais Fracos, Precursores e Incidentes",
    "4. Barreiras Críticas e Estado das Barreiras",
    "5. Cenários, Mecanismos e Pareto 80/20",
    "6. Curadoria e Qualidade da Classificação",
    "7. Qualidade dos Dados",
    "Modelos preditivos - preparação",
    "Dados filtrados",
]


def get_configured_password() -> str:
    try:
        return st.secrets.get("APP_PASSWORD", DEFAULT_PASSWORD)
    except Exception:
        return DEFAULT_PASSWORD


def check_password() -> bool:
    st.sidebar.header("Área protegida")
    password = st.sidebar.text_input("Senha", type="password")
    if password == get_configured_password():
        return True
    if password:
        st.sidebar.error("Senha incorreta.")
    return False


def clean_series(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip()
    return cleaned.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "NaT": pd.NA})


def display_series(series: pd.Series, missing_label: str = "Não informado") -> pd.Series:
    return clean_series(series).fillna(missing_label)


def has_column(df: pd.DataFrame, column: str) -> bool:
    return column in df.columns


def distinct_events(df: pd.DataFrame) -> pd.DataFrame:
    if has_column(df, EVENT_ID_COL):
        return df.drop_duplicates(subset=[EVENT_ID_COL]).copy()
    return df.copy()


def contains_any(df: pd.DataFrame, column: str, terms: Iterable[str]) -> pd.Series:
    if not has_column(df, column):
        return pd.Series(False, index=df.index)
    escaped_terms = [re.escape(term.lower()) for term in terms]
    pattern = "|".join(escaped_terms)
    return clean_series(df[column]).str.lower().str.contains(pattern, na=False, regex=True)


def equals_any(df: pd.DataFrame, column: str, values: Iterable[str]) -> pd.Series:
    if not has_column(df, column):
        return pd.Series(False, index=df.index)
    normalized = {value.strip().lower() for value in values}
    return clean_series(df[column]).str.lower().isin(normalized).fillna(False)


def missing_like(series: pd.Series) -> pd.Series:
    cleaned = clean_series(series)
    lowered = cleaned.str.lower()
    missing_tokens = {
        "não informado",
        "nao informado",
        "não avaliado",
        "nao avaliado",
        "não identificada",
        "nao identificada",
        "não identificado",
        "nao identificado",
        "sem classificação",
        "sem classificacao",
        "sem dados",
    }
    return (
        cleaned.isna()
        | lowered.isin(missing_tokens).fillna(False)
        | lowered.str.startswith("não identificad", na=False)
        | lowered.str.startswith("nao identificad", na=False)
    )


def fmt_int(value: int | float) -> str:
    try:
        return f"{int(value):,}".replace(",", ".")
    except Exception:
        return "0"


def fmt_pct(value: float) -> str:
    if pd.isna(value):
        return "0,0%"
    return f"{value * 100:.1f}%".replace(".", ",")


def fmt_ratio(numerator: float, denominator: float) -> str:
    if not denominator:
        return "n/d"
    return f"{numerator / denominator:.1f}x".replace(".", ",")


def shorten(value: object, limit: int = 58) -> str:
    text = "" if pd.isna(value) else str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


@st.cache_data(show_spinner="Carregando planilha Sphera...")
def load_workbook_from_path(path: str, modified_at: float) -> dict[str, pd.DataFrame]:
    del modified_at
    return load_workbook(BytesIO(Path(path).read_bytes()))


@st.cache_data(show_spinner="Carregando planilha enviada...")
def load_workbook_from_bytes(content: bytes) -> dict[str, pd.DataFrame]:
    return load_workbook(BytesIO(content))


def load_workbook(source: BytesIO) -> dict[str, pd.DataFrame]:
    sheets = pd.read_excel(source, sheet_name=None, engine="openpyxl")
    if SHEET_EVENTS not in sheets:
        available = ", ".join(sheets.keys())
        raise ValueError(f"Aba obrigatória '{SHEET_EVENTS}' não encontrada. Abas disponíveis: {available}")

    events = prepare_events(sheets[SHEET_EVENTS])
    variables = sheets.get(SHEET_VARIABLES, pd.DataFrame())
    analyses = sheets.get(SHEET_ANALYSES, pd.DataFrame())

    return {
        "events": events,
        "variables": variables,
        "analyses": analyses,
        "pareto": sheets.get("Pareto", pd.DataFrame()),
        "crossings": sheets.get("Cruzamentos", pd.DataFrame()),
    }


def prepare_events(df: pd.DataFrame) -> pd.DataFrame:
    events = df.copy()
    events.columns = [str(column).strip() for column in events.columns]

    if has_column(events, DATE_COL):
        events[DATE_COL] = pd.to_datetime(events[DATE_COL], errors="coerce")
        events["Ano-Mês"] = events[DATE_COL].dt.to_period("M").astype("string")
        events["Trimestre"] = events[DATE_COL].dt.to_period("Q").astype("string")

    if has_column(events, YEAR_COL):
        events[YEAR_COL] = pd.to_numeric(events[YEAR_COL], errors="coerce").astype("Int64")
    elif has_column(events, DATE_COL):
        events[YEAR_COL] = events[DATE_COL].dt.year.astype("Int64")

    text_columns = events.select_dtypes(include=["object", "string"]).columns
    for column in text_columns:
        events[column] = clean_series(events[column])

    return events


def value_counts_df(
    df: pd.DataFrame,
    column: str,
    top: int | None = None,
    missing_label: str = "Não informado",
) -> pd.DataFrame:
    if not has_column(df, column):
        return pd.DataFrame(columns=[column, "Quantidade", "%"])
    base = distinct_events(df)
    counts = display_series(base[column], missing_label).value_counts().reset_index()
    counts.columns = [column, "Quantidade"]
    total = counts["Quantidade"].sum()
    counts["%"] = counts["Quantidade"] / total if total else 0
    if top:
        counts = counts.head(top)
    return counts


def pareto_df(df: pd.DataFrame, column: str) -> pd.DataFrame:
    counts = value_counts_df(df, column)
    if counts.empty:
        return counts
    counts["% acumulado"] = counts["%"].cumsum()
    counts["Pareto 80?"] = counts["% acumulado"].le(0.8)
    if not counts["Pareto 80?"].any() and len(counts) > 0:
        counts.loc[counts.index[0], "Pareto 80?"] = True
    return counts


def percent_top_n(df: pd.DataFrame, column: str, n: int) -> float:
    counts = value_counts_df(df, column)
    total = counts["Quantidade"].sum()
    if not total:
        return 0.0
    return counts.head(n)["Quantidade"].sum() / total


def metric_cards(cards: list[tuple[str, str, str | None]], columns: int = 4) -> None:
    for start in range(0, len(cards), columns):
        row = cards[start : start + columns]
        cols = st.columns(columns)
        for index, card in enumerate(row):
            label, value, delta = card
            cols[index].metric(label, value, delta=delta)


def bar_chart(
    data: pd.DataFrame,
    category_col: str,
    title: str,
    top: int | None = None,
    color: str = "#2A6F97",
) -> None:
    chart_data = data.head(top).copy() if top else data.copy()
    if chart_data.empty:
        st.info(f"Sem dados para {title}.")
        return
    chart_data["Categoria curta"] = chart_data[category_col].map(shorten)
    chart_data = chart_data.sort_values("Quantidade", ascending=True)
    height = max(360, min(760, 90 + 34 * len(chart_data)))
    fig = px.bar(
        chart_data,
        x="Quantidade",
        y="Categoria curta",
        orientation="h",
        text="Quantidade",
        hover_data={category_col: True, "%": ":.1%"},
        color_discrete_sequence=[color],
        title=title,
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=40, t=55, b=10),
        xaxis_title="Eventos",
        yaxis_title="",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def donut_chart(data: pd.DataFrame, category_col: str, title: str) -> None:
    if data.empty:
        st.info(f"Sem dados para {title}.")
        return
    fig = px.pie(
        data,
        names=category_col,
        values="Quantidade",
        hole=0.48,
        title=title,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=55, b=10))
    st.plotly_chart(fig, use_container_width=True)


def line_by_month(df: pd.DataFrame, category_col: str, title: str, top_categories: int = 6) -> None:
    if not has_column(df, "Ano-Mês") or not has_column(df, category_col):
        st.info(f"Sem dados para {title}.")
        return
    base = distinct_events(df).dropna(subset=["Ano-Mês"])
    if base.empty:
        st.info(f"Sem dados para {title}.")
        return

    top_values = display_series(base[category_col]).value_counts().head(top_categories).index
    working = base[base[category_col].isin(top_values)].copy()
    grouped = (
        working.groupby(["Ano-Mês", category_col], observed=True)[EVENT_ID_COL]
        .nunique()
        .reset_index(name="Quantidade")
        if has_column(working, EVENT_ID_COL)
        else working.groupby(["Ano-Mês", category_col], observed=True).size().reset_index(name="Quantidade")
    )
    fig = px.line(
        grouped,
        x="Ano-Mês",
        y="Quantidade",
        color=category_col,
        markers=True,
        title=title,
        color_discrete_sequence=px.colors.qualitative.Dark24,
    )
    fig.update_layout(height=460, margin=dict(l=10, r=10, t=55, b=10), xaxis_title="", yaxis_title="Eventos")
    st.plotly_chart(fig, use_container_width=True)


def heatmap(df: pd.DataFrame, row_col: str, col_col: str, title: str, top_rows: int = 12, top_cols: int = 8) -> None:
    if not has_column(df, row_col) or not has_column(df, col_col):
        st.info(f"Sem dados para {title}.")
        return
    base = distinct_events(df)
    working = pd.DataFrame(
        {
            row_col: display_series(base[row_col]),
            col_col: display_series(base[col_col]),
        }
    )
    table = pd.crosstab(working[row_col], working[col_col])
    if table.empty:
        st.info(f"Sem dados para {title}.")
        return
    top_row_index = table.sum(axis=1).sort_values(ascending=False).head(top_rows).index
    top_col_index = table.sum(axis=0).sort_values(ascending=False).head(top_cols).index
    table = table.loc[top_row_index, top_col_index]
    fig = px.imshow(
        table,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="YlGnBu",
        title=title,
    )
    fig.update_layout(
        height=max(420, min(760, 120 + 36 * len(table))),
        margin=dict(l=10, r=10, t=55, b=10),
        xaxis_title=col_col,
        yaxis_title=row_col,
    )
    st.plotly_chart(fig, use_container_width=True)


def pareto_chart(data: pd.DataFrame, category_col: str, title: str, top: int = 15) -> None:
    chart_data = data.head(top).copy()
    if chart_data.empty:
        st.info(f"Sem dados para {title}.")
        return
    chart_data["Categoria curta"] = chart_data[category_col].map(shorten)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=chart_data["Categoria curta"],
            y=chart_data["Quantidade"],
            name="Eventos",
            marker_color="#2A6F97",
            customdata=chart_data[[category_col, "%"]],
            hovertemplate="<b>%{customdata[0]}</b><br>Eventos: %{y}<br>%: %{customdata[1]:.1%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=chart_data["Categoria curta"],
            y=chart_data["% acumulado"] * 100,
            name="% acumulado",
            mode="lines+markers",
            yaxis="y2",
            line=dict(color="#C44536", width=3),
            hovertemplate="% acumulado: %{y:.1f}%<extra></extra>",
        )
    )
    fig.add_hline(y=80, line_dash="dash", line_color="#C44536", yref="y2")
    fig.update_layout(
        title=title,
        height=560,
        margin=dict(l=10, r=30, t=55, b=120),
        xaxis_tickangle=-35,
        yaxis=dict(title="Eventos"),
        yaxis2=dict(title="% acumulado", overlaying="y", side="right", range=[0, 105]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df.copy()

    with st.sidebar.expander("Filtros", expanded=True):
        if has_column(filtered, DATE_COL):
            valid_dates = filtered[DATE_COL].dropna()
            if not valid_dates.empty:
                min_date = valid_dates.min().date()
                max_date = valid_dates.max().date()
                selected_period = st.date_input(
                    "Período",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                )
                if isinstance(selected_period, tuple) and len(selected_period) == 2:
                    start_date, end_date = selected_period
                    filtered = filtered[
                        (filtered[DATE_COL].dt.date >= start_date)
                        & (filtered[DATE_COL].dt.date <= end_date)
                    ]

        for label, column in [
            ("Localidade", LOCATION_COL),
            ("Tipo de evento", EVENT_TYPE_COL),
            ("Tipo ontológico", ONTOLOGY_COL),
            ("Potencial FPI/SIF", FPI_COL),
            ("Área de risco", RISK_AREA_COL),
            ("Cenário acidental", SCENARIO_COL),
        ]:
            if not has_column(filtered, column):
                continue
            options = sorted(display_series(df[column]).dropna().unique(), key=lambda value: str(value).lower())
            selected = st.multiselect(label, options, default=[])
            if selected:
                filtered = filtered[display_series(filtered[column]).isin(selected)]

        search_text = st.text_input("Busca textual")
        if search_text:
            mask = pd.Series(False, index=filtered.index)
            for column in [TITLE_COL, DESCRIPTION_COL, OBSERVED_EVENT_COL, EVIDENCE_COL]:
                if has_column(filtered, column):
                    mask = mask | clean_series(filtered[column]).str.contains(search_text, case=False, na=False)
            filtered = filtered[mask]

    return filtered


def render_sidebar_source() -> bytes | None:
    st.sidebar.header("Fonte de dados")
    uploaded = st.sidebar.file_uploader("Planilha Sphera (.xlsx)", type=["xlsx"])
    if st.sidebar.button("Atualizar cache"):
        st.cache_data.clear()
        st.rerun()
    if uploaded is not None:
        st.sidebar.caption(f"Arquivo enviado: {uploaded.name}")
        return uploaded.getvalue()
    if DATA_PATH.exists():
        st.sidebar.caption(f"Arquivo local: {DATA_PATH.name}")
    return None


def load_data(uploaded_content: bytes | None) -> dict[str, pd.DataFrame]:
    if uploaded_content is not None:
        return load_workbook_from_bytes(uploaded_content)
    if not DATA_PATH.exists():
        st.error(f"Arquivo de dados não encontrado: {DATA_PATH}")
        st.stop()
    return load_workbook_from_path(str(DATA_PATH), DATA_PATH.stat().st_mtime)


def page_header(df: pd.DataFrame) -> None:
    total_events = distinct_events(df)[EVENT_ID_COL].nunique() if has_column(df, EVENT_ID_COL) else len(df)
    if has_column(df, DATE_COL) and df[DATE_COL].notna().any():
        min_date = df[DATE_COL].min().date()
        max_date = df[DATE_COL].max().date()
        period = f"{min_date:%d/%m/%Y} a {max_date:%d/%m/%Y}"
    else:
        period = "Sem data válida"
    st.title("Análise de Eventos Sphera")
    st.caption(f"Base filtrada: {fmt_int(total_events)} eventos | Período: {period}")


def dashboard_executive(df: pd.DataFrame) -> None:
    st.header("1. Visão Executiva da Ontologia")
    base = distinct_events(df)
    total = len(base)
    weak = int(contains_any(base, ONTOLOGY_COL, ["sinal fraco"]).sum())
    precursors = int(contains_any(base, ONTOLOGY_COL, ["precursor"]).sum())
    incidents = int(contains_any(base, ONTOLOGY_COL, ["incidente realizado"]).sum())
    fpi_yes = int(equals_any(base, FPI_COL, ["Sim"]).sum())
    fpi_possible = int(contains_any(base, FPI_COL, ["possível"]).sum())
    high_curation = int(equals_any(base, CURATION_PRIORITY_COL, ["Alta"]).sum())
    degraded_pct = contains_any(
        base,
        BARRIER_STATE_COL,
        ["degradada", "falhou", "bypassada", "vencida", "potencialmente degradada"],
    ).mean()

    metric_cards(
        [
            ("Eventos analisados", fmt_int(total), None),
            ("Sinais fracos", fmt_int(weak), fmt_pct(weak / total) if total else None),
            ("Precursores", fmt_int(precursors), fmt_pct(precursors / total) if total else None),
            ("Incidentes realizados", fmt_int(incidents), fmt_pct(incidents / total) if total else None),
            ("FPI/SIF Sim", fmt_int(fpi_yes), fmt_pct(fpi_yes / total) if total else None),
            ("FPI/SIF Possível", fmt_int(fpi_possible), fmt_pct(fpi_possible / total) if total else None),
            ("Curadoria alta", fmt_int(high_curation), fmt_pct(high_curation / total) if total else None),
            ("Barreiras degradadas", fmt_pct(degraded_pct), None),
        ]
    )

    col1, col2 = st.columns([1.1, 0.9])
    with col1:
        bar_chart(value_counts_df(base, ONTOLOGY_COL), ONTOLOGY_COL, "Distribuição por tipo ontológico", color="#2A6F97")
    with col2:
        donut_chart(value_counts_df(base, FPI_COL), FPI_COL, "Potencial FPI/SIF - Pessoas")

    line_by_month(base, ONTOLOGY_COL, "Evolução mensal por tipo ontológico")

    col1, col2, col3 = st.columns(3)
    with col1:
        bar_chart(value_counts_df(base, SCENARIO_COL, top=10), SCENARIO_COL, "Top cenários acidentais", color="#5E8C61")
    with col2:
        bar_chart(value_counts_df(base, BARRIER_COL, top=10), BARRIER_COL, "Top barreiras críticas", color="#B08968")
    with col3:
        bar_chart(value_counts_df(base, LOCATION_COL, top=10), LOCATION_COL, "Eventos por localidade", color="#6D597A")


def dashboard_fpi(df: pd.DataFrame) -> None:
    st.header("2. FPI/SIF e Severidade Potencial")
    base = distinct_events(df)
    total = len(base)
    fpi_counts = value_counts_df(base, FPI_COL)
    fpi_yes = int(equals_any(base, FPI_COL, ["Sim"]).sum())
    fpi_possible = int(equals_any(base, FPI_COL, ["Possível", "Possível/Conflito"]).sum())
    fpi_no = int(equals_any(base, FPI_COL, ["Não"]).sum())
    fpi_not_evaluated = int(equals_any(base, FPI_COL, ["Não avaliado"]).sum())

    metric_cards(
        [
            ("FPI/SIF Sim", fmt_int(fpi_yes), fmt_pct(fpi_yes / total) if total else None),
            ("FPI/SIF Possível", fmt_int(fpi_possible), fmt_pct(fpi_possible / total) if total else None),
            ("FPI/SIF Não", fmt_int(fpi_no), fmt_pct(fpi_no / total) if total else None),
            ("Não avaliado", fmt_int(fpi_not_evaluated), fmt_pct(fpi_not_evaluated / total) if total else None),
        ]
    )

    col1, col2 = st.columns(2)
    with col1:
        donut_chart(fpi_counts, FPI_COL, "Distribuição FPI/SIF")
    with col2:
        bar_chart(
            value_counts_df(base, POTENTIAL_SEVERITY_PEOPLE_COL, top=12),
            POTENTIAL_SEVERITY_PEOPLE_COL,
            "Severidade potencial - pessoas",
            color="#C44536",
        )

    fpi_focus = base[equals_any(base, FPI_COL, ["Sim", "Possível", "Possível/Conflito"])]
    col1, col2 = st.columns([1, 1])
    with col1:
        bar_chart(value_counts_df(fpi_focus, SCENARIO_COL, top=12), SCENARIO_COL, "Top cenários FPI/SIF", color="#2F7D6D")
    with col2:
        bar_chart(value_counts_df(fpi_focus, FPI_DAMAGE_COL, top=12), FPI_DAMAGE_COL, "Tipo de dano FPI/SIF potencial", color="#9A6B4F")

    heatmap(base, SCENARIO_COL, FPI_COL, "Cenário acidental x Potencial FPI/SIF", top_rows=14, top_cols=6)


def dashboard_progression(df: pd.DataFrame) -> None:
    st.header("3. Sinais Fracos, Precursores e Incidentes")
    base = distinct_events(df)
    weak = int(contains_any(base, ONTOLOGY_COL, ["sinal fraco"]).sum())
    operational_precursors = int(contains_any(base, ONTOLOGY_COL, ["precursor operacional"]).sum())
    systemic_precursors = int(contains_any(base, ONTOLOGY_COL, ["precursor sistêmico", "precursor sistemico"]).sum())
    incidents = int(contains_any(base, ONTOLOGY_COL, ["incidente realizado"]).sum())
    precursors = operational_precursors + systemic_precursors + int(contains_any(base, ONTOLOGY_COL, ["precursor de barreira"]).sum())

    metric_cards(
        [
            ("Sinais fracos", fmt_int(weak), None),
            ("Precursores operacionais", fmt_int(operational_precursors), None),
            ("Precursores sistêmicos", fmt_int(systemic_precursors), None),
            ("Incidentes realizados", fmt_int(incidents), None),
            ("Razão precursores/incidentes", fmt_ratio(precursors, incidents), None),
            ("Razão sinais fracos/precursores", fmt_ratio(weak, precursors), None),
        ],
        columns=3,
    )

    line_by_month(base, ONTOLOGY_COL, "Evolução mensal da cadeia ontológica", top_categories=8)

    col1, col2 = st.columns(2)
    with col1:
        bar_chart(value_counts_df(base, ONTOLOGY_COL), ONTOLOGY_COL, "Distribuição ontológica", color="#2A6F97")
    with col2:
        heatmap(base, ONTOLOGY_COL, EVENT_TYPE_COL, "Tipo ontológico x tipo de evento", top_rows=8, top_cols=5)


def dashboard_barriers(df: pd.DataFrame) -> None:
    st.header("4. Barreiras Críticas e Estado das Barreiras")
    base = distinct_events(df)
    total = len(base)
    absent = int(contains_any(base, BARRIER_STATE_COL, ["ausente", "insuficiente"]).sum())
    degraded = int(contains_any(base, BARRIER_STATE_COL, ["degradada", "falhou", "bypassada", "vencida"]).sum())
    demanded = int(contains_any(base, BARRIER_STATE_COL, ["demandada", "atuada"]).sum())
    worked = int(contains_any(base, BARRIER_STATE_COL, ["funcionou", "capacidade resiliente"]).sum())

    metric_cards(
        [
            ("Barreiras ausentes", fmt_int(absent), fmt_pct(absent / total) if total else None),
            ("Barreiras degradadas", fmt_int(degraded), fmt_pct(degraded / total) if total else None),
            ("Barreiras demandadas", fmt_int(demanded), fmt_pct(demanded / total) if total else None),
            ("Barreiras que funcionaram", fmt_int(worked), fmt_pct(worked / total) if total else None),
        ]
    )

    col1, col2 = st.columns([1.15, 0.85])
    with col1:
        bar_chart(value_counts_df(base, BARRIER_COL, top=10), BARRIER_COL, "Top 10 barreiras críticas", color="#B08968")
    with col2:
        donut_chart(value_counts_df(base, BARRIER_STATE_COL), BARRIER_STATE_COL, "Estado das barreiras")

    heatmap(base, BARRIER_COL, BARRIER_STATE_COL, "Barreira crítica x estado da barreira", top_rows=12, top_cols=8)


def dashboard_pareto(df: pd.DataFrame) -> None:
    st.header("5. Cenários, Mecanismos e Pareto 80/20")
    base = distinct_events(df)
    scenario_counts = value_counts_df(base, SCENARIO_COL)
    mechanism_counts = value_counts_df(base, ESCALATION_COL)
    domain_counts = value_counts_df(base, DOMAIN_COL)

    top_scenario = scenario_counts.iloc[0][SCENARIO_COL] if not scenario_counts.empty else "n/d"
    top_mechanism = mechanism_counts.iloc[0][ESCALATION_COL] if not mechanism_counts.empty else "n/d"
    top_domain = domain_counts.iloc[0][DOMAIN_COL] if not domain_counts.empty else "n/d"

    metric_cards(
        [
            ("Top cenário acidental", shorten(top_scenario, 44), None),
            ("Top mecanismo de escalada", shorten(top_mechanism, 44), None),
            ("Top domínio", shorten(top_domain, 44), None),
            ("% nos 5 principais cenários", fmt_pct(percent_top_n(base, SCENARIO_COL, 5)), None),
            ("% nos 10 principais cenários", fmt_pct(percent_top_n(base, SCENARIO_COL, 10)), None),
        ],
        columns=3,
    )

    tab1, tab2, tab3, tab4 = st.tabs(["Cenários", "Mecanismos", "Domínios", "Tasks"])
    with tab1:
        pareto_chart(pareto_df(base, SCENARIO_COL), SCENARIO_COL, "Pareto de cenários acidentais")
    with tab2:
        pareto_chart(pareto_df(base, ESCALATION_COL), ESCALATION_COL, "Pareto de mecanismos de escalada")
    with tab3:
        pareto_chart(pareto_df(base, DOMAIN_COL), DOMAIN_COL, "Pareto de domínios de manifestação")
    with tab4:
        pareto_chart(pareto_df(base, TASK_COL), TASK_COL, "Pareto de Task / Activity")


def curation_review_mask(df: pd.DataFrame) -> pd.Series:
    high_priority = equals_any(df, CURATION_PRIORITY_COL, ["Alta"])
    low_confidence = equals_any(df, CONFIDENCE_COL, ["Baixa"])
    high_inference = contains_any(df, INFERENCE_COL, ["alta inferência", "alta inferencia"])
    no_direct_evidence = equals_any(df, DIRECT_EVIDENCE_COL, ["Não"])
    fpi_conflict = contains_any(df, CURATION_FLAG_COL, ["revisar", "conflit"])
    has_gap = contains_any(df, GAP_COL, ["revisar", "faltam", "pouco claros"])
    return high_priority | low_confidence | high_inference | no_direct_evidence | fpi_conflict | has_gap


def dashboard_curation(df: pd.DataFrame) -> None:
    st.header("6. Curadoria e Qualidade da Classificação")
    base = distinct_events(df)
    total = len(base)
    high_priority = int(equals_any(base, CURATION_PRIORITY_COL, ["Alta"]).sum())
    low_confidence = int(equals_any(base, CONFIDENCE_COL, ["Baixa"]).sum())
    high_inference = int(contains_any(base, INFERENCE_COL, ["alta inferência", "alta inferencia"]).sum())
    no_direct_evidence = int(equals_any(base, DIRECT_EVIDENCE_COL, ["Não"]).sum())
    fpi_conflict = int(contains_any(base, CURATION_FLAG_COL, ["revisar", "conflit"]).sum())
    gaps = int(contains_any(base, GAP_COL, ["revisar", "faltam", "pouco claros"]).sum())
    review_mask = curation_review_mask(base)

    metric_cards(
        [
            ("Prioridade alta", fmt_int(high_priority), fmt_pct(high_priority / total) if total else None),
            ("Baixa confiança", fmt_int(low_confidence), fmt_pct(low_confidence / total) if total else None),
            ("Alta inferência", fmt_int(high_inference), fmt_pct(high_inference / total) if total else None),
            ("Sem evidência direta", fmt_int(no_direct_evidence), fmt_pct(no_direct_evidence / total) if total else None),
            ("FPI/SIF com revisão/conflito", fmt_int(fpi_conflict), fmt_pct(fpi_conflict / total) if total else None),
            ("Com lacunas de dados", fmt_int(gaps), fmt_pct(gaps / total) if total else None),
            ("Exigem revisão especializada", fmt_int(int(review_mask.sum())), fmt_pct(review_mask.mean()) if total else None),
        ],
        columns=4,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        donut_chart(value_counts_df(base, CURATION_PRIORITY_COL), CURATION_PRIORITY_COL, "Prioridade de curadoria")
    with col2:
        donut_chart(value_counts_df(base, CONFIDENCE_COL), CONFIDENCE_COL, "Confiança")
    with col3:
        bar_chart(value_counts_df(base, QUALITY_EVIDENCE_COL, top=8), QUALITY_EVIDENCE_COL, "Qualidade da evidência", color="#5E8C61")

    col1, col2 = st.columns(2)
    with col1:
        bar_chart(value_counts_df(base, INFERENCE_COL), INFERENCE_COL, "Nível de inferência", color="#6D597A")
    with col2:
        bar_chart(value_counts_df(base, GAP_COL), GAP_COL, "Lacunas de dados", color="#B08968")

    review_cols = [
        EVENT_ID_COL,
        DATE_COL,
        LOCATION_COL,
        EVENT_TYPE_COL,
        ONTOLOGY_COL,
        SCENARIO_COL,
        FPI_COL,
        CURATION_PRIORITY_COL,
        CONFIDENCE_COL,
        INFERENCE_COL,
        DIRECT_EVIDENCE_COL,
        GAP_COL,
    ]
    available_cols = [column for column in review_cols if has_column(base, column)]
    st.subheader("Eventos prioritários para curadoria")
    st.dataframe(base.loc[review_mask, available_cols].head(200), use_container_width=True, hide_index=True)


def completeness_table(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        if not has_column(df, column):
            continue
        missing = int(missing_like(df[column]).sum())
        total = len(df)
        rows.append(
            {
                "Variável": column,
                "Completude": 1 - (missing / total if total else 0),
                "Nulos/não informados": missing,
                "Valores distintos": int(display_series(df[column]).nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows).sort_values("Completude", ascending=True)


def dashboard_data_quality(df: pd.DataFrame, variables: pd.DataFrame) -> None:
    st.header("7. Qualidade dos Dados")
    base = distinct_events(df)
    completeness = completeness_table(base, CORE_COMPLETENESS_COLUMNS)
    avg_completeness = completeness["Completude"].mean() if not completeness.empty else 0
    critical_vars = int((completeness["Completude"] < 0.7).sum()) if not completeness.empty else 0
    insufficient_evidence = int(contains_any(base, QUALITY_EVIDENCE_COL, ["baixa", "insuficiente"]).sum())
    low_description = int(contains_any(base, QUALITY_DESCRIPTION_COL, ["baixa"]).sum())
    no_fpi = int(missing_like(base[FPI_COL]).sum()) if has_column(base, FPI_COL) else 0
    no_scenario = int(missing_like(base[SCENARIO_COL]).sum()) if has_column(base, SCENARIO_COL) else 0
    no_barrier = int(missing_like(base[BARRIER_COL]).sum()) if has_column(base, BARRIER_COL) else 0
    no_escalation = int(missing_like(base[ESCALATION_COL]).sum()) if has_column(base, ESCALATION_COL) else 0

    metric_cards(
        [
            ("Completude média", fmt_pct(avg_completeness), None),
            ("Variáveis críticas (<70%)", fmt_int(critical_vars), None),
            ("Evidência insuficiente", fmt_int(insufficient_evidence), None),
            ("Baixa qualidade de descrição", fmt_int(low_description), None),
            ("Sem FPI/SIF classificado", fmt_int(no_fpi), None),
            ("Sem cenário acidental", fmt_int(no_scenario), None),
            ("Sem barreira crítica", fmt_int(no_barrier), None),
            ("Sem mecanismo de escalada", fmt_int(no_escalation), None),
        ]
    )

    if not completeness.empty:
        chart_data = completeness.copy()
        chart_data["Completude %"] = chart_data["Completude"] * 100
        chart_data["Variável curta"] = chart_data["Variável"].map(lambda value: shorten(value, 42))
        fig = px.bar(
            chart_data,
            x="Completude %",
            y="Variável curta",
            orientation="h",
            text=chart_data["Completude"].map(fmt_pct),
            hover_data={"Variável": True, "Nulos/não informados": True, "Valores distintos": True},
            title="Completude das variáveis principais",
            color="Completude %",
            color_continuous_scale="Teal",
            range_x=[0, 100],
        )
        fig.update_layout(height=max(420, 80 + 34 * len(chart_data)), margin=dict(l=10, r=20, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(completeness, use_container_width=True, hide_index=True)

    if not variables.empty:
        st.subheader("Metadados da aba Variaveis_Utilidade")
        selected_cols = [
            "Atributo",
            "Qtde. nulos/não informados",
            "% Completude",
            "Pode ser variável-alvo?",
            "Pode ser variável preditora?",
            "Uso em Dashboard",
            "Recomendação final de uso",
        ]
        available = [column for column in selected_cols if column in variables.columns]
        st.dataframe(variables[available], use_container_width=True, hide_index=True)


def model_readiness(df: pd.DataFrame, target: str) -> dict[str, object]:
    if not has_column(df, target):
        return {
            "Alvo": target,
            "Registros úteis": 0,
            "Completude alvo": 0.0,
            "Classes": 0,
            "Maior classe": "n/d",
            "% maior classe": 0.0,
            "Status": "Coluna-alvo ausente",
        }
    base = distinct_events(df)
    valid = base[~missing_like(base[target])].copy()
    counts = display_series(valid[target]).value_counts()
    useful = int(len(valid))
    classes = int(counts.shape[0])
    top_class = str(counts.index[0]) if not counts.empty else "n/d"
    top_pct = float(counts.iloc[0] / counts.sum()) if not counts.empty and counts.sum() else 0.0
    completeness = useful / len(base) if len(base) else 0.0
    if useful < 200 or classes < 2:
        status = "Requer revisão do alvo"
    elif top_pct >= 0.8:
        status = "Viável com balanceamento"
    else:
        status = "Viável para baseline"
    return {
        "Alvo": target,
        "Registros úteis": useful,
        "Completude alvo": completeness,
        "Classes": classes,
        "Maior classe": shorten(top_class, 40),
        "% maior classe": top_pct,
        "Status": status,
    }


def dashboard_models(df: pd.DataFrame) -> None:
    st.header("Modelos preditivos - preparação")
    models = [
        ("Prever Potencial FPI/SIF", FPI_COL),
        ("Classificar Tipo Ontológico", ONTOLOGY_COL),
        ("Prever Cenário Acidental", SCENARIO_COL),
        ("Prever Barreira Crítica", BARRIER_COL),
        ("Prever Incidente Futuro", EVENT_TYPE_COL),
        ("Prever Potential Severity - Pessoas", POTENTIAL_SEVERITY_PEOPLE_COL),
        ("Prever Tipo de Dano FPI/SIF", FPI_DAMAGE_COL),
        ("Prever Escopo de Risco", SCOPE_COL),
    ]
    rows = []
    for model, target in models:
        readiness = model_readiness(df, target)
        rows.append({"Modelo": model, **readiness})
    readiness_df = pd.DataFrame(rows)

    display = readiness_df.copy()
    display["Completude alvo"] = display["Completude alvo"].map(fmt_pct)
    display["% maior classe"] = display["% maior classe"].map(fmt_pct)
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.subheader("Variáveis candidatas a preditoras tabulares")
    candidate_cols = [
        DATE_COL,
        YEAR_COL,
        EVENT_TYPE_COL,
        LOCATION_COL,
        RAM_POTENTIAL_COL,
        TASK_COL,
        RISK_AREA_COL,
        HUMAN_FACTOR_COL,
        SCENARIO_COL,
        BARRIER_COL,
        BARRIER_STATE_COL,
        ESCALATION_COL,
        DOMAIN_COL,
        QUALITY_DESCRIPTION_COL,
    ]
    candidate_cols = [column for column in candidate_cols if has_column(df, column)]
    st.dataframe(completeness_table(distinct_events(df), candidate_cols), use_container_width=True, hide_index=True)

    st.subheader("Campos textuais para modelos NLP")
    text_cols = [column for column in [TITLE_COL, DESCRIPTION_COL, OBSERVED_EVENT_COL, EVIDENCE_COL] if has_column(df, column)]
    text_summary = []
    base = distinct_events(df)
    for column in text_cols:
        lengths = clean_series(base[column]).fillna("").str.len()
        text_summary.append(
            {
                "Campo": column,
                "Completude": fmt_pct((lengths > 0).mean()),
                "Tamanho médio": round(float(lengths.mean()), 1),
                "Tamanho mediano": round(float(lengths.median()), 1),
            }
        )
    st.dataframe(pd.DataFrame(text_summary), use_container_width=True, hide_index=True)


def dashboard_filtered_data(df: pd.DataFrame) -> None:
    st.header("Dados filtrados")
    base = distinct_events(df)
    st.dataframe(base, use_container_width=True, hide_index=True)


def render_page(page: str, df: pd.DataFrame, variables: pd.DataFrame) -> None:
    if page == "1. Visão Executiva da Ontologia":
        dashboard_executive(df)
    elif page == "2. FPI/SIF e Severidade Potencial":
        dashboard_fpi(df)
    elif page == "3. Sinais Fracos, Precursores e Incidentes":
        dashboard_progression(df)
    elif page == "4. Barreiras Críticas e Estado das Barreiras":
        dashboard_barriers(df)
    elif page == "5. Cenários, Mecanismos e Pareto 80/20":
        dashboard_pareto(df)
    elif page == "6. Curadoria e Qualidade da Classificação":
        dashboard_curation(df)
    elif page == "7. Qualidade dos Dados":
        dashboard_data_quality(df, variables)
    elif page == "Modelos preditivos - preparação":
        dashboard_models(df)
    elif page == "Dados filtrados":
        dashboard_filtered_data(df)


def main() -> None:
    if not check_password():
        st.stop()

    uploaded_content = render_sidebar_source()
    try:
        workbook = load_data(uploaded_content)
    except Exception as exc:
        st.error(f"Não foi possível carregar a planilha: {exc}")
        st.stop()

    events = workbook["events"]
    variables = workbook["variables"]
    filtered = filter_dataframe(events)

    page = st.sidebar.radio("Dashboard", PAGE_OPTIONS, index=0)
    st.sidebar.divider()
    st.sidebar.caption("Sem seleção nos filtros equivale a todos os valores.")

    page_header(filtered)
    if filtered.empty:
        st.warning("Nenhum evento encontrado para os filtros selecionados.")
        st.stop()

    render_page(page, filtered, variables)
    st.caption("App por @titetodesco & ChatGPT - Atualização para a nova base Sphera.")


if __name__ == "__main__":
    main()
