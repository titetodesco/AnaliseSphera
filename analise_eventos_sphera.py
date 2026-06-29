from __future__ import annotations

import re
import warnings
from io import BytesIO
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder


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
OBSERVATION_TYPE_COL = "Observation Type"
HUMAN_FACTOR_COL = "Human Factors"
CONSEQUENCES_COL = "Consequences"
EVENT_CONSEQUENCES_COL = "Event Consequences"
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
POTENTIAL_SEVERITY_ASSET_COL = "Potential Severity - Asset"
POTENTIAL_SEVERITY_COMMUNITY_COL = "Potential Severity - Community"
POTENTIAL_SEVERITY_ENVIRONMENTAL_COL = "Potential Severity - Environmental"
RELATED_DOCUMENTS_COL = "Related documents"
SCORE_RULE_COL = "Score regra"

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

MODEL_TARGETS = {
    "1. Prever Potencial FPI/SIF": FPI_COL,
    "2. Classificar Tipo Ontológico": ONTOLOGY_COL,
    "3. Prever Cenário Acidental": SCENARIO_COL,
    "4. Prever Barreira Crítica": BARRIER_COL,
    "5. Prever Incidente Futuro": EVENT_TYPE_COL,
    "6. Prever Potential Severity - Pessoas": POTENTIAL_SEVERITY_PEOPLE_COL,
    "7. Prever Tipo de Dano FPI/SIF": FPI_DAMAGE_COL,
    "8. Prever Escopo de Risco": SCOPE_COL,
}

MODEL_EXPLANATIONS = {
    "1. Prever Potencial FPI/SIF": {
        "aba": "1. FPI/SIF",
        "objetivo": "Estimar se um evento tem potencial de fatalidade ou lesão séria/permanente em pessoas.",
        "pergunta": "Quais eventos aparentemente simples podem esconder potencial de consequência grave para pessoas?",
        "alvo": FPI_COL,
        "uso": "Priorização preventiva, triagem de eventos com maior potencial de severidade e apoio à curadoria de segurança.",
        "cuidado": "O resultado deve ser usado como sinal de priorização. Variáveis de severidade e ontologia podem representar avaliação posterior e devem ser usadas com cautela em produção.",
    },
    "2. Classificar Tipo Ontológico": {
        "aba": "2. Tipo ontológico",
        "objetivo": "Classificar automaticamente o evento como sinal fraco, precursor, incidente, capacidade resiliente ou classe equivalente.",
        "pergunta": "Este novo evento parece apenas um sinal fraco, um precursor operacional/sistêmico ou um incidente realizado?",
        "alvo": ONTOLOGY_COL,
        "uso": "Organização da base por maturidade do sinal de risco e apoio à leitura da cadeia sinal fraco -> precursor -> incidente.",
        "cuidado": "Não use variáveis ontológicas como preditoras quando o objetivo for simular classificação antes da curadoria, pois elas podem vazar a resposta.",
    },
    "3. Prever Cenário Acidental": {
        "aba": "3. Cenário",
        "objetivo": "Classificar o cenário acidental mais provável associado ao evento.",
        "pergunta": "Se a condição observada evoluir, qual tipo de acidente ela pode representar?",
        "alvo": SCENARIO_COL,
        "uso": "Pareto de riscos, roteamento para especialistas e identificação de famílias de cenários dominantes.",
        "cuidado": "Cenários raros podem ter pouco suporte estatístico. O filtro de mínimo de registros por classe evita métricas artificiais em classes muito pequenas.",
    },
    "4. Prever Barreira Crítica": {
        "aba": "4. Barreira",
        "objetivo": "Identificar qual barreira crítica está envolvida, degradada, ausente, demandada ou deveria ser avaliada.",
        "pergunta": "Qual barreira provavelmente precisa ser verificada ou reforçada?",
        "alvo": BARRIER_COL,
        "uso": "Direcionamento de ações preventivas, revisão de controles críticos e priorização de inspeções ou auditorias.",
        "cuidado": "A barreira crítica pode ter sido inferida pela própria ontologia; use o modo exploratório para investigação e o modo conservador para avaliação mais realista.",
    },
    "5. Prever Incidente Futuro": {
        "aba": "5. Incidente",
        "objetivo": "Classificar o tipo administrativo do evento usando `Event Type` como proxy para evolução para incidentes, near misses ou observações.",
        "pergunta": "Quais sinais fracos ou precursores podem ter perfil semelhante ao de eventos classificados como incidentes?",
        "alvo": EVENT_TYPE_COL,
        "uso": "Triagem inicial de registros e identificação de perfis que se aproximam de incidentes registrados.",
        "cuidado": "Este é um proxy, não uma previsão temporal real de futuro. Para previsão temporal, seria necessário construir uma variável-alvo baseada em janelas de tempo e recorrência.",
    },
    "6. Prever Potential Severity - Pessoas": {
        "aba": "6. Severidade",
        "objetivo": "Estimar a severidade potencial para pessoas conforme o campo de severidade do Sphera.",
        "pergunta": "Qual nível de severidade potencial para pessoas é compatível com este evento?",
        "alvo": POTENTIAL_SEVERITY_PEOPLE_COL,
        "uso": "Apoio à consistência de classificação de severidade e identificação de eventos subavaliados.",
        "cuidado": "A severidade pode refletir avaliação humana posterior ao registro; valide com separação temporal antes de usar operacionalmente.",
    },
    "7. Prever Tipo de Dano FPI/SIF": {
        "aba": "7. Dano FPI/SIF",
        "objetivo": "Classificar o tipo de dano potencial associado a fatalidade ou lesão séria/permanente.",
        "pergunta": "Qual tipo de dano FPI/SIF é mais plausível caso o evento evolua?",
        "alvo": FPI_DAMAGE_COL,
        "uso": "Apoio à prevenção direcionada por tipo de dano e ao planejamento de barreiras específicas.",
        "cuidado": "Classes de dano muito específicas podem precisar de mais dados ou agrupamento para melhorar robustez.",
    },
    "8. Prever Escopo de Risco": {
        "aba": "8. Escopo",
        "objetivo": "Indicar a dimensão principal afetada caso o evento evolua ou revele uma condição de risco relevante.",
        "pergunta": "O principal domínio de consequência é pessoas, processo, ativo, ambiente, marítimo ou outro escopo?",
        "alvo": SCOPE_COL,
        "uso": "Roteamento por domínio de risco, priorização por especialidade e apoio à taxonomia do SafetyChat.",
        "cuidado": "É o alvo com mais classes e maior dispersão; espere desempenho menor e avalie agrupamentos de escopo se necessário.",
    },
}

BASIC_CATEGORICAL_FEATURES = [
    EVENT_TYPE_COL,
    LOCATION_COL,
    RAM_POTENTIAL_COL,
    TASK_COL,
    RISK_AREA_COL,
    OBSERVATION_TYPE_COL,
    HUMAN_FACTOR_COL,
    CONSEQUENCES_COL,
    EVENT_CONSEQUENCES_COL,
    POTENTIAL_SEVERITY_PEOPLE_COL,
    POTENTIAL_SEVERITY_ASSET_COL,
    POTENTIAL_SEVERITY_COMMUNITY_COL,
    POTENTIAL_SEVERITY_ENVIRONMENTAL_COL,
]

ONTOLOGY_CATEGORICAL_FEATURES = [
    ONTOLOGY_COL,
    SCENARIO_COL,
    BARRIER_COL,
    BARRIER_STATE_COL,
    ESCALATION_COL,
    FPI_COL,
    FPI_DAMAGE_COL,
    DOMAIN_COL,
    SCOPE_COL,
    QUALITY_DESCRIPTION_COL,
]

NUMERIC_FEATURES = [
    YEAR_COL,
    RELATED_DOCUMENTS_COL,
]

TEXT_FEATURES = [
    TITLE_COL,
    DESCRIPTION_COL,
    OBSERVED_EVENT_COL,
]

PAGE_OPTIONS = [
    "1. Visão Executiva da Ontologia",
    "2. FPI/SIF e Severidade Potencial",
    "3. Sinais Fracos, Precursores e Incidentes",
    "4. Barreiras Críticas e Estado das Barreiras",
    "5. Cenários, Mecanismos e Pareto 80/20",
    "6. Curadoria e Qualidade da Classificação",
    "7. Qualidade dos Dados",
    "Modelos preditivos",
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


def dataframe_signature(df: pd.DataFrame) -> tuple[int, int]:
    if has_column(df, EVENT_ID_COL):
        values = df[EVENT_ID_COL].astype("string")
    else:
        values = pd.Series(df.index.astype("string"), index=df.index)
    hash_sum = int(pd.util.hash_pandas_object(values, index=False).sum())
    return len(df), hash_sum


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


def combine_text_columns(frame: pd.DataFrame | pd.Series) -> pd.Series:
    if isinstance(frame, pd.Series):
        frame = frame.to_frame()
    return frame.fillna("").astype(str).agg(" ".join, axis=1)


def available_columns(df: pd.DataFrame, columns: list[str], exclude: str | None = None) -> list[str]:
    return [column for column in columns if has_column(df, column) and column != exclude]


def build_model_frame(
    df: pd.DataFrame,
    target: str,
    feature_mode: str,
    min_class_count: int,
) -> dict[str, object]:
    base = distinct_events(df).copy()
    if not has_column(base, target):
        raise ValueError(f"Coluna-alvo ausente: {target}")

    valid_target = ~missing_like(base[target])
    base = base[valid_target].copy()
    y = display_series(base[target])
    class_counts = y.value_counts()
    kept_classes = class_counts[class_counts >= min_class_count].index
    base = base[y.isin(kept_classes)].copy()
    y = display_series(base[target])

    if base.empty or y.nunique() < 2:
        raise ValueError("O alvo selecionado não tem classes suficientes após o filtro de classes raras.")

    base["Mes"] = base[DATE_COL].dt.month if has_column(base, DATE_COL) else pd.NA
    base["Trimestre_num"] = base[DATE_COL].dt.quarter if has_column(base, DATE_COL) else pd.NA
    base["Dia_semana"] = base[DATE_COL].dt.dayofweek if has_column(base, DATE_COL) else pd.NA

    categorical_cols = available_columns(base, BASIC_CATEGORICAL_FEATURES, exclude=target)
    if "ontologia" in feature_mode.lower():
        categorical_cols += [
            column
            for column in available_columns(base, ONTOLOGY_CATEGORICAL_FEATURES, exclude=target)
            if column not in categorical_cols
        ]

    numeric_candidates = NUMERIC_FEATURES + ["Mes", "Trimestre_num", "Dia_semana"]
    numeric_cols = available_columns(base, numeric_candidates, exclude=target)
    text_cols = available_columns(base, TEXT_FEATURES, exclude=target)
    feature_cols = categorical_cols + numeric_cols + text_cols

    if not feature_cols:
        raise ValueError("Não há variáveis preditoras disponíveis para este alvo.")

    X = base[feature_cols].copy()
    for column in categorical_cols + text_cols:
        X[column] = display_series(X[column])
    for column in numeric_cols:
        X[column] = pd.to_numeric(X[column], errors="coerce")

    metadata_cols = [EVENT_ID_COL, DATE_COL, LOCATION_COL, TITLE_COL, target]
    metadata_cols = [column for column in metadata_cols if has_column(base, column)]

    return {
        "base": base,
        "X": X,
        "y": y,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "text_cols": text_cols,
        "metadata": base[metadata_cols].copy(),
        "class_counts": y.value_counts(),
    }


def make_model_pipeline(categorical_cols: list[str], numeric_cols: list[str], text_cols: list[str]) -> Pipeline:
    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_cols,
            )
        )

    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="Não informado")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="infrequent_if_exist",
                                min_frequency=5,
                            ),
                        ),
                    ]
                ),
                categorical_cols,
            )
        )

    if text_cols:
        transformers.append(
            (
                "text",
                Pipeline(
                    steps=[
                        ("combine", FunctionTransformer(combine_text_columns, validate=False)),
                        (
                            "tfidf",
                            TfidfVectorizer(
                                max_features=1800,
                                min_df=2,
                                ngram_range=(1, 2),
                                strip_accents="unicode",
                            ),
                        ),
                    ]
                ),
                text_cols,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers, sparse_threshold=0.35)
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=1200,
        n_jobs=-1,
        random_state=42,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("classifier", classifier)])


def train_predictive_model(
    df: pd.DataFrame,
    target: str,
    feature_mode: str,
    test_size: float,
    min_class_count: int,
) -> dict[str, object]:
    prepared = build_model_frame(df, target, feature_mode, min_class_count)
    X = prepared["X"]
    y = prepared["y"]

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=stratify,
    )

    pipeline = make_model_pipeline(
        prepared["categorical_cols"],
        prepared["numeric_cols"],
        prepared["text_cols"],
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    labels = list(pipeline.named_steps["classifier"].classes_)
    report = classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0)
    majority_accuracy = float(y_test.value_counts(normalize=True).max())

    prediction_proba = None
    if hasattr(pipeline, "predict_proba"):
        prediction_proba = pipeline.predict_proba(X_test)

    metadata = prepared["metadata"].loc[X_test.index].copy()
    metadata["Valor real"] = y_test
    metadata["Predição"] = y_pred
    metadata["Acertou?"] = np.where(y_test.to_numpy() == y_pred, "Sim", "Não")
    if prediction_proba is not None:
        metadata["Confiança predição"] = prediction_proba.max(axis=1)

    return {
        "pipeline": pipeline,
        "target": target,
        "feature_mode": feature_mode,
        "X_all": X,
        "y_all": y,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "labels": labels,
        "report": report,
        "confusion": confusion_matrix(y_test, y_pred, labels=labels),
        "metadata": metadata,
        "class_counts": prepared["class_counts"],
        "features": {
            "Numéricas": prepared["numeric_cols"],
            "Categóricas": prepared["categorical_cols"],
            "Texto": prepared["text_cols"],
        },
        "metrics": {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            "majority_accuracy": majority_accuracy,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "classes": int(y.nunique()),
        },
    }


def report_to_dataframe(report: dict[str, object]) -> pd.DataFrame:
    rows = []
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            rows.append(
                {
                    "Classe": label,
                    "Precision": metrics.get("precision", 0),
                    "Recall": metrics.get("recall", 0),
                    "F1-score": metrics.get("f1-score", 0),
                    "Support": int(metrics.get("support", 0)),
                }
            )
    return pd.DataFrame(rows)


def clean_feature_name(name: str) -> str:
    for prefix in ["num__", "cat__", "text__"]:
        name = name.replace(prefix, "")
    return name.replace("onehot__", "").replace("tfidf__", "")


def top_model_features(model_result: dict[str, object], top_n: int = 25) -> pd.DataFrame:
    pipeline = model_result["pipeline"]
    classifier = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocess"]
    try:
        names = preprocessor.get_feature_names_out()
    except Exception:
        return pd.DataFrame()

    coef = classifier.coef_
    importance = np.mean(np.abs(coef), axis=0) if coef.ndim == 2 else np.abs(coef)
    feature_df = pd.DataFrame(
        {
            "Variável/termo": [clean_feature_name(str(name)) for name in names],
            "Importância média": importance,
        }
    )
    return feature_df.sort_values("Importância média", ascending=False).head(top_n)


def render_model_metrics(model_result: dict[str, object]) -> None:
    metrics = model_result["metrics"]
    metric_cards(
        [
            ("Acurácia", fmt_pct(metrics["accuracy"]), None),
            ("Acurácia balanceada", fmt_pct(metrics["balanced_accuracy"]), None),
            ("F1 macro", fmt_pct(metrics["macro_f1"]), None),
            ("F1 ponderado", fmt_pct(metrics["weighted_f1"]), None),
            ("Baseline maior classe", fmt_pct(metrics["majority_accuracy"]), None),
            ("Treino", fmt_int(metrics["train_rows"]), None),
            ("Teste", fmt_int(metrics["test_rows"]), None),
            ("Classes", fmt_int(metrics["classes"]), None),
        ]
    )


def render_confusion_matrix(model_result: dict[str, object]) -> None:
    labels = model_result["labels"]
    confusion = pd.DataFrame(model_result["confusion"], index=labels, columns=labels)
    display_confusion = confusion.copy()
    display_confusion.index = [shorten(label, 38) for label in display_confusion.index]
    display_confusion.columns = [shorten(label, 38) for label in display_confusion.columns]
    fig = px.imshow(
        display_confusion,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="YlGnBu",
        title="Matriz de confusão - conjunto de teste",
    )
    fig.update_layout(height=max(430, min(820, 120 + 34 * len(labels))), margin=dict(l=10, r=10, t=55, b=10))
    st.plotly_chart(fig, use_container_width=True)


def render_prediction_simulator(model_result: dict[str, object], key_prefix: str = "prediction_simulator") -> None:
    st.subheader("Predição para evento existente")
    metadata = model_result["metadata"].copy()
    if metadata.empty or not has_column(metadata, EVENT_ID_COL):
        st.info("Não há eventos disponíveis para simulação.")
        return

    all_rows = pd.concat(
        [
            model_result["X_all"],
            model_result["y_all"].rename(model_result["target"]),
        ],
        axis=1,
    )
    event_options = metadata[EVENT_ID_COL].dropna().astype(str).unique().tolist()
    selected_event_id = st.selectbox("Evento do conjunto de teste", event_options, key=f"{key_prefix}_event")
    selected_index = metadata[metadata[EVENT_ID_COL].astype(str) == selected_event_id].index[0]
    sample_X = model_result["X_all"].loc[[selected_index]]
    predicted = model_result["pipeline"].predict(sample_X)[0]
    actual = all_rows.loc[selected_index, model_result["target"]]

    cols = st.columns(3)
    cols[0].metric("Evento", selected_event_id)
    cols[1].metric("Real", shorten(actual, 42))
    cols[2].metric("Predição", shorten(predicted, 42))

    if hasattr(model_result["pipeline"], "predict_proba"):
        proba = model_result["pipeline"].predict_proba(sample_X)[0]
        classes = model_result["pipeline"].named_steps["classifier"].classes_
        proba_df = (
            pd.DataFrame({"Classe": classes, "Probabilidade": proba})
            .sort_values("Probabilidade", ascending=False)
            .head(8)
        )
        fig = px.bar(
            proba_df.sort_values("Probabilidade", ascending=True),
            x="Probabilidade",
            y="Classe",
            orientation="h",
            text=proba_df.sort_values("Probabilidade", ascending=True)["Probabilidade"].map(fmt_pct),
            title="Probabilidades estimadas",
            color_discrete_sequence=["#2A6F97"],
        )
        fig.update_layout(height=max(320, 70 + 34 * len(proba_df)), xaxis_tickformat=".0%", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    detail_cols = [column for column in [DATE_COL, LOCATION_COL, TITLE_COL] if has_column(metadata, column)]
    if detail_cols:
        st.dataframe(metadata.loc[[selected_index], detail_cols], use_container_width=True, hide_index=True)


def model_state_key(model_name: str) -> str:
    model_number = list(MODEL_TARGETS.keys()).index(model_name) + 1
    return f"model_{model_number}"


def model_overview_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, target in MODEL_TARGETS.items():
        info = MODEL_EXPLANATIONS[model]
        readiness = model_readiness(df, target)
        rows.append(
            {
                "Modelo": model,
                "Variável-alvo": target,
                "Objetivo": info["objetivo"],
                "Pergunta respondida": info["pergunta"],
                "Registros úteis": readiness["Registros úteis"],
                "Completude alvo": fmt_pct(readiness["Completude alvo"]),
                "Classes": readiness["Classes"],
                "Maior classe": readiness["Maior classe"],
                "% maior classe": fmt_pct(readiness["% maior classe"]),
                "Status": readiness["Status"],
            }
        )
    return pd.DataFrame(rows)


def render_model_explanation(model_name: str) -> None:
    info = MODEL_EXPLANATIONS[model_name]
    st.markdown(
        f"""
**Objetivo:** {info["objetivo"]}

**Pergunta que responde:** {info["pergunta"]}

**Variável-alvo:** `{info["alvo"]}`

**Uso esperado:** {info["uso"]}

**Cuidados de interpretação:** {info["cuidado"]}
"""
    )


def render_target_distribution(df: pd.DataFrame, target: str) -> None:
    if not has_column(df, target):
        return
    counts = value_counts_df(df, target, top=12)
    if counts.empty:
        return
    chart_data = counts.sort_values("Quantidade", ascending=True).copy()
    chart_data["Classe curta"] = chart_data[target].map(lambda value: shorten(value, 42))
    fig = px.bar(
        chart_data,
        x="Quantidade",
        y="Classe curta",
        orientation="h",
        text="Quantidade",
        title="Distribuição das classes do alvo",
        color_discrete_sequence=["#2A6F97"],
    )
    fig.update_layout(height=max(320, 90 + 32 * len(chart_data)), margin=dict(l=10, r=30, t=55, b=10), yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)


def render_model_result_tabs(model_result: dict[str, object], key_prefix: str) -> None:
    st.subheader("Desempenho no conjunto de teste")
    render_model_metrics(model_result)

    tab_metrics, tab_confusion, tab_errors, tab_features, tab_simulator = st.tabs(
        ["Relatório", "Matriz de confusão", "Erros e acertos", "Variáveis", "Simulador"]
    )
    with tab_metrics:
        report_df = report_to_dataframe(model_result["report"])
        st.dataframe(report_df, use_container_width=True, hide_index=True)
        st.caption("F1 macro trata todas as classes com o mesmo peso; F1 ponderado considera o tamanho de cada classe.")

    with tab_confusion:
        render_confusion_matrix(model_result)

    with tab_errors:
        metadata = model_result["metadata"].copy()
        if "Confiança predição" in metadata.columns:
            metadata["Confiança predição"] = metadata["Confiança predição"].map(fmt_pct)
        ordered = metadata.sort_values("Acertou?")
        st.dataframe(ordered.head(200), use_container_width=True, hide_index=True)

    with tab_features:
        feature_groups = pd.DataFrame(
            [
                {"Grupo": group, "Variáveis": ", ".join(columns) if columns else "Nenhuma"}
                for group, columns in model_result["features"].items()
            ]
        )
        st.dataframe(feature_groups, use_container_width=True, hide_index=True)
        feature_importance = top_model_features(model_result)
        if not feature_importance.empty:
            st.subheader("Variáveis e termos mais influentes")
            st.dataframe(feature_importance, use_container_width=True, hide_index=True)
        else:
            st.info("Importância de variáveis indisponível para este pipeline.")

    with tab_simulator:
        render_prediction_simulator(model_result, key_prefix=key_prefix)


def render_single_model_panel(
    df: pd.DataFrame,
    model_name: str,
    feature_mode: str,
    test_size: float,
    min_class_count: int,
) -> None:
    target = MODEL_TARGETS[model_name]
    state_key = model_state_key(model_name)
    result_key = f"{state_key}_result"
    signature_key = f"{state_key}_signature"
    model_signature = (model_name, target, feature_mode, test_size, min_class_count, dataframe_signature(df))

    render_model_explanation(model_name)
    readiness = model_readiness(df, target)
    metric_cards(
        [
            ("Registros úteis", fmt_int(readiness["Registros úteis"]), None),
            ("Completude do alvo", fmt_pct(readiness["Completude alvo"]), None),
            ("Classes", fmt_int(readiness["Classes"]), None),
            ("Maior classe", fmt_pct(readiness["% maior classe"]), shorten(readiness["Maior classe"], 32)),
        ]
    )

    col_distribution, col_action = st.columns([1.25, 0.75])
    with col_distribution:
        render_target_distribution(df, target)
    with col_action:
        st.subheader("Treino")
        st.write(f"Status: {readiness['Status']}")
        should_train = st.button("Treinar este modelo", type="primary", key=f"{state_key}_train")
        if should_train:
            with st.spinner("Treinando modelo baseline..."):
                try:
                    st.session_state[result_key] = train_predictive_model(
                        df,
                        target=target,
                        feature_mode=feature_mode,
                        test_size=test_size,
                        min_class_count=min_class_count,
                    )
                    st.session_state[signature_key] = model_signature
                except Exception as exc:
                    st.error(f"Não foi possível treinar este modelo: {exc}")
                    st.session_state.pop(result_key, None)
                    st.session_state.pop(signature_key, None)

    model_result = st.session_state.get(result_key)
    if not model_result:
        st.info("Modelo ainda não treinado nesta sessão.")
        return
    if st.session_state.get(signature_key) != model_signature:
        st.warning("A configuração ou os filtros mudaram desde o último treino. Treine novamente para atualizar as métricas.")
    render_model_result_tabs(model_result, key_prefix=state_key)


def dashboard_models(df: pd.DataFrame) -> None:
    st.header("Modelos preditivos")
    st.subheader("Viabilidade dos alvos")
    st.dataframe(model_overview_dataframe(df), use_container_width=True, hide_index=True)

    st.subheader("Configuração comum dos treinos")
    col1, col2 = st.columns([1.3, 1])
    with col1:
        feature_mode = st.radio(
            "Conjunto de variáveis",
            [
                "Campos Sphera/texto inicial (menor vazamento)",
                "Campos Sphera + ontologia (exploratório)",
            ],
            horizontal=False,
            key="model_feature_mode",
        )
    with col2:
        test_size = st.slider("Percentual para teste", 0.15, 0.35, 0.25, 0.05, key="model_test_size")
        min_class_count = st.slider("Mínimo de registros por classe", 2, 50, 20, 1, key="model_min_class_count")

    if "ontologia" in feature_mode.lower():
        st.warning(
            "Modo exploratório: algumas variáveis ontológicas podem carregar informação derivada do próprio alvo. "
            "Use esse resultado para investigação, não como evidência final de desempenho em produção."
        )

    model_tabs = st.tabs([MODEL_EXPLANATIONS[model]["aba"] for model in MODEL_TARGETS])
    for model_tab, model_name in zip(model_tabs, MODEL_TARGETS.keys()):
        with model_tab:
            render_single_model_panel(
                df,
                model_name=model_name,
                feature_mode=feature_mode,
                test_size=test_size,
                min_class_count=min_class_count,
            )


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
    elif page == "Modelos preditivos":
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
