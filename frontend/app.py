from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "transactions.csv"
PUBLIC_DATA_PATH = BASE_DIR / "data" / "public_fraud_sample.csv"
DEFAULT_API_URL = "http://127.0.0.1:8000"

MERCHANT_ENCODER = {
    "Amazon": 0,
    "ATM": 1,
    "Netflix": 2,
    "Unknown": 3,
    "Uber": 4,
    "Swiggy": 5,
    "Zomato": 6,
    "Flipkart": 7,
    "Paytm": 8,
    "Airbnb": 9,
    "Apple": 10,
    "IRCTC": 11,
}

TYPE_ENCODER = {
    "Shopping": 0,
    "Cash": 1,
    "Transfer": 2,
    "Subscription": 3,
    "Travel": 4,
    "Food": 5,
    "Bill": 6,
    "Wallet": 7,
}

MERCHANT_RISK = {
    "Unknown": 35,
    "ATM": 24,
    "Paytm": 20,
    "Airbnb": 18,
    "IRCTC": 16,
    "Flipkart": 14,
    "Uber": 12,
    "Amazon": 10,
    "Apple": 10,
    "Zomato": 8,
    "Swiggy": 8,
    "Netflix": 6,
}

TYPE_RISK = {
    "Transfer": 34,
    "Cash": 28,
    "Wallet": 22,
    "Travel": 16,
    "Shopping": 13,
    "Bill": 9,
    "Food": 7,
    "Subscription": 5,
}


st.set_page_config(
    page_title="FinGuard AI",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    :root {
        --bg: #08111f;
        --panel: rgba(255, 255, 255, 0.08);
        --panel-strong: rgba(255, 255, 255, 0.13);
        --line: rgba(255, 255, 255, 0.18);
        --text: #f6f8fb;
        --muted: #aebbd0;
        --cyan: #3dd6c6;
        --mint: #8be9bd;
        --amber: #f3bf5f;
        --red: #f06d78;
        --blue: #7aa8ff;
    }

    .stApp {
        color: var(--text);
        background:
            radial-gradient(circle at 14% 8%, rgba(61, 214, 198, 0.22), transparent 30%),
            radial-gradient(circle at 86% 18%, rgba(122, 168, 255, 0.18), transparent 28%),
            linear-gradient(135deg, #08111f 0%, #0f1b2b 48%, #17222f 100%);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2.5rem;
        max-width: 1400px;
    }

    header[data-testid="stHeader"],
    div[data-testid="stToolbar"],
    #MainMenu {
        visibility: hidden;
        height: 0;
    }

    [data-testid="stSidebar"] {
        background: rgba(8, 17, 31, 0.86);
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] * {
        color: var(--text);
    }

    h1, h2, h3, p {
        letter-spacing: 0;
    }

    .hero {
        padding: 2rem;
        border: 1px solid var(--line);
        border-radius: 8px;
        background: linear-gradient(135deg, rgba(255,255,255,0.13), rgba(255,255,255,0.055));
        box-shadow: 0 24px 70px rgba(0, 0, 0, 0.32);
        backdrop-filter: blur(22px);
    }

    .eyebrow {
        color: var(--mint);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
    }

    .hero h1 {
        font-size: clamp(2.1rem, 4vw, 4.5rem);
        line-height: 1;
        margin: 0 0 1rem 0;
        color: var(--text);
    }

    .hero p {
        color: var(--muted);
        font-size: 1rem;
        max-width: 780px;
        margin: 0;
    }

    .glass-card {
        min-height: 100%;
        padding: 1.15rem;
        border-radius: 8px;
        border: 1px solid var(--line);
        background: var(--panel);
        box-shadow: 0 18px 55px rgba(0, 0, 0, 0.24);
        backdrop-filter: blur(18px);
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid var(--line);
        border-radius: 8px;
        background: var(--panel);
        box-shadow: 0 18px 55px rgba(0, 0, 0, 0.24);
        backdrop-filter: blur(18px);
    }

    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        background: transparent;
    }

    .metric-label {
        color: var(--muted);
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    .metric-value {
        color: var(--text);
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.1;
        margin-top: 0.35rem;
    }

    .metric-note {
        color: var(--muted);
        font-size: 0.86rem;
        margin-top: 0.35rem;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.42rem 0.7rem;
        border-radius: 999px;
        border: 1px solid var(--line);
        background: var(--panel-strong);
        color: var(--text);
        font-size: 0.82rem;
        font-weight: 700;
    }

    .status-dot {
        width: 0.55rem;
        height: 0.55rem;
        border-radius: 999px;
        background: var(--mint);
    }

    .section-title {
        color: var(--text);
        font-size: 1.15rem;
        font-weight: 800;
        margin: 0 0 0.8rem 0;
    }

    .small-muted {
        color: var(--muted);
        font-size: 0.88rem;
    }

    .risk-low { color: var(--mint); }
    .risk-medium { color: var(--amber); }
    .risk-high { color: var(--red); }

    .stButton > button,
    .stDownloadButton > button {
        border: 1px solid rgba(61, 214, 198, 0.4);
        border-radius: 8px;
        background: linear-gradient(135deg, rgba(61,214,198,0.92), rgba(122,168,255,0.85));
        color: #06111f;
        font-weight: 800;
        min-height: 2.8rem;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        border-color: rgba(255, 255, 255, 0.7);
        color: #06111f;
        filter: brightness(1.05);
    }

    div[data-testid="stMetricValue"] {
        color: var(--text);
    }

    div[data-testid="stMetricLabel"] {
        color: var(--muted);
    }

    .stDataFrame {
        border: 1px solid var(--line);
        border-radius: 8px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_transactions() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
    df["is_fraud"] = pd.to_numeric(df["is_fraud"], errors="coerce").fillna(0).astype(int)
    df["risk_segment"] = pd.cut(
        df["amount"],
        bins=[-1, 1000, 10000, 30000, float("inf")],
        labels=["Low", "Medium", "High", "Critical"],
    )
    return df


@st.cache_data(show_spinner=False)
def load_public_sample() -> pd.DataFrame:
    if not PUBLIC_DATA_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(PUBLIC_DATA_PATH)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
    df["is_fraud"] = pd.to_numeric(df["is_fraud"], errors="coerce").fillna(0).astype(int)
    return df


def api_get(api_url: str, route: str) -> dict[str, Any] | None:
    try:
        response = requests.get(f"{api_url.rstrip('/')}{route}", timeout=2.5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def api_post(api_url: str, route: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    try:
        response = requests.post(f"{api_url.rstrip('/')}{route}", json=payload, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def local_risk_score(amount: float, merchant: str, transaction_type: str) -> float:
    amount_component = min(amount / 80000, 1) * 42
    merchant_component = MERCHANT_RISK.get(merchant, 18)
    type_component = TYPE_RISK.get(transaction_type, 12)
    score = (amount_component + merchant_component + type_component) / 100
    return max(0.01, min(score, 0.98))


def risk_band(score: float) -> str:
    if score >= 0.75:
        return "Critical"
    if score >= 0.5:
        return "High"
    if score >= 0.3:
        return "Medium"
    return "Low"


def local_reason_codes(amount: float, merchant: str, transaction_type: str, score: float) -> list[str]:
    reasons = []
    if amount >= 50000:
        reasons.append("High value transaction")
    elif amount >= 25000:
        reasons.append("Elevated amount")
    if merchant == "Unknown":
        reasons.append("Unrecognized merchant")
    if transaction_type in {"Transfer", "Cash"}:
        reasons.append("High risk transaction type")
    if score >= 0.5:
        reasons.append("Manual review threshold crossed")
    return reasons or ["No major risk indicators"]


def build_gauge(score: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(score * 100, 1),
            number={"suffix": "%", "font": {"size": 42}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#aebbd0"},
                "bar": {"color": "#3dd6c6"},
                "bgcolor": "rgba(255,255,255,0.04)",
                "borderwidth": 1,
                "bordercolor": "rgba(255,255,255,0.2)",
                "steps": [
                    {"range": [0, 30], "color": "rgba(139,233,189,0.35)"},
                    {"range": [30, 50], "color": "rgba(243,191,95,0.38)"},
                    {"range": [50, 75], "color": "rgba(240,109,120,0.34)"},
                    {"range": [75, 100], "color": "rgba(240,109,120,0.58)"},
                ],
            },
        )
    )
    fig.update_layout(
        height=310,
        margin={"l": 24, "r": 24, "t": 20, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f6f8fb"},
    )
    return fig


def chart_layout(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f6f8fb"},
        margin={"l": 20, "r": 20, "t": 36, "b": 28},
        legend={"orientation": "h", "y": 1.08, "x": 0},
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.16)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.16)")
    return fig


def metric_card(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


transactions = load_transactions()
public_sample = load_public_sample()

with st.sidebar:
    st.markdown("### FinGuard AI")
    st.caption("Fraud scoring cockpit")
    api_url = st.text_input("FastAPI endpoint", value=DEFAULT_API_URL)
    mode = st.segmented_control("View mode", ["Executive", "Analyst"], default="Executive")
    review_threshold = st.slider("Review threshold", 0.1, 0.9, 0.5, 0.05)
    st.divider()
    st.caption("Preset scenarios")
    preset = st.selectbox(
        "Load scenario",
        [
            "Custom transaction",
            "Low value subscription",
            "Large ATM withdrawal",
            "Unknown merchant transfer",
            "Travel booking spike",
        ],
    )

health = api_get(api_url, "/health")
api_online = health is not None

preset_values = {
    "Custom transaction": (101, 18500, "Amazon", "Shopping"),
    "Low value subscription": (102, 799, "Netflix", "Subscription"),
    "Large ATM withdrawal": (103, 64000, "ATM", "Cash"),
    "Unknown merchant transfer": (104, 72000, "Unknown", "Transfer"),
    "Travel booking spike": (105, 42000, "Airbnb", "Travel"),
}
default_transaction_id, default_amount, default_merchant, default_type = preset_values[preset]

st.markdown(
    """
    <section class="hero">
        <div class="eyebrow">Realtime fraud intelligence</div>
        <h1>FinGuard AI</h1>
        <p>
            A polished fraud detection workspace for scoring transactions, exploring risk trends,
            reviewing high-risk cases, and presenting an end-to-end machine learning project.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

st.write("")

status_text = "API online" if api_online else "Local fallback active"
st.markdown(
    f"""
    <div class="status-pill">
        <span class="status-dot"></span>
        <span>{status_text}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

total_transactions = len(transactions)
fraud_count = int(transactions["is_fraud"].sum())
fraud_rate = fraud_count / max(total_transactions, 1)
exposure = transactions.loc[transactions["is_fraud"] == 1, "amount"].sum()
avg_amount = transactions["amount"].mean()

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    metric_card("Transactions", f"{total_transactions:,}", "Sample ledger loaded")
with kpi2:
    metric_card("Fraud rate", f"{fraud_rate:.1%}", f"{fraud_count} flagged cases")
with kpi3:
    metric_card("Exposure", f"${exposure:,.0f}", "Fraud amount in sample")
with kpi4:
    metric_card("Average ticket", f"${avg_amount:,.0f}", "Mean transaction value")

st.write("")

score_col, result_col = st.columns([1, 1], gap="large")

with score_col:
    with st.container(border=True):
        st.markdown('<div class="section-title">Transaction Scoring</div>', unsafe_allow_html=True)

        form_col_a, form_col_b = st.columns(2)
        with form_col_a:
            transaction_id = st.number_input(
                "Transaction ID",
                min_value=1,
                value=default_transaction_id,
                step=1,
            )
            merchant = st.selectbox(
                "Merchant",
                list(MERCHANT_ENCODER.keys()),
                index=list(MERCHANT_ENCODER.keys()).index(default_merchant),
            )
        with form_col_b:
            amount = st.number_input(
                "Amount",
                min_value=0.0,
                value=float(default_amount),
                step=500.0,
                format="%.2f",
            )
            transaction_type = st.selectbox(
                "Transaction type",
                list(TYPE_ENCODER.keys()),
                index=list(TYPE_ENCODER.keys()).index(default_type),
            )

        analyze = st.button("Analyze transaction", width="stretch")

with result_col:
    with st.container(border=True):
        st.markdown('<div class="section-title">Risk Decision</div>', unsafe_allow_html=True)

        payload = {
            "transaction_id": int(transaction_id),
            "amount": float(amount),
            "merchant": merchant,
            "type": transaction_type,
        }

        result = None
        if analyze:
            result = api_post(api_url, "/predict", payload) if api_online else None

        if result is None:
            fallback_score = local_risk_score(float(amount), merchant, transaction_type)
            result = {
                "transaction_id": int(transaction_id),
                "prediction": int(fallback_score >= review_threshold),
                "result": "Fraud" if fallback_score >= review_threshold else "Safe Transaction",
                "risk_score": fallback_score,
                "risk_band": risk_band(fallback_score),
                "reason_codes": local_reason_codes(float(amount), merchant, transaction_type, fallback_score),
            }

        risk_score = float(result["risk_score"])
        band = result["risk_band"]
        band_class = "risk-high" if risk_score >= review_threshold else "risk-low"
        st.markdown(
            f"""
            <div class="metric-label">Decision</div>
            <div class="metric-value {band_class}">{result["result"]}</div>
            <div class="metric-note">Risk band: {band}</div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(build_gauge(risk_score), width="stretch")
        st.markdown('<div class="small-muted">Reason codes</div>', unsafe_allow_html=True)
        for reason in result["reason_codes"]:
            st.write(reason)

st.write("")

tab_overview, tab_cases, tab_batch, tab_data = st.tabs(
    ["Portfolio analytics", "Risk queue", "Batch analyzer", "Dataset lab"]
)

with tab_overview:
    chart_left, chart_right = st.columns([1.1, 0.9], gap="large")
    type_summary = (
        transactions.groupby("type", as_index=False)
        .agg(total_amount=("amount", "sum"), fraud_rate=("is_fraud", "mean"), transactions=("transaction_id", "count"))
        .sort_values("fraud_rate", ascending=False)
    )

    with chart_left:
        with st.container(border=True):
            fig = px.bar(
                type_summary,
                x="type",
                y="fraud_rate",
                color="total_amount",
                color_continuous_scale=["#8be9bd", "#f3bf5f", "#f06d78"],
                labels={"type": "Type", "fraud_rate": "Fraud rate", "total_amount": "Amount"},
                title="Fraud rate by transaction type",
            )
            st.plotly_chart(chart_layout(fig), width="stretch")

    with chart_right:
        with st.container(border=True):
            merchant_summary = (
                transactions.groupby("merchant", as_index=False)
                .agg(amount=("amount", "sum"), fraud=("is_fraud", "sum"))
                .sort_values("amount", ascending=False)
                .head(8)
            )
            fig = px.scatter(
                merchant_summary,
                x="merchant",
                y="amount",
                size="amount",
                color="fraud",
                color_continuous_scale=["#7aa8ff", "#f06d78"],
                title="Merchant exposure map",
                labels={"merchant": "Merchant", "amount": "Amount", "fraud": "Fraud cases"},
            )
            st.plotly_chart(chart_layout(fig), width="stretch")

with tab_cases:
    with st.container(border=True):
        case_df = transactions.copy()
        case_df["estimated_risk"] = case_df.apply(
            lambda row: local_risk_score(row["amount"], str(row["merchant"]), str(row["type"])),
            axis=1,
        )
        case_df["review"] = case_df["estimated_risk"] >= review_threshold
        case_df = case_df.sort_values(["review", "estimated_risk", "amount"], ascending=False)
        st.dataframe(
            case_df[["transaction_id", "amount", "merchant", "type", "is_fraud", "estimated_risk", "review"]],
            width="stretch",
            hide_index=True,
        )
        csv = case_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download review queue", data=csv, file_name="finguard_review_queue.csv", mime="text/csv")

with tab_batch:
    with st.container(border=True):
        st.markdown('<div class="section-title">Batch Analyzer</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload CSV with transaction_id, amount, merchant, type", type=["csv"])
        batch_df = pd.read_csv(uploaded) if uploaded is not None else transactions[["transaction_id", "amount", "merchant", "type"]]

        required = {"transaction_id", "amount", "merchant", "type"}
        if required.issubset(batch_df.columns):
            scored = batch_df.copy()
            scored["estimated_risk"] = scored.apply(
                lambda row: local_risk_score(float(row["amount"]), str(row["merchant"]), str(row["type"])),
                axis=1,
            )
            scored["risk_band"] = scored["estimated_risk"].apply(risk_band)
            scored["review"] = scored["estimated_risk"] >= review_threshold
            st.dataframe(scored, width="stretch", hide_index=True)
            st.download_button(
                "Download scored batch",
                data=scored.to_csv(index=False).encode("utf-8"),
                file_name="finguard_batch_scores.csv",
                mime="text/csv",
            )
        else:
            st.warning("The CSV must include transaction_id, amount, merchant, and type columns.")

with tab_data:
    with st.container(border=True):
        st.markdown('<div class="section-title">Dataset Lab</div>', unsafe_allow_html=True)
        st.caption(
            "The project includes a compact demo ledger plus a small public-data sample fetched from the Hugging Face credit-card-fraud dataset for schema reference."
        )
        demo_tab, public_tab = st.tabs(["Demo ledger", "Public sample"])
        with demo_tab:
            if mode == "Analyst":
                st.dataframe(transactions, width="stretch", hide_index=True)
            else:
                sample_view = transactions.sort_values("amount", ascending=False).head(12)
                st.dataframe(sample_view, width="stretch", hide_index=True)
        with public_tab:
            if public_sample.empty:
                st.info("Public sample file is not available yet.")
            else:
                st.dataframe(public_sample, width="stretch", hide_index=True)

st.caption("FinGuard AI uses Streamlit, FastAPI, XGBoost, Plotly, and a glassmorphism interface.")
