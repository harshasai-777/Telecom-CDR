"""
dashboard.py  —  Enterprise Telecom CDR Analytics & Fraud Intelligence Platform
================================================================================
Client-ready, modern Streamlit UI with executive analytics, interactive subscriber
investigation, fraud risk scoring, behavioral clustering, and distributed engine insights.

Run:
    streamlit run dashboard.py
"""

import os
import sys
import pickle
import tempfile
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, DATASET_SIZES, MODEL_DIR, N_CLUSTERS, REPORT_DIR

# ── Streamlit Setup ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telecom Intelligence | CDR Analytics",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Polished Enterprise CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Global Typography & Background */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #F8FAFC !important;
        color: #0F172A !important;
    }
    
    /* Hide default elements */
    #MainMenu, footer, header { visibility: hidden; }

    /* Universal Dark High-Contrast Text for all Markdown, Headers, and Content */
    p, span, label, li, h1, h2, h3, h4, h5, h6,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4,
    [data-testid="stMarkdownContainer"] h5,
    [data-testid="stMarkdownContainer"] h6,
    [data-testid="stMarkdownContainer"] span {
        color: #0F172A !important;
    }

    /* Sidebar Background and High-Contrast Text */
    [data-testid="stSidebar"], [data-testid="stSidebar"] section {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    [data-testid="stSidebar"] * {
        color: #0F172A !important;
    }
    
    /* Top Brand Navigation Bar — High-Contrast Clean Light Enterprise */
    .brand-navbar {
        background: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 16px;
        padding: 1.25rem 2rem;
        margin-bottom: 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
    }
    
    .brand-title, .brand-title *,
    [data-testid="stMarkdownContainer"] .brand-title,
    [data-testid="stMarkdownContainer"] .brand-title * {
        color: #0F172A !important;
        font-size: 1.65rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.025em;
        margin: 0 !important;
    }
    
    .brand-subtitle, .brand-subtitle *,
    [data-testid="stMarkdownContainer"] .brand-subtitle,
    [data-testid="stMarkdownContainer"] .brand-subtitle * {
        color: #475569 !important;
        font-size: 0.92rem !important;
        margin-top: 0.35rem !important;
        font-weight: 500 !important;
    }
    
    .status-badge {
        background: #F0FDF4 !important;
        border: 1px solid #86EFAC !important;
        padding: 0.45rem 1.1rem;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }
    .status-badge, .status-badge * {
        color: #15803D !important;
    }
    
    .status-dot {
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background-color: #16A34A;
        box-shadow: 0 0 8px rgba(22, 163, 74, 0.6);
        display: inline-block;
    }
    
    /* Modern Metric Cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .glass-card {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.2rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04), 0 4px 12px rgba(0, 0, 0, 0.02);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.06);
    }
    
    .glass-card.card-blue { border-top: 4px solid #3B82F6; }
    .glass-card.card-emerald { border-top: 4px solid #10B981; }
    .glass-card.card-rose { border-top: 4px solid #F43F5E; }
    .glass-card.card-amber { border-top: 4px solid #F59E0B; }
    .glass-card.card-purple { border-top: 4px solid #8B5CF6; }
    
    .card-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B !important;
        margin-bottom: 0.4rem;
    }
    
    .card-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0F172A !important;
        line-height: 1.1;
        margin-bottom: 0.35rem;
    }
    
    .card-meta {
        font-size: 0.78rem;
        color: #64748B !important;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    
    .card-pill {
        font-size: 0.7rem;
        padding: 0.15rem 0.5rem;
        border-radius: 6px;
        font-weight: 600;
    }
    .pill-green { background: #ECFDF5; color: #059669 !important; }
    .pill-red { background: #FFF1F2; color: #E11D48 !important; }
    .pill-blue { background: #EFF6FF; color: #2563EB !important; }
    
    /* Native Metrics */
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] * {
        color: #0F172A !important;
        font-weight: 800 !important;
    }
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {
        color: #475569 !important;
        font-weight: 600 !important;
    }

    /* Section Headers */
    .panel-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #E2E8F0;
    }
    
    .panel-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0F172A !important;
    }
    
    /* Segment Profiles */
    .segment-card {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    
    .segment-badge {
        font-size: 1.75rem;
        margin-bottom: 0.4rem;
    }
    
    .segment-name {
        font-size: 0.95rem;
        font-weight: 700;
        margin-bottom: 0.6rem;
    }
    
    .segment-stat-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.8rem;
        padding: 0.25rem 0;
        color: #334155 !important;
        border-bottom: 1px dashed #F1F5F9;
    }
    .segment-stat-row * {
        color: #334155 !important;
    }
    
    /* Investigation Box */
    .investigate-box {
        background: #FFFFFF !important;
        border-radius: 14px;
        border: 1px solid #E2E8F0;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        margin-bottom: 1.5rem;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background-color: transparent;
        padding: 0.25rem 0;
        border-bottom: 2px solid #E2E8F0;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.65rem 1.25rem;
        font-weight: 600;
        font-size: 0.9rem;
        color: #475569 !important;
        background-color: transparent;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        color: #2563EB !important;
        border-bottom: 2px solid #2563EB !important;
        background-color: rgba(37, 99, 235, 0.04) !important;
    }

    /* Input fields and Selectboxes */
    [data-baseweb="select"] * {
        color: #0F172A !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Cached Pipeline Engine ─────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_process_pipeline(custom_path: str, file_fingerprint: str = ""):
    import importlib
    import preprocessing
    import clustering
    import anomaly_detection
    import xgb_classifier

    # Hot-reload underlying modules to prevent stale cached bytecode
    importlib.reload(preprocessing)
    importlib.reload(clustering)
    importlib.reload(anomaly_detection)
    importlib.reload(xgb_classifier)

    raw_df, sub_df = preprocessing.run_pipeline(custom_path=custom_path, verbose=False)
    sub_df, km, scaler, pca, elbow = clustering.run_clustering(sub_df, n_clusters=N_CLUSTERS, run_elbow=False)
    sub_df = anomaly_detection.run_anomaly_pipeline(sub_df)
    clf, metrics = xgb_classifier.run_xgb_pipeline(sub_df, run_cv=False)

    # fraud_probability is already computed and attached by run_xgb_pipeline
    if "fraud_probability" not in sub_df.columns:
        sub_df["fraud_probability"] = sub_df["is_fraud"].astype(float)

    return raw_df, sub_df, metrics, elbow


# ── Session State for User CSV File ──────────────────────────────────────────
if "active_csv_path" not in st.session_state:
    st.session_state.active_csv_path = None
if "active_file_name" not in st.session_state:
    st.session_state.active_file_name = None
if "file_fingerprint" not in st.session_state:
    st.session_state.file_fingerprint = ""

# ── Global Sidebar ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/cell-tower.png", width=56)
    st.markdown("### 🎛️ Analytics Engine Controls")
    
    st.markdown("#### 📂 Dataset Status")
    if st.session_state.active_file_name:
        st.markdown(
            f"""
            <div style="background:#F0FDF4; border:1px solid #BBF7D0; border-radius:8px; padding:0.6rem 0.8rem; margin-bottom:0.8rem;">
                <span style="font-size:0.75rem; color:#15803D; font-weight:700;">🟢 ACTIVE CSV</span><br>
                <span style="font-size:0.85rem; color:#0F172A; font-weight:600; word-break:break-all;">{st.session_state.active_file_name}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("📤 Load Another CSV", use_container_width=True):
            st.session_state.active_csv_path = None
            st.session_state.active_file_name = None
            st.session_state.file_fingerprint = ""
            st.cache_data.clear()
            st.rerun()

    uploaded_file = st.file_uploader(
        "Upload New CSV",
        type=["csv"],
        help="Upload any real CDR dataset or customer call details CSV file"
    )
    
    if uploaded_file is not None:
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            custom_csv_path = os.path.join(DATA_DIR, "uploaded_cdr.csv")
        except Exception:
            custom_csv_path = os.path.join(tempfile.gettempdir(), "uploaded_cdr.csv")
        buffer = uploaded_file.getbuffer()
        fingerprint = f"{uploaded_file.name}_{len(buffer)}"
        if st.session_state.file_fingerprint != fingerprint:
            with open(custom_csv_path, "wb") as f:
                f.write(buffer)
            st.session_state.active_csv_path = custom_csv_path
            st.session_state.active_file_name = uploaded_file.name
            st.session_state.file_fingerprint = fingerprint
            st.success(f"Loaded: {uploaded_file.name}")
            st.rerun()

    if st.button("🔄 Clear Cache & Reload", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("#### 🔍 Quick Filter")
    selected_call_types = st.multiselect(
        "Filter by Call Type",
        options=["voice", "sms", "data"],
        default=["voice", "sms", "data"]
    )
    
    st.markdown("---")
    st.markdown(
        """
        <div style="background:#F1F5F9; padding:0.8rem; border-radius:10px; font-size:0.75rem; color:#475569;">
            <b>Distributed Core:</b> Dask 4-Core Worker<br>
            <b>Classifier:</b> XGBoost Histogram Trees<br>
            <b>Clustering:</b> K-Means++ (K=5) + PCA<br>
            <b>Anomaly Engine:</b> Isolation Forest
        </div>
        """,
        unsafe_allow_html=True
    )


# ── Data Ingestion & Mandatory User CSV Prompt ─────────────────────────────────
active_custom_path = st.session_state.active_csv_path
dataset_source_desc = st.session_state.active_file_name or "Custom File"

if active_custom_path is None or not os.path.exists(active_custom_path):
    existing_uploaded = os.path.join(DATA_DIR, "uploaded_cdr.csv")
    has_existing = os.path.exists(existing_uploaded)

    st.markdown("""
    <div style="background: white; border: 2px dashed #CBD5E1; border-radius: 16px; padding: 2.5rem 2rem; text-align: center; margin: 1.5rem 0;">
        <img src="https://img.icons8.com/fluency/96/cell-tower.png" width="76" style="margin-bottom: 0.8rem;">
        <h2 style="color: #0F172A; font-weight: 800; margin-bottom: 0.5rem; letter-spacing: -0.02em;">
            Upload Telecom CDR Dataset to Run Analytics
        </h2>
        <p style="color: #64748B; font-size: 1rem; max-width: 650px; margin: 0 auto 1.5rem auto; line-height: 1.6;">
            Please provide your <b>Call Detail Records (CDR)</b> or <b>Customer Call Details CSV</b> file to run the platform.
            The engine will dynamically adapt your dataset schema, partition the data with Dask, cluster subscriber profiles, and train distributed XGBoost fraud detection models.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### 📂 Option 1: Drag & Drop CSV")
        main_upload = st.file_uploader(
            "Upload Telecom CDR / Call Details CSV",
            type=["csv"],
            key="main_page_csv_uploader"
        )
        if main_upload is not None:
            try:
                os.makedirs(DATA_DIR, exist_ok=True)
                custom_csv_path = os.path.join(DATA_DIR, "uploaded_cdr.csv")
            except Exception:
                custom_csv_path = os.path.join(tempfile.gettempdir(), "uploaded_cdr.csv")
            buffer = main_upload.getbuffer()
            with open(custom_csv_path, "wb") as f:
                f.write(buffer)
            st.session_state.active_csv_path = custom_csv_path
            st.session_state.active_file_name = main_upload.name
            st.session_state.file_fingerprint = f"{main_upload.name}_{len(buffer)}"
            st.success(f"Successfully loaded {main_upload.name}! Initializing pipeline...")
            st.rerun()

    with col2:
        st.markdown("#### 🚀 Option 2: Instant Demo Mode")
        st.markdown("""
        <div style="background:#F0FDF4; border:1px solid #BBF7D0; border-radius:10px; padding:0.9rem; font-size:0.85rem; color:#166534; margin-bottom:0.8rem;">
            Don't have a CDR CSV on hand right now? Click below to generate a <b>1,000-call synthetic telecom dataset</b> and test the live intelligence platform immediately!
        </div>
        """, unsafe_allow_html=True)
        if st.button("⚡ Launch Demo with Sample Dataset", use_container_width=True, type="primary"):
            from data_generator import generate_cdr
            demo_df = generate_cdr(1000)
            try:
                os.makedirs(DATA_DIR, exist_ok=True)
                demo_path = os.path.join(DATA_DIR, "demo_cdr.csv")
            except Exception:
                demo_path = os.path.join(tempfile.gettempdir(), "demo_cdr.csv")
            demo_df.to_csv(demo_path, index=False)
            st.session_state.active_csv_path = demo_path
            st.session_state.active_file_name = "demo_cdr.csv (Live Simulation)"
            st.session_state.file_fingerprint = f"demo_1000_{len(demo_df)}"
            st.success("Sample dataset generated! Loading analytics...")
            st.rerun()

        if has_existing:
            st.markdown("---")
            if st.button("⚡ Continue with Previously Uploaded CSV", use_container_width=True):
                st.session_state.active_csv_path = existing_uploaded
                st.session_state.active_file_name = "uploaded_cdr.csv"
                st.session_state.file_fingerprint = f"uploaded_cdr_{os.path.getsize(existing_uploaded)}"
                st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 1.2rem; font-size: 0.85rem; color: #475569;">
        <b>Accepted Formats & Schema Support:</b><br>
        • Supports any CDR or Customer Call Details CSV (Kaggle datasets, telecom operator logs, network dumps).<br>
        • Automatically maps caller phone numbers, call durations/minutes, call charges, and churn/fraud indicators.<br>
        • Ingests standard telecom schema (MSISDN, timestamp, call type, cell towers, billing charges).
    </div>
    """, unsafe_allow_html=True)

    # Strictly halt execution here so no data is processed before user chooses
    st.stop()

# ── Execute Full Pipeline on User CSV ──────────────────────────────────────────
# Cloud safety: cap dataset to 50 000 rows to avoid CPU throttling on free tier
_MAX_ROWS_CLOUD = 50_000

def _maybe_downsample(path: str, max_rows: int = _MAX_ROWS_CLOUD):
    if not path or not os.path.exists(path):
        return path, False
    try:
        with open(path, "rb") as f:
            total_lines = sum(1 for _ in f)
        if total_lines > max_rows:
            df = pd.read_csv(path, nrows=max_rows)
            sampled_path = path.replace(".csv", "_sampled.csv")
            df.to_csv(sampled_path, index=False)
            return sampled_path, True
        return path, False
    except Exception:
        return path, False

_pipeline_path, _was_sampled = _maybe_downsample(active_custom_path)
if _was_sampled:
    st.info(f"⚡ Dataset was large — running on a 50 000-row sample to stay within Streamlit Cloud limits.", icon="ℹ️")

with st.spinner(f"Executing Dask Preprocessing, Clustering & XGBoost Pipeline on {dataset_source_desc}..."):
    raw_df, sub_df, metrics, _ = load_and_process_pipeline(
        custom_path=_pipeline_path,
        file_fingerprint=st.session_state.file_fingerprint
    )

# Robust guard: ensure segment_name and fraud_probability are always present
if "cluster" not in sub_df.columns:
    sub_df["cluster"] = 0

if "segment_name" not in sub_df.columns:
    fallback_map = {
        0: "Standard / Casual Users",
        1: "Suspicious / High-Risk",
        2: "Heavy Callers",
        3: "International Travellers",
        4: "Night-Shift / Off-Peak",
    }
    cluster_names_path = os.path.join(MODEL_DIR, "cluster_names.pkl")
    if os.path.exists(cluster_names_path):
        try:
            with open(cluster_names_path, "rb") as f:
                c_map = pickle.load(f)
            sub_df["segment_name"] = sub_df["cluster"].map(c_map)
        except Exception:
            sub_df["segment_name"] = sub_df["cluster"].map(fallback_map)
    else:
        sub_df["segment_name"] = sub_df["cluster"].map(fallback_map)

sub_df["segment_name"] = sub_df["segment_name"].fillna("Standard / Casual Users")

if "fraud_probability" not in sub_df.columns:
    sub_df["fraud_probability"] = sub_df["is_fraud"].astype(float)

# Pre-compute common filters and aggregations
if selected_call_types:
    filtered_raw_df = raw_df[raw_df["call_type"].isin(selected_call_types)].copy()
else:
    filtered_raw_df = raw_df.copy()

filtered_raw_df["date"] = pd.to_datetime(filtered_raw_df["call_start"]).dt.date
filtered_raw_df["hour"] = pd.to_datetime(filtered_raw_df["call_start"]).dt.hour
filtered_raw_df["dow"] = pd.to_datetime(filtered_raw_df["call_start"]).dt.dayofweek

total_records = len(filtered_raw_df)
total_subscribers = len(sub_df)
total_revenue = filtered_raw_df["charge_usd"].sum()
fraud_subscribers = int(sub_df["is_fraud"].sum())
anomalies_detected = int(sub_df["is_anomaly"].sum())


# ── Top Enterprise Header Bar ──────────────────────────────────────────────────
st.markdown(f"""
<div class="brand-navbar">
    <div style="display: flex; align-items: center; gap: 1.25rem;">
        <span style="font-size: 2.3rem; line-height: 1;">📡</span>
        <div>
            <h1 class="brand-title" style="color: #0F172A !important; font-size: 1.65rem; font-weight: 800; margin: 0; padding: 0; letter-spacing: -0.02em;">
                TELECOM CDR INTELLIGENCE PLATFORM
            </h1>
            <div class="brand-subtitle" style="color: #475569 !important; font-size: 0.92rem; font-weight: 500; margin-top: 0.3rem;">
                Distributed Big Data Clustering &middot; Behavioral Customer Analytics &middot; XGBoost Fraud Shield
            </div>
        </div>
    </div>
    <div style="display: flex; gap: 1rem; align-items: center;">
        <span class="status-badge" style="background: #F0FDF4 !important; border: 1px solid #86EFAC !important; color: #15803D !important; padding: 0.45rem 1.1rem; border-radius: 9999px; font-weight: 700; font-size: 0.82rem; display: inline-flex; align-items: center; gap: 0.5rem;">
            <span class="status-dot" style="width: 9px; height: 9px; border-radius: 50%; background-color: #16A34A; display: inline-block;"></span>
            ACTIVE &middot; {dataset_source_desc.upper()}
        </span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Top KPI Stat Cards ─────────────────────────────────────────────────────────
st.markdown(f"""
<div class="kpi-container">
    <div class="glass-card card-blue">
        <div class="card-label">Total Call Records</div>
        <div class="card-value">{total_records:,}</div>
        <div class="card-meta">
            <span class="card-pill pill-blue">Processed</span> Dask 4-Partition Parallel
        </div>
    </div>
    <div class="glass-card card-emerald">
        <div class="card-label">Active Subscribers</div>
        <div class="card-value">{total_subscribers:,}</div>
        <div class="card-meta">
            <span class="card-pill pill-green">Aggregated</span> 19 Derived Feature Signals
        </div>
    </div>
    <div class="glass-card card-purple">
        <div class="card-label">Billed Revenue</div>
        <div class="card-value">${total_revenue:,.2f}</div>
        <div class="card-meta">
            <span class="card-pill pill-blue">USD</span> Voice, SMS & Data Sessions
        </div>
    </div>
    <div class="glass-card card-rose">
        <div class="card-label">Confirmed Fraud</div>
        <div class="card-value">{fraud_subscribers}</div>
        <div class="card-meta">
            <span class="card-pill pill-red">{fraud_subscribers/total_subscribers:.1%}</span> High Risk Accounts
        </div>
    </div>
    <div class="glass-card card-amber">
        <div class="card-label">XGBoost Accuracy</div>
        <div class="card-value">{metrics['accuracy']*100:.1f}%</div>
        <div class="card-meta">
            <span class="card-pill pill-green">ROC-AUC {metrics['roc_auc']:.3f}</span> 5-Fold Stratified
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Navigation Tabs ────────────────────────────────────────────────────
tab_overview, tab_investigate, tab_traffic, tab_segments, tab_architecture = st.tabs([
    "📈 Executive Overview",
    "🔍 Subscriber Investigation",
    "📊 Network Traffic & Revenue",
    "👥 Customer Segmentation",
    "⚡ Distributed Architecture",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: EXECUTIVE OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    has_tod_metrics = all(c in filtered_raw_df.columns for c in ["Day Charge", "Eve Charge", "Night Charge", "Intl Charge"])
    col_left, col_right = st.columns([7, 5])
    
    with col_left:
        if has_tod_metrics and "Account Length" in filtered_raw_df.columns:
            st.markdown('<div class="panel-header"><div class="panel-title">📈 Customer Account Tenure & Loyalty Distribution</div></div>', unsafe_allow_html=True)
            fig_trend = px.histogram(
                filtered_raw_df, x="Account Length",
                nbins=40,
                color_discrete_sequence=["#2563EB"],
                labels={"Account Length": "Account Tenure (Billing Cycles)"}
            )
            fig_trend.update_layout(
                height=320,
                margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                xaxis=dict(showgrid=False, title="Account Tenure (Billing Cycles)"),
                yaxis=dict(showgrid=True, gridcolor="#F1F5F9", title="Subscribers"),
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.markdown('<div class="panel-header"><div class="panel-title">📈 Daily Call Volume & Traffic Trends</div></div>', unsafe_allow_html=True)
            daily_traffic = filtered_raw_df.groupby("date").size().reset_index(name="Volume")
            fig_trend = px.area(
                daily_traffic, x="date", y="Volume",
                color_discrete_sequence=["#3B82F6"]
            )
            fig_trend.update_traces(
                line=dict(width=2.5, color="#2563EB"),
                fillcolor="rgba(37, 99, 235, 0.12)"
            )
            fig_trend.update_layout(
                height=320,
                margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                xaxis=dict(showgrid=False, title=""),
                yaxis=dict(showgrid=True, gridcolor="#F1F5F9", title="Calls / Day"),
                hovermode="x unified",
            )
            st.plotly_chart(fig_trend, use_container_width=True)

    with col_right:
        if has_tod_metrics:
            st.markdown('<div class="panel-header"><div class="panel-title">📞 Network Traffic Composition by Window</div></div>', unsafe_allow_html=True)
            day_k = int(filtered_raw_df["Day Calls"].sum()) if "Day Calls" in filtered_raw_df.columns else 0
            eve_k = int(filtered_raw_df["Eve Calls"].sum()) if "Eve Calls" in filtered_raw_df.columns else 0
            night_k = int(filtered_raw_df["Night Calls"].sum()) if "Night Calls" in filtered_raw_df.columns else 0
            intl_k = int(filtered_raw_df["Intl Calls"].sum()) if "Intl Calls" in filtered_raw_df.columns else 0
            cs_k = int(filtered_raw_df["CustServ Calls"].sum()) if "CustServ Calls" in filtered_raw_df.columns else 0
            mix_df = pd.DataFrame([
                {"Category": "Daytime", "Calls": day_k},
                {"Category": "Evening", "Calls": eve_k},
                {"Category": "Night-time", "Calls": night_k},
                {"Category": "International", "Calls": intl_k},
                {"Category": "Customer Care", "Calls": cs_k},
            ])
            fig_donut = px.pie(
                mix_df, names="Category", values="Calls",
                hole=0.60,
                color="Category",
                color_discrete_map={
                    "Daytime": "#3B82F6",
                    "Evening": "#F59E0B",
                    "Night-time": "#8B5CF6",
                    "International": "#EC4899",
                    "Customer Care": "#10B981"
                }
            )
            fig_donut.update_layout(
                height=320,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="#FFFFFF",
                legend=dict(orientation="h", yanchor="bottom", y=-0.18, x=0.0),
            )
            fig_donut.update_traces(textinfo="percent+label", textfont_size=11)
            st.plotly_chart(fig_donut, use_container_width=True)
        else:
            st.markdown('<div class="panel-header"><div class="panel-title">📞 Service Composition & Mix</div></div>', unsafe_allow_html=True)
            call_types_agg = filtered_raw_df["call_type"].value_counts().reset_index()
            call_types_agg.columns = ["Type", "Count"]
            fig_donut = px.pie(
                call_types_agg, names="Type", values="Count",
                hole=0.62,
                color="Type",
                color_discrete_map={"voice": "#3B82F6", "sms": "#10B981", "data": "#F59E0B"}
            )
            fig_donut.update_layout(
                height=320,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="#FFFFFF",
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, x=0.2),
            )
            fig_donut.update_traces(textinfo="percent+label", textfont_size=12)
            st.plotly_chart(fig_donut, use_container_width=True)

    st.markdown('<div class="panel-header"><div class="panel-title">🛡️ Fraud Defense Model Capabilities</div></div>', unsafe_allow_html=True)
    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    c_m1.metric("Precision", f"{metrics['precision']*100:.1f}%", help="True fraud among flagged users")
    c_m2.metric("Recall", f"{metrics['recall']*100:.1f}%", help="Percentage of all real fraud caught")
    c_m3.metric("F1 Score", f"{metrics['f1']*100:.1f}%", help="Harmonic mean of precision and recall")
    c_m4.metric("Unsupervised Isolation Catch", f"{sub_df[(sub_df['is_anomaly']==1) & (sub_df['is_fraud']==1)].shape[0]}/{fraud_subscribers}", help="Fraud identified with 0 ground-truth labels")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: SUBSCRIBER INVESTIGATION (Interactive Drill-down)
# ══════════════════════════════════════════════════════════════════════════════
with tab_investigate:
    st.markdown('<div class="panel-header"><div class="panel-title">🔍 Individual Subscriber Risk Investigation Console</div></div>', unsafe_allow_html=True)
    st.caption("Select or search any phone number (MSISDN) to inspect their complete behavioural profile, AI risk verdict, and call logs.")
    
    # Sort subscribers by fraud probability descending
    sorted_subscribers = sub_df.sort_values("fraud_probability", ascending=False)["caller_id"].tolist()
    
    col_select, col_meta = st.columns([4, 8])
    with col_select:
        selected_subscriber = st.selectbox(
            "Select Subscriber MSISDN:",
            options=sorted_subscribers,
            format_func=lambda s: f"{s} {'🚨 [FRAUD]' if sub_df.loc[sub_df['caller_id']==s, 'is_fraud'].values[0]==1 else '✅ [NORMAL]'}"
        )
        
    sub_row = sub_df[sub_df["caller_id"] == selected_subscriber].iloc[0]
    prob = sub_row["fraud_probability"]
    is_fraud_val = sub_row["is_fraud"]
    is_anomaly_val = sub_row["is_anomaly"]
    
    # Subscriber summary card
    status_color = "#EF4444" if is_fraud_val == 1 else "#10B981"
    status_label = "HIGH RISK FRAUDSTER" if is_fraud_val == 1 else "TRUSTED SUBSCRIBER"
    
    st.markdown(f"""
    <div class="investigate-box">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem;">
            <div>
                <span style="font-size:1.4rem; font-weight:800; color:#0F172A;">{selected_subscriber}</span>
                <span style="margin-left:1rem; background:{status_color}18; color:{status_color}; border:1px solid {status_color}40; padding:0.25rem 0.75rem; border-radius:999px; font-weight:700; font-size:0.78rem;">
                    {status_label}
                </span>
            </div>
            <div style="text-align:right;">
                <span style="font-size:0.8rem; color:#64748B;">AI Fraud Risk Probability:</span><br>
                <span style="font-size:1.6rem; font-weight:800; color:{status_color};">{prob*100:.1f}%</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Key Stats for this subscriber
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Total Calls", f"{int(sub_row['total_calls']):,}")
    s2.metric("Avg Call Duration", f"{sub_row['avg_duration_min']:.1f} min")
    s3.metric("International Calls", f"{sub_row['pct_international']*100:.1f}%")
    s4.metric("Night-time Calls", f"{sub_row['pct_night_calls']*100:.1f}%")
    s5.metric("Total Charges", f"${sub_row['total_charge_usd']:.2f}")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Drill-down into actual raw call records for this subscriber
    if active_custom_path is not None and os.path.exists(active_custom_path):
        st.markdown(f"##### 📜 Authentic Records from Uploaded Dataset for Subscriber `{selected_subscriber}`")
        try:
            orig_df = pd.read_csv(active_custom_path)
            phone_col = next(
                (c for c in orig_df.columns if str(c).strip().lower().replace(" ", "_").replace("-", "_").replace(".", "_") in ["phone_number", "phone", "caller_id", "caller", "calling_number", "msisdn", "subscriber_id"]),
                None
            )
            if phone_col:
                sub_records = orig_df[orig_df[phone_col].astype(str) == str(selected_subscriber)].reset_index(drop=True)
            else:
                sub_records = filtered_raw_df[filtered_raw_df["caller_id"] == selected_subscriber].reset_index(drop=True)
        except Exception:
            sub_records = filtered_raw_df[filtered_raw_df["caller_id"] == selected_subscriber].reset_index(drop=True)

        st.caption(f"Displaying all **{len(sub_records)}** native activity records matching phone number **{selected_subscriber}** from `{os.path.basename(active_custom_path)}`")
        st.dataframe(
            sub_records,
            use_container_width=True,
            height=280
        )
    else:
        st.markdown("##### 📜 Recent Call Detail Records (CDRs) for this Subscriber")
        sub_records = filtered_raw_df[filtered_raw_df["caller_id"] == selected_subscriber][[
            "record_id", "callee_id", "call_start", "duration_sec", "call_type",
            "direction", "is_international", "cell_tower_id", "charge_usd"
        ]].sort_values("call_start", ascending=False).reset_index(drop=True)

        st.dataframe(
            sub_records,
            column_config={
                "record_id": st.column_config.TextColumn("Record ID"),
                "callee_id": st.column_config.TextColumn("Recipient"),
                "call_start": st.column_config.DatetimeColumn("Timestamp", format="D MMM YYYY, h:mm a"),
                "duration_sec": st.column_config.NumberColumn("Duration (s)"),
                "call_type": st.column_config.TextColumn("Type"),
                "direction": st.column_config.TextColumn("Direction"),
                "is_international": st.column_config.CheckboxColumn("International"),
                "charge_usd": st.column_config.NumberColumn("Charge ($)", format="$%.2f"),
            },
            use_container_width=True,
            height=260
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: NETWORK TRAFFIC & REVENUE
# ══════════════════════════════════════════════════════════════════════════════
with tab_traffic:
    st.markdown('<div class="panel-header"><div class="panel-title">📊 Network Traffic Load & Financial Revenue Intelligence</div></div>', unsafe_allow_html=True)

    # Detect if dataset contains account-level Time-Of-Day billing & call metrics
    has_tod_metrics = all(c in filtered_raw_df.columns for c in ["Day Charge", "Eve Charge", "Night Charge", "Intl Charge"])

    if has_tod_metrics:
        # ── Account Usage & Billing Dataset Mode (e.g. CALL DETAILS-DATA.CSV) ──
        day_c = float(filtered_raw_df["Day Charge"].sum())
        eve_c = float(filtered_raw_df["Eve Charge"].sum())
        night_c = float(filtered_raw_df["Night Charge"].sum())
        intl_c = float(filtered_raw_df["Intl Charge"].sum())
        tot_c = max(1e-6, day_c + eve_c + night_c + intl_c)

        day_k = int(filtered_raw_df["Day Calls"].sum()) if "Day Calls" in filtered_raw_df.columns else 0
        eve_k = int(filtered_raw_df["Eve Calls"].sum()) if "Eve Calls" in filtered_raw_df.columns else 0
        night_k = int(filtered_raw_df["Night Calls"].sum()) if "Night Calls" in filtered_raw_df.columns else 0
        intl_k = int(filtered_raw_df["Intl Calls"].sum()) if "Intl Calls" in filtered_raw_df.columns else 0
        cs_k = int(filtered_raw_df["CustServ Calls"].sum()) if "CustServ Calls" in filtered_raw_df.columns else 0

        # KPI Summary Row
        kp1, kp2, kp3, kp4 = st.columns(4)
        kp1.metric("Daytime Revenue", f"${day_c:,.2f}", f"{day_c/tot_c*100:.1f}% share")
        kp2.metric("Evening Revenue", f"${eve_c:,.2f}", f"{eve_c/tot_c*100:.1f}% share")
        kp3.metric("Night-time Revenue", f"${night_c:,.2f}", f"{night_c/tot_c*100:.1f}% share")
        kp4.metric("International Revenue", f"${intl_c:,.2f}", f"{intl_c/tot_c*100:.1f}% share")

        st.markdown("<br>", unsafe_allow_html=True)

        c_tr1, c_tr2 = st.columns(2)

        with c_tr1:
            st.markdown("##### 💰 Billed Revenue Distribution by Calling Window")
            rev_df = pd.DataFrame([
                {"Calling Window": "Daytime", "Revenue ($)": day_c},
                {"Calling Window": "Evening", "Revenue ($)": eve_c},
                {"Calling Window": "Night-time", "Revenue ($)": night_c},
                {"Calling Window": "International", "Revenue ($)": intl_c},
            ])
            fig_rev = px.pie(
                rev_df, names="Calling Window", values="Revenue ($)",
                hole=0.55,
                color="Calling Window",
                color_discrete_map={
                    "Daytime": "#3B82F6",
                    "Evening": "#F59E0B",
                    "Night-time": "#8B5CF6",
                    "International": "#EC4899"
                }
            )
            fig_rev.update_traces(
                textinfo="percent+label",
                textfont=dict(size=13, color="#0F172A"),
                hovertemplate="<b>%{label}</b><br>Billed Revenue: $%{value:,.2f}<br>Share: %{percent}"
            )
            fig_rev.update_layout(
                height=330,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="#FFFFFF",
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, x=0.1)
            )
            st.plotly_chart(fig_rev, use_container_width=True)

        with c_tr2:
            st.markdown("##### 📞 Total Call Traffic Distribution by Calling Window")
            calls_df = pd.DataFrame([
                {"Time Window": "Daytime", "Total Calls": day_k},
                {"Time Window": "Evening", "Total Calls": eve_k},
                {"Time Window": "Night-time", "Total Calls": night_k},
                {"Time Window": "International", "Total Calls": intl_k},
                {"Time Window": "Customer Service", "Total Calls": cs_k},
            ])
            fig_calls = px.bar(
                calls_df, x="Time Window", y="Total Calls",
                color="Time Window",
                color_discrete_map={
                    "Daytime": "#3B82F6",
                    "Evening": "#F59E0B",
                    "Night-time": "#8B5CF6",
                    "International": "#EC4899",
                    "Customer Service": "#10B981"
                },
                text_auto=".2s"
            )
            fig_calls.update_traces(
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Total Calls: %{y:,}"
            )
            fig_calls.update_layout(
                height=330,
                margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                showlegend=False,
                xaxis=dict(showgrid=False, title=""),
                yaxis=dict(showgrid=True, gridcolor="#F1F5F9", title="Calls Count")
            )
            st.plotly_chart(fig_calls, use_container_width=True)

        c_tr3, c_tr4 = st.columns(2)

        with c_tr3:
            st.markdown("##### ⏱ Total Minutes of Network Usage by Time Window")
            day_m = float(filtered_raw_df["Day Mins"].sum()) if "Day Mins" in filtered_raw_df.columns else 0
            eve_m = float(filtered_raw_df["Eve Mins"].sum()) if "Eve Mins" in filtered_raw_df.columns else 0
            night_m = float(filtered_raw_df["Night Mins"].sum()) if "Night Mins" in filtered_raw_df.columns else 0
            intl_m = float(filtered_raw_df["Intl Mins"].sum()) if "Intl Mins" in filtered_raw_df.columns else 0

            mins_df = pd.DataFrame([
                {"Window": "Daytime", "Talk Minutes": day_m},
                {"Window": "Evening", "Talk Minutes": eve_m},
                {"Window": "Night-time", "Talk Minutes": night_m},
                {"Window": "International", "Talk Minutes": intl_m},
            ])
            fig_mins = px.bar(
                mins_df, y="Window", x="Talk Minutes", orientation="h",
                color="Window",
                color_discrete_map={
                    "Daytime": "#3B82F6",
                    "Evening": "#F59E0B",
                    "Night-time": "#8B5CF6",
                    "International": "#EC4899"
                },
                text_auto=".2s"
            )
            fig_mins.update_traces(
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Talk Minutes: %{x:,.1f} mins"
            )
            fig_mins.update_layout(
                height=320,
                margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                showlegend=False,
                xaxis=dict(showgrid=True, gridcolor="#F1F5F9", title="Total Talk Minutes"),
                yaxis=dict(showgrid=False, title="")
            )
            st.plotly_chart(fig_mins, use_container_width=True)

        with c_tr4:
            st.markdown("##### 🎯 Customer Service Friction vs Account Risk Status")
            if "CustServ Calls" in filtered_raw_df.columns and "is_fraud" in filtered_raw_df.columns:
                cs_comp = filtered_raw_df.groupby("is_fraud")[["CustServ Calls", "Day Mins"]].mean().reset_index()
                cs_comp["Account Status"] = cs_comp["is_fraud"].map({0: "Normal Accounts", 1: "High-Risk / Churn"})
                fig_cs = px.bar(
                    cs_comp, x="Account Status", y="CustServ Calls",
                    color="Account Status",
                    color_discrete_map={"Normal Accounts": "#10B981", "High-Risk / Churn": "#EF4444"},
                    text_auto=".2f",
                    labels={"CustServ Calls": "Avg Complaints"}
                )
                fig_cs.update_traces(
                    textposition="outside",
                    hovertemplate="<b>%{x}</b><br>Avg Customer Service Inquiries: %{y:.2f}"
                )
                fig_cs.update_layout(
                    height=320,
                    margin=dict(l=10, r=10, t=10, b=10),
                    plot_bgcolor="#FFFFFF",
                    paper_bgcolor="#FFFFFF",
                    showlegend=False,
                    xaxis=dict(showgrid=False, title=""),
                    yaxis=dict(showgrid=True, gridcolor="#F1F5F9", title="Avg Customer Service Inquiries")
                )
                st.plotly_chart(fig_cs, use_container_width=True)
            else:
                st.info("Customer Service data not present in current schema.")

    else:
        # ── Event-Level CDR Dataset Mode (Timestamps & Tower IDs) ──
        c_tr1, c_tr2 = st.columns(2)
        with c_tr1:
            st.markdown("##### 🕐 Hourly Traffic Distribution (Peak vs Off-Peak)")
            hourly_counts = filtered_raw_df.groupby("hour").size().reset_index(name="Calls")
            fig_hourly = px.bar(
                hourly_counts, x="hour", y="Calls",
                color_discrete_sequence=["#2563EB"],
                labels={"hour": "Hour of Day (24h)", "Calls": "Total Calls"}
            )
            fig_hourly.update_layout(
                height=320,
                margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                xaxis=dict(dtick=2, showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="#F1F5F9")
            )
            st.plotly_chart(fig_hourly, use_container_width=True)

        with c_tr2:
            st.markdown("##### ⏱ Call Duration Distribution")
            dur_cap = filtered_raw_df["duration_sec"].quantile(0.99)
            clipped_dur = filtered_raw_df[filtered_raw_df["duration_sec"] <= dur_cap]
            fig_dur = px.histogram(
                clipped_dur, x="duration_sec", color="call_type",
                nbins=35, opacity=0.8,
                color_discrete_map={"voice": "#3B82F6", "sms": "#10B981", "data": "#F59E0B"},
                labels={"duration_sec": "Duration (Seconds)"}
            )
            fig_dur.update_layout(
                height=320,
                margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                legend=dict(title=""),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="#F1F5F9")
            )
            st.plotly_chart(fig_dur, use_container_width=True)

        c_tr3, c_tr4 = st.columns(2)
        with c_tr3:
            st.markdown("##### 🗓 Weekly Traffic Density (Day of Week × Hour)")
            hmap_data = filtered_raw_df.groupby(["dow", "hour"]).size().reset_index(name="Volume")
            hmap_pivot = hmap_data.pivot(index="dow", columns="hour", values="Volume").fillna(0)
            hmap_pivot.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][:len(hmap_pivot)]
            fig_hmap = px.imshow(
                hmap_pivot,
                color_continuous_scale="Blues",
                aspect="auto",
                labels=dict(x="Hour of Day", y="", color="Calls")
            )
            fig_hmap.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="#FFFFFF"
            )
            st.plotly_chart(fig_hmap, use_container_width=True)

        with c_tr4:
            st.markdown("##### 💰 Total Billed Revenue by Service Type")
            rev_by_type = filtered_raw_df.groupby("call_type")["charge_usd"].sum().reset_index()
            rev_by_type.columns = ["Call Type", "Revenue"]
            fig_rev = px.bar(
                rev_by_type, x="Call Type", y="Revenue",
                color="Call Type",
                color_discrete_map={"voice": "#3B82F6", "sms": "#10B981", "data": "#F59E0B"},
                text_auto="$.2f"
            )
            fig_rev.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                showlegend=False,
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="#F1F5F9")
            )
            fig_rev.update_traces(textposition="outside")
            st.plotly_chart(fig_rev, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: CUSTOMER SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_segments:
    st.markdown('<div class="panel-header"><div class="panel-title">👥 Unsupervised Behavioral Segmentation (K-Means + PCA)</div></div>', unsafe_allow_html=True)
    st.caption("Subscribers are categorized into 5 behavioral clusters based on 19 multidimensional metrics, reinforced by XGBoost feature importance.")

    # Dynamic Segment Cards
    profile_cols = [
        "total_calls", "avg_duration_min", "pct_night_calls",
        "pct_international", "pct_roaming", "total_charge_usd", "is_fraud"
    ]
    seg_summary = sub_df.groupby("segment_name")[profile_cols].mean().round(2)
    seg_summary["subscribers"] = sub_df.groupby("segment_name").size()
    seg_summary["fraud_rate"] = (seg_summary["is_fraud"] * 100).round(1)

    icon_dict = {
        "Suspicious / High-Risk": ("🚨", "#EF4444"),
        "International Travellers": ("🌍", "#3B82F6"),
        "Night-Shift / Off-Peak": ("🌙", "#8B5CF6"),
        "Heavy Callers": ("📞", "#10B981"),
        "Standard / Casual Users": ("👤", "#64748B"),
    }

    cols_seg = st.columns(len(seg_summary))
    for idx, (sname, srow) in enumerate(seg_summary.iterrows()):
        icon, color = icon_dict.get(sname, ("📊", "#3B82F6"))
        with cols_seg[idx]:
            st.markdown(f"""
            <div class="segment-card" style="border-top: 3px solid {color};">
                <div>
                    <div class="segment-badge">{icon}</div>
                    <div class="segment-name" style="color:{color};">{sname}</div>
                </div>
                <div>
                    <div class="segment-stat-row"><span>Subscribers:</span><b>{int(srow['subscribers'])}</b></div>
                    <div class="segment-stat-row"><span>Avg Calls:</span><b>{srow['total_calls']:.0f}</b></div>
                    <div class="segment-stat-row"><span>Avg Duration:</span><b>{srow['avg_duration_min']:.1f} m</b></div>
                    <div class="segment-stat-row"><span>Intl Ratio:</span><b>{srow['pct_international']*100:.0f}%</b></div>
                    <div class="segment-stat-row"><span>Fraud Rate:</span><b style="color:{color};">{srow['fraud_rate']:.1f}%</b></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    col_pca, col_bar = st.columns([7, 5])
    with col_pca:
        st.markdown("##### 🗺 2-D PCA Projection of Heterogeneous Subscriber Space")
        fig_pca = px.scatter(
            sub_df, x="pca_x", y="pca_y",
            color="segment_name",
            symbol=sub_df["is_fraud"].map({0: "circle", 1: "x"}),
            hover_data={
                "caller_id": True,
                "total_calls": True,
                "pct_international": ":.1%",
                "total_charge_usd": ":.2f",
                "fraud_probability": ":.1%",
                "pca_x": False, "pca_y": False,
            },
            color_discrete_sequence=px.colors.qualitative.Bold,
            opacity=0.8,
            labels={"pca_x": "Principal Component 1", "pca_y": "Principal Component 2", "segment_name": "Cluster"}
        )
        fig_pca.update_traces(marker=dict(size=7))
        fig_pca.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            legend=dict(orientation="h", yanchor="top", y=-0.12),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
        )
        st.plotly_chart(fig_pca, use_container_width=True)
        st.caption("● Normal Subscribers &nbsp;·&nbsp; ❌ Verified Fraudsters")
        
    with col_bar:
        st.markdown("##### 🚨 Fraud Concentration by Cluster")
        fig_fr_bar = px.bar(
            seg_summary.reset_index(),
            x="segment_name", y="fraud_rate",
            color="fraud_rate",
            color_continuous_scale="Reds",
            text_auto=".1f",
            labels={"segment_name": "", "fraud_rate": "Fraud Rate (%)"}
        )
        fig_fr_bar.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            coloraxis_showscale=False,
            xaxis=dict(showgrid=False, tickangle=-15),
            yaxis=dict(showgrid=True, gridcolor="#F1F5F9")
        )
        fig_fr_bar.update_traces(textposition="outside")
        st.plotly_chart(fig_fr_bar, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: DISTRIBUTED ARCHITECTURE & BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════
with tab_architecture:
    st.markdown('<div class="panel-header"><div class="panel-title">⚡ Distributed Big Data Architecture & Execution Pipeline</div></div>', unsafe_allow_html=True)
    
    col_arch1, col_arch2 = st.columns(2)
    
    with col_arch1:
        st.markdown("""
        #### 🏛️ Distributed Processing Model
        This platform is engineered to handle **millions of records** without memory overflow:
        
        1. **Dask DataFrame Engine (`preprocessing.py`)**:
           - Splits raw CSV/Parquet files across **4 parallel partitions**.
           - Processes out-of-core temporal feature transformations.
           - Parallelizes groupby aggregations per subscriber.
           
        2. **Histogram-Based XGBoost (`xgb_classifier.py`)**:
           - Configured with `tree_method="hist"`, which discretizes continuous features into bins.
           - Enables parallel multi-threaded training across CPU cores (`n_jobs=-1`).
           - Incorporates **unsupervised cluster IDs** into the supervised decision tree boundaries.
           
        3. **Enterprise Apache Spark Script (`spark_pipeline.py`)**:
           - Production PySpark implementation ready for Databricks, YARN, or AWS EMR clusters.
           - Employs `pyspark.ml.clustering.KMeans` and `xgboost.spark.SparkXGBClassifier`.
        """)
        
    with col_arch2:
        st.markdown("#### ⏱ Pipeline Execution Time Benchmark")
        bench_path = os.path.join(REPORT_DIR, "benchmark_results.csv")
        if os.path.exists(bench_path):
            bench_df = pd.read_csv(bench_path)
            st.dataframe(
                bench_df,
                column_config={
                    "size": "Size",
                    "n_records": st.column_config.NumberColumn("Records", format="%d"),
                    "t_preprocess_s": st.column_config.NumberColumn("Dask Preprocess", format="%.2fs"),
                    "t_kmeans_s": st.column_config.NumberColumn("K-Means", format="%.2fs"),
                    "t_xgb_s": st.column_config.NumberColumn("XGBoost", format="%.2fs"),
                    "t_total_s": st.column_config.NumberColumn("Total Time", format="%.2fs"),
                    "xgb_accuracy": st.column_config.NumberColumn("Accuracy", format="%.1%"),
                    "xgb_roc_auc": st.column_config.NumberColumn("ROC-AUC", format="%.3f"),
                },
                use_container_width=True
            )
        else:
            st.info("Run `python run_pipeline.py --size small` to generate benchmark stats.")

    st.markdown("---")
    st.markdown("##### 📌 Title to Code Alignment Reference")
    st.table(pd.DataFrame([
        {"Title Component": "Scalable", "Mechanism": "Small (10K), Medium (50K), Large (200K) dataset scalability"},
        {"Title Component": "Heterogeneous Big Data", "Mechanism": "Ingests mixed types (timestamps, categoricals, tower coordinates, billing USD)"},
        {"Title Component": "Distributed Computing", "Mechanism": "Dask multi-partition processing + Spark MLlib module (spark_pipeline.py)"},
        {"Title Component": "Clustering", "Mechanism": "K-Means++ with Elbow & Silhouette analysis + Dynamic behavioral naming"},
        {"Title Component": "Ensemble Gradient-Boosting", "Mechanism": "XGBoost with histogram tree binning, scale_pos_weight & cluster tags"},
        {"Title Component": "Telecom CDR Analysis", "Mechanism": "Revenue monitoring, call type ratio, peak load management, and fraud mitigation"},
    ]))
