import streamlit as st
import pandas as pd
import json
import time
import html
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple, Optional
import os

from inference import run_inference, InvalidTokopediaURLError, ReviewsNotFoundError, ScraperUnavailableError
from utils import model_weights_available

# ======================================================================
# 1. KONFIGURASI HALAMAN
# ======================================================================
st.set_page_config(
    page_title="ABSA - Aspect-Based Sentiment Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================================
# 2. SESSION STATE INITIALIZATION
# ======================================================================
if "input_mode" not in st.session_state:
    st.session_state.input_mode = "Teks"
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

# ======================================================================
# 3. CSS KUSTOM (Identik dengan app.py + Custom Result Styles)
# ======================================================================
st.markdown("""
<style>
:root {
    --bg-main: #131314;
    --bg-sidebar: #1E1F22;
    --text-primary: #E3E3E3;
    --text-muted: #C4C7C5;
    --border-color: #333538;
    --hover-bg: #2A2B2E;
    --color-positive: #10B981;
    --color-neutral: #6B7280;
    --color-negative: #EF4444;
}

/* Base Styles */
.stApp { background-color: var(--bg-main) !important; color: var(--text-primary); }
[data-testid="stSidebar"] { background-color: var(--bg-sidebar) !important; border-right: none; }
h1, h2, h3, p, span { color: var(--text-primary) !important; }
header[data-testid="stHeader"] { background-color: transparent !important; box-shadow: none !important; }

/* ---------- SIDEBAR LAYOUT ---------- */
[data-testid="stSidebarUserContent"] { height: 100vh !important; position: relative !important; display: flex !important; flex-direction: column !important; }
[data-testid="stSidebarUserContent"] > div { gap: 0px !important; padding-top: 15px !important; padding-bottom: 60px !important; }
[data-testid="stSidebar"] [data-testid="stElementContainer"],
[data-testid="stSidebar"] .stElementContainer,
[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div { margin-bottom: 0px !important; padding-bottom: 0px !important; padding-top: 0px !important; }
[data-testid="stSidebar"] [data-testid="stWidgetFormWrapper"],
[data-testid="stSidebar"] [data-testid="stSidebarRadio"] { gap: 0px !important; }

.sidebar-title-icon { display: inline-flex; align-items: center; gap: 8px; font-size: 14px; font-weight: 700; color: var(--text-primary); padding: 4px 8px; margin-top: 6px; }
.sidebar-title-icon svg { flex-shrink: 0; }
.sidebar-hr { border: 0; height: 1px; background-color: var(--border-color); margin: 4px 0px; }
[data-testid="stSidebar"] div[role="radiogroup"] { padding-left: 10px !important; padding-top: 1px !important; padding-bottom: 5px !important; }

[data-testid="stSidebar"] .stButton button { background-color: transparent !important; border: none !important; box-shadow: none !important; padding: 2px 8px !important; color: var(--text-primary) !important; font-size: 14px !important; font-weight: 700 !important; height: auto !important; width: auto !important; display: inline-flex; align-items: center; gap: 8px; margin-bottom: 2px !important; }
[data-testid="stSidebar"] .stButton button:hover { color: var(--text-muted) !important; }

.sidebar-bottom-absolute { position: fixed !important; bottom: 20px !important; left: 0 !important; width: 244px !important; padding-left: 10px !important; background-color: var(--bg-sidebar); z-index: 999999 !important; box-sizing: border-box !important; }
.btn-back-link { display: inline-flex; align-items: center; gap: 8px; font-size: 14px; font-weight: 700; color: var(--text-primary) !important; text-decoration: none !important; padding: 4px 8px; cursor: pointer; transition: color 0.15s; }
.btn-back-link:hover { color: var(--text-muted) !important; }

/* ---------- WELCOME PAGE ---------- */
.main-center-wrapper { text-align: center !important; max-width: 1000px !important; margin: 0 auto !important; padding-top: 40px !important; }
.welcome-text-large { font-size: 26px !important; color: var(--text-muted) !important; font-weight: 500 !important; margin-bottom: 6px !important; letter-spacing: -0.5px !important; }
.judul-skripsi-large { font-size: 40px !important; font-weight: 800 !important; line-height: 1.25 !important; color: var(--text-primary) !important; margin-top: 0px !important; margin-bottom: 28px !important; letter-spacing: -1px !important; }
.identitas-mahasiswa-large { font-size: 18px !important; color: var(--text-muted) !important; font-weight: 400 !important; line-height: 1.65 !important; letter-spacing: -0.2px !important; }

/* ---------- INPUT DOCK STYLES ---------- */
[data-testid="stChatInputContainer"], .st-key-absa_alt_input_dock {
    position: fixed !important;
    bottom: 60px !important; 
    left: auto !important;
    right: auto !important;
    width: calc(100% - 370px) !important; 
    background-color: transparent !important;
    padding: 0px 20px !important; 
    z-index: 99999 !important;
    border: none !important;
    display: flex !important;
    justify-content: center !important; 
}

.st-key-absa_alt_input_dock > div {
    width: 100% !important;
    display: flex !important;
    justify-content: center !important;
}

.st-key-absa_alt_input_dock [data-testid="stForm"] {
    width: 100% !important;
    max-width: var(--st-sizes-content-width, 730px) !important; 
    background-color: #262730 !important; 
    border: 1px solid rgba(250, 250, 250, 0.1) !important;
    border-radius: 8px !important; 
    padding: 6px 10px 6px 16px !important; 
    margin: 0 !important;
    display: flex !important;
    align-items: center !important;
    min-height: 52px !important; 
    box-sizing: border-box !important;
}

.st-key-absa_alt_input_dock [data-testid="stHorizontalBlock"] { 
    align-items: center !important; 
    width: 100% !important; 
    gap: 8px !important; 
}
.st-key-absa_alt_input_dock [data-testid="stHorizontalBlock"] > div:first-child {
    flex-grow: 1 !important;
}

.st-key-absa_alt_input_dock [data-testid="stFileUploader"] { width: 100% !important; }
.st-key-absa_alt_input_dock [data-testid="stFileUploaderDropzone"] {
    background-color: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
    min-height: 40px !important;
    display: flex !important;
    align-items: center !important;
}

.st-key-absa_alt_input_dock [data-testid="stFileUploaderDropzone"] label {
    font-size: 15px !important;
    font-weight: 400 !important;
    padding: 8px 0px !important;
    width: 100% !important;
    display: flex !important;
    justify-content: flex-start !important;
}
.st-key-absa_alt_input_dock [data-testid="stFileUploaderDropzone"] button:hover {
    color: #FFF !important;
    box-shadow: none !important;
}

/* TOMBOL KIRIM SERAGAM */
.st-key-absa_alt_input_dock [data-testid="stFormSubmitButton"] {
    display: flex;
    justify-content: flex-end;
}
.st-key-absa_alt_input_dock [data-testid="stFormSubmitButton"] button {
    background-color: #3B3C44 !important;
    color: #C4C7C5 !important;
    border: none !important;
    border-radius: 6px !important;
    width: 32px !important;
    height: 32px !important;
    min-height: 32px !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
.st-key-absa_alt_input_dock [data-testid="stFormSubmitButton"] button p {
    font-size: 18px !important;
    margin: 0 !important;
    padding-bottom: 2px !important;
}
.st-key-absa_alt_input_dock [data-testid="stFormSubmitButton"] button:hover {
    background-color: #555760 !important;
    color: #FFF !important;
}

/* ========== RESULT PAGE STYLES ========== */
.result-container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 20px;
}

.result-title {
    font-size: 28px;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 30px;
}

.result-header {
    margin-bottom: 24px;
}

.result-header-title {
    font-size: 26px;
    font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.3px;
}

.result-header-subtitle {
    font-size: 14px;
    color: var(--text-muted);
    margin-top: 4px;
}

.summary-row {
    display: flex;
    gap: 20px;
    margin-bottom: 30px;
    width: 100%;
}

.summary-card {
    flex: 1;
    background-color: #262730;
    border: 1px solid rgba(250, 250, 250, 0.1);
    border-radius: 12px;
    padding: 20px;
}

.card-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 15px;
}

.card-label {
    font-size: 14px;
    color: var(--text-muted);
    font-weight: 500;
}

.card-percentage {
    font-size: 28px;
    font-weight: 800;
    color: var(--text-primary);
}

.card-reviews-count {
    font-size: 12px;
    color: var(--text-muted);
    margin-top: 4px;
}

.sentiment-bar {
    width: 100%;
    height: 24px;
    background-color: #1E1F22;
    border-radius: 6px;
    display: flex;
    overflow: hidden;
    border: 1px solid rgba(250, 250, 250, 0.1);
}

.sentiment-bar-segment {
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    font-weight: 600;
    color: white;
    cursor: help;
}

/* ========== CHART SECTION ========== */
.chart-title {
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 15px;
    margin-top: 10px;
}

/* ========== DROPDOWN SECTION ========== */
.aspect-title {
    font-size: 16px;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 15px;
}

.review-item {
    background-color: #1E1F22;
    padding: 10px 12px;
    border-radius: 6px;
    font-size: 13px;
    color: var(--text-muted);
    border-left: 3px solid var(--border-color);
    margin-bottom: 6px;
}

.review-item.positive { border-left-color: var(--color-positive); }
.review-item.neutral { border-left-color: var(--color-neutral); }
.review-item.negative { border-left-color: var(--color-negative); }

[data-testid="stAppViewMainFrame"] {
    display: flex;
    flex-direction: column;
    align-items: center;
}

</style>
""", unsafe_allow_html=True)

# ======================================================================
# 4. UTILITY FUNCTIONS
# ======================================================================

def create_sentiment_bar(positive: int, neutral: int, negative: int, total: int) -> str:
    if total == 0:
        return "<div class='sentiment-bar'></div>"

    pos_pct = (positive / total) * 100
    neu_pct = (neutral / total) * 100
    neg_pct = (negative / total) * 100

    html = '<div class="sentiment-bar">'
    if pos_pct > 0:
        html += f'<div class="sentiment-bar-segment" style="width: {pos_pct}%; background-color: var(--color-positive);" title="Positif: {pos_pct:.1f}%"></div>'
    if neu_pct > 0:
        html += f'<div class="sentiment-bar-segment" style="width: {neu_pct}%; background-color: var(--color-neutral);" title="Netral: {neu_pct:.1f}%"></div>'
    if neg_pct > 0:
        html += f'<div class="sentiment-bar-segment" style="width: {neg_pct}%; background-color: var(--color-negative);" title="Negatif: {neg_pct:.1f}%"></div>'
    html += '</div>'

    return html

# ======================================================================
# 5. SIDEBAR CONFIGURATION
# ======================================================================
with st.sidebar:
    if st.button("Percakapan Baru", icon=":material/chat:"):
        for k in list(st.session_state.keys()):
            if k.startswith("expand_"):
                del st.session_state[k]
        st.session_state.analysis_results = None
        st.rerun()

    st.markdown('<div class="sidebar-title-icon"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path fill-rule="evenodd" d="M1.5 1a.5.5 0 0 1 .5.5v13a.5.5 0 0 1-1 0v-13a.5.5 0 0 1 .5-.5"/><path d="M3 7a1 1 0 0 1 1-1h10a1 1 0 0 1 1 1v2a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1z"/></svg>Model</div>', unsafe_allow_html=True)
    pilihan_model = st.radio("Pilih Model", options=["IndoBERT", "mBERT"], index=0, label_visibility="collapsed")

    if not model_weights_available(pilihan_model):
        st.caption(f"Bobot model {pilihan_model} belum ditemukan di folder model/.")

    st.markdown('<div class="sidebar-title-icon"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M14 5a1 1 0 0 1 1 1v5a1 1 0 0 1-1 1H2a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1zM2 4a2 2 0 0 0-2 2v5a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V6a2 2 0 0 0-2-2z"/><path d="M13 10.25a.25.25 0 0 1 .25-.25h.5a.25.25 0 0 1 .25.25v.5a.25.25 0 0 1-.25.25h-.5a.25.25 0 0 1-.25-.25z"/></svg>Metode Input</div>', unsafe_allow_html=True)
    mode = st.radio("Mode Input", options=["Teks", "Upload File", "URL Tokopedia"], key="input_mode", label_visibility="collapsed")

    st.markdown('<div class="sidebar-bottom-absolute"><a href="#" class="btn-back-link"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path fill-rule="evenodd" d="M12.5 15a.5.5 0 0 1-.5-.5v-13a.5.5 0 0 1 1 0v13a.5.5 0 0 1-.5.5M10 8a.5.5 0 0 1-.5-.5H3.707l2.147 2.146a.5.5 0 0 1-.708.708l-3-3a.5.5 0 0 1 0-.708l3-3a.5.5 0 1 1 .708.708L3.707 7.5H9.5a.5.5 0 0 1 .5.5"/></svg>Kembali</a></div>', unsafe_allow_html=True)

# ======================================================================
# 6. MAIN CONTENT AREA
# ======================================================================
if st.session_state.analysis_results:
    st.markdown('<div class="result-container">', unsafe_allow_html=True)

    results = st.session_state.analysis_results

    mode_label = html.escape(str(results.get('mode', '')))
    judul_ulasan = html.escape(str(results.get('judul_ulasan', '')))
    nama_model = html.escape(str(results.get('model', '')))
    st.markdown(f'''
    <div class="result-header">
        <div class="result-header-title">ABSA - {nama_model}</div>
        <div class="result-header-subtitle">{mode_label} - {judul_ulasan}</div>
    </div>
    ''', unsafe_allow_html=True)

    # ========== BARIS 1: ASPEK PALING DIBAHAS & PALING DIKELUHKAN ==========
    st.markdown('<div class="summary-row">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        td = results.get('top_discussed', {})
        st.markdown(f'''
        <div class="summary-card">
            <div class="card-header">
                <div>
                    <div class="card-label">Aspek Paling Sering Dibahas</div>
                    <div class="card-reviews-count">{td.get('aspect', 'N/A')}</div>
                </div>
                <div class="card-percentage">{td.get('percentage', 0):.0f}%</div>
            </div>
            {create_sentiment_bar(td.get('positive', 0), td.get('neutral', 0), td.get('negative', 0), td.get('total', 0))}
            <div class="card-reviews-count" style="margin-top: 12px;">Total Ulasan: {td.get('total', 0)}</div>
        </div>
        ''', unsafe_allow_html=True)

    with col2:
        tc = results.get('top_complained', {})
        st.markdown(f'''
        <div class="summary-card">
            <div class="card-header">
                <div>
                    <div class="card-label">Aspek Paling Sering Dikeluhkan</div>
                    <div class="card-reviews-count">{tc.get('aspect', 'N/A')}</div>
                </div>
                <div class="card-percentage">{tc.get('percentage', 0):.0f}%</div>
            </div>
            {create_sentiment_bar(tc.get('positive', 0), tc.get('neutral', 0), tc.get('negative', 0), tc.get('total', 0))}
            <div class="card-reviews-count" style="margin-top: 12px;">Total Ulasan: {tc.get('total', 0)}</div>
        </div>
        ''', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ========== BARIS 2: DISTRIBUSI ASPEK & SENTIMEN (plot langsung, tanpa card) ==========
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="chart-title">Distribusi Aspek</div>', unsafe_allow_html=True)
        aspect_dist = results.get('aspect_distribution', {})
        if aspect_dist:
            fig = go.Figure(data=[go.Pie(
                labels=list(aspect_dist.keys()),
                values=list(aspect_dist.values()),
                hole=0,
                domain=dict(x=[0, 0.62], y=[0.02, 0.98]),
            )])
            fig.update_traces(marker=dict(line=dict(color='#131314', width=2)))
            fig.update_layout(
                showlegend=True,
                legend=dict(orientation='v', yanchor='middle', y=0.5, xanchor='left', x=0.68, font=dict(size=11)),
                margin=dict(l=10, r=10, t=10, b=10),
                height=420,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#E3E3E3', size=12)
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key="pie_aspect_distribution")

    with col2:
        st.markdown('<div class="chart-title">Distribusi Sentimen Total</div>', unsafe_allow_html=True)
        sentiment_dist = results.get('overall_sentiment_distribution', {})
        if sentiment_dist:
            colors_map = {'Positif': '#10B981', 'Netral': '#6B7280', 'Negatif': '#EF4444'}
            colors = [colors_map.get(k, '#6B7280') for k in sentiment_dist.keys()]
            fig = go.Figure(data=[go.Pie(
                labels=list(sentiment_dist.keys()),
                values=list(sentiment_dist.values()),
                hole=0,
                domain=dict(x=[0, 0.62], y=[0.02, 0.98]),
                marker=dict(colors=colors)
            )])
            fig.update_traces(marker=dict(line=dict(color='#131314', width=2)))
            fig.update_layout(
                showlegend=True,
                legend=dict(orientation='v', yanchor='middle', y=0.5, xanchor='left', x=0.68, font=dict(size=11)),
                margin=dict(l=10, r=10, t=10, b=10),
                height=420,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#E3E3E3', size=12)
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key="pie_sentiment_distribution")

    # ========== BARIS 3: SEBARAN SENTIMEN PER ASPEK (plot langsung, tanpa card) ==========
    st.markdown('<div class="chart-title">Sebaran Sentimen per Aspek</div>', unsafe_allow_html=True)

    aspects_data = results.get('aspects_detailed', [])
    if aspects_data:
        aspect_names = [a.get('aspect', 'Unknown') for a in aspects_data]
        positive_counts = [a.get('positive_count', 0) for a in aspects_data]
        neutral_counts = [a.get('neutral_count', 0) for a in aspects_data]
        negative_counts = [a.get('negative_count', 0) for a in aspects_data]

        fig = go.Figure(data=[
            go.Bar(name='Positif', x=aspect_names, y=positive_counts, marker_color='#10B981'),
            go.Bar(name='Netral', x=aspect_names, y=neutral_counts, marker_color='#6B7280'),
            go.Bar(name='Negatif', x=aspect_names, y=negative_counts, marker_color='#EF4444'),
        ])
        fig.update_layout(
            barmode='stack',
            xaxis_title='Aspek',
            yaxis_title='Jumlah Ulasan',
            hovermode='x unified',
            margin=dict(l=0, r=0, t=10, b=0),
            height=520,
            showlegend=True,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(color='#E3E3E3'),
            yaxis=dict(color='#E3E3E3'),
            font=dict(color='#E3E3E3', size=12)
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key="bar_sentiment_per_aspect")

    # ========== BARIS 4: DETAIL ULASAN PER ASPEK ==========
    st.markdown('<h3 style="margin-top: 30px; margin-bottom: 20px;">Detail Ulasan per Aspek</h3>', unsafe_allow_html=True)

    for aspect_idx, aspect in enumerate(aspects_data):
        aspect_name = aspect.get('aspect', 'Unknown')

        with st.expander(f"**{aspect_name}**", expanded=False):
            col_details, col_chart = st.columns([3, 1])

            with col_details:
                for field_key, label, css_class in [
                    ("positive_reviews", "Positif", "positive"),
                    ("neutral_reviews", "Netral", "neutral"),
                    ("negative_reviews", "Negatif", "negative"),
                ]:
                    reviews_all = aspect.get(field_key, [])
                    count_key = field_key.replace("_reviews", "_count")
                    st.markdown(f"**{label}** ({aspect.get(count_key, 0)} ulasan)")

                    toggle_key = f"expand_{aspect_idx}_{field_key}"
                    if toggle_key not in st.session_state:
                        st.session_state[toggle_key] = False

                    shown = reviews_all if st.session_state[toggle_key] else reviews_all[:3]
                    for review in shown:
                        st.markdown(f'<div class="review-item {css_class}">{review}</div>', unsafe_allow_html=True)

                    if len(reviews_all) > 3:
                        if st.session_state[toggle_key]:
                            if st.button("Sembunyikan", key=f"btn_{toggle_key}"):
                                st.session_state[toggle_key] = False
                                st.rerun()
                        else:
                            if st.button(f"Lihat selengkapnya ({len(reviews_all) - 3} lainnya)", key=f"btn_{toggle_key}"):
                                st.session_state[toggle_key] = True
                                st.rerun()

                    st.markdown("")

            with col_chart:
                sentiment_data = {
                    'Positif': aspect.get('positive_count', 0),
                    'Netral': aspect.get('neutral_count', 0),
                    'Negatif': aspect.get('negative_count', 0),
                }
                fig = go.Figure(data=[go.Pie(
                    labels=list(sentiment_data.keys()),
                    values=list(sentiment_data.values()),
                    hole=0.4,
                    marker=dict(colors=['#10B981', '#6B7280', '#EF4444'])
                )])
                fig.update_traces(marker=dict(line=dict(color='#262730', width=2)))
                fig.update_layout(
                    showlegend=True,
                    margin=dict(l=0, r=0, t=0, b=0),
                    height=300,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#E3E3E3', size=10)
                )
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key=f"donut_aspect_{aspect_idx}")

    # ========== TUTUP RESULT CONTAINER ==========
    st.markdown('</div>', unsafe_allow_html=True)

else:
    with st.container():
        st.markdown('<div class="main-center-wrapper">', unsafe_allow_html=True)
        st.markdown('<p class="welcome-text-large">Selamat Datang</p>', unsafe_allow_html=True)
        st.markdown('<h1 class="judul-skripsi-large">IMPLEMENTASI ASPECT-BASED SENTIMENT ANALYSIS (ABSA) BERBASIS INDOBERT UNTUK PEMETAAN KELUHAN PADA ULASAN PRODUK</h1>', unsafe_allow_html=True)
        st.markdown('<div class="identitas-mahasiswa-large">Louders Yoakim Telaumbanua - 4223250023</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ======================================================================
# 7. INPUT DOCK LOGIC
# ======================================================================
file_terupload_list = []
pesan_user = None
url_terkirim = None

if st.session_state.input_mode == "Teks":
    pesan_user = st.chat_input("Masukkan ulasan teks di sini...", key="teks_input")

elif st.session_state.input_mode == "URL Tokopedia":
    url_input = st.chat_input("Tempel link/URL produk Tokopedia di sini...", key="url_input")
    if url_input:
        pesan_user = url_input
        url_terkirim = url_input

elif st.session_state.input_mode == "Upload File":
    with st.container(key="absa_alt_input_dock"):
        with st.form(key="form_upload_absa", clear_on_submit=True, border=False):
            col_field, col_submit = st.columns([15, 1])
            with col_field:
                file_terupload = st.file_uploader(
                    "Upload File",
                    type=["csv", "txt", "xlsx"],
                    accept_multiple_files=True,
                    label_visibility="collapsed",
                    key="uploader_absa"
                )
            with col_submit:
                kirim = st.form_submit_button("↑")

            if kirim and file_terupload:
                file_terupload_list = file_terupload
                nama_files = [f.name for f in file_terupload]
                pesan_user = f"[File dilampirkan: {', '.join(nama_files)}]"

# ======================================================================
# 8. PEMROSESAN PESAN DAN PEMANGGILAN INFERENCE
# ======================================================================
if pesan_user:
    if st.session_state.input_mode == "Teks":
        data_untuk_inference = pesan_user
    elif st.session_state.input_mode == "URL Tokopedia":
        data_untuk_inference = url_terkirim
    else:
        data_untuk_inference = file_terupload_list

    with st.status("Menyiapkan data ulasan", expanded=True) as status:
        def progress(msg: str) -> None:
            status.update(label=msg)
            st.write(msg)
            time.sleep(1)

        try:
            hasil = run_inference(
                input_data=data_untuk_inference,
                model_type=pilihan_model,
                mode=st.session_state.input_mode,
                progress=progress,
            )
            status.update(label="Analisis selesai, berikut hasilnya", state="complete")
            st.session_state.analysis_results = hasil

        except InvalidTokopediaURLError:
            status.update(label="URL tidak valid", state="error")
            st.error("URL yang diinput tidak valid.")
        except ReviewsNotFoundError:
            status.update(label="Ulasan tidak ditemukan", state="error")
            st.error("Mohon maaf, ulasan tidak berhasil didapatkan.")
        except ScraperUnavailableError as e:
            status.update(label="Layanan scraping tidak tersedia", state="error")
            st.error(str(e))
        except FileNotFoundError:
            status.update(label="Bobot model tidak ditemukan", state="error")
            st.error(f"Bobot model {pilihan_model} belum tersedia di folder model/.")
        except ValueError as e:
            status.update(label="Input tidak valid", state="error")
            st.error(str(e))
        except Exception as e:
            status.update(label="Analisis gagal", state="error")
            st.error(f"Terjadi kesalahan saat memproses ulasan: {e}")

    if st.session_state.analysis_results is not None:
        st.rerun()

# ======================================================================
# 9. FOOTER
# ======================================================================
st.markdown("""
<style>
.footer {
    text-align: center;
    color: var(--text-muted);
    font-size: 12px;
    padding: 20px;
    border-top: 1px solid var(--border-color);
    margin-top: 40px;
}
</style>
<div class="footer">
</div>
""", unsafe_allow_html=True)